"""Parallel candidate verification on the kappa_path delta (reduced two-step)
branch.

``DiscretePathEnum._run_reduced`` fans the per-candidate dReal check across a
worker pool when ``[common]`` parallel is enabled, while proposal, the linear
feasibility screen and block bookkeeping stay on the calling thread. Feasible
candidates are proposed ahead under a provisional pair block and their checks
run on the pool; results commit in proposal order, a pooled word adds its radius
block, and the radius separation is re-checked at commit. These tests drive the
loop with a scripted structure search and scripted dReal verifiers (no dReal, no
benchmark model), and pin the properties a wider pool must preserve:

* a serial pool (parallel-core = 1) and a wide pool reach the identical pooled
  words, in the identical order, with the identical witness per word, and the
  identical verdict, whatever order the checks complete in -- the scripted
  search decides exclusion deterministically from the installed blocks, so it
  isolates the pipeline logic from the real scenario solver's z3 model choice
  (which is what reorders a real pooling depth, and is accepted);
* a word is pooled from its EARLIEST-proposed structure and a later sibling
  sharing the word is dropped at commit, even when it was proposed and verified
  before the earlier word committed (a word-keyed check alone would not expose
  the distinct witnesses);
* a radius ball established at a commit excludes a word proposed before the
  block landed (the commit-time separation check);
* an infeasible word is screened before any check is dispatched;
* a verification UNKNOWN leaves the depth unresolved rather than a false absence.
"""
import os
import random
import re
import threading
import time

import pytest
import z3

from stlmc.constraints.constraints import (
    Bool,
    Eq,
    Real,
    RealVal,
)
from stlmc.generation import pathenum as _pathenum
from stlmc.generation.oracle import SAT, UNKNOWN, UNSAT
from stlmc.generation.pathenum import DiscretePathEnum
from stlmc.solver.z3 import z3Obj

_SID = Real("sid")


@pytest.fixture(autouse=True)
def _ample_cpus(monkeypatch):
    # _path_verify_workers caps a requested pool at the cpu count; give the tests
    # plenty so a requested width is not shrunk. The cap itself is exercised in
    # the _path_verify_workers tests, which set their own cpu count.
    monkeypatch.setattr(os, "cpu_count", lambda: 64)


# --------------------------------------------------------------------------- #
#  scripted config / printer / encoder
# --------------------------------------------------------------------------- #
class _Section:
    def __init__(self, values):
        self._values = values

    def is_argument_in(self, key):
        return key in self._values

    def get_value(self, key):
        return self._values[key]


class _Config:
    """A [gen] section plus an optional [common] section (parallel-core)."""
    def __init__(self, common=None, **gen):
        self._gen = _Section({k.replace("_", "-"): v for k, v in gen.items()})
        self._common = None if common is None else _Section(
            {k.replace("_", "-"): v for k, v in common.items()})

    def is_section_in(self, name):
        return name == "gen" or (name == "common" and self._common is not None)

    def get_section(self, name):
        if name == "gen":
            return self._gen
        if name == "common" and self._common is not None:
            return self._common
        raise KeyError(name)


class _SilentPrinter:
    def __getattr__(self, _):
        return lambda *a, **k: None


class _StubEncoder:
    """enumerate_components_at returns the depth, which the scripted
    _make_reduced_search uses to select that depth's script; reset is a no-op."""
    model = None

    def enumerate_components_at(self, depth):
        return depth

    def reset(self):
        pass


# --------------------------------------------------------------------------- #
#  scripted structure search
# --------------------------------------------------------------------------- #
class _StructSearch:
    """A scenario-search stand-in driven by an explicit list of structures.

    Each structure is ``{"sid": int, "word": tuple[int, ...]}``. ``sid`` gives a
    distinct ``path_const`` (``sid == i``) so a pair block excludes exactly that
    structure, while two structures may share a ``word``. ``propose`` returns the
    first structure in list order that the installed blocks still admit, deciding
    admission with real z3 over the actual block formulas -- so pair blocks,
    word blocks and radius balls behave exactly as on the real solver, and a
    fresh search built from replayed blocks resumes identically.
    """
    def __init__(self, script):
        self._script = list(script)
        self._blocks = []
        self._v = None

    def _pin(self, s):
        pin = [z3Obj(Eq(_SID, RealVal(str(s["sid"]))))]
        for k, w in enumerate(s["word"]):
            pin.append(z3Obj(Eq(Real(f"currentMode_{k}"), RealVal(str(w)))))
        return pin

    def _admitted(self, s):
        solver = z3.Solver()
        for c in self._pin(s):
            solver.add(c)
        for b in self._blocks:
            solver.add(z3Obj(b))
        return solver.check() == z3.sat

    def propose(self):
        for s in self._script:
            if self._admitted(s):
                self._v = SAT
                total = Bool(f"STRUCT_{s['sid']}")
                path = Eq(_SID, RealVal(str(s["sid"])))
                assn = {_SID: RealVal(str(s["sid"]))}
                for k, w in enumerate(s["word"]):
                    assn[Real(f"currentMode_{k}")] = RealVal(str(w))
                return total, path, assn
        self._v = UNSAT
        return None

    def add_block(self, formula):
        self._blocks.append(formula)

    def last_verdict(self):
        return self._v


# --------------------------------------------------------------------------- #
#  scripted dReal verifier (make_oracle seam), keyed by STRUCTURE
# --------------------------------------------------------------------------- #
_STRUCT_RE = re.compile(r"STRUCT_(\d+)")
_MODE_RE = re.compile(r"currentMode_(\d+) = ([0-9.]+)")


class _Verifier:
    """A per-candidate verifier. Reads the structure id from total_const and the
    word from the pinned modes, and answers by structure so distinct structures
    sharing a word return distinct witnesses. ``spec`` maps sid ->
    (verdict, marker); the SAT witness carries the word plus a ``wmark`` value so
    a test can tell WHICH structure was pooled."""
    def __init__(self, spec, delays, ledger, drop=(), wrong=None, block=()):
        self._spec = spec
        self._delays = delays
        self._ledger = ledger
        self._drop = set(drop)
        # sid -> a complete integral word (dotted) the SAT witness spells INSTEAD
        # of the pinned one, to exercise the pinned-word check.
        self._wrong = dict(wrong or {})
        # sids whose check() blocks until terminate() releases it, to exercise
        # the in-flight kill a cap or run-end performs.
        self._block = set(block)
        self._released = threading.Event()
        self._sid = None
        self._word = None
        self.budgets = []

    def set_budget(self, seconds):
        self.budgets.append(seconds)

    def assert_(self, formula):
        s = str(formula)
        m = _STRUCT_RE.search(s)
        if m:
            self._sid = int(m.group(1))
        pins = _MODE_RE.findall(s)
        if pins:
            self._word = [(int(k), v) for k, v in pins]

    def check(self):
        led = self._ledger
        with led["lock"]:
            led["active"] += 1
            led["max_active"] = max(led["max_active"], led["active"])
        try:
            if self._sid in self._block:
                # A real dReal check cannot be cancelled once running; this
                # scripted stand-in only returns when terminate() releases it.
                # The safety timeout keeps a broken hook a slow return, not a
                # hung test.
                self._released.wait(timeout=10.0)
            delay = self._delays.get(self._sid, 0.0)
            if delay:
                time.sleep(delay)
            with led["lock"]:
                led["checked"].append(self._sid)
            return self._spec[self._sid][0]
        finally:
            with led["lock"]:
                led["active"] -= 1

    def terminate(self):
        with self._ledger["lock"]:
            self._ledger["terminated"].append(self._sid)
        self._released.set()

    def model(self):
        if self._sid in self._drop:
            return {}
        if self._sid in self._wrong:
            src = list(enumerate(self._wrong[self._sid].split(".")))
        else:
            src = sorted(self._word, key=lambda p: p[0])
        d = {}
        for k, v in src:
            d[Real(f"currentMode_{k}")] = RealVal(v)
        d[Real("wmark")] = RealVal(str(self._spec[self._sid][1]))
        return d

    def unknown_reason(self):
        return "scripted"


def _new_ledger():
    return {"lock": threading.Lock(), "checked": [], "active": 0,
            "max_active": 0, "terminated": []}


# --------------------------------------------------------------------------- #
#  harness
# --------------------------------------------------------------------------- #
def _extract(pool):
    """(word, marker) per pooled counterexample, in pool order."""
    out = []
    for p in pool:
        steps = sorted(
            (int(re.match(r"currentMode_(\d+)", v.id).group(1)), c.value)
            for v, c in p.items() if re.match(r"currentMode_(\d+)$", v.id))
        word = ".".join(c for _, c in steps)
        mark = next((c.value for v, c in p.items() if v.id == "wmark"), None)
        out.append((word, mark))
    return out


def _run(scripts, spec, *, workers=None, delays=None, radius=0, per_depth=64,
         depths=(0,), drop=(), wrong=None, block=(), budget=None,
         filter_infeasible=None, monkeypatch=None):
    if isinstance(scripts, list):
        scripts = {depths[0]: scripts}
    dmap = dict(delays or {})
    ledger = _new_ledger()

    def factory(**kwargs):
        return _Verifier(spec, dmap, ledger, drop=drop, wrong=wrong, block=block)

    monkeypatch.setattr(_pathenum, "make_oracle", factory)
    if filter_infeasible is not None:
        class _Filter:
            def __init__(self, model, tau_max, horizon, cache=None):
                pass

            def word_is_infeasible(self, depth, mode_seq):
                return tuple(mode_seq) in filter_infeasible
        monkeypatch.setattr(_pathenum, "LinearWordFeasibilityFilter", _Filter)

    class _Alg(DiscretePathEnum):
        def _make_reduced_search(self, components, model, seed, timeout_ms):
            return _StructSearch(scripts[components])

    common = (None if workers is None
              else {"parallel": "true", "parallel-core": str(workers)})
    gen = {}
    if budget is not None:
        gen["pivot_budget"] = str(budget)
    cfg = _Config(common=common, **gen)
    alg = _Alg()
    result, _t, first_depth, pool = alg._run_reduced(
        _StubEncoder(), target_depths=list(depths), per_depth=per_depth,
        radius=radius, seed=0, logic="QF_LRA", config=cfg, logger=None,
        printer=_SilentPrinter(), max_depth=max(depths), tau_max=1.0)
    return result, _extract(pool), ledger


# --------------------------------------------------------------------------- #
#  1. serial and a wide pool agree on words, order, witnesses and verdict
# --------------------------------------------------------------------------- #
def _mixed_script():
    # words 0..5; some words carry two structures with distinct markers.
    return [
        {"sid": 0, "word": (0,)},   # UNSAT
        {"sid": 1, "word": (1,)},   # SAT   marker 11
        {"sid": 2, "word": (1,)},   # sibling of word 1 (never reached serially)
        {"sid": 3, "word": (2,)},   # UNKNOWN
        {"sid": 4, "word": (3,)},   # SAT   marker 33
        {"sid": 5, "word": (4,)},   # UNSAT
        {"sid": 6, "word": (5,)},   # SAT   marker 55
    ]


_MIXED_SPEC = {0: (UNSAT, 0), 1: (SAT, 11), 2: (SAT, 22), 3: (UNKNOWN, 0),
               4: (SAT, 33), 5: (UNSAT, 0), 6: (SAT, 55)}


def test_serial_and_wide_pool_agree_on_words_order_and_witness(monkeypatch):
    r1, p1, _ = _run(_mixed_script(), _MIXED_SPEC, workers=1,
                     monkeypatch=monkeypatch)
    for w in (2, 4, 8):
        rn, pn, _ = _run(_mixed_script(), _MIXED_SPEC, workers=w,
                         monkeypatch=monkeypatch)
        assert pn == p1, (w, pn, p1)
        assert rn == r1, (w, rn, r1)
    # word 1 pooled from its earliest structure (marker 11, not 22); an UNKNOWN
    # word leaves the depth unresolved.
    assert p1 == [("1", "11"), ("3", "33"), ("5", "55")]
    assert r1 == "False"


def test_completion_order_does_not_change_the_pool(monkeypatch):
    ref = None
    for delays in ({}, {1: 0.05}, {4: 0.05, 6: 0.02}, {1: 0.06, 2: 0.0, 6: 0.03}):
        _r, p, _ = _run(_mixed_script(), _MIXED_SPEC, workers=6, delays=delays,
                        monkeypatch=monkeypatch)
        ref = p if ref is None else ref
        assert p == ref, (delays, p, ref)


# --------------------------------------------------------------------------- #
#  2. a word is pooled from its earliest-proposed structure, not a sibling
# --------------------------------------------------------------------------- #
def _substitution_script():
    # word "0" is pooled first and its radius-0 ball does NOT cover word "1".
    # Two structures spell word "1" with distinct witnesses; the earliest (sid 1)
    # is the one serial pools. Both siblings are proposed into the window and
    # verified before their in-order commit, so the pool must take sid 1 (which
    # commits first) and drop sid 2 at the commit-time de-duplication check.
    return [
        {"sid": 0, "word": (0,)},
        {"sid": 1, "word": (1,)},
        {"sid": 2, "word": (1,)},
    ]


_SUBST_SPEC = {0: (SAT, 100), 1: (SAT, 1), 2: (SAT, 2)}


def test_pool_takes_the_earliest_structure_not_a_speculative_sibling(monkeypatch):
    # Delay word 0 so it completes LAST though it commits first: the siblings are
    # verified and waiting while it commits.
    r_ser, p_ser, _ = _run(_substitution_script(), _SUBST_SPEC, workers=1,
                           monkeypatch=monkeypatch)
    r_par, p_par, led = _run(_substitution_script(), _SUBST_SPEC, workers=4,
                             delays={0: 0.15}, monkeypatch=monkeypatch)
    assert p_ser == [("0", "100"), ("1", "1")], p_ser
    assert p_par == p_ser, p_par
    # word 1 pooled from sid 1 (marker 1), never sid 2 (marker 2).
    assert ("1", "2") not in p_par
    assert r_par == r_ser == "False"


def test_a_witness_spelling_a_different_word_is_not_pooled(monkeypatch):
    # The candidate pins word (0,0), but its delta-sat witness spells a complete
    # integral (0,1). Pooling it while blocking the pinned word would leave the
    # pool entry and the exclusion disagreeing, so it must be rejected like the
    # schema-incomplete case: not pooled, depth left unresolved.
    script = [{"sid": 0, "word": (0, 0)}]
    spec = {0: (SAT, 5)}
    for w in (1, 4):
        r, p, _ = _run(script, spec, workers=w, depths=(1,),
                       wrong={0: "0.1"}, monkeypatch=monkeypatch)
        assert p == [], (w, p)
        assert r == "Unknown", (w, r)


# --------------------------------------------------------------------------- #
#  3. a radius ball set at a barrier excludes a speculative later word
# --------------------------------------------------------------------------- #
def test_a_radius_ball_from_a_barrier_excludes_a_speculative_word(monkeypatch):
    # depth 1, words of length 2. Word (0,0) is pooled; with radius 1 its ball
    # covers (0,1) at Hamming distance 1. (0,1) is proposed into the window
    # before (0,0) commits, so the barrier's radius block must drop it.
    script = [
        {"sid": 0, "word": (0, 0)},
        {"sid": 1, "word": (0, 1)},
    ]
    spec = {0: (SAT, 7), 1: (SAT, 9)}
    r_ser, p_ser, _ = _run(script, spec, workers=1, radius=1, depths=(1,),
                           monkeypatch=monkeypatch)
    r_par, p_par, _ = _run(script, spec, workers=4, radius=1, depths=(1,),
                           delays={0: 0.1}, monkeypatch=monkeypatch)
    assert p_ser == [("0.0", "7")], p_ser   # (0,1) excluded by the ball
    assert p_par == p_ser, p_par
    assert r_par == r_ser


# --------------------------------------------------------------------------- #
#  3b. the commit-time separation uses the CAPPED radius, not the configured one
# --------------------------------------------------------------------------- #
def test_radius_capped_to_zero_at_depth_0_pools_every_distinct_word(monkeypatch):
    # depth 0: a word has one position, so the block caps radius to 0. Two
    # distinct SAT words must both pool -- the commit check must use the capped
    # radius (0), not the configured 1, which would drop the second word.
    script = [{"sid": 0, "word": (0,)}, {"sid": 1, "word": (1,)}]
    spec = {0: (SAT, 5), 1: (SAT, 6)}
    for w in (1, 4):
        _r, p, _ = _run(script, spec, workers=w, radius=1, depths=(0,),
                        delays={0: 0.05} if w > 1 else None,
                        monkeypatch=monkeypatch)
        assert p == [("0", "5"), ("1", "6")], (w, p)


def test_radius_capped_to_depth_keeps_complementary_words_eligible(monkeypatch):
    # depth 1 (word length 2): the block caps radius to 1. Two complementary
    # words at Hamming distance 2 are outside each other's effective (radius-1)
    # ball, so both must pool even under a configured radius of 2.
    script = [{"sid": 0, "word": (0, 0)}, {"sid": 1, "word": (1, 1)}]
    spec = {0: (SAT, 8), 1: (SAT, 9)}
    for w in (1, 4):
        _r, p, _ = _run(script, spec, workers=w, radius=2, depths=(1,),
                        delays={0: 0.05} if w > 1 else None,
                        monkeypatch=monkeypatch)
        assert sorted(p) == [("0.0", "8"), ("1.1", "9")], (w, p)


# --------------------------------------------------------------------------- #
#  4. multiple SATs inside one window, committed in order
# --------------------------------------------------------------------------- #
def test_multiple_sats_in_one_window_pool_in_proposal_order(monkeypatch):
    script = [{"sid": i, "word": (i,)} for i in range(4)]
    spec = {0: (SAT, 10), 1: (SAT, 20), 2: (SAT, 30), 3: (SAT, 40)}
    r_ser, p_ser, _ = _run(script, spec, workers=1, monkeypatch=monkeypatch)
    r_par, p_par, _ = _run(script, spec, workers=8,
                           delays={0: 0.06, 1: 0.03}, monkeypatch=monkeypatch)
    assert p_ser == [("0", "10"), ("1", "20"), ("2", "30"), ("3", "40")]
    assert p_par == p_ser
    assert r_par == r_ser == "False"


# --------------------------------------------------------------------------- #
#  5. an infeasible word is screened before any check is dispatched
# --------------------------------------------------------------------------- #
def test_an_infeasible_word_is_screened_before_dispatch(monkeypatch):
    script = [{"sid": 0, "word": (0,)}, {"sid": 1, "word": (1,)},
              {"sid": 2, "word": (2,)}]
    spec = {0: (UNSAT, 0), 1: (SAT, 5), 2: (UNSAT, 0)}
    _r, p, led = _run(script, spec, workers=4, filter_infeasible={(1,)},
                      monkeypatch=monkeypatch)
    # word 1 is screened out: it reaches no verifier and is not pooled.
    assert p == []
    assert 1 not in led["checked"]
    assert set(led["checked"]) <= {0, 2}


# --------------------------------------------------------------------------- #
#  6. an UNKNOWN before exhaustion is not a false absence
# --------------------------------------------------------------------------- #
def test_an_unknown_leaves_the_depth_unresolved(monkeypatch):
    script = [{"sid": 0, "word": (0,)}, {"sid": 1, "word": (1,)}]
    spec = {0: (UNKNOWN, 0), 1: (UNSAT, 0)}
    r, p, _ = _run(script, spec, workers=4, monkeypatch=monkeypatch)
    assert p == []
    assert r == "Unknown"


def test_a_fully_refuted_space_is_absence(monkeypatch):
    script = [{"sid": 0, "word": (0,)}, {"sid": 1, "word": (1,)}]
    spec = {0: (UNSAT, 0), 1: (UNSAT, 0)}
    r, p, _ = _run(script, spec, workers=4, monkeypatch=monkeypatch)
    assert p == []
    assert r == "True"


# --------------------------------------------------------------------------- #
#  7. the pipeline actually overlaps checks (not a disguised serial loop)
# --------------------------------------------------------------------------- #
def test_the_pipeline_overlaps_checks(monkeypatch):
    script = [{"sid": i, "word": (i,)} for i in range(8)]
    spec = {i: (UNSAT, 0) for i in range(8)}   # full exhaustion, no pool
    _r, p, led = _run(script, spec, workers=4, delays={i: 0.03 for i in range(8)},
                      monkeypatch=monkeypatch)
    assert p == []
    assert led["max_active"] >= 2, led["max_active"]
    assert set(led["checked"]) == set(range(8))


# --------------------------------------------------------------------------- #
#  8. per-depth k-paths budget under the pool
# --------------------------------------------------------------------------- #
def test_the_k_paths_budget_caps_the_pool_under_a_wide_pool(monkeypatch):
    script = [{"sid": i, "word": (i,)} for i in range(6)]
    spec = {i: (SAT, i * 10) for i in range(6)}
    r_ser, p_ser, _ = _run(script, spec, workers=1, per_depth=2,
                           monkeypatch=monkeypatch)
    r_par, p_par, _ = _run(script, spec, workers=8, per_depth=2,
                           delays={0: 0.05}, monkeypatch=monkeypatch)
    assert p_ser == [("0", "0"), ("1", "10")]
    assert p_par == p_ser
    assert r_ser == r_par   # subset of depths decided -> Unknown


def test_a_cap_terminates_an_in_flight_check_and_returns_promptly(monkeypatch):
    # A speculative check dispatched ahead of the cap is still running when the
    # k-paths budget is met on an earlier word. Future.cancel cannot stop a
    # running check, so the depth must actively terminate it rather than wait out
    # its per-call budget. sid 0 pools and meets per_depth = 1 while sid 1 is
    # blocked mid-check; the cap must terminate sid 1 and the run must return
    # without waiting for it.
    script = [{"sid": 0, "word": (0,)}, {"sid": 1, "word": (1,)}]
    spec = {0: (SAT, 7), 1: (SAT, 8)}
    t0 = time.perf_counter()
    r, p, led = _run(script, spec, workers=2, per_depth=1, block={1},
                     delays={0: 0.1}, monkeypatch=monkeypatch)
    elapsed = time.perf_counter() - t0
    assert p == [("0", "7")]            # sid 0 pooled, cap met
    assert 1 in led["terminated"]       # sid 1's in-flight check was killed
    assert 0 not in led["terminated"]   # sid 0 completed on its own
    assert elapsed < 5.0                # not held for sid 1's safety timeout


def test_an_interrupt_mid_run_terminates_in_flight_checks(monkeypatch):
    # A run interrupted mid-flight (Ctrl+C) must not leave a check running: the
    # run-scoped cleanup terminates every live verifier as it unwinds. A scripted
    # search raises KeyboardInterrupt from the proposing thread once a blocking
    # verifier is registered live; the interrupt must propagate AND that verifier
    # must have been terminated by the finally rather than left to its budget.
    spec = {0: (SAT, 1), 1: (SAT, 2)}
    ledger = _new_ledger()

    def factory(**kwargs):
        return _Verifier(spec, {}, ledger, block={0})

    monkeypatch.setattr(_pathenum, "make_oracle", factory)

    class _Boom(_StructSearch):
        def __init__(self, script):
            super().__init__(script)
            self._calls = 0

        def propose(self):
            self._calls += 1
            if self._calls == 1:
                return super().propose()   # sid 0 dispatched; its check blocks
            # Interrupt only once sid 0 is registered live, so the unwinding
            # cleanup has an in-flight check to terminate.
            t0 = time.perf_counter()
            while ledger["active"] < 1 and time.perf_counter() - t0 < 5.0:
                time.sleep(0.005)
            raise KeyboardInterrupt

    class _Alg(DiscretePathEnum):
        def _make_reduced_search(self, components, model, seed, timeout_ms):
            return _Boom([{"sid": 0, "word": (0,)}, {"sid": 1, "word": (1,)}])

    cfg = _Config(common={"parallel": "true", "parallel-core": "2"})
    alg = _Alg()
    with pytest.raises(KeyboardInterrupt):
        alg._run_reduced(
            _StubEncoder(), target_depths=[0], per_depth=64, radius=0, seed=0,
            logic="QF_LRA", config=cfg, logger=None, printer=_SilentPrinter(),
            max_depth=0, tau_max=1.0)
    assert 0 in ledger["terminated"]   # the in-flight check was killed on unwind


# --------------------------------------------------------------------------- #
#  9. multi-depth agreement
# --------------------------------------------------------------------------- #
def test_k_paths_zero_poses_no_query_and_terminates(monkeypatch):
    # The degenerate [gen] k-paths = 0: the depth is visited, no query is posed,
    # nothing is settled. The pool is empty and the run is Unknown -- and the
    # wide pool must terminate rather than spin with nothing to propose.
    script = [{"sid": 0, "word": (0,)}, {"sid": 1, "word": (1,)}]
    spec = {0: (SAT, 1), 1: (SAT, 2)}
    r, p, led = _run(script, spec, workers=4, per_depth=0, monkeypatch=monkeypatch)
    assert p == []
    assert r == "Unknown"
    assert led["checked"] == []   # no candidate ever verified


def test_multi_depth_serial_and_wide_agree(monkeypatch):
    scripts = {
        0: [{"sid": 0, "word": (0,)}, {"sid": 1, "word": (1,)}],
        1: [{"sid": 2, "word": (0, 0)}, {"sid": 3, "word": (0, 1)}],
    }
    spec = {0: (SAT, 1), 1: (UNSAT, 0), 2: (UNSAT, 0), 3: (SAT, 4)}
    r1, p1, _ = _run(scripts, spec, workers=1, depths=(0, 1),
                     monkeypatch=monkeypatch)
    rn, pn, _ = _run(scripts, spec, workers=6, depths=(0, 1),
                     delays={0: 0.04, 3: 0.02}, monkeypatch=monkeypatch)
    assert pn == p1, (pn, p1)
    assert rn == r1


# --------------------------------------------------------------------------- #
#  10. adversarial determinism stress across worker counts
# --------------------------------------------------------------------------- #
def _random_case(rng):
    n = rng.randint(3, 9)
    words = [(rng.randint(0, 3),) for _ in range(n)]
    script = [{"sid": i, "word": words[i]} for i in range(n)]
    spec = {}
    for i in range(n):
        roll = rng.random()
        v = SAT if roll < 0.5 else (UNSAT if roll < 0.85 else UNKNOWN)
        spec[i] = (v, i * 7 + 1)
    delays = {i: rng.choice([0.0, 0.0, 0.01, 0.03]) for i in range(n)}
    return script, spec, delays


def test_adversarial_determinism_stress(monkeypatch):
    rng = random.Random(20260830)
    for _ in range(60):
        script, spec, delays = _random_case(rng)
        r_ref, p_ref, _ = _run([dict(s) for s in script], spec, workers=1,
                               monkeypatch=monkeypatch)
        for w in (2, 3, 5, 8):
            r, p, _ = _run([dict(s) for s in script], spec, workers=w,
                           delays=delays, monkeypatch=monkeypatch)
            assert p == p_ref, (w, script, spec, p, p_ref)
            assert r == r_ref, (w, script, spec, r, r_ref)


# --------------------------------------------------------------------------- #
#  11. the worker-count policy (mirrors the sibling strategy's contract)
# --------------------------------------------------------------------------- #
def _workers_for(common):
    return DiscretePathEnum()._path_verify_workers(_Config(common=common))


def test_verify_workers_honours_the_parallel_switch(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert _workers_for({"parallel": "true", "parallel-core": "1"}) == 1
    assert _workers_for({"parallel": "true", "parallel-core": "3"}) == 3
    assert _workers_for({"parallel": "false", "parallel-core": "4"}) == 1
    assert _workers_for({"parallel": "true"}) == 1
    assert _workers_for(None) == 1


def test_verify_workers_defaults_safe_but_honours_an_explicit_request(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert _workers_for({"parallel": "true", "parallel-core": "25"}) == 2
    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    assert _workers_for({"parallel": "true", "parallel-core": "25"}) == 5
    assert _workers_for({"parallel": "true", "parallel-core": "2"}) == 2
    assert _workers_for({"parallel": "true", "parallel-core": "10"}) == 10