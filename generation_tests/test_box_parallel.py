"""Parallel candidate verification in kappa_box's two-step pivot search.

The proposal step of ``_pivot_two_step`` is sequential; the per-candidate dReal
check is fanned out across a pool sized by ``[common]`` parallel-core. These
tests drive the loop with a scripted pivot search and scripted candidate oracles
(no dReal, no model), and pin the properties a wider pool must not disturb:

* the earliest-PROPOSED satisfiable candidate is accepted, whatever order the
  checks complete in;
* only the proposals up to the accepted one count toward the metrics and the
  undecided tally, so a candidate proposed past the acceptance point cannot
  perturb the per-depth accounting;
* a serial pool (parallel-core = 1) and a wide pool reach the identical accepted
  candidate and the identical counters on the same scripted sequence;
* an infeasible word is filtered before it is dispatched;
* the accepted candidate's oracle is returned unbudgeted with the pivot-binding
  blocks popped, ready for growth.

Verdicts are keyed to a candidate's identity (its total_const), not to the order
the oracles are constructed, so completion order can be shuffled with per-word
delays without mispairing a verdict.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from stlmc.constraints.constraints import (
    And,
    Bool,
    Eq,
    Not,
    Real,
    RealVal,
)
from stlmc.generation import box as _box
from stlmc.generation.box import RegionBoxDiscovery, _Counter
from stlmc.generation.oracle import SAT, UNKNOWN, UNSAT


class _SilentPrinter:
    def __getattr__(self, _):
        return lambda *a, **k: None


class _Section:
    def __init__(self, values):
        self._values = values

    def is_argument_in(self, key):
        return key in self._values

    def get_value(self, key):
        return self._values[key]


class _Config:
    """A [gen] section plus an optional [common] section carrying parallel-core."""

    def __init__(self, common=None, **gen):
        self._gen = _Section({k.replace("_", "-"): v for k, v in gen.items()})
        self._common = (
            None
            if common is None
            else _Section({k.replace("_", "-"): v for k, v in common.items()})
        )

    def is_section_in(self, name):
        return name == "gen" or (name == "common" and self._common is not None)

    def get_section(self, name):
        if name == "gen":
            return self._gen
        if name == "common" and self._common is not None:
            return self._common
        raise KeyError(name)


class _Encoding:
    bound = 0


def _tag(word):
    return "CAND_" + "_".join(str(d) for d in word)


def _new_ledger():
    return {
        "lock": threading.Lock(),
        "checked": [],
        "built": [],
        "active": 0,
        "max_active": 0,
    }


@pytest.fixture(autouse=True)
def _ample_cpus(monkeypatch):
    # The scripted batching logic is machine-independent, but _verify_workers
    # caps the pool at the cpu count. Give the tests plenty of cpus so a
    # requested batch width is not shrunk below what the scenario needs; the cap
    # itself is exercised explicitly in the _verify_workers tests, which set
    # their own cpu count.
    monkeypatch.setattr(os, "cpu_count", lambda: 64)


class _PivotSearch:
    """A ReducedPivotSearch stand-in whose pivots are given.

    Each ``next`` yields a distinct ``total_const`` (tagged with the word) and
    blocks it structure-wide, as the real ``next`` does, then ``None`` once
    spent. ``model`` is None by default, which disables the feasibility filter;
    a test that exercises the filter sets it truthy and patches the filter."""

    def __init__(self, words, exhausted=UNSAT):
        self._words = list(words)
        self._i = 0
        self._exhausted = exhausted
        self.blocks = []
        self.model = None
        self.tau_max = 1.0

    def add_block(self, formula):
        self.blocks.append(formula)

    def last_verdict(self):
        return self._exhausted

    def next(self):
        if self._i >= len(self._words):
            return None
        word = self._words[self._i]
        self._i += 1
        total = Bool(_tag(word))
        assn = {Real(f"currentMode_{k}"): RealVal(str(d)) for k, d in enumerate(word)}
        self.add_block(Not(total))  # structure-wide, mirroring the real next()
        return total, total, assn


class _Candidate:
    """A candidate oracle whose verdict and completion delay are looked up from
    the total_const it is asked to verify, so a thread completing out of
    proposal order still returns that candidate's scripted verdict."""

    def __init__(self, verdicts, delays, ledger):
        self._verdicts = verdicts
        self._delays = delays
        self._ledger = ledger
        self.asserted = []
        self.budgets = []
        self._frames = [[]]
        self.tag = None

    def set_budget(self, seconds):
        self.budgets.append(seconds)

    def assert_(self, formula):
        self.asserted.append(formula)
        self._frames[-1].append(formula)
        fid = getattr(formula, "id", "")
        if isinstance(fid, str) and fid.startswith("CAND_"):
            self.tag = fid

    def push(self):
        self._frames.append([])

    def pop(self):
        self._frames.pop()

    @property
    def live(self):
        return [f for frame in self._frames for f in frame]

    def check(self):
        with self._ledger["lock"]:
            self._ledger["active"] += 1
            self._ledger["max_active"] = max(
                self._ledger["max_active"], self._ledger["active"]
            )
        try:
            delay = self._delays.get(self.tag, 0.0)
            if delay:
                time.sleep(delay)
            with self._ledger["lock"]:
                self._ledger["checked"].append(self.tag)  # completion order
            return self._verdicts.get(self.tag, UNSAT)
        finally:
            with self._ledger["lock"]:
                self._ledger["active"] -= 1

    def model(self):
        return {"tag": self.tag}


class _ScriptedBox(RegionBoxDiscovery):
    def __init__(self, search, verdicts, delays):
        super().__init__()
        self._search = search
        self._verdicts = verdicts
        self._delays = delays
        self.ledger = _new_ledger()

    def _reduced_pivot_search(self, encoding, encoder, seed):
        return self._search

    def _candidate_oracle(self, logic, seed):
        oracle = _Candidate(self._verdicts, self._delays, self.ledger)
        with self.ledger["lock"]:
            self.ledger["built"].append(oracle)
        return oracle


def _run(
    words,
    verdicts,
    *,
    delays=None,
    workers=None,
    blocks=(),
    exhausted=UNSAT,
    budget="30",
    timeout="45",
):
    vmap = {_tag(w): v for w, v in verdicts.items()}
    dmap = {_tag(w): d for w, d in (delays or {}).items()}
    search = _PivotSearch(words, exhausted=exhausted)
    alg = _ScriptedBox(search, vmap, dmap)
    alg._printer = _SilentPrinter()
    common = (
        None if workers is None else {"parallel": "true", "parallel-core": str(workers)}
    )
    alg._config = _Config(common=common, pivot_budget=budget, pivot_timeout=timeout)
    alg._logger = None
    alg._tau_max = "8"
    alg._underlying = "dreal"
    alg._time_horizon = 8.0
    alg._feasibility_cache = {}
    alg._undecided_candidates = 0
    alg._metrics = _Counter()
    result = alg._pivot_two_step(_Encoding(), "LRA", 0, list(blocks), object())
    return alg, result


# --------------------------------------------------------------------------- #
#  earliest-proposed acceptance, independent of completion order
# --------------------------------------------------------------------------- #
def test_earliest_proposed_sat_is_accepted_despite_later_proposal_finishing_first():
    # Two satisfiable candidates; the second-proposed one is made to finish
    # first. The earliest-proposed must still win.
    alg, (oracle, model, _) = _run(
        ["0", "1", "2", "3"],
        {"0": UNSAT, "1": SAT, "2": SAT, "3": UNSAT},
        delays={"1": 0.20},  # word 1 (earliest SAT) completes last
        workers=4,
    )
    assert oracle is not None
    assert model["tag"] == _tag("1")
    # word 2 verified before word 1, yet word 1 is the accepted pivot.
    checked = alg.ledger["checked"]
    assert checked.index(_tag("2")) < checked.index(_tag("1"))


def test_completion_order_does_not_change_the_accepted_candidate():
    # Whatever the delays, the accepted candidate is fixed by proposal order.
    for delays in ({}, {"1": 0.05}, {"2": 0.05, "3": 0.05}, {"1": 0.1, "2": 0.0}):
        alg, (oracle, model, _) = _run(
            ["0", "1", "2", "3"],
            {"0": UNSAT, "1": SAT, "2": SAT, "3": SAT},
            delays=delays,
            workers=4,
        )
        assert model["tag"] == _tag("1"), delays


# --------------------------------------------------------------------------- #
#  serial / parallel agreement on the accepted candidate AND the counters
# --------------------------------------------------------------------------- #
def test_serial_and_wide_pool_agree_on_candidate_and_counters():
    words = ["0", "1", "2", "3", "4", "5"]
    verdicts = {"0": UNSAT, "1": UNKNOWN, "2": UNSAT, "3": SAT, "4": UNKNOWN, "5": SAT}
    serial, (o1, m1, _) = _run(words, verdicts, workers=1)
    wide, (o2, m2, _) = _run(words, verdicts, workers=8)
    assert m1["tag"] == m2["tag"] == _tag("3")
    for alg in (serial, wide):
        # candidates counted = proposals up to the accepted one (0..3).
        assert alg._metrics["candidates"] == 4
        assert alg._metrics["accepted"] == 1
        # only the undecided proposed BEFORE the acceptance counts (word 1);
        # word 4, proposed after, must not.
        assert alg._undecided_candidates == 1


def test_absent_common_section_defaults_to_serial():
    # No [common] section: parallel-core cannot be read, so the pool is 1 and
    # the loop still decides correctly.
    alg, (oracle, model, _) = _run(["0", "1"], {"0": UNSAT, "1": SAT}, workers=None)
    assert model["tag"] == _tag("1")
    assert alg._metrics["candidates"] == 2 and alg._metrics["accepted"] == 1


# --------------------------------------------------------------------------- #
#  the index gate: an over-proposed candidate cannot perturb accounting
# --------------------------------------------------------------------------- #
def test_unknown_proposed_after_acceptance_is_not_counted_undecided():
    alg, (oracle, model, _) = _run(
        ["0", "1", "2", "3"],
        {"0": UNSAT, "1": SAT, "2": UNKNOWN, "3": UNKNOWN},
        workers=4,
    )
    assert model["tag"] == _tag("1")
    assert alg._metrics["candidates"] == 2  # words 0 and 1 only
    assert alg._metrics["accepted"] == 1
    assert alg._undecided_candidates == 0  # words 2, 3 are past acceptance


# --------------------------------------------------------------------------- #
#  multi-batch accounting when no batch before the last holds a SAT
# --------------------------------------------------------------------------- #
def test_acceptance_in_a_later_batch_counts_every_prior_proposal():
    # Pool of 2: batch {0,1} both refuted, batch {2,3} accepts 3.
    alg, (oracle, model, _) = _run(
        ["0", "1", "2", "3"], {"0": UNSAT, "1": UNSAT, "2": UNSAT, "3": SAT}, workers=2
    )
    assert model["tag"] == _tag("3")
    assert alg._metrics["candidates"] == 4
    assert alg._metrics["accepted"] == 1


def test_a_fully_refuted_space_exhausts_as_unsat_across_batches():
    alg, (oracle, _, _) = _run(
        ["0", "1", "2"],
        {"0": UNSAT, "1": UNSAT, "2": UNSAT},
        workers=2,
        exhausted=UNSAT,
    )
    assert oracle is None
    assert alg._last_pivot_verdict == UNSAT
    assert alg._metrics["candidates"] == 3


def test_a_scenario_give_up_is_unknown_not_unsat():
    alg, (oracle, _, _) = _run(["0"], {"0": UNSAT}, workers=2, exhausted=UNKNOWN)
    assert oracle is None
    assert alg._last_pivot_verdict == UNKNOWN


# --------------------------------------------------------------------------- #
#  the feasibility filter runs before dispatch, under a pool too
# --------------------------------------------------------------------------- #
def test_an_infeasible_word_is_filtered_before_it_is_dispatched(monkeypatch):
    infeasible = (1,)  # word "1" -> mode_seq [1]

    class _Filter:
        def __init__(self, model, tau_max, horizon, cache=None):
            pass

        def word_is_infeasible(self, depth, mode_seq):
            return tuple(mode_seq) == infeasible

    monkeypatch.setattr(_box, "LinearWordFeasibilityFilter", _Filter)
    search = _PivotSearch(["0", "1", "2"], exhausted=UNSAT)
    search.model = object()  # enable the filter
    alg = _ScriptedBox(search, {_tag("0"): UNSAT, _tag("1"): SAT, _tag("2"): UNSAT}, {})
    alg._printer = _SilentPrinter()
    alg._config = _Config(
        common={"parallel-core": "4"}, pivot_budget="30", pivot_timeout="45"
    )
    alg._logger = None
    alg._tau_max = "8"
    alg._underlying = "dreal"
    alg._time_horizon = 8.0
    alg._feasibility_cache = {}
    alg._undecided_candidates = 0
    alg._metrics = _Counter()
    oracle, _model, _ = alg._pivot_two_step(_Encoding(), "LRA", 0, [], object())
    # The infeasible word "1" reaches no candidate oracle...
    assert all(o.tag != _tag("1") for o in alg.ledger["built"])
    # ...and is blocked word-wide (a Not(And(...)) over its mode literals).
    mode_block = Not(And([Eq(Real("currentMode_0"), RealVal("1"))]))
    assert any(str(b) == str(mode_block) for b in search.blocks)
    # SAT never arrives, so the fully-filtered/refuted space exhausts.
    assert oracle is None and alg._last_pivot_verdict == UNSAT


# --------------------------------------------------------------------------- #
#  the accepted oracle survives for growth: unbudgeted, pivot-blocks popped
# --------------------------------------------------------------------------- #
def test_accepted_oracle_from_the_pool_is_unbudgeted_and_unwalled():
    block = Eq(Real("x1_0_0"), RealVal("0"))
    alg, (oracle, model, _) = _run(
        ["0", "1"], {"0": UNSAT, "1": SAT}, workers=4, blocks=[block]
    )
    assert model is not None
    # The pivot-binding block was asserted to the accepted candidate...
    assert any(f is block for f in oracle.asserted)
    # ...but popped, so growth on the same oracle is not walled by it.
    assert not any(f is block for f in oracle.live)
    # and the per-candidate budget was released for growth.
    assert oracle.budgets and oracle.budgets[-1] == "unset"
    # the pivot search was told the IC block as well.
    assert block in alg._search.blocks


# --------------------------------------------------------------------------- #
#  acceptance does not wait on candidates proposed after the earliest SAT
# --------------------------------------------------------------------------- #
def test_acceptance_does_not_wait_on_a_later_long_running_candidate():
    # Earliest-proposed candidate is SAT at once; a later one runs long. The
    # method must return on the earliest SAT without waiting out the later one.
    long = 1.0
    started = time.monotonic()
    alg, (oracle, model, _) = _run(
        ["0", "1"], {"0": SAT, "1": UNSAT}, delays={"1": long}, workers=4
    )
    elapsed = time.monotonic() - started
    assert model["tag"] == _tag("0")
    assert elapsed < long / 2, elapsed
    # the later candidate is neither accepted nor counted.
    assert alg._metrics["candidates"] == 1 and alg._metrics["accepted"] == 1
    assert alg._undecided_candidates == 0


# --------------------------------------------------------------------------- #
#  the parallel switch gates the pool
# --------------------------------------------------------------------------- #
def _workers_for(common):
    alg = RegionBoxDiscovery()
    alg._config = _Config(common=common)
    return alg._verify_workers()


def test_verify_workers_honours_the_parallel_switch(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    # parallel enabled: an in-range parallel-core is honored.
    assert _workers_for({"parallel": "true", "parallel-core": "1"}) == 1
    assert _workers_for({"parallel": "true", "parallel-core": "3"}) == 3
    # parallel disabled -> serial, whatever parallel-core says.
    assert _workers_for({"parallel": "false", "parallel-core": "4"}) == 1
    # parallel enabled but no core, or no [common] at all -> serial.
    assert _workers_for({"parallel": "true"}) == 1
    assert _workers_for(None) == 1


def test_verify_workers_defaults_safe_but_honours_an_explicit_request(monkeypatch):
    # A value above the core count (the generous config default) -> a memory-safe
    # fraction: a quarter of the cpus on a multiple of four, half otherwise.
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert _workers_for({"parallel": "true", "parallel-core": "25"}) == 2
    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    assert _workers_for({"parallel": "true", "parallel-core": "25"}) == 5
    # An explicit request from 1 up to the core count is honored as-is, even
    # above the safe fraction (the user's own risk); the core count itself too.
    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    assert _workers_for({"parallel": "true", "parallel-core": "2"}) == 2
    assert _workers_for({"parallel": "true", "parallel-core": "8"}) == 8
    assert _workers_for({"parallel": "true", "parallel-core": "10"}) == 10


def test_parallel_false_stays_serial_and_matches_the_reference():
    # parallel-core is large but parallel is off: the loop must behave exactly
    # as the serial reference (parallel enabled, core 1) on the same sequence.
    words = ["0", "1", "2", "3"]
    verdicts = {"0": UNSAT, "1": UNKNOWN, "2": SAT, "3": SAT}
    vmap = {_tag(w): v for w, v in verdicts.items()}
    off = _ScriptedBox(_PivotSearch(words), vmap, {})
    off._printer = _SilentPrinter()
    off._config = _Config(
        common={"parallel": "false", "parallel-core": "25"},
        pivot_budget="30",
        pivot_timeout="45",
    )
    off._logger = None
    off._tau_max = "8"
    off._underlying = "dreal"
    off._time_horizon = 8.0
    off._feasibility_cache = {}
    off._undecided_candidates = 0
    off._metrics = _Counter()
    _, model_off, _ = off._pivot_two_step(_Encoding(), "LRA", 0, [], object())

    ref, (_o, model_ref, _) = _run(words, verdicts, workers=1)
    assert model_off["tag"] == model_ref["tag"] == _tag("2")
    assert off._metrics["candidates"] == ref._metrics["candidates"] == 3
    assert off._undecided_candidates == ref._undecided_candidates == 1


# --------------------------------------------------------------------------- #
#  a run-scoped pool caps outstanding checks across searches
# --------------------------------------------------------------------------- #
class _MultiSearchBox(RegionBoxDiscovery):
    """Serves a fresh scripted search per _pivot_two_step call, as run() does,
    so several searches can be driven against one shared verification pool."""

    def __init__(self, word_lists, verdicts, delays, ledger):
        super().__init__()
        self._queue = list(word_lists)
        self._verdicts = verdicts
        self._delays = delays
        self.ledger = ledger

    def _reduced_pivot_search(self, encoding, encoder, seed):
        return _PivotSearch(self._queue.pop(0))

    def _candidate_oracle(self, logic, seed):
        return _Candidate(self._verdicts, self._delays, self.ledger)


def test_a_run_scoped_pool_caps_concurrent_checks_across_searches():
    # Each search accepts an instant SAT and leaves a slow refutation draining.
    # With a fresh pool per search those would pile up; with the run-scoped pool
    # the next search's batch queues behind the drainers, so the number of
    # concurrently active checks never exceeds the pool size.
    core = 2
    ledger = _new_ledger()
    verdicts = {_tag("0"): SAT, _tag("1"): UNSAT}  # "0" accepts, "1" drains
    delays = {_tag("1"): 0.2}
    alg = _MultiSearchBox([["0", "1"]] * 4, verdicts, delays, ledger)
    alg._printer = _SilentPrinter()
    alg._config = _Config(
        common={"parallel": "true", "parallel-core": str(core)},
        pivot_budget="30",
        pivot_timeout="45",
    )
    alg._logger = None
    alg._tau_max = "8"
    alg._underlying = "dreal"
    alg._time_horizon = 8.0
    alg._feasibility_cache = {}
    alg._undecided_candidates = 0
    alg._metrics = _Counter()
    pool = ThreadPoolExecutor(max_workers=core)
    alg._verify_pool = pool
    try:
        for _ in range(4):
            _o, model, _ = alg._pivot_two_step(_Encoding(), "LRA", 0, [], object())
            assert model["tag"] == _tag("0")
    finally:
        pool.shutdown(wait=True)
    assert ledger["max_active"] <= core, ledger["max_active"]


def test_shared_pool_and_local_pool_agree():
    # The run-scoped pool path and the direct-call (local pool) path must reach
    # the same accepted candidate and the same counters.
    words = ["0", "1", "2", "3"]
    verdicts = {"0": UNSAT, "1": UNKNOWN, "2": SAT, "3": SAT}
    local, (_o, m_local, _) = _run(words, verdicts, workers=4)

    vmap = {_tag(w): v for w, v in verdicts.items()}
    shared = _ScriptedBox(_PivotSearch(words), vmap, {})
    shared._printer = _SilentPrinter()
    shared._config = _Config(
        common={"parallel": "true", "parallel-core": "4"},
        pivot_budget="30",
        pivot_timeout="45",
    )
    shared._logger = None
    shared._tau_max = "8"
    shared._underlying = "dreal"
    shared._time_horizon = 8.0
    shared._feasibility_cache = {}
    shared._undecided_candidates = 0
    shared._metrics = _Counter()
    pool = ThreadPoolExecutor(max_workers=4)
    shared._verify_pool = pool
    try:
        _o2, m_shared, _ = shared._pivot_two_step(_Encoding(), "LRA", 0, [], object())
    finally:
        pool.shutdown(wait=True)
    assert m_local["tag"] == m_shared["tag"] == _tag("2")
    assert local._metrics["candidates"] == shared._metrics["candidates"] == 3
    assert local._undecided_candidates == shared._undecided_candidates == 1


# --------------------------------------------------------------------------- #
#  a queued candidate must not hold the pivot past its budget
# --------------------------------------------------------------------------- #
def test_a_candidate_queued_past_the_deadline_does_not_overrun_the_budget():
    # Both pool threads are held by long draining work, so this pivot's
    # candidates only ever queue. The pivot must not wait for a worker to free:
    # once its budget elapses it cancels the queued work and gives up, without
    # launching dReal and without blocking on the older jobs.
    core = 2
    blocker_secs = 1.0
    budget = 0.15
    verdicts = {_tag("0"): SAT, _tag("1"): SAT}  # would accept if they ran
    alg = _ScriptedBox(_PivotSearch(["0", "1"]), verdicts, {})
    alg._printer = _SilentPrinter()
    alg._config = _Config(
        common={"parallel": "true", "parallel-core": str(core)},
        pivot_budget=str(budget),
        pivot_timeout="45",
    )
    alg._logger = None
    alg._tau_max = "8"
    alg._underlying = "dreal"
    alg._time_horizon = 8.0
    alg._feasibility_cache = {}
    alg._undecided_candidates = 0
    alg._metrics = _Counter()
    pool = ThreadPoolExecutor(max_workers=core)
    alg._verify_pool = pool
    blockers = [pool.submit(time.sleep, blocker_secs) for _ in range(core)]
    try:
        start = time.monotonic()
        oracle, _model, _ = alg._pivot_two_step(_Encoding(), "LRA", 0, [], object())
        elapsed = time.monotonic() - start
    finally:
        for b in blockers:
            b.result()
        pool.shutdown(wait=True)
    # Returned near the budget, NOT after the blockers drained.
    assert elapsed < blocker_secs / 2, elapsed
    # No dReal was launched, and the region is reported unresolved (giveup),
    # so absence is never established from candidates the deadline cut off.
    assert alg.ledger["built"] == []
    assert oracle is None
    assert alg._last_pivot_verdict == UNKNOWN


# --------------------------------------------------------------------------- #
#  the pipeline overlaps a slow in-order candidate with later work
# --------------------------------------------------------------------------- #
def test_pipeline_runs_later_candidates_while_a_slow_first_one_is_pending():
    # The earliest-proposed candidate is slow; the rest are fast. A batch
    # barrier would idle the pool until the slow one commits. The pipeline must
    # keep verifying later candidates meanwhile, so several finish before it.
    words = [str(i) for i in range(8)]
    verdicts = {w: UNSAT for w in words}  # full exhaustion, no accept
    alg, (oracle, _m, _) = _run(
        words, verdicts, delays={"0": 0.3}, workers=4, exhausted=UNSAT
    )
    assert oracle is None  # space exhausted
    checked = alg.ledger["checked"]  # completion order
    # Multiple checks ran at once, and the slow first-proposed one finished
    # only after several later ones -- the pool did not idle waiting on it.
    assert alg.ledger["max_active"] >= 2
    assert checked.index(_tag("0")) >= 4
    # Every structure was still verified and counted (clean exhaustion).
    assert set(checked) == {_tag(w) for w in words}
    assert alg._metrics["candidates"] == 8
