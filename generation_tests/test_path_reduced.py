"""z3-only behavioural tests for the kappa_path delta (reduced two-step) branch.

The kappa_path end-to-end tests run on the exact backend and the reduced-loop
tests in ``test_reduced_pin.py`` exercise kappa_box; neither covers the delta
word-enumeration loop in ``DiscretePathEnum._run_reduced``. These tests drive
that loop directly with a scripted structure search (substituted through the
``_make_reduced_search`` seam) and a scripted dReal verifier (through the
``make_oracle`` seam), so no dReal and no benchmark model are needed.

The central case is a reduced property structure (``path_const``) shared by two
distinct location words. Because ``path_const`` is not guaranteed to pin the whole
word, excluding it structure-wide after verifying ONE word would drop the other
untested. The loop must exclude only ``path_const AND word``, so a sibling word
that shares the structure is still enumerated and can still be pooled.
"""
import os

from stlmc.constraints.constraints import Eq, Geq, Leq, Or, Real, RealVal
from stlmc.generation import encode as _encode  # noqa: F401  resolve import order
from stlmc.generation.oracle import SAT, UNKNOWN, UNSAT
from stlmc.generation.pathenum import DiscretePathEnum
from stlmc.solver.z3 import z3Obj

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURE_MODEL = os.path.join(
    _REPO, "generation_tests", "fixtures", "path_branch.model")


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


class _GenConfig:
    def __init__(self, **gen):
        self._gen = _Section({k.replace("_", "-"): v for k, v in gen.items()})

    def is_section_in(self, name):
        return name == "gen"

    def get_section(self, name):
        if name != "gen":
            raise KeyError(name)
        return self._gen


class _SilentPrinter:
    def __getattr__(self, _):
        return lambda *a, **k: None


class _StubEncoder:
    """enumerate_components_at is ignored by the scripted search; reset is a no-op."""
    model = None

    def enumerate_components_at(self, depth):
        return None

    def reset(self):
        pass


# --------------------------------------------------------------------------- #
#  scripted structure search: two words share one word-agnostic path_const
# --------------------------------------------------------------------------- #
_CM0 = Real("currentMode_0")
_AUX = Real("aux")


class _SharedPathSearch:
    """A depth-0 search whose words 0..len-1 all share one path_const P.

    P constrains an auxiliary variable, never ``currentMode_0``, so it is
    word-agnostic: every word coexists with P. ``propose`` returns the words in a
    FIXED order (so a chosen word is verified before its siblings) as long as
    ``P AND currentMode_0=w`` survives the accumulated blocks; the blocks are real
    z3, so structure-wide vs word-aware exclusion behave exactly as they would on
    a real scenario solver.
    """
    def __init__(self, words):
        self._words = list(words)
        self.P = Eq(_AUX, RealVal("1"))
        import z3
        self._solver = z3.Solver()
        self._solver.add(z3Obj(self.P))
        self._solver.add(z3Obj(Geq(_CM0, RealVal("0"))))
        self._solver.add(z3Obj(Leq(_CM0, RealVal(str(len(words) - 1)))))
        self._solver.add(z3Obj(Or([Eq(_CM0, RealVal(w)) for w in words])))
        self._v = None

    def _feasible(self, w):
        import z3
        self._solver.push()
        self._solver.add(z3Obj(Eq(_CM0, RealVal(w))))
        r = self._solver.check()
        self._solver.pop()
        return r == z3.sat

    def propose(self):
        for w in self._words:
            if self._feasible(w):
                # a scenario assignment carrying the word (depth-0: one position)
                assn = {_CM0: RealVal(w)}
                self._v = SAT
                return self.P, self.P, assn
        self._v = UNSAT
        return None

    def add_block(self, formula):
        self._solver.add(z3Obj(formula))

    def last_verdict(self):
        return self._v


class _NoneSearch:
    """Immediately exhausted with a chosen verdict (UNSAT spent / UNKNOWN gave up)."""
    def __init__(self, verdict):
        self._v = verdict

    def propose(self):
        return None

    def add_block(self, formula):
        pass

    def last_verdict(self):
        return self._v


class _ForeverUnsatSearch:
    """Always proposes the same word; every verification will refute it. Models a
    depth whose skeleton space never runs dry within a budget."""
    def __init__(self, word="0"):
        self.P = Eq(_AUX, RealVal("1"))
        self._word = word
        self._v = SAT

    def propose(self):
        self._v = SAT
        return self.P, self.P, {_CM0: RealVal(self._word)}

    def add_block(self, formula):
        pass  # deliberately ineffective: the same word keeps coming back

    def last_verdict(self):
        return self._v


# --------------------------------------------------------------------------- #
#  scripted dReal verifier (make_oracle seam)
# --------------------------------------------------------------------------- #
class _Verifier:
    """Reads the pinned word from the fix_modes assertion and answers by word.

    ``verdicts`` maps a word string to SAT/UNSAT/UNKNOWN. ``witness_drop`` names a
    word whose SAT witness omits the mode variable (to exercise the schema guard).
    """
    def __init__(self, verdicts, witness_drop=None):
        self._verdicts = verdicts
        self._drop = witness_drop
        self._word = None
        self.budgets = []

    def set_budget(self, seconds):
        # Match DrealReSolveOracle's interface: the loop clamps the per-call
        # budget to the time left in the depth.
        self.budgets.append(seconds)

    def assert_(self, formula):
        import re
        pins = re.findall(r"currentMode_(\d+) = ([0-9.]+)", str(formula))
        if pins:
            self._word = ".".join(v for _, v in sorted(pins, key=lambda p: int(p[0])))

    def check(self):
        return self._verdicts.get(self._word, UNSAT)

    def model(self):
        if self._word == self._drop:
            return {}  # schema-incomplete witness: no mode variable
        d = {}
        for k, v in enumerate(self._word.split(".")):
            d[Real(f"currentMode_{k}")] = RealVal(v)
        return d

    def unknown_reason(self):
        return "scripted"


def _make_verifier(verdicts, witness_drop=None):
    def factory(**kwargs):
        return _Verifier(verdicts, witness_drop)
    return factory


# --------------------------------------------------------------------------- #
#  harness
# --------------------------------------------------------------------------- #
def _pooled_words(pool):
    import re
    out = []
    for p in pool:
        steps = sorted(
            (int(re.match(r"currentMode_(\d+)", v.id).group(1)), c.value)
            for v, c in p.items() if re.match(r"currentMode_(\d+)$", v.id))
        out.append(".".join(c for _, c in steps))
    return out


def _run(search, verdicts, monkeypatch, *, radius=0, per_depth=64,
         witness_drop=None, **gen):
    import stlmc.generation.pathenum as pathenum
    monkeypatch.setattr(pathenum, "make_oracle",
                        _make_verifier(verdicts, witness_drop))

    class _Alg(DiscretePathEnum):
        def _make_reduced_search(self, components, model, seed, timeout_ms):
            return search

    alg = _Alg()
    result, _t, first_depth, pool = alg._run_reduced(
        _StubEncoder(), target_depths=[0], per_depth=per_depth, radius=radius,
        seed=0, logic="QF_LRA", config=_GenConfig(**gen), logger=None,
        printer=_SilentPrinter(), max_depth=0, tau_max=1.0)
    return result, first_depth, pool


# --------------------------------------------------------------------------- #
#  1. the shared-path_const regression (his highest-value case)
# --------------------------------------------------------------------------- #
def test_a_sibling_word_sharing_the_structure_is_still_pooled(monkeypatch):
    """Two words share one path_const. Word 0 refutes, word 1 satisfies. Word 1
    must be pooled: the refutation of word 0 excludes only (path_const AND word 0),
    not path_const, so word 1 -- which shares that structure -- is still reached.
    A structure-wide block would have made the search UNSAT after word 0 and
    dropped word 1, wrongly reporting the depth exhausted with an empty pool."""
    search = _SharedPathSearch(["0", "1"])
    result, _fd, pool = _run(search, {"0": UNSAT, "1": SAT}, monkeypatch)
    assert _pooled_words(pool) == ["1"], _pooled_words(pool)
    assert result == "False"


def test_both_words_pooled_when_both_satisfy(monkeypatch):
    """Both words share the structure and both satisfy: both are pooled, and the
    depth still exhausts (verdict False, pool of two)."""
    search = _SharedPathSearch(["0", "1"])
    result, _fd, pool = _run(search, {"0": SAT, "1": SAT}, monkeypatch)
    assert sorted(_pooled_words(pool)) == ["0", "1"]
    assert result == "False"


def test_no_word_satisfies_is_exhaustion_absence(monkeypatch):
    """Both words refute: the depth exhausts empty. With depths = full coverage
    (0..0) that is an absence verdict, not Unknown."""
    search = _SharedPathSearch(["0", "1"])
    result, _fd, pool = _run(search, {"0": UNSAT, "1": UNSAT}, monkeypatch)
    assert pool == []
    assert result == "True"


# --------------------------------------------------------------------------- #
#  2. undecided results never become a false absence
# --------------------------------------------------------------------------- #
def test_a_verification_unknown_leaves_the_depth_unresolved(monkeypatch):
    """A structure that verifies UNKNOWN is not evidence of absence: the depth is
    enumerated but cannot claim exhaustion, so the verdict is Unknown even though
    the search ran dry."""
    search = _SharedPathSearch(["0", "1"])
    result, _fd, pool = _run(search, {"0": UNKNOWN, "1": UNKNOWN}, monkeypatch)
    assert pool == []
    assert result == "Unknown"


def test_a_reduction_unknown_leaves_the_depth_unresolved(monkeypatch):
    """The reduced-query minimizer giving up (propose returns None with an UNKNOWN
    last_verdict) must not be read as exhaustion."""
    result, _fd, pool = _run(_NoneSearch(UNKNOWN), {}, monkeypatch)
    assert pool == []
    assert result == "Unknown"


def test_a_spent_search_is_absence(monkeypatch):
    """propose None with an UNSAT last_verdict is a spent structure space -> the
    depth is decided empty (absence)."""
    result, _fd, pool = _run(_NoneSearch(UNSAT), {}, monkeypatch)
    assert pool == []
    assert result == "True"


# --------------------------------------------------------------------------- #
#  3. per-depth budget -> honest Unknown, not an external kill
# --------------------------------------------------------------------------- #
def test_the_pivot_budget_stops_a_runaway_depth_as_unresolved(monkeypatch):
    """A depth whose search never runs dry is bounded by [gen] pivot-budget, and
    hitting it leaves the depth unresolved (Unknown) rather than looping until an
    external timeout kills the process."""
    result, _fd, pool = _run(
        _ForeverUnsatSearch("0"), {"0": UNSAT}, monkeypatch, pivot_budget="0")
    assert pool == []
    assert result == "Unknown"


def test_the_verify_budget_is_clamped_to_the_depth_budget(monkeypatch):
    """Under a tight [gen] pivot-budget, each verification call is clamped to the
    time left in the depth -- as the box strategy clamps its candidate calls --
    so a call that starts just under the budget cannot run a full query-timeout
    past it. The per-call budget never exceeds the pivot-budget."""
    import stlmc.generation.pathenum as pathenum
    made = []

    def factory(**kwargs):
        v = _Verifier({"0": UNSAT})
        made.append(v)
        return v

    monkeypatch.setattr(pathenum, "make_oracle", factory)

    class _Alg(DiscretePathEnum):
        def _make_reduced_search(self, components, model, seed, timeout_ms):
            return _ForeverUnsatSearch("0")

    _Alg()._run_reduced(
        _StubEncoder(), target_depths=[0], per_depth=64, radius=0, seed=0,
        logic="QF_LRA",
        config=_GenConfig(pivot_budget="0.2", query_timeout="45"),
        logger=None, printer=_SilentPrinter(), max_depth=0, tau_max=1.0)
    budgets = [b for v in made for b in v.budgets]
    assert budgets, "expected at least one clamped verification call"
    assert all(b <= 0.2 for b in budgets), budgets


# --------------------------------------------------------------------------- #
#  4. schema guard: a witness missing part of the word is not pooled
# --------------------------------------------------------------------------- #
def test_a_witness_missing_the_word_is_not_pooled(monkeypatch):
    """A satisfiable structure whose witness omits the location word is not pooled
    (the pool and downstream validation need the full word), and the depth is left
    unresolved rather than claiming exhaustion around a dropped witness."""
    search = _SharedPathSearch(["0", "1"])
    result, _fd, pool = _run(
        search, {"0": SAT, "1": UNSAT}, monkeypatch, witness_drop="0")
    assert pool == []
    assert result == "Unknown"


# --------------------------------------------------------------------------- #
#  5. the reduced query retains the complete model execution (guards + resets)
# --------------------------------------------------------------------------- #
def test_the_reduced_query_carries_guards_and_resets_on_a_real_model():
    """On a real guarded model, every reconstructed total_const contains the whole
    model execution as a conjunct -- the jump guards and the identity resets -- so a
    pooled witness is a run of the automaton, not just a trajectory that satisfies
    the reduced property. The backend decides the query itself; this reads the
    structural precondition for the no-guard-violation check.

    The scripted tests above cannot catch a dropped guard because their verifier
    returns a supplied verdict without evaluating model dynamics; this reads the
    actual reconstructed query on a real parsed model instead."""
    from stlmc.generation.encode import Encoder
    from stlmc.generation.reduced import ReducedPivotSearch
    from stlmc.objects.object_factory import ObjectFactory

    om = ObjectFactory("model-with-goal-enhanced").generate_object_manager()
    model, prop_dict, goals, labels = om.generate_objects(_FIXTURE_MODEL)
    goal = next(g for g in goals if labels.get(g.get_formula()) == "f2")
    enc = Encoder(model, goal, prop_dict, 0.1, 8.0)
    rp = ReducedPivotSearch(enc.enumerate_components_at(2), enc.model, seed=0)
    total_const, _path, _assn = rp.propose()

    me = str(rp._model_execution)
    # the whole model execution is a conjunct of the query, verbatim
    assert me in str(total_const)
    # ... and it carries the real jump guards and an identity reset
    assert "<= 4" in me            # a branch guard
    assert ">= 15" in me           # a later guard
    assert "x1_1_0 = x1_0_t" in me  # an identity reset across a jump