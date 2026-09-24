"""z3-only tests for the reduced-query delta pivot (the two-step reduced path).

Two layers, both without dReal or a benchmark model:

1. Contract of the reconstruction, synthetically: ``ReducedPivotSearch``
   minimizes against the flattened RECURSIVE falsification target
   (current_minimize_info, enumerate.py:199), NOT the monolithic stl_const, with
   stl_final excluded -- the property that keeps the core (and the reconstructed
   property path) small.

2. Behaviour of the candidate loop, against a scripted pivot search substituted
   through the ``_reduced_pivot_search`` seam (the analogue of the old skeleton
   oracle seam): verdict recording, accepted-returns-its-oracle, the blocks
   binding the candidate oracle, the per-candidate budget clamp, and the delta
   backend routing through the two-step. The oracles are scripted because the
   search's own decisions do not depend on a solver.

The real-encoding numbers (the reduced query carries fewer forall_t than
encoding.consts on the actual benchmarks) are confirmed by the delta runs on the
mac, not by a check here -- these tests need neither dReal nor a model.
"""

from stlmc.constraints.constraints import Bool, BoolVal, Eq, Real, RealVal
from stlmc.generation import encode as _encode  # noqa: F401  resolve import order
from stlmc.generation.box import RegionBoxDiscovery
from stlmc.generation.encode import StlComponents
from stlmc.generation.oracle import SAT, UNKNOWN, UNSAT
from stlmc.generation.reduced import ReducedPivotSearch


# --------------------------------------------------------------------------- #
#  1. reconstruction contract (synthetic StlComponents, no model, no solver)
# --------------------------------------------------------------------------- #
def _components(bound=2):
    return StlComponents(
        bound=bound,
        tau_max=1.0,
        delta=0.1,
        sub_formulas=set(),
        initial_stl_f=Bool("INIT_STL"),
        initial_model_f=Bool("INIT_MODEL"),
        initial_track_const=Bool("INIT_TRACK"),
        model_consts=[Bool(f"MODELNEXT_{b}") for b in range(bound + 1)],
        model_track_consts=[Bool(f"MTRK_{b}") for b in range(bound + 1)],
        model_f_k_final=Bool("MODELFINAL"),
        model_track_f_k_final=Bool("MTRKFINAL"),
        stl_consts=[Bool(f"STL_{b}") for b in range(bound + 1)],
        stl_time_consts=[Bool(f"STLTIME_{b}") for b in range(bound + 1)],
        final_f_k=Bool("STLFINAL"),
        time_order_const=Bool("TIMEORDER"),
        boolean_abstract={},
    )


def _reduced(bound=2):
    # model=None is fine: the constructor never dereferences the model (only
    # next()/_reconstruct do, and these tests do not call them).
    return ReducedPivotSearch(_components(bound), model=None, seed=0)


def test_minimize_target_excludes_stl_final():
    """stl_final reaches the scenario solver and total_const, but is NOT in the
    minimize target F. Minimizing against F rather than the full stl_const is what
    keeps the core -- and hence the reconstructed property path -- small."""
    rp = _reduced()
    assert "STLFINAL" not in str(rp._not_F)
    assert rp.stl_final.id == "STLFINAL"


def test_minimize_target_unrolls_the_recursion():
    """F is init AND (non-final structural path for every EARLIER bound) AND
    (FINAL model consts + time order at the pivot bound) -- the two
    current_minimize_info shapes (enumerate.py:199 and :341), flattened."""
    s = str(_reduced(bound=2)._not_F)
    for marker in (
        "INIT_MODEL",
        "INIT_STL",
        "INIT_TRACK",
        "MODELNEXT_0",
        "STL_0",
        "STLTIME_0",
        "MODELNEXT_1",
        "STL_1",
        "STLTIME_1",
        "MODELFINAL",
        "STL_2",
        "STLTIME_2",
        "TIMEORDER",
    ):
        assert marker in s, marker


def test_pivot_bound_uses_final_model_not_the_next_form():
    s = str(_reduced(bound=2)._not_F)
    assert "MODELFINAL" in s
    assert "MODELNEXT_2" not in s


def test_target_scales_with_bound():
    s1 = str(_reduced(bound=1)._not_F)
    s3 = str(_reduced(bound=3)._not_F)
    assert "MODELNEXT_2" not in s1 and "MODELNEXT_2" in s3
    assert "MODELFINAL" in s1 and "MODELFINAL" in s3


def test_model_execution_is_the_complete_model_not_the_abstraction():
    """The reduced query retains the COMPLETE model execution -- the initial
    condition, every non-final step's model consts, and the final step's -- so a
    jump guard or reset dropped from the property core cannot let a witness
    satisfy the reduced property along a trajectory the automaton cannot produce.
    The abstraction map alone (boolean_abstract) only DEFINES the ODE-integral
    and invariant Bools; it asserts no guard or reset, which is what let the
    minimizer produce guard-violating witnesses before this was retained whole."""
    me = str(_reduced(bound=2)._model_execution)
    # init + every step's model consts (guards, resets, flows live here)
    for marker in ("INIT_MODEL", "MODELNEXT_0", "MODELNEXT_1", "MODELFINAL"):
        assert marker in me, marker
    # the property / timing subformulas are NOT part of the model execution:
    # those are the only thing the reduction is allowed to drop.
    for absent in ("STL_0", "STL_1", "STLFINAL", "STLTIME_0", "TIMEORDER"):
        assert absent not in me, absent


def test_the_model_execution_scales_with_bound():
    """Every non-final step 0..N-1 plus the final step is kept, so no step's
    guards or resets are missing at a deeper bound."""
    me1 = str(_reduced(bound=1)._model_execution)
    me3 = str(_reduced(bound=3)._model_execution)
    assert "MODELNEXT_2" not in me1 and "MODELNEXT_2" in me3
    assert "MODELFINAL" in me1 and "MODELFINAL" in me3


# --------------------------------------------------------------------------- #
#  2. candidate-loop behaviour (scripted pivot search + scripted candidate oracle)
# --------------------------------------------------------------------------- #
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


class _GenConfig:
    """Just the [gen] section; budgets are given, not read from a benchmark."""

    def __init__(self, **gen):
        self._gen = _Section({k.replace("_", "-"): v for k, v in gen.items()})

    def is_section_in(self, name):
        return name == "gen"

    def get_section(self, name):
        if name != "gen":
            raise KeyError(name)
        return self._gen


class _Encoding:
    consts = BoolVal("True")
    skeleton = BoolVal("True")
    bound = 0


class _OneDepthEncoder:
    def encode_at(self, depth):
        return _Encoding()


class _ScriptedReducedSearch:
    """A ReducedPivotSearch stand-in whose pivots are given.

    Each ``next`` yields ``(total_const, path_const, assn)`` from a script, then
    ``None`` once spent (``forever`` repeats the last). ``last_verdict`` is what
    the exhausted search reports -- UNSAT (structure space spent) vs UNKNOWN (the
    scenario solver gave up), the distinction the caller must preserve."""

    def __init__(self, words, forever=False, exhausted_verdict=UNSAT):
        self._words = list(words)
        self._forever = forever
        self._exhausted = exhausted_verdict
        self.blocks = []

    def add_block(self, formula):
        self.blocks.append(formula)

    def last_verdict(self):
        return self._exhausted

    def next(self):
        if not self._words:
            return None
        word = self._words[0] if self._forever else self._words.pop(0)
        assn = {Real(f"currentMode_{k}"): RealVal(str(d)) for k, d in enumerate(word)}
        return BoolVal("True"), BoolVal("True"), assn


class _ScriptedCandidate:
    def __init__(self, verdict):
        self.verdict = verdict
        self.asserted = []
        self.budgets = []
        self._frames = [[]]

    def set_budget(self, seconds):
        self.budgets.append(seconds)

    def assert_(self, formula):
        self.asserted.append(formula)
        self._frames[-1].append(formula)

    def push(self):
        self._frames.append([])

    def pop(self):
        self._frames.pop()

    @property
    def live(self):
        return [f for frame in self._frames for f in frame]

    def check(self):
        return self.verdict

    def model(self):
        return {}


class _ScriptedReducedTwoStep(RegionBoxDiscovery):
    """kappa_box with the reduced pivot search and candidate oracle given."""

    def __init__(self, words, verdicts, forever=False, exhausted_verdict=UNSAT):
        super().__init__()
        self.search = _ScriptedReducedSearch(
            words, forever=forever, exhausted_verdict=exhausted_verdict
        )
        self._verdicts = list(verdicts)
        self.candidates = []

    def _reduced_pivot_search(self, encoding, encoder, seed):
        return self.search

    def _candidate_oracle(self, logic, seed):
        verdict = self._verdicts.pop(0) if self._verdicts else UNSAT
        oracle = _ScriptedCandidate(verdict)
        self.candidates.append(oracle)
        return oracle


def _run(words, verdicts, blocks=(), forever=False, exhausted_verdict=UNSAT, **gen):
    from stlmc.generation.box import _Counter

    alg = _ScriptedReducedTwoStep(
        words, verdicts, forever=forever, exhausted_verdict=exhausted_verdict
    )
    alg._printer = _SilentPrinter()
    alg._config = _GenConfig(**gen)
    alg._logger = None
    alg._tau_max = "8"
    alg._underlying = "dreal"
    alg._time_horizon = 8.0
    alg._undecided_candidates = 0
    alg._metrics = _Counter()
    return alg, alg._pivot_two_step(_Encoding(), "LRA", 0, list(blocks), object())


class TestReducedVerdictRecording:
    """UNSAT (structure space spent) must stay distinct from UNKNOWN (the
    scenario solver gave up), so a resource failure is never reported as
    absence."""

    def test_a_spent_structure_space_records_unsat(self):
        alg, (oracle, _, _) = _run([], [], pivot_budget=30, exhausted_verdict=UNSAT)
        assert oracle is None
        assert alg._last_pivot_verdict == UNSAT

    def test_a_search_give_up_records_unknown_not_unsat(self):
        alg, (oracle, _, _) = _run([], [], pivot_budget=30, exhausted_verdict=UNKNOWN)
        assert oracle is None
        assert alg._last_pivot_verdict == UNKNOWN


class TestReducedCandidateLoop:
    def test_an_accepted_candidate_returns_its_oracle(self):
        alg, (oracle, model, encoding) = _run(["012"], [SAT], pivot_budget=30)
        assert oracle is not None and model is not None and encoding is not None
        assert alg._metrics["accepted"] == 1

    def test_a_refuted_candidate_advances_the_search(self):
        # two refutations then accept: three candidates, one accepted.
        alg, (oracle, _, _) = _run(
            ["012", "013", "014"], [UNSAT, UNSAT, SAT], pivot_budget=30
        )
        assert oracle is not None
        assert alg._metrics["candidates"] == 3 and alg._metrics["accepted"] == 1

    def test_the_candidate_budget_is_clamped_to_the_deadline(self):
        """One candidate must not overrun the search deadline by a whole
        pivot-timeout: the per-candidate budget never exceeds the time left."""
        alg, _ = _run(
            ["01"], [UNSAT] * 100, forever=True, pivot_budget=0.2, pivot_timeout=45
        )
        numeric = [b for c in alg.candidates for b in c.budgets if isinstance(b, float)]
        assert numeric and all(b <= 0.2 for b in numeric), numeric


class TestReducedBlocksBindTheCandidate:
    """The blocks bind the pivot query (the candidate oracle chooses the pivot),
    in a pushed frame so growth on the same oracle is not walled."""

    def _block(self):
        return Eq(Real("x1_0_0"), RealVal("0"))

    def test_the_candidate_oracle_sees_the_blocks(self):
        block = self._block()
        alg, (oracle, model, _) = _run(["01"], [SAT], blocks=[block], pivot_budget=30)
        assert model is not None
        assert any(f is block for f in oracle.asserted)

    def test_growth_is_not_walled_by_the_blocks(self):
        block = self._block()
        alg, (oracle, _, _) = _run(["01"], [SAT], blocks=[block], pivot_budget=30)
        assert not any(f is block for f in oracle.live)

    def test_the_search_is_told_the_blocks_too(self):
        block = self._block()
        alg, _ = _run(["01"], [SAT], blocks=[block], pivot_budget=30)
        assert block in alg.search.blocks


class TestReducedBackendRouting:
    def test_dreal_routes_through_the_reduced_two_step(self):
        from stlmc.generation.box import _Counter

        alg = _ScriptedReducedTwoStep(["01"], [SAT])
        alg._printer = _SilentPrinter()
        alg._config = _GenConfig(pivot_budget="30")
        alg._logger = None
        alg._tau_max = "8"
        alg._underlying = "dreal"
        alg._time_horizon = 8.0
        alg._undecided_candidates = 0
        alg._metrics = _Counter()
        _, model, _ = alg._pivot_at(_OneDepthEncoder(), 0, "LRA", 0, [])
        assert model is not None, "dreal must route through the two-step search"
