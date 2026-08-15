"""Tests for the per-call solver bound shared by both generation strategies.

The bound is what makes an undecidable query a reportable result rather than a
hang. Both backends resolve it from the same key, and an exceeded bound must
surface as UNKNOWN -- never as UNSAT, which callers read as a claim about the
model.

Only the z3 backend is exercised; dReal's budget is enforced around a subprocess
and the suite runs no external solver, so its resolution is tested and its
enforcement is not.
"""

import inspect
from fractions import Fraction

from stlmc.constraints.constraints import And, Eq, Geq, Int, IntVal, Leq, Mul
from stlmc.generation.oracle import (
    DEFAULT_QUERY_TIMEOUT,
    SAT,
    UNKNOWN,
    UNSAT,
    DrealReSolveOracle,
    Z3IncrementalOracle,
    query_timeout,
)


class FakeSection:
    def __init__(self, values):
        self._values = values

    def get_value(self, key):
        return self._values[key]


class FakeConfig:
    """Just enough of a Configuration for the [gen] lookup."""

    def __init__(self, **values):
        self._values = values

    def get_section(self, name):
        if name != "gen" or not self._values:
            raise KeyError(name)
        return FakeSection(self._values)


# =============================================================== resolution

class TestQueryTimeoutResolution:
    """One key, one meaning, on either backend."""

    def test_absent_falls_back_to_the_default(self):
        assert query_timeout(FakeConfig()) == DEFAULT_QUERY_TIMEOUT

    def test_no_configuration_falls_back_to_the_default(self):
        """Not to "unbounded": an unbounded call is what this prevents."""
        assert query_timeout(None) == DEFAULT_QUERY_TIMEOUT

    def test_a_value_is_seconds(self):
        assert query_timeout(FakeConfig(**{"query-timeout": "2.5"})) == 2.5

    def test_zero_and_its_spellings_disable_the_bound(self):
        for spelling in ("0", "off", "none", " off "):
            assert query_timeout(
                FakeConfig(**{"query-timeout": spelling})) is None, spelling

    def test_a_caller_override_wins_over_the_configuration(self):
        """kappa_box bounds one candidate more tightly than a growth query."""
        oracle = DrealReSolveOracle(FakeConfig(**{"query-timeout": "60"}))
        assert oracle._query_budget() == 60.0
        oracle.set_budget(3.0)
        assert oracle._query_budget() == 3.0
        oracle.set_budget("unset")
        assert oracle._query_budget() == 60.0


# ============================================================== enforcement

def semiprime_query():
    """`x * y = p * q` over the integers, with both factors non-trivial.

    Reliably beyond a millisecond and satisfiable, so a bounded check reports
    UNKNOWN for the bound rather than because the query has no model.
    """
    x, y = Int("x"), Int("y")
    product = 32416187567 * 32416189381
    return And([Geq(x, IntVal("2")), Geq(y, IntVal("2")),
                Leq(x, IntVal(str(product))), Leq(y, IntVal(str(product))),
                Eq(Mul(x, y), IntVal(str(product)))])


class TestZ3Enforcement:
    def test_an_exceeded_bound_is_unknown_and_not_unsat(self):
        oracle = Z3IncrementalOracle("QF_NRA", seed=0, timeout=0.05)
        oracle.assert_(semiprime_query())
        assert oracle.check() == UNKNOWN

    def test_an_exceeded_bound_leaves_no_model_to_read(self):
        oracle = Z3IncrementalOracle("QF_NRA", seed=0, timeout=0.05)
        oracle.assert_(semiprime_query())
        oracle.check()
        try:
            oracle.model()
        except RuntimeError:
            return
        raise AssertionError("model() must not return after a bounded check")

    def test_the_bound_does_not_disturb_a_query_that_decides(self):
        x = Int("x")
        oracle = Z3IncrementalOracle("QF_NRA", seed=0, timeout=10)
        oracle.assert_(And([Geq(x, IntVal("1")), Leq(x, IntVal("1"))]))
        assert oracle.check() == SAT
        assert oracle.model()[Int("x")].value == "1"

    def test_unsat_is_still_unsat_under_a_bound(self):
        x = Int("x")
        oracle = Z3IncrementalOracle("QF_NRA", seed=0, timeout=10)
        oracle.assert_(And([Geq(x, IntVal("2")), Leq(x, IntVal("1"))]))
        assert oracle.check() == UNSAT

    def test_the_bound_can_be_removed(self):
        """[gen] query-timeout = 0 restores the unbounded behaviour. None is
        what `query_timeout` returns for it; a non-positive number means the
        same, so a value cannot accidentally read as a sub-resolution bound."""
        x = Int("x")
        for disabled in (None, 0, 0.0):
            oracle = Z3IncrementalOracle("QF_NRA", seed=0, timeout=disabled)
            oracle.assert_(And([Geq(x, IntVal("3")), Leq(x, IntVal("3"))]))
            assert oracle.check() == SAT, disabled

    def test_the_default_lives_in_the_constructor(self):
        """kappa_box builds this class directly rather than through make_oracle,
        so a default injected by the factory would leave its queries unbounded."""
        default = inspect.signature(
            Z3IncrementalOracle.__init__).parameters["timeout"].default
        assert default == DEFAULT_QUERY_TIMEOUT

    def test_a_directly_constructed_oracle_enforces_its_bound(self):
        """Same query, bound lowered so the test does not wait for the default."""
        oracle = Z3IncrementalOracle("QF_NRA", seed=0, timeout=0.05)
        oracle.assert_(semiprime_query())
        assert oracle.check() == UNKNOWN


# ============================================================== diagnostics

class TestUnknownReason:
    """An UNRESOLVED report should say why: a per-call bound expiry reads very
    differently from a solver give-up. The hook is diagnostic only -- callers
    never reclassify a verdict from it."""

    def test_a_bound_expiry_names_the_query_timeout(self):
        oracle = Z3IncrementalOracle("QF_NRA", seed=0, timeout=0.05)
        oracle.assert_(semiprime_query())
        assert oracle.check() == UNKNOWN
        assert "query-timeout" in oracle.unknown_reason()

    def test_a_dreal_budget_expiry_names_the_query_timeout(self, monkeypatch):
        oracle = DrealReSolveOracle(FakeConfig(**{"query-timeout": "60"}))

        def expired(consts):
            oracle.timeouts += 1
            oracle._unknown_reason = (
                "dReal exceeded the per-call [gen] query-timeout (60s)")
            return "Unknown", None

        monkeypatch.setattr(oracle, "_solve_once", expired)
        assert oracle.check() == UNKNOWN
        assert "query-timeout" in oracle.unknown_reason()

    def test_a_dreal_give_up_reports_a_generic_reason(self, monkeypatch):
        oracle = DrealReSolveOracle(FakeConfig(**{"query-timeout": "60"}))
        monkeypatch.setattr(oracle, "_solve_once",
                            lambda consts: ("Unknown", None))
        assert oracle.check() == UNKNOWN
        assert oracle.unknown_reason() == "dReal did not decide"


# =========================================================== classification

class FakeProc:
    """A finished subprocess: exit status and captured streams."""

    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self._stdout = stdout.encode()
        self._stderr = stderr.encode()

    def communicate(self):
        return self._stdout, self._stderr


def classify(returncode, stdout, stderr):
    """Run dRealSolver.parallel_check_sat over a fake finished process."""
    import queue
    import threading

    from stlmc.solver.dreal import dRealSolver

    solver = dRealSolver.__new__(dRealSolver)  # the method touches no state
    main_queue: queue.Queue = queue.Queue()
    dRealSolver.parallel_check_sat(solver, main_queue, threading.Semaphore(0),
                                   FakeProc(returncode, stdout, stderr))
    result, assignment, _ = main_queue.get_nowait()
    return result, assignment


class TestDrealClassification:
    """The verdict is gated on the exit status. An error transcript that quotes
    the encoding necessarily contains "currentMode"; before the gate it was
    classified as satisfiable, and the caller pooled a "model" parsed from an
    error message before crashing on it."""

    SAT_OUT = "Solution:\ncurrentMode_0 : Int = [1, 1]\nx_0_0 : [0.5, 0.6]\n"

    def test_a_clean_solution_is_satisfiable(self):
        result, assignment = classify(0, self.SAT_OUT, "b@goal : Bool = true\n")
        assert result == "False"
        assert assignment._dreal_model

    def test_a_clean_unsat_is_unsatisfiable(self):
        assert classify(0, "unsat\n", "")[0] == "True"

    def test_an_error_transcript_is_no_verdict(self):
        result, _ = classify(1, "", "parse error near '(= currentMode_0 1)'\n")
        assert result == "Unknown"

    def test_a_killed_process_is_no_verdict(self):
        assert classify(-9, "", "")[0] == "Unknown"

    def test_a_clean_exit_with_no_recognisable_output_is_no_verdict(self):
        assert classify(0, "something else", "")[0] == "Unknown"

    def test_a_transcript_without_a_blank_line_does_not_crash(self):
        """.remove("") raised ValueError when the transcript had none."""
        result, assignment = classify(
            0, "Solution:\ncurrentMode_0 : Int = [1, 1]", "no-newline")
        assert result == "False"
        assert assignment._dreal_model

class TestDrealSolverArgs:
    """What the binary is actually told.

    The [dreal] section was read for the ODE settings and the executable path,
    but the delta was never passed on, so a configured value described a run it
    did not reach. These pin the command line rather than the configuration.
    """

    def _solver(self):
        from stlmc.solver.dreal import dRealSolver

        return dRealSolver()

    def test_a_solver_that_was_not_given_a_delta_keeps_its_command_line(self):
        """Upstream's arms never ask for one, so they must be unaffected."""
        solver = self._solver()
        assert solver._solver_args("dReal", "q.smt2") == [
            "dReal", "q.smt2", "--short_sat", "--model"]

    def test_a_configured_delta_reaches_the_binary(self):
        solver = self._solver()
        solver.set_precision(Fraction(1, 100))
        assert solver._solver_args("dReal", "q.smt2") == [
            "dReal", "q.smt2", "--short_sat", "--model", "--precision", "0.01"]

    def test_the_delta_is_rendered_as_a_decimal(self):
        """dReal3's parser has no p/q literal, so an exact rational must not
        reach the command line as one."""
        solver = self._solver()
        solver.set_precision(Fraction(1, 1000))
        assert "--precision" in solver._solver_args("dReal", "q.smt2")
        assert "1/1000" not in solver._solver_args("dReal", "q.smt2")

    def test_the_delta_can_be_cleared(self):
        solver = self._solver()
        solver.set_precision(Fraction(1, 100))
        solver.set_precision(None)
        assert solver._solver_args("dReal", "q.smt2") == [
            "dReal", "q.smt2", "--short_sat", "--model"]