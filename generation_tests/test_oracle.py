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