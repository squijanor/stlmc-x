"""Witness classification, reconstruction at a jump, and what a report says
about its own reliability.

The bands are three comparisons against two numbers the run supplies, so they
are tested against values placed on and around each edge rather than against a
recorded run: an edge case is exactly what a recorded run does not contain. The
reconstruction is tested the same way, on crafted traces whose every value is
chosen by hand: a jump on the evaluation instant and a non-finite robustness are
cases a recorded pool of well-formed witnesses does not contain either.
"""

import pytest

from stlmc.constraints.constraints import (
    And,
    BoolVal,
    Eq,
    Function,
    Geq,
    GloballyFormula,
    Int,
    Interval,
    Ode,
    Real,
    RealVal,
    Sqrt,
)
from stlmc.generation.validate import (
    _band_margin,
    _classify,
    _own_post_jump,
    _trace_faults,
    pool_backend_delta,
    validate_ce,
    validate_pool,
)

TAU = 0.1
DELTA = 0.001


class TestClassify:
    def test_a_clear_violation_is_a_falsifier(self):
        assert _classify(-1.0, TAU, DELTA) == "falsifier"

    def test_a_violation_inside_the_backend_slack_is_marginal(self):
        assert _classify(-0.0005, TAU, DELTA) == "marginal"

    def test_the_slack_edge_belongs_to_the_falsifier_band(self):
        """delta is the backend's own error, so a violation exceeding it is a
        violation; at exactly delta the two are the same size."""
        assert _classify(-0.002, TAU, DELTA) == "falsifier"
        assert _classify(-DELTA, TAU, DELTA) == "marginal"

    def test_zero_is_threshold_only(self):
        assert _classify(0.0, TAU, DELTA) == "threshold-only"

    def test_the_threshold_edge_is_closed(self):
        """The encoding applies tau by relaxing the negated goal, so it asks for
        rho(0) <= tau. A witness at exactly tau is what the search was told to
        find, and must not depend on rounding in the reconstruction."""
        assert _classify(TAU, TAU, DELTA) == "threshold-only"

    def test_above_the_threshold_is_unverified(self):
        assert _classify(TAU * 1.5, TAU, DELTA) == "unverified"

    def test_an_exact_backend_collapses_the_marginal_band(self):
        assert _classify(-1e-12, TAU, 0.0) == "falsifier"


class TestClassifyNonFinite:
    """A non-finite rho(0) is not a band. NaN fails every comparison and would
    fall through to `unverified`; -inf compares below every edge and would read
    as a clean `falsifier`. Both are routed to `error` instead, so a divergent
    integration is never reported as a clean counterexample."""

    def test_nan_is_an_error_not_unverified(self):
        assert _classify(float("nan"), TAU, DELTA) == "error"

    def test_negative_infinity_is_an_error_not_a_falsifier(self):
        assert _classify(float("-inf"), TAU, DELTA) == "error"

    def test_positive_infinity_is_an_error(self):
        assert _classify(float("inf"), TAU, DELTA) == "error"

    def test_a_finite_value_is_unaffected(self):
        assert _classify(-1.0, TAU, DELTA) == "falsifier"


class TestBandMargin:
    def test_the_margin_is_the_distance_to_the_nearest_edge(self):
        # edges at -delta, 0 and tau; 0.02 is nearest 0.
        assert _band_margin(0.02, TAU, DELTA) == 0.02
        # 0.09 is nearest tau.
        assert abs(_band_margin(0.09, TAU, DELTA) - 0.01) < 1e-12

    def test_a_value_on_an_edge_has_no_margin(self):
        assert _band_margin(TAU, TAU, DELTA) == 0.0
        assert _band_margin(0.0, TAU, DELTA) == 0.0

    def test_the_margin_is_measured_beyond_the_outermost_edges_too(self):
        assert abs(_band_margin(-0.5, TAU, DELTA) - 0.499) < 1e-12


class TestPoolBackendDelta:
    def test_a_pool_that_records_a_relaxation_reports_it(self):
        payload = tuple(range(10)) + (0.001,)
        assert pool_backend_delta(payload) == 0.001

    def test_a_pool_predating_the_element_reads_as_exact(self):
        """A shorter payload is read as it was read before the element existed,
        rather than as an unknown relaxation."""
        assert pool_backend_delta(tuple(range(10))) == 0.0
        assert pool_backend_delta(tuple(range(9))) == 0.0


# ---- crafted traces -------------------------------------------------------
#
# A trace is reconstructed from the tau values, the per-segment initial x and
# the mode of each segment. A constant flow x(t) = x makes every sample of a
# segment equal to that segment's own initial x, so the robustness of an atom
# over x at any instant is fixed by which segment owns that instant -- which is
# exactly what the post-jump rule decides. The expected values below are read
# straight off the trace, not recorded from a run.


def _const_flow():
    return Function([Real("x")], [Real("x")])  # x(t) = x


def _nan_flow():
    return Ode([Real("x")], [Sqrt(RealVal("-1"))])  # dx/dt = sqrt(-1) = nan


def _trace(seg_x, taus, flow):
    """One x and one mode per segment; identity elsewhere. Returns (assn, rest),
    the two arguments `_reconstruct` reads (rest[0:5] = modules, mode_var_dict,
    propositions, cont_var_dict, prop_dict)."""
    assn = {}
    for k, t in enumerate(taus):
        assn[Real(f"tau_{k}")] = RealVal(str(t))
    for k, value in enumerate(seg_x):
        assn[Real(f"currentMode_{k}")] = RealVal(str(k))
        assn[Real(f"x_{k}_0")] = RealVal(str(value))
        assn[Real(f"x_{k}_t")] = RealVal(str(value))
    modules = [{"flow": flow} for _ in seg_x]
    rest = [modules, {}, {}, {Real("x"): None}, {}, None, None, None]
    return assn, rest


def _global_at(instant):
    """[][instant,instant] (x >= 5): rho(0) is (x - 5) at that single instant,
    so it reads whichever segment owns the instant."""
    edge = RealVal(str(instant))
    local = Interval(True, edge, True, edge)
    horizon = Interval(True, RealVal("0"), True, RealVal("100"))
    return GloballyFormula(local, horizon, Geq(Real("x"), RealVal("5")))


def _payload(assn, rest, formula, tau=TAU, delta=0.0):
    modules, mode_var_dict, propositions, cont_var_dict, prop_dict = rest[0:5]
    return (
        [assn],
        modules,
        mode_var_dict,
        propositions,
        cont_var_dict,
        prop_dict,
        formula,
        None,
        tau,
        [""],
        delta,
    )


class TestPostJumpOwnership:
    """The reconstruction gives a shared variable point to the segment that
    begins at it. `_own_post_jump` is the mechanism, checked directly."""

    def test_a_shared_variable_point_goes_to_the_later_segment(self):
        times = [[0.0, 0.5, 1.0], [1.0, 1.5, 2.0], [2.0, 2.5, 3.0]]
        point_samples = {"x": [[0, 0, 9], [1, 1, 1], [2, 2, 2]]}
        discrete = {"m": [[0, 0, 0], [1, 1, 1], [2, 2, 2]]}
        _own_post_jump(times, point_samples, discrete)
        # 1.0 and 2.0 survive only in the segment that begins at them; the last
        # segment keeps its closed right endpoint, the end of the trace.
        assert times == [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5, 3.0]]
        assert point_samples["x"] == [[0, 0], [1, 1], [2, 2, 2]]
        assert discrete["m"] == [[0, 0], [1, 1], [2, 2, 2]]

    def test_a_zero_duration_segment_cedes_its_instant(self):
        times = [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]
        point_samples = {"x": [[9, 9, 9], [1, 1, 1]]}
        discrete = {}
        _own_post_jump(times, point_samples, discrete)
        assert times == [[], [0.0, 1.0, 2.0]]
        assert point_samples["x"] == [[], [1, 1, 1]]


class TestJumpBoundaryValue:
    """At a jump the pre- and post-jump states differ; the instant is read in
    the mode entered at it. Here x drops from 10 to 0 across the jump, and the
    property x >= 5 holds before and fails after, so the post-jump reading is a
    falsifier where the pre-jump reading was not."""

    def test_a_jump_boundary_is_read_post_jump(self):
        # x: 10 on [0,1], 0 on [1,2]; evaluate x >= 5 at the jump instant 1.
        assn, rest = _trace([10, 0], [0, 1, 2], _const_flow())
        verdict, rho0, _, _ = validate_ce(assn, rest, TAU, 0.0, _global_at(1))
        assert rho0 == -5.0
        assert verdict == "falsifier"

    def test_a_jump_at_the_final_variable_point_is_read_post_jump(self):
        # taus 0,1,2,2: the last segment is [2,2]; x is 0 there. The jump at the
        # final variable point is read in that segment.
        assn, rest = _trace([10, 10, 0], [0, 1, 2, 2], _const_flow())
        verdict, rho0, _, _ = validate_ce(assn, rest, TAU, 0.0, _global_at(2))
        assert rho0 == -5.0
        assert verdict == "falsifier"

    def test_a_jump_on_the_evaluation_instant_is_read_post_jump(self):
        # taus 0,0,2: the first segment is [0,0] and holds no sample, so rho(0)
        # is read in the segment entered at 0, where x is 0.
        assn, rest = _trace([10, 0], [0, 0, 2], _const_flow())
        verdict, rho0, _, _ = validate_ce(assn, rest, TAU, 0.0, _global_at(0))
        assert rho0 == -5.0
        assert verdict == "falsifier"


class TestNonFiniteRobustness:
    """A reconstruction that yields a non-finite rho(0) -- a divergent flow, or
    a formula that is identically true or false at the instant -- is an error,
    not a band, and the refinement it cannot inform is skipped."""

    def test_a_negative_infinity_robustness_is_an_error_not_a_falsifier(self):
        # []_1 False is -inf everywhere; the pre-fix reading was `falsifier`.
        assn, rest = _trace([1], [0, 1], _const_flow())
        horizon = Interval(True, RealVal("0"), True, RealVal("100"))
        phi = GloballyFormula(
            Interval(True, RealVal("1"), True, RealVal("1")), horizon, BoolVal("False")
        )
        verdict, rho0, _, _ = validate_ce(assn, rest, TAU, 0.0, phi)
        assert verdict == "error"

    @pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
    def test_a_nan_flow_is_an_error(self):
        assn, rest = _trace([1], [0, 1], _nan_flow())
        verdict, _, _, _ = validate_ce(assn, rest, TAU, 0.0, _global_at(1))
        assert verdict == "error"

    @pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
    def test_validate_pool_records_a_non_finite_rho0_as_error(self):
        assn, rest = _trace([1], [0, 1], _nan_flow())
        rec = validate_pool(_payload(assn, rest, _global_at(1)))[0]
        assert rec["verdict"] == "error"
        assert "non-finite" in rec["note"]
        # the band comparisons a non-finite number cannot inform are skipped
        assert rec["stable"] == ""
        assert rec["resolved"] == ""


# ---- trace consistency ----------------------------------------------------
#
# The trace check reads the assignment's own boundary variables and asks whether
# a jump is one the automaton admits. A two-mode model with one guarded jump
# places a witness on either side of the guard; the values are chosen by hand,
# not recorded from a run.

_MODE = {"m": Int("m")}
_RANGE = [Real("x")]


def _jump_model():
    """Two modes; mode 0 jumps to mode 1 under x >= 1.3, carrying x. Mode 1 is
    terminal. A primed name carries a suffix, as the encoding writes it."""
    guard = Geq(Real("x"), RealVal("1.3"))
    reset = And([Eq(Int("m'"), RealVal("1")), Eq(Real("x'"), Real("x"))])
    return [{"jump": {guard: reset}}, {"jump": {}}]


def _two_segment_assn(x_exit, x_entry, post_mode=1, tau1=0.7, dwell=0.7):
    return {
        Real("m_0"): RealVal("0"),
        Real("m_1"): RealVal(str(post_mode)),
        Real("x_0_t"): RealVal(str(x_exit)),
        Real("x_1_0"): RealVal(str(x_entry)),
        Real("tau_0"): RealVal("0"),
        Real("tau_1"): RealVal(str(tau1)),
        Real("time_0"): RealVal(str(dwell)),
    }


class TestTraceCheck:
    def test_a_jump_below_the_guard_is_flagged(self):
        trace, guard_margin, _ = _trace_faults(
            _two_segment_assn(1.12, 1.12), _jump_model(), _MODE, _RANGE, 0.0
        )
        assert trace == "guard-violating"
        assert guard_margin < 0

    def test_a_jump_that_meets_the_guard_is_consistent(self):
        trace, guard_margin, _ = _trace_faults(
            _two_segment_assn(1.4, 1.4), _jump_model(), _MODE, _RANGE, 0.0
        )
        assert trace == "consistent"
        assert guard_margin > 0

    def test_a_guard_met_within_the_backend_slack_is_not_a_violation(self):
        # x_0_t = 1.2995 misses the guard by 5e-4, inside a 1e-3 backend delta.
        trace, _, _ = _trace_faults(
            _two_segment_assn(1.2995, 1.2995), _jump_model(), _MODE, _RANGE, 1e-3
        )
        assert trace == "consistent"

    def test_a_reset_no_edge_produces_is_a_mismatch(self):
        # mode changes 0 -> 1 but x is not carried, so reset x' = x fails.
        trace, _, _ = _trace_faults(
            _two_segment_assn(1.4, 0.2), _jump_model(), _MODE, _RANGE, 0.0
        )
        assert trace == "reset-mismatch"

    def test_an_unchanged_mode_carried_identically_is_a_stutter(self):
        # no declared jump reaches mode 0 from mode 0; identity makes it a
        # stutter, which needs no guard.
        trace, _, _ = _trace_faults(
            _two_segment_assn(0.5, 0.5, post_mode=0), _jump_model(), _MODE, _RANGE, 0.0
        )
        assert trace == "consistent"

    def test_a_dwell_that_disagrees_with_its_endpoints_is_flagged(self):
        # tau_1 - tau_0 = 0.7 but time_0 = 0.9: a 0.2 disagreement.
        trace, _, dwell_slack = _trace_faults(
            _two_segment_assn(1.4, 1.4, dwell=0.9), _jump_model(), _MODE, _RANGE, 0.0
        )
        assert trace == "time-mismatch"
        assert abs(dwell_slack) > 0.1

    def test_a_payload_without_mode_structure_is_left_unchecked(self):
        assn, rest = _trace([1], [0, 1], _const_flow())
        trace, guard_margin, dwell = _trace_faults(assn, rest[0], {}, [], 0.0)
        assert trace == "" and guard_margin is None and dwell is None
