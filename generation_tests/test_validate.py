"""Witness classification and what a report says about its own reliability.

The bands are three comparisons against two numbers the run supplies, so they
are tested against values placed on and around each edge rather than against a
recorded run: an edge case is exactly what a recorded run does not contain.
"""

from stlmc.generation.validate import (
    _band_margin,
    _classify,
    pool_backend_delta,
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