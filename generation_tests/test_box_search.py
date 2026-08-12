"""Unit tests for kappa_box's geometric searches.

Everything here runs against the FakeOracle from conftest, so each expected
value is derived from a known falsifying set rather than recorded from a solver
run. That is deliberate: these are the properties the strategy claims, and they
should hold independently of which backend answers the queries and of whether a
particular benchmark happens to exercise them.
"""

from fractions import Fraction

from conftest import FakeOracle, beyond, covers, probe_at

from stlmc.constraints.constraints import BoolVal, Real, RealVal
from stlmc.generation.box import (
    _BOUNDARY,
    _DEEP,
    _DOMAIN,
    RegionBoxDiscovery,
    _block_box,
    _merge_markers,
    _Theta,
    _too_close,
    _value_of,
    _verdict,
)
from stlmc.generation.oracle import UNKNOWN

TRUE = BoolVal("True")


def assn(**values):
    """An assignment dict keyed by Real variables, as a solver would return."""
    return {Real(name): RealVal(str(Fraction(str(value))))
            for name, value in values.items()}


# ===================================================================== faces

class TestSearchFace:
    """_search_face locates the largest falsifying value towards a wall.

    The four statuses are the contract: a caller distinguishing them wrongly
    turns a resource failure into a geometric claim, which is the error this
    module cares most about.
    """

    def test_frontier_is_bracketed_and_within_tolerance(self, x):
        # falsifying on [0, 4]; wall at 10 is not falsifying, so the search must
        # bracket the frontier at 4 and report it to within tol.
        alg = RegionBoxDiscovery()
        oracle = FakeOracleFor(x, 0, 4)
        tol = Fraction(1, 100)
        bound, status, calls = alg._search_face(
            oracle, x, TRUE, Fraction(0), Fraction(10), tol)
        assert status == "frontier"
        assert bound <= 4 and 4 - bound <= tol, "bound must be falsifying-side"
        assert calls > 1, "a bracketed frontier requires more than the wall probe"

    def test_domain_when_the_falsifying_set_reaches_the_wall(self, x):
        alg = RegionBoxDiscovery()
        oracle = FakeOracleFor(x, 0, 10)          # falsifying up to the wall
        bound, status, calls = alg._search_face(
            oracle, x, TRUE, Fraction(0), Fraction(10), Fraction(1, 100))
        assert status == "domain"
        assert bound == 10
        assert calls == 1, "the wall probe alone settles a domain face"

    def test_partial_when_every_detour_is_undecided(self, x):
        # Falsifying on [0, 4], but the solver cannot decide anything at or
        # beyond 2. The search may still confirm growth below 2, and must then
        # report a LOWER BOUND rather than a frontier.
        alg = RegionBoxDiscovery()
        oracle = FakeOracleFor(x, 0, 4, undecided=beyond(x, 2))
        bound, status, _ = alg._search_face(
            oracle, x, TRUE, Fraction(0), Fraction(10), Fraction(1, 100))
        assert status in ("partial", "frontier")
        if status == "partial":
            assert bound > 0, "partial means growth happened"
            assert bound <= 4, "a reported bound must be falsifying"

    def test_unresolved_when_nothing_beyond_the_pivot_is_decided(self, x):
        # Undecided everywhere at or above the pivot: no growth is confirmable.
        alg = RegionBoxDiscovery()
        oracle = AlwaysUnknown()
        bound, status, _ = alg._search_face(
            oracle, x, TRUE, Fraction(1), Fraction(10), Fraction(1, 100))
        assert status == "unresolved"
        assert bound == 1, "an unresolved face keeps the pivot as its bound"

    def test_unknown_at_the_wall_does_not_abandon_the_face(self, x):
        """An UNKNOWN answer about the wall says nothing about points closer
        in, so the search must continue rather than return the pivot."""
        alg = RegionBoxDiscovery()
        oracle = FakeOracleFor(x, 0, 4, undecided=probe_at(x, 10))
        bound, status, _ = alg._search_face(
            oracle, x, TRUE, Fraction(0), Fraction(10), Fraction(1, 100))
        assert status in ("partial", "frontier")
        assert bound > 0, "growth must still be found below the undecided wall"

    def test_reported_bound_is_always_falsifying(self, x):
        """Across every status: the bound is a point the oracle confirmed."""
        alg = RegionBoxDiscovery()
        for undecided in (None, probe_at(x, 3), beyond(x, Fraction(1, 2)),
                          probe_at(x, 10)):
            oracle = FakeOracleFor(x, 0, 4, undecided=undecided)
            bound, status, _ = alg._search_face(
                oracle, x, TRUE, Fraction(0), Fraction(10), Fraction(1, 100))
            assert bound <= 4, f"status={status} returned a non-falsifying bound"


# ================================================================== harvest

class TestHarvest:
    def test_points_are_spread_and_counted(self, x):
        alg = RegionBoxDiscovery()
        oracle = FakeOracleFor(x, 0, 10)
        got, requested, undecided = alg._harvest(
            oracle, x, TRUE, Fraction(0), Fraction(1), Fraction(1, 10), 8,
            Fraction(1, 20))
        assert requested == 8, "budget binds before extent/theta here"
        assert undecided == 0
        assert len(got) == 8

    def test_extent_binds_when_smaller_than_the_budget(self, x):
        alg = RegionBoxDiscovery()
        oracle = FakeOracleFor(x, 0, 10)
        _, requested, _ = alg._harvest(
            oracle, x, TRUE, Fraction(0), Fraction(1, 4), Fraction(1, 10), 8,
            Fraction(1, 20))
        assert requested == 2, "floor(0.25 / 0.1) = 2"

    def test_undecided_points_are_counted(self, x):
        """A short pool must be attributable: solver failure or empty region."""
        alg = RegionBoxDiscovery()
        oracle = FakeOracleFor(x, 0, 10, undecided=covers(x, Fraction(1, 16)))
        got, requested, undecided = alg._harvest(
            oracle, x, TRUE, Fraction(0), Fraction(1), Fraction(1, 10), 8,
            Fraction(1, 20))
        assert undecided >= 1
        assert len(got) + undecided <= requested

    def test_degenerate_extent_yields_nothing(self, x):
        alg = RegionBoxDiscovery()
        oracle = FakeOracleFor(x, 0, 10)
        got, requested, undecided = alg._harvest(
            oracle, x, TRUE, Fraction(1), Fraction(1), Fraction(1, 10), 8,
            Fraction(1, 20))
        assert (got, requested, undecided) == ([], 0, 0)


# ============================================================ theta per axis

class TestTheta:
    def test_absolute_applies_to_every_axis(self, x, y):
        theta = _Theta(Fraction(1, 20))
        assert theta.of(x) == theta.of(y) == Fraction(1, 20)

    def test_relative_scales_by_declared_range(self, x, y):
        # range_dict: state variable -> (lo_incl, lo, hi, hi_incl)
        range_dict = {Real("x"): (True, "0", "10", True),
                      Real("y"): (True, "-20", "20", True)}
        theta = _Theta(Fraction(1, 100), Fraction(1, 200), range_dict)
        assert theta.of(x) == Fraction(10, 200)     # 0.05
        assert theta.of(y) == Fraction(40, 200)     # 0.20

    def test_relative_falls_back_where_no_range_is_declared(self, x):
        theta = _Theta(Fraction(1, 100), Fraction(1, 200), {})
        assert theta.of(x) == Fraction(1, 100)

    def test_degenerate_range_falls_back(self, x):
        range_dict = {Real("x"): (True, "3", "3", True)}
        theta = _Theta(Fraction(1, 100), Fraction(1, 200), range_dict)
        assert theta.of(x) == Fraction(1, 100)

    def test_relative_equals_absolute_when_the_numbers_coincide(self, x):
        range_dict = {Real("x"): (True, "0", "10", True)}
        assert (_Theta(Fraction(1, 100), Fraction(1, 200), range_dict).of(x)
                == _Theta(Fraction(1, 20)).of(x))


# ================================================================== blocking

class TestBlocking:
    def test_a_point_inside_the_box_is_excluded(self, x, y):
        block = _block_box({x: (Fraction(1), Fraction(2)),
                            y: (Fraction(0), Fraction(1))})
        assert not _satisfies(block, {x: Fraction(3, 2), y: Fraction(1, 2)})

    def test_a_point_outside_on_any_axis_is_admitted(self, x, y):
        block = _block_box({x: (Fraction(1), Fraction(2)),
                            y: (Fraction(0), Fraction(1))})
        assert _satisfies(block, {x: Fraction(5), y: Fraction(1, 2)})
        assert _satisfies(block, {x: Fraction(3, 2), y: Fraction(9)})

    def test_the_faces_themselves_stay_blocked(self, x):
        """Bounds are inclusive in the box, so its faces are excluded too."""
        block = _block_box({x: (Fraction(1), Fraction(2))})
        assert not _satisfies(block, {x: Fraction(1)})
        assert not _satisfies(block, {x: Fraction(2)})


# ============================================================ marker merging

class TestMergeMarkers:
    def test_a_coincident_marker_relabels_instead_of_duplicating(self, x):
        witnesses = [assn(x_0_0=4.0)]
        labels = [_DEEP]
        merged = _merge_markers(
            witnesses, labels, [assn(x_0_0=4.00001)], [_BOUNDARY], [x],
            lambda v: Fraction(1, 100), None)
        assert merged == 1
        assert len(witnesses) == 1, "no duplicate entry for one initial condition"
        assert labels == [_BOUNDARY], "the frontier annotation survives"

    def test_a_distinct_marker_is_appended(self, x):
        witnesses = [assn(x_0_0=4.0)]
        labels = [_DEEP]
        merged = _merge_markers(
            witnesses, labels, [assn(x_0_0=5.0)], [_BOUNDARY], [x],
            lambda v: Fraction(1, 100), None)
        assert merged == 0
        assert labels == [_DEEP, _BOUNDARY]

    def test_an_existing_marker_label_is_not_downgraded(self, x):
        witnesses = [assn(x_0_0=4.0)]
        labels = [_DOMAIN]
        _merge_markers(witnesses, labels, [assn(x_0_0=4.0)], [_BOUNDARY], [x],
                       lambda v: Fraction(1, 100), None)
        assert labels == [_DOMAIN], "domain outranks a coincident boundary"

    def test_coincidence_requires_every_axis(self, x, y):
        witnesses = [assn(x_0_0=4.0, y_0_0=1.0)]
        labels = [_DEEP]
        _merge_markers(witnesses, labels,
                       [assn(x_0_0=4.0, y_0_0=9.0)], [_BOUNDARY], [x, y],
                       lambda v: Fraction(1, 100), None)
        assert len(witnesses) == 2, "far apart on y is not a coincidence"

    def test_tolerance_may_differ_per_axis(self, x, y):
        witnesses = [assn(x_0_0=0.0, y_0_0=0.0)]
        labels = [_DEEP]
        tol = {x: Fraction(1, 10), y: Fraction(1, 1000)}
        _merge_markers(witnesses, labels,
                       [assn(x_0_0=0.05, y_0_0=0.01)], [_BOUNDARY], [x, y],
                       lambda v: tol[v], None)
        assert len(witnesses) == 2, "inside tol on x, outside on y -> distinct"


# =================================================================== thinning

def test_too_close_is_an_l_infinity_ball(x, y):
    pool = [assn(x_0_0=0.0, y_0_0=0.0)]
    assert _too_close(assn(x_0_0=0.01, y_0_0=0.01), pool, [x, y], Fraction(1, 20))
    assert not _too_close(assn(x_0_0=0.01, y_0_0=1.0), pool, [x, y], Fraction(1, 20))


# =================================================================== helpers

def _satisfies(formula, point):
    """Evaluate a block formula (an Or of strict inequalities) at a point."""
    from stlmc.constraints.constraints import And, Gt, Lt, Or

    def go(node):
        if isinstance(node, Or):
            return any(go(child) for child in node.children)
        if isinstance(node, And):
            return all(go(child) for child in node.children)
        if isinstance(node, BoolVal):
            return str(node.value) == "True"
        left, right = node.left, node.right
        value = point[left]
        bound = Fraction(str(right.value))
        if isinstance(node, Lt):
            return value < bound
        if isinstance(node, Gt):
            return value > bound
        raise AssertionError(f"unexpected node {type(node).__name__}")

    return go(formula)


class FakeOracleFor:
    """conftest.FakeOracle for a single variable, with a readable signature."""

    def __new__(cls, var, lo, hi, undecided=None):
        return FakeOracle({var: (Fraction(str(lo)), Fraction(str(hi)))},
                          undecided=undecided)


class AlwaysUnknown:
    is_exact = False
    tolerance = Fraction(1, 1000)

    def rv(self, f):
        return RealVal(str(f))

    def push(self):
        pass

    def pop(self):
        pass

    def assert_(self, formula):
        pass

    def check(self):
        return UNKNOWN

    def check_with(self, formula):
        return UNKNOWN

    def model(self):
        raise AssertionError("model() must not be called after UNKNOWN")


# =================================================================== verdicts

class TestVerdict:
    """A verdict may never cover ground the run did not visit."""

    def test_a_non_empty_pool_is_false(self):
        result, note = _verdict(["a witness"], False, [1, 2], 2)
        assert (result, note) == ("False", None)

    def test_unresolved_beats_true(self):
        result, note = _verdict([], True, [1, 2], 2)
        assert result == "Unknown"
        assert "unresolved" in note

    def test_skipped_depths_forbid_true(self):
        result, note = _verdict([], False, [1], 5)
        assert result == "Unknown"
        assert "2/3/4/5" in note, "the note must name what was skipped"

    def test_exhaustive_absence_is_true(self):
        assert _verdict([], False, [0, 1, 2, 3], 3) == ("True", None)

    def test_a_pool_outranks_a_skipped_depth(self):
        """Finding a CE is a positive result: it does not need full coverage."""
        assert _verdict(["a witness"], False, [5], 5)[0] == "False"

    def test_order_and_duplicates_in_the_depth_list_do_not_matter(self):
        assert _verdict([], False, [3, 1, 0, 2, 2], 3) == ("True", None)


# =================================================================== lattice

class TestHarvestLattice:
    """The lattice pins every axis, so a witness lands in a known cell.

    The per-axis sweep leaves the other axes free anywhere in the box, which is
    why theta could not control per-axis spacing and why witness count grew as
    the sum over axes rather than the product.
    """

    def test_one_dimension_agrees_with_the_sweep(self, x):
        """Single-variable models must be unaffected by the change."""
        alg = RegionBoxDiscovery()
        box = {x: [Fraction(0), Fraction(1)]}
        theta = _Theta(Fraction(1, 10))

        lattice_oracle = FakeOracle({x: (Fraction(0), Fraction(10))})
        got_l, req_l, _ = alg._harvest_lattice(lattice_oracle, box, theta, 8, 200)
        sweep_oracle = FakeOracle({x: (Fraction(0), Fraction(10))})
        got_s, req_s, _ = alg._harvest(
            sweep_oracle, x, TRUE, Fraction(0), Fraction(1), Fraction(1, 10), 8,
            Fraction(1, 20))

        assert req_l == req_s
        assert ([_value_of(w, x) for w in got_l]
                == [_value_of(w, x) for w in got_s])

    def test_two_dimensions_query_the_product_grid(self, x, y):
        alg = RegionBoxDiscovery()
        box = {x: [Fraction(0), Fraction(1)], y: [Fraction(0), Fraction(1)]}
        theta = _Theta(Fraction(1, 4))            # 4 cells per axis
        oracle = FakeOracle({x: (Fraction(0), Fraction(10)),
                             y: (Fraction(0), Fraction(10))})
        got, requested, _ = alg._harvest_lattice(oracle, box, theta, 8, 200)
        assert requested == 16, "4 x 4 cells"
        assert len(got) == 16

    def test_every_witness_lies_inside_the_box(self, x, y):
        alg = RegionBoxDiscovery()
        box = {x: [Fraction(1), Fraction(2)], y: [Fraction(-1), Fraction(1)]}
        theta = _Theta(Fraction(1, 2))
        oracle = FakeOracle({x: (Fraction(0), Fraction(10)),
                             y: (Fraction(-10), Fraction(10))})
        got, _, _ = alg._harvest_lattice(oracle, box, theta, 8, 200)
        assert got
        for witness in got:
            assert box[x][0] <= _value_of(witness, x) <= box[x][1]
            assert box[y][0] <= _value_of(witness, y) <= box[y][1]

    def test_cells_shrink_proportionally_under_the_cap(self, x, y):
        alg = RegionBoxDiscovery()
        box = {x: [Fraction(0), Fraction(1)], y: [Fraction(0), Fraction(1)]}
        theta = _Theta(Fraction(1, 100))          # 100 cells per axis uncapped
        oracle = FakeOracle({x: (Fraction(0), Fraction(10)),
                             y: (Fraction(0), Fraction(10))})
        _, requested, _ = alg._harvest_lattice(oracle, box, theta, 100, 50)
        assert requested <= 50, "the cap must bound the product"
        assert requested >= 25, "and must not collapse the lattice to a line"

    def test_undecided_cells_are_counted(self, x, y):
        alg = RegionBoxDiscovery()
        box = {x: [Fraction(0), Fraction(1)], y: [Fraction(0), Fraction(1)]}
        theta = _Theta(Fraction(1, 2))
        oracle = FakeOracle({x: (Fraction(0), Fraction(10)),
                             y: (Fraction(0), Fraction(10))},
                            undecided=covers(x, Fraction(1, 4)))
        got, requested, undecided = alg._harvest_lattice(oracle, box, theta, 8, 200)
        assert undecided > 0
        assert len(got) + undecided == requested

    def test_a_degenerate_axis_yields_nothing(self, x, y):
        alg = RegionBoxDiscovery()
        box = {x: [Fraction(1), Fraction(1)], y: [Fraction(0), Fraction(1)]}
        theta = _Theta(Fraction(1, 4))
        oracle = FakeOracle({x: (Fraction(0), Fraction(10)),
                             y: (Fraction(0), Fraction(10))})
        assert alg._harvest_lattice(oracle, box, theta, 8, 200) == ([], 0, 0)


# ====================================================== the shared procedure

class TestGrowBoxUnderEitherOracle:
    """_grow_box is one procedure for an exact and a partial oracle.

    The end-to-end tests exercise it against an exact backend; these exercise
    the partial instantiation, which no real backend is needed to reach: the
    oracle's tolerance and its undecided answers are what distinguish the two,
    and both are properties of the oracle rather than of the dynamics.
    """

    @staticmethod
    def _encoding(box_vars):
        class _Encoding:
            range_dict = {Real(v.id[:-4]): (True, "0", "10", True) for v in box_vars}
            bound = 1
        return _Encoding()

    @staticmethod
    def _pivot(**values):
        return assn(**values)

    def _grow(self, oracle, theta, budget=4, iters=20):
        alg = RegionBoxDiscovery()
        alg._config = None
        pivot = self._pivot(x_0_0=5)
        return alg._grow_box(oracle, pivot, self._encoding([Real("x_0_0")]),
                             theta, iters, 1, budget, _SilentPrinter())

    def test_a_partial_oracle_yields_a_labeled_box(self, x):
        oracle = FakeOracle({x: (Fraction(4), Fraction(6))},
                            tolerance=Fraction(1, 1000))
        witnesses, labels, box = self._grow(oracle, _Theta(Fraction(1, 4)))
        assert witnesses and len(witnesses) == len(labels)
        assert set(labels) <= {_DEEP, _BOUNDARY, _DOMAIN}
        lo, hi = box[x]
        assert Fraction(4) <= lo <= Fraction(5) <= hi <= Fraction(6)

    def test_an_exact_oracle_locates_a_tighter_frontier(self, x):
        """Precision follows the oracle: theta/2**iters against theta/8."""
        theta = _Theta(Fraction(1, 4))
        exact = FakeOracle({x: (Fraction(4), Fraction(6))}, tolerance=Fraction(0))
        exact.is_exact = True
        partial = FakeOracle({x: (Fraction(4), Fraction(6))},
                             tolerance=Fraction(1, 1000))
        _, _, box_exact = self._grow(exact, theta, iters=12)
        _, _, box_partial = self._grow(partial, theta)
        assert abs(box_exact[x][1] - 6) <= abs(box_partial[x][1] - 6)

    def test_undecided_probes_do_not_produce_a_wrong_box(self, x):
        """Whatever the oracle refuses to decide, the box stays falsifying."""
        oracle = FakeOracle({x: (Fraction(4), Fraction(6))},
                            undecided=beyond(x, Fraction(11, 2)),
                            tolerance=Fraction(1, 1000))
        witnesses, labels, box = self._grow(oracle, _Theta(Fraction(1, 4)))
        lo, hi = box[x]
        assert Fraction(4) <= lo and hi <= Fraction(6), (
            "an undecided probe must cost extent, never correctness")
        for witness in witnesses:
            value = _value_of(witness, x)
            assert Fraction(4) <= value <= Fraction(6)

    def test_witnesses_are_theta_separated_under_a_partial_oracle(self, x):
        theta = _Theta(Fraction(1, 4))
        oracle = FakeOracle({x: (Fraction(4), Fraction(6))},
                            tolerance=Fraction(1, 1000))
        witnesses, labels, _ = self._grow(oracle, theta, budget=8)
        deep = sorted(_value_of(w, x)
                      for w, label in zip(witnesses, labels) if label == _DEEP)
        gaps = [b - a for a, b in zip(deep, deep[1:])]
        assert all(g >= Fraction(1, 4) for g in gaps), gaps


class _SilentPrinter:
    def print_normal(self, *_a, **_k):
        pass

    def print_verbose(self, *_a, **_k):
        pass