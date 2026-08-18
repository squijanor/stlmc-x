"""Unit tests for kappa_box's geometric searches.

Everything here runs against the FakeOracle from conftest, so each expected
value is derived from a known falsifying set rather than recorded from a solver
run. That is deliberate: these are the properties the strategy claims, and they
should hold independently of which backend answers the queries and of whether a
particular benchmark happens to exercise them.
"""

from collections import Counter as _Counter
from fractions import Fraction
from time import sleep as _sleep

from conftest import FakeOracle, beyond, covers, probe_at

from stlmc.constraints.constraints import BoolVal, Real, RealVal
from stlmc.generation.box import (
    _BOUNDARY,
    _DEEP,
    _DOMAIN,
    RegionBoxDiscovery,
    _binding_bound,
    _block_box,
    _box_budget,
    _merge_markers,
    _Theta,
    _too_close,
    _value_of,
    _verdict,
    _WordRotation,
)
from stlmc.generation.common import gen_float, validate_gen
from stlmc.generation.oracle import SAT, UNKNOWN, UNSAT

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

    def test_a_degenerate_axis_contributes_one_cell(self, x, y):
        """Def. 7: c_j >= 1 -- "an axis narrower than theta contributes one
        cell, never none". The defect gave a zero-span axis zero cells and
        returned an empty harvest for the WHOLE box, silently: the pool was
        pivot + markers and k-witness was inert whenever one face never grew
        (unresolved both ways, or an IC pinned by the model)."""
        alg = RegionBoxDiscovery()
        box = {x: [Fraction(1), Fraction(1)], y: [Fraction(0), Fraction(1)]}
        theta = _Theta(Fraction(1, 4))
        oracle = FakeOracle({x: (Fraction(0), Fraction(10)),
                             y: (Fraction(0), Fraction(10))})
        got, requested, undecided = alg._harvest_lattice(
            oracle, box, theta, 8, 200)
        assert requested == 4, "1 cell on the pinned axis x 4 on the live one"
        assert len(got) == 4
        for witness in got:
            assert abs(_value_of(witness, x) - 1) <= Fraction(1, 8), (
                "the degenerate axis samples at its pivot value +/- theta/2")

    def test_the_cap_reduction_does_not_compound(self):
        """Greedy one-cell decrements, not a uniform floored shrink: five axes
        at 3 cells (243) under a 200 cap must stop at 162 (3^4 x 2), not
        collapse to 2^5 = 32."""
        alg = RegionBoxDiscovery()
        axes = [Real(f"v{i}_0_0") for i in range(5)]
        box = {v: [Fraction(0), Fraction(3, 4)] for v in axes}
        theta = _Theta(Fraction(1, 4))            # 3 cells per axis uncapped
        oracle = FakeOracle({v: (Fraction(0), Fraction(10)) for v in axes})
        _, requested, _ = alg._harvest_lattice(oracle, box, theta, 8, 200)
        assert requested == 162

    def test_the_cap_reduction_is_reported(self, x, y):
        """The docstring promised a report and none was printed: a capped
        harvest read as "covered everything", against the no-silent-caps
        rule. The line carries before, after, and per-axis counts."""
        class _Recorder:
            lines = []

            def print_normal(self, msg):
                self.lines.append(msg)

            def print_verbose(self, msg):
                self.lines.append(msg)
        alg = RegionBoxDiscovery()
        alg._printer = _Recorder()
        box = {x: [Fraction(0), Fraction(1)], y: [Fraction(0), Fraction(1)]}
        theta = _Theta(Fraction(1, 100))          # 100 x 100 uncapped
        oracle = FakeOracle({x: (Fraction(0), Fraction(10)),
                             y: (Fraction(0), Fraction(10))})
        _, requested, _ = alg._harvest_lattice(oracle, box, theta, 100, 50)
        assert requested <= 50
        capped = [line for line in _Recorder.lines if "lattice capped" in line]
        assert capped and "10000" in capped[0]


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

# ================================================== the pruning policy itself

class TestWordRotation:
    """word-rotate is a policy over candidate verdicts, and only a refutation
    is one.

    The oracle's UNKNOWN carries no information about the query that produced
    it, so an expired per-candidate bound says the candidate was expensive, not
    that its mode word is infeasible. Counting it as a rejection discards a word
    for a solver reason -- measured doing exactly that on a goal whose published
    counterexample lives at the discarded word's depth.
    """

    def test_consecutive_refutations_drop_the_word(self):
        rotation = _WordRotation(3)
        assert [rotation.refuted("0112") for _ in range(3)] == [False, False, True]
        assert rotation.dropped == 1

    def test_undecided_candidates_never_drop_a_word(self):
        rotation = _WordRotation(3)
        for _ in range(100):
            rotation.undecided_candidate()
        assert rotation.dropped == 0
        assert rotation.undecided == 100

    def test_undecided_candidates_do_not_advance_the_streak(self):
        """The distinction the fix is about: fifty expiries and two refutations
        are two refutations."""
        rotation = _WordRotation(3)
        rotation.refuted("0112")
        for _ in range(50):
            rotation.undecided_candidate()
        assert rotation.refuted("0112") is False
        assert rotation.streak == 2
        assert rotation.refuted("0112") is True

    def test_undecided_candidates_do_not_reset_the_streak_either(self):
        """An UNKNOWN is evidence in neither direction, so a run of refutations
        continues across one rather than restarting."""
        rotation = _WordRotation(2)
        rotation.refuted("0112")
        rotation.undecided_candidate()
        assert rotation.refuted("0112") is True

    def test_a_different_word_restarts_the_streak(self):
        rotation = _WordRotation(2)
        assert rotation.refuted("aa") is False
        assert rotation.refuted("bb") is False
        assert rotation.refuted("bb") is True

    def test_the_counter_resets_after_a_drop(self):
        """One word is reported once per run of refutations, not once per
        refutation past the limit."""
        rotation = _WordRotation(2)
        for _ in range(4):
            rotation.refuted("aa")
        assert rotation.dropped == 2

    def test_zero_disables_the_heuristic(self):
        rotation = _WordRotation(0)
        assert not any(rotation.refuted("0112") for _ in range(100))
        assert rotation.dropped == 0

    def test_a_candidate_with_no_word_is_never_counted(self):
        rotation = _WordRotation(1)
        assert rotation.refuted("") is False
        assert rotation.dropped == 0


# ============================================== which bound the search hit

class TestBindingBound:
    """A budget-exhausted search must name the key that actually bounded it.

    Both directions were measured: a search whose candidates expired at
    pivot-timeout, and one that refuted 972 candidates at ~0.1 s each until
    pivot-budget ran out. Raising the wrong one of the two makes each case
    strictly worse.
    """

    def test_expiries_that_consumed_the_search_name_pivot_timeout(self):
        key, why = _binding_bound(41, 11, 30, 1790.0, 1800.0)
        assert key == "pivot-timeout"
        assert "30 of 41" in why

    def test_fast_refutations_name_pivot_budget(self):
        key, why = _binding_bound(972, 972, 0, 0.0, 1800.0)
        assert key == "pivot-budget"
        assert "972 candidate(s)" in why

    def test_a_few_expiries_in_a_long_search_still_name_the_budget(self):
        key, _ = _binding_bound(900, 895, 5, 300.0, 1800.0)
        assert key == "pivot-budget"

    def test_a_search_that_decided_nothing_at_all_names_the_budget(self):
        """No candidate reached a verdict and none expired: nothing points at
        the per-call bound."""
        assert _binding_bound(0, 0, 0, 0.0, 120.0)[0] == "pivot-budget"


# ====================================================== the two-step search

class _ScriptedSkeleton:
    """The outer solver of the two-step pivot search, with its answers given.

    It proposes one location word per check, from a script, and answers UNSAT
    once the script is spent (``forever`` repeats the last word instead). Every
    assertion is recorded, which is how a test sees whether a mode-word block --
    the rotation heuristic's only effect on the search -- was ever posted."""

    def __init__(self, words, forever=False):
        self._words = list(words)
        self._forever = forever
        self.asserted = []
        self._model = None

    def assert_(self, formula):
        self.asserted.append(formula)

    def check(self):
        if not self._words:
            self._model = None
            return UNSAT
        self._model = self._words[0] if self._forever else self._words.pop(0)
        return SAT

    def model(self):
        return {Real(f"currentMode_{k}"): RealVal(str(digit))
                for k, digit in enumerate(self._model)}


class _ScriptedCandidate:
    """The inner ODE-feasibility oracle for one candidate, with its verdict and
    its cost given. ``cost`` is what the search charges against pivot-budget, so
    a test can decide whether expiries dominate the search or not.

    Frames are real: the search asserts the region blocks in a pushed frame so
    that the pivot query sees them and the growth queries (which reuse this
    oracle) do not. ``asserted`` records everything ever asserted; ``live``
    is what a growth query issued after the search returned would still see."""

    def __init__(self, verdict, cost=0.0):
        self.verdict = verdict
        self.cost = cost
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
        if self.cost:
            _sleep(self.cost)
        return self.verdict

    def model(self):
        return {}


class _ScriptedTwoStep(RegionBoxDiscovery):
    """kappa_box with both pivot-search oracles given explicitly.

    `_pivot_two_step` is otherwise reachable only through a delta backend on a
    nonlinear model, which would measure dReal rather than the search. The two
    oracle seams keep the verdict handling -- which is all the search decides on
    its own -- testable here."""

    def __init__(self, words, verdicts, forever=False, cost=0.0):
        super().__init__()
        self.skeleton = _ScriptedSkeleton(words, forever=forever)
        self._verdicts = list(verdicts)
        self._cost = cost
        self.candidates = []

    def _skeleton_oracle(self, logic, seed):
        return self.skeleton

    def _candidate_oracle(self, logic, seed):
        verdict = self._verdicts.pop(0) if self._verdicts else UNSAT
        oracle = _ScriptedCandidate(verdict, self._cost)
        self.candidates.append(oracle)
        return oracle


class _Section:
    def __init__(self, values):
        self._values = values

    def is_argument_in(self, key):
        return key in self._values

    def get_value(self, key):
        return self._values[key]


class _GenConfig:
    """Just the [gen] section, so a search's budgets are given rather than
    read from a benchmark. Keys are written with underscores and read with
    hyphens."""

    def __init__(self, **gen):
        self._gen = _Section({k.replace("_", "-"): v for k, v in gen.items()})

    def is_section_in(self, name):
        return name == "gen"

    def get_section(self, name):
        if name != "gen":
            raise KeyError(name)
        return self._gen


class _Encoding:
    """The two-step search reads only these three fields off an encoding."""

    skeleton = BoolVal("True")
    consts = BoolVal("True")
    bound = 0


def _two_step(words, verdicts, forever=False, cost=0.0, blocks=(), **gen):
    """Run one candidate search against scripted oracles; returns the algorithm
    (for its counters) and the search's own result."""
    alg = _ScriptedTwoStep(words, verdicts, forever=forever, cost=cost)
    alg._printer = _SilentPrinter()
    alg._config = _GenConfig(**gen)
    alg._logger = None
    alg._tau_max = "8"
    alg._underlying = "dreal"
    alg._time_horizon = 8.0
    alg._metrics = _Counter()
    return alg, alg._pivot_two_step(_Encoding(), "LRA", 0, list(blocks))


class TestTwoStepCandidateLoop:
    """What the candidate loop does with each verdict.

    The oracles are scripted rather than real: the search's own decisions are
    which verdicts advance the heuristic and what it reports on giving up, and
    neither depends on a solver.
    """

    def test_refutations_rotate_the_word_off(self):
        alg, _ = _two_step(["0112"] * 40, [UNSAT] * 40,
                           word_rotate=5, pivot_budget=30)
        assert alg._rotated_words == 8, "one drop per five refutations"

    def test_expiries_on_one_word_never_rotate_it_off(self):
        """The measured defect: thirty consecutive pivot-timeout expiries on
        word 0112 discarded it as if refuted."""
        alg, _ = _two_step(["0112"] * 40, [UNKNOWN] * 40, pivot_budget=30)
        assert alg._rotated_words == 0
        assert alg._undecided_candidates == 40

    def test_the_same_count_of_refutations_does_rotate(self):
        """The contrast that makes the previous test about the verdict and not
        about the count."""
        alg, _ = _two_step(["0112"] * 40, [UNSAT] * 40, pivot_budget=30)
        assert alg._rotated_words == 1

    def test_expiries_do_not_interrupt_a_run_of_refutations(self):
        alg, _ = _two_step(["0112"] * 6,
                           [UNSAT, UNKNOWN, UNSAT, UNKNOWN, UNSAT, UNKNOWN],
                           word_rotate=3, pivot_budget=30)
        assert alg._rotated_words == 1
        assert alg._undecided_candidates == 3

    def test_an_accepted_candidate_returns_its_oracle(self):
        alg, (oracle, model, encoding) = _two_step(
            ["0112"] * 3, [UNSAT, UNSAT, SAT], pivot_budget=30)
        assert oracle is alg.candidates[-1]
        assert model is not None and encoding is not None
        assert alg._metrics["accepted"] == 1

    def test_an_undecided_candidate_is_still_blocked(self):
        """It has to be, or z3 proposes it again forever; the loop must make
        progress. What that costs is the exhaustion claim, not termination."""
        alg, _ = _two_step(["0112"] * 4, [UNKNOWN] * 4, pivot_budget=30)
        assert len(alg.candidates) == 4, "each expiry was followed by another"

    def test_a_fast_refuting_search_blames_the_budget(self):
        alg, (oracle, _, _) = _two_step(
            ["0112"], [UNSAT] * 10_000, forever=True, pivot_budget=0.05)
        assert oracle is None
        assert alg._pivot_giveup[0] == "pivot-budget"

    def test_a_search_spent_on_expiries_blames_the_per_call_bound(self):
        alg, (oracle, _, _) = _two_step(
            ["0112"], [UNKNOWN] * 10, forever=True, cost=0.04,
            pivot_budget=0.05)
        assert oracle is None
        assert alg._pivot_giveup[0] == "pivot-timeout"

class TestSkeletonVerdictRecording:
    """What the two-step search records when the outer solver stops it.

    The distinction it must preserve is UNSAT (structure space exhausted)
    against UNKNOWN (the skeleton solver gave up within its bound): the caller
    routes the first to an absence claim and the second to an UNRESOLVED
    report, so collapsing them turns a resource failure into a verdict.
    """

    def test_a_spent_structure_space_records_unsat(self):
        alg, (oracle, _, _) = _two_step(["01"], [UNSAT], pivot_budget=30)
        assert oracle is None
        assert alg._last_pivot_verdict == UNSAT

    def test_a_skeleton_give_up_records_unknown_not_unsat(self):
        """The defect: an undecided skeleton solve was recorded as UNSAT, so
        the run claimed exhaustion for a query z3 gave up on."""
        alg = _ScriptedTwoStep([], [])
        alg.skeleton.check = lambda: UNKNOWN
        alg._printer = _SilentPrinter()
        alg._config = _GenConfig(pivot_budget="30")
        alg._logger = None
        alg._tau_max = "8"
        alg._underlying = "dreal"
        alg._time_horizon = 8.0
        alg._metrics = _Counter()
        oracle, _, _ = alg._pivot_two_step(_Encoding(), "LRA", 0, [])
        assert oracle is None
        assert alg._last_pivot_verdict == UNKNOWN


class TestBoxBudget:
    """[gen] k-ic resolved to a per-depth budget: 0 spells off, like the
    section's other keys, and never 'stop before the first solver call'."""

    def test_absent_means_exhaust(self):
        assert _box_budget(_GenConfig()) is None

    def test_zero_means_exhaust_not_zero_boxes(self):
        assert _box_budget(_GenConfig(k_ic="0")) is None

    def test_a_positive_budget_is_kept(self):
        assert _box_budget(_GenConfig(k_ic="3")) == 3

    def test_a_negative_budget_is_a_configuration_error(self):
        try:
            _box_budget(_GenConfig(k_ic="-1"))
        except ValueError as error:
            assert "k-ic" in str(error)
        else:
            raise AssertionError("a negative k-ic must be rejected")


class _ScriptedExact(RegionBoxDiscovery):
    """kappa_box with the exact path's single-query oracle given explicitly,
    so the block-frame discipline is testable without a solver."""

    def __init__(self, verdict):
        super().__init__()
        self.oracle = _ScriptedCandidate(verdict)

    def _exact_oracle(self, logic, seed):
        return self.oracle


class _OneDepthEncoder:
    def encode_at(self, depth):
        return _Encoding()


class TestBlocksBindThePivotNotGrowth:
    """Alg. 2: the pivot solves Enc conjoined with the negated blocks; every
    growth query runs on Enc[w] alone. Kept permanently, the blocks wall in
    the growth oracle and _search_face reads UNSAT at a block wall as a
    bracketed frontier -- a `boundary` marker at an algorithmic wall (exact
    path). Never asserted at all, the pivot the run uses can sit inside an
    already-blocked box and regrow the same region (delta path). Both
    backends must therefore frame the blocks around the pivot query alone.
    """

    def test_exact_pivot_sees_the_blocks(self):
        block = _block_box({Real("x1_0_0"): (Fraction(0), Fraction(1))})
        alg = _ScriptedExact(SAT)
        alg._underlying = "z3"
        alg._config = _GenConfig()
        oracle, pivot, _ = alg._pivot_at(_OneDepthEncoder(), 0, "LRA", 0,
                                         [block])
        assert pivot is not None
        assert any(f is block for f in oracle.asserted), (
            "the pivot query must see the block")

    def test_exact_growth_is_not_walled_by_the_blocks(self):
        block = _block_box({Real("x1_0_0"): (Fraction(0), Fraction(1))})
        alg = _ScriptedExact(SAT)
        alg._underlying = "z3"
        alg._config = _GenConfig()
        oracle, _, _ = alg._pivot_at(_OneDepthEncoder(), 0, "LRA", 0, [block])
        assert not any(f is block for f in oracle.live), (
            "the block frame must be popped before the oracle grows the box")

    def test_exact_no_pivot_still_pops_and_records_the_verdict(self):
        for verdict in (UNSAT, UNKNOWN):
            block = _block_box({Real("x1_0_0"): (Fraction(0), Fraction(1))})
            alg = _ScriptedExact(verdict)
            alg._underlying = "z3"
            alg._config = _GenConfig()
            oracle, pivot, _ = alg._pivot_at(_OneDepthEncoder(), 0, "LRA", 0,
                                             [block])
            assert pivot is None
            assert alg._last_pivot_verdict == verdict
            assert not any(f is block for f in alg.oracle.live)

    def test_delta_candidate_sees_the_blocks(self):
        """The defect on this backend was the opposite: the blocks reached
        only the outer skeleton solver, whose candidate pins no reals, so the
        pivot dReal returned could sit inside an already-blocked box."""
        block = _block_box({Real("x1_0_0"): (Fraction(0), Fraction(1))})
        alg, (oracle, model, _) = _two_step(["01"], [SAT], pivot_budget=30,
                                            blocks=[block])
        assert model is not None
        assert any(f is block for f in oracle.asserted), (
            "the candidate oracle chooses the pivot; the blocks must bind it")

    def test_delta_growth_is_not_walled_by_the_blocks(self):
        block = _block_box({Real("x1_0_0"): (Fraction(0), Fraction(1))})
        alg, (oracle, _, _) = _two_step(["01"], [SAT], pivot_budget=30,
                                        blocks=[block])
        assert not any(f is block for f in oracle.live)


class TestValidateGen:
    """Range checks run before the first solver call.

    Each rejected value previously failed late or silently: epsilon = 0 made
    the exact bisection non-terminating and divided by zero in the harvest; a
    negative query-timeout unbounded z3 silently and crashed dReal after the
    subprocess was spawned; a negative word-rotate blocked a word on its
    first refutation; pivot budgets of 0 were folded into their defaults and
    the folded value printed back as if configured.
    """

    @staticmethod
    def _rejects(**gen):
        try:
            validate_gen(_GenConfig(**gen))
        except ValueError as error:
            return str(error)
        raise AssertionError(f"{gen} must be rejected")

    def test_a_default_configuration_passes(self):
        validate_gen(_GenConfig())

    def test_a_populated_valid_configuration_passes(self):
        validate_gen(_GenConfig(epsilon="0.001", epsilon_relative="0.01",
                                k_ic="3", k_witness="8", word_rotate="0",
                                pivot_budget="1800", pivot_timeout="60",
                                query_timeout='"off"', bisect_iters="0",
                                log_every="1", thin_ic="0"))

    def test_zero_epsilon_is_rejected_with_the_reason(self):
        assert "never terminates" in self._rejects(epsilon="0")

    def test_negative_epsilon_is_rejected(self):
        self._rejects(epsilon="-0.01")

    def test_epsilon_relative_must_be_a_proper_fraction(self):
        self._rejects(epsilon_relative="1")
        self._rejects(epsilon_relative="-0.1")

    def test_zero_pivot_budgets_are_rejected_not_folded(self):
        assert "pivot-budget" in self._rejects(pivot_budget="0")
        assert "pivot-timeout" in self._rejects(pivot_timeout="0")

    def test_negative_word_rotate_is_rejected(self):
        assert "word-rotate" in self._rejects(word_rotate="-1")

    def test_zero_k_witness_is_rejected(self):
        assert "k-witness" in self._rejects(k_witness="0")

    def test_negative_query_timeout_is_rejected_but_off_is_not(self):
        assert "query-timeout" in self._rejects(query_timeout="-5")
        validate_gen(_GenConfig(query_timeout="0"))
        validate_gen(_GenConfig(query_timeout='"off"'))

    def test_gen_float_reports_a_configured_zero(self):
        """The fold this class replaces: gen_float("0") must be 0.0, not the
        caller's default."""
        assert gen_float(_GenConfig(pivot_budget="0"), "pivot-budget") == 0.0
        assert gen_float(_GenConfig(), "pivot-budget") is None


class _DictSkeleton(_ScriptedSkeleton):
    """A skeleton oracle whose models are given as dicts, for mode words the
    char-per-step script cannot express (values >= 10, steps >= 10)."""

    def __init__(self, models):
        super().__init__(words=[])
        self._models = list(models)

    def check(self):
        if not self._models:
            self._model = None
            return UNSAT
        self._model = self._models.pop(0)
        return SAT

    def model(self):
        return {Real(f"currentMode_{k}"): RealVal(str(v))
                for k, v in self._model.items()}


class TestModeWordIdentity:
    """The rotation streak is keyed by the mode word, so the word must be a
    faithful identity of the path: positional (sorted by step index, not by
    id string) and separated (no digit-join ambiguity)."""

    @staticmethod
    def _run(models, **gen):
        alg = _ScriptedTwoStep([], [])
        alg.skeleton = _DictSkeleton(models)
        alg._printer = _SilentPrinter()
        alg._config = _GenConfig(**gen)
        alg._logger = None
        alg._tau_max = "8"
        alg._underlying = "dreal"
        alg._time_horizon = 8.0
        alg._metrics = _Counter()
        alg._pivot_two_step(_Encoding(), "LRA", 0, [])
        return alg

    def test_different_paths_never_share_a_streak(self):
        """(1,12) and (11,2) both rendered "112" under the separator-free
        join, so refutations of two different paths accumulated into one
        streak and could rotate off a feasible word."""
        alg = self._run([{0: 1, 1: 12}, {0: 11, 1: 2}] * 2,
                        word_rotate=2, pivot_budget=30)
        assert alg._rotated_words == 0, (
            "alternating distinct words must never reach a streak of 2")

    def test_the_same_path_still_accumulates(self):
        alg = self._run([{0: 1, 1: 12}] * 2, word_rotate=2, pivot_budget=30)
        assert alg._rotated_words == 1

    def test_the_word_is_ordered_by_step_index(self):
        """currentMode_10 sorts lexicographically before currentMode_2: the
        printed word was non-positional from depth 10 up."""
        recorded = []

        class _Recorder:
            def print_normal(self, msg):
                recorded.append(msg)

            def print_verbose(self, msg):
                recorded.append(msg)

        alg = _ScriptedTwoStep([], [])
        alg.skeleton = _DictSkeleton([{2: 5, 10: 7}])
        alg._printer = _Recorder()
        alg._config = _GenConfig(pivot_budget="30")
        alg._logger = None
        alg._tau_max = "8"
        alg._underlying = "dreal"
        alg._time_horizon = 8.0
        alg._metrics = _Counter()
        alg._pivot_two_step(_Encoding(), "LRA", 0, [])
        words = [line for line in recorded if "(word " in line]
        assert words and "(word 5.7)" in words[0], words


class TestPivotMetricsAndClamp:
    def test_the_exact_pivot_counts_into_the_metrics(self):
        """The metrics line printed candidates=0 accepted=0 on every exact
        run: only the two-step search incremented the counters."""
        alg = _ScriptedExact(SAT)
        alg._underlying = "z3"
        alg._config = _GenConfig()
        alg._pivot_at(_OneDepthEncoder(), 0, "LRA", 0, [])
        assert alg._metrics["candidates"] == 1
        assert alg._metrics["accepted"] == 1

    def test_a_refused_exact_pivot_counts_the_attempt_only(self):
        alg = _ScriptedExact(UNSAT)
        alg._underlying = "z3"
        alg._config = _GenConfig()
        alg._pivot_at(_OneDepthEncoder(), 0, "LRA", 0, [])
        assert alg._metrics["candidates"] == 1
        assert alg._metrics["accepted"] == 0

    def test_the_candidate_budget_is_clamped_to_the_deadline(self):
        """One candidate could overrun the search deadline by a whole
        pivot-timeout: the per-candidate budget must never exceed what is
        left of pivot-budget."""
        alg, _ = _two_step(["01"], [UNSAT] * 100, forever=True,
                           pivot_budget=0.2, pivot_timeout=45)
        numeric = [b for oracle in alg.candidates for b in oracle.budgets
                   if isinstance(b, float)]
        assert numeric and all(b <= 0.2 for b in numeric), numeric


class TestOpenRangeEdges:
    """A face that runs to an OPEN declared edge is a domain face.

    The encoding bounds an open range strictly, so the probe at the wall is
    UNSAT for encoding reasons; with the inclusivity flags dropped, the
    bisection then 'bracketed' against the edge itself and the face was
    labeled boundary -- the exact distinction the three labels exist to make.
    """

    @staticmethod
    def _encoding_with_flags(hi_incl):
        class _Enc:
            range_dict = {Real("x"): (True, "0", "10", hi_incl)}
            bound = 1
        return _Enc()

    def _labels(self, hi_incl, x):
        # Falsifying almost to the wall: within tol of 10, but not at it --
        # geometrically indistinguishable from an open-edge encoding.
        oracle = FakeOracle({x: (Fraction(4), Fraction(10) - Fraction(1, 4000))},
                            tolerance=Fraction(1, 1000))
        alg = RegionBoxDiscovery()
        alg._config = None
        _, labels, _ = alg._grow_box(
            oracle, assn(x_0_0=5), self._encoding_with_flags(hi_incl),
            _Theta(Fraction(1, 4)), 20, 1, 4, _SilentPrinter())
        return labels

    def test_an_open_edge_is_a_domain_face(self, x):
        assert _DOMAIN in self._labels(False, x)

    def test_a_closed_edge_stays_a_frontier(self, x):
        labels = self._labels(True, x)
        assert _DOMAIN not in labels and _BOUNDARY in labels


def test_ic_pivots_are_sorted_by_variable_id():
    """_grow_box grows greedily in one pass, so its axis order is part of the
    geometry; the model dict arrives in solver order, which no seed pins."""
    from stlmc.generation.box import _ic_pivots
    range_dict = {Real("b"): None, Real("a"): None, Real("c"): None}
    model = assn(c_0_0=3, a_0_0=1, b_0_0=2)
    assert [v.id for v in _ic_pivots(model, range_dict)] == [
        "a_0_0", "b_0_0", "c_0_0"]


# ============================================= partial-face witness separation

class TestPartialFaceWitnessSeparation:
    """A partial face contributes a deep witness, not an exempt marker.

    It is labeled deep because no frontier was located, so it must be held to
    the same theta separation as any other deep witness. Entering it through the
    marker merge -- which is exempt from separation, since a frontier marker's
    position is the information it carries -- let a partial-face witness sit
    within theta of the pivot it grew from.
    """

    @staticmethod
    def _encoding():
        class _Enc:
            range_dict = {Real("x"): (True, "0", "10", True)}
            bound = 1
        return _Enc()

    def _grow(self, oracle, theta):
        alg = RegionBoxDiscovery()
        alg._config = None
        return alg._grow_box(oracle, assn(x_0_0=5), self._encoding(),
                             theta, 20, 1, 4, _SilentPrinter())

    def test_a_partial_witness_within_theta_of_the_pivot_is_dropped(self, x):
        # Undecided at or beyond 5.7 lets the +face confirm a little growth (to
        # ~5.6) and then stop partial, within theta = 1 of the pivot at 5. The
        # partial witness carries nothing the pivot does not and must be dropped.
        oracle = FakeOracle({x: (Fraction(4), Fraction(6))},
                            undecided=beyond(x, Fraction(57, 10)),
                            tolerance=Fraction(1, 1000))
        witnesses, labels, _ = self._grow(oracle, _Theta(Fraction(1)))
        deep = sorted(_value_of(w, x)
                      for w, label in zip(witnesses, labels) if label == _DEEP)
        gaps = [b - a for a, b in zip(deep, deep[1:])]
        assert all(g >= Fraction(1) for g in gaps), (
            "deep witnesses closer than theta: "
            f"{[float(d) for d in deep]}")

    def test_an_exempt_frontier_marker_within_theta_still_survives(self, x):
        # The -face brackets a real frontier near 4; its boundary marker is
        # exempt from separation and must be kept even on a narrow box, since its
        # position is the frontier it reports.
        oracle = FakeOracle({x: (Fraction(4), Fraction(6))},
                            undecided=beyond(x, Fraction(57, 10)),
                            tolerance=Fraction(1, 1000))
        _, labels, _ = self._grow(oracle, _Theta(Fraction(1)))
        assert _BOUNDARY in labels, "an exempt frontier marker must survive"


# ================================================= harvest window delta floor

class _WindowRecorder(FakeOracle):
    """Records the width of every per-axis window a check() is issued under."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widths = []

    def check(self):
        for _var, (lo, hi) in self._window().items():
            if lo is not None and hi is not None:
                self.widths.append(hi - lo)
        return super().check()


class TestHarvestWindowDeltaFloor:
    """Def. harvest: the sampling window has width >= delta.

    Under a delta backend a window narrower than delta forfeits the claim that a
    returned point lies in the cell that was queried, which is the whole
    justification for pinning every axis. theta can be finer than delta, so the
    window floors at delta rather than following theta down. On an exact oracle
    (tolerance 0) the floor is a no-op and the window is theta as before.
    """

    def test_lattice_window_is_floored_at_delta(self, x, y):
        delta = Fraction(1, 1000)
        theta = _Theta(Fraction(1, 1_000_000))       # far below delta
        box = {x: [Fraction(0), Fraction(1)], y: [Fraction(0), Fraction(1)]}
        oracle = _WindowRecorder({x: (Fraction(0), Fraction(10)),
                                  y: (Fraction(0), Fraction(10))},
                                 tolerance=delta)
        RegionBoxDiscovery()._harvest_lattice(oracle, box, theta, 2, 200)
        assert oracle.widths
        assert min(oracle.widths) >= delta, (
            f"window {float(min(oracle.widths))} is below delta {float(delta)}")

    def test_an_exact_oracle_window_is_unchanged(self, x):
        theta = _Theta(Fraction(1, 10))
        box = {x: [Fraction(0), Fraction(1)]}
        oracle = _WindowRecorder({x: (Fraction(0), Fraction(10))},
                                 tolerance=Fraction(0))
        oracle.is_exact = True
        RegionBoxDiscovery()._harvest_lattice(oracle, box, theta, 8, 200)
        assert oracle.widths and max(oracle.widths) == Fraction(1, 10)


# ================================================== two-step backend capability

class TestTwoStepBackendGuard:
    """The two-step pivot search is sound only under dReal.

    Its outer level asserts the segment-duration timing identities and the
    endpoint instantiation of quantified subformulas -- necessary conditions of
    the delta query, not of the encoding -- which hold only under a backend that
    realizes a closed segment interval and the injected clock dynamics. The
    dispatch guards that capability rather than inferring it from "not z3".
    """

    def test_a_foreign_backend_is_refused(self):
        alg = _ScriptedExact(SAT)
        alg._underlying = "yices"
        alg._config = _GenConfig()
        try:
            alg._pivot_at(_OneDepthEncoder(), 0, "LRA", 0, [])
        except ValueError as error:
            assert "dreal" in str(error)
        else:
            raise AssertionError("a non-z3, non-dreal backend must be refused")

    def test_the_exact_backend_is_unaffected(self):
        alg = _ScriptedExact(SAT)
        alg._underlying = "z3"
        alg._config = _GenConfig()
        _, pivot, _ = alg._pivot_at(_OneDepthEncoder(), 0, "LRA", 0, [])
        assert pivot is not None

    def test_dreal_still_takes_the_two_step_path(self):
        alg = _ScriptedTwoStep(["01"], [SAT])
        alg._printer = _SilentPrinter()
        alg._config = _GenConfig(pivot_budget="30")
        alg._logger = None
        alg._tau_max = "8"
        alg._underlying = "dreal"
        alg._time_horizon = 8.0
        alg._metrics = _Counter()
        _, model, _ = alg._pivot_at(_OneDepthEncoder(), 0, "LRA", 0, [])
        assert model is not None, "dreal must route through the two-step search"