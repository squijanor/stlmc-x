"""Unit tests for kappa_path's blocking predicate and verdict.

The block is checked by its *meaning*, not its shape: for a word alphabet given
explicitly, the words a clause still admits are enumerated with z3 and compared
against the Hamming ball computed in the test. Expected values are therefore
derived from the definition rather than recorded from a run, and a change that
preserves the semantics while rewriting the encoding stays green.
"""

import itertools
from functools import reduce

import z3

from stlmc.constraints.constraints import Add, And, Geq, Int, IntVal, Leq
from stlmc.generation.pathenum import _verdict, block_radius
from stlmc.solver.z3 import z3Obj

# A small alphabet keeps the enumeration exhaustive: MODES modes over LEN steps
# is MODES**LEN words, all of which are checked.
MODES = 3

# Every word of length 1, for the depth-0 case.
ALL_WORDS_1 = {(m,) for m in range(MODES)}


def word_assignment(word):
    """An assignment dict as a solver returns it, for a location word."""
    return {Int(f"currentMode_{k}"): IntVal(str(m)) for k, m in enumerate(word)}


def admitted(clause, length):
    """Every word of ``length`` the clause still admits, by enumeration in z3."""
    modes = [Int(f"currentMode_{k}") for k in range(length)]
    domain = And([And([Geq(v, IntVal("0")), Leq(v, IntVal(str(MODES - 1)))])
                  for v in modes])
    solver = z3.Solver()
    solver.add(z3Obj(domain))
    solver.add(z3Obj(clause))
    out = set()
    for candidate in itertools.product(range(MODES), repeat=length):
        solver.push()
        for var, value in zip(modes, candidate):
            solver.add(z3Obj(Geq(var, IntVal(str(value)))))
            solver.add(z3Obj(Leq(var, IntVal(str(value)))))
        if solver.check() == z3.sat:
            out.add(candidate)
        solver.pop()
    return out


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


class TestBlockSemantics:
    """A radius-r block excludes exactly the Hamming ball of radius r."""

    def test_radius_zero_excludes_one_word(self):
        word = (0, 1, 2)
        block = block_radius(word_assignment(word), 0, uid=0)
        every = set(itertools.product(range(MODES), repeat=len(word)))
        assert admitted(block.clause, len(word)) == every - {word}

    def test_radius_zero_introduces_no_auxiliary_variables(self):
        block = block_radius(word_assignment((0, 1, 2)), 0, uid=0)
        assert block.indicators == ()
        assert block.radius == 0

    def test_radius_r_excludes_the_hamming_ball(self):
        word = (0, 1, 2)
        for radius in (1, 2):
            block = block_radius(word_assignment(word), radius, uid=radius)
            expected = {w for w in itertools.product(range(MODES), repeat=len(word))
                        if hamming(w, word) > radius}
            assert admitted(block.clause, len(word)) == expected, radius

    def test_a_negative_radius_is_radius_zero(self):
        word = (1, 0)
        block = block_radius(word_assignment(word), -3, uid=0)
        assert block.radius == 0
        every = set(itertools.product(range(MODES), repeat=len(word)))
        assert admitted(block.clause, len(word)) == every - {word}


class TestRadiusCap:
    """A radius at or above the word length would block every word.

    Uncapped, `sum >= r+1` over L indicators is unsatisfiable for r >= L, so the
    next solve returns UNSAT and the depth reports as exhausted after a single
    path. The cap keeps the block a ball.
    """

    def test_radius_is_capped_at_the_word_length(self):
        word = (0, 1)                      # depth 1: two positions
        block = block_radius(word_assignment(word), 5, uid=0)
        assert block.radius == len(word) - 1

    def test_a_capped_block_still_admits_words(self):
        word = (0, 1)
        block = block_radius(word_assignment(word), 5, uid=0)
        still = admitted(block.clause, len(word))
        assert still, "a capped block must not exclude the whole lattice"
        assert all(hamming(w, word) == len(word) for w in still)

    def test_the_cap_is_the_coarsest_ball_that_admits_a_word(self):
        """Capping to L-1 rather than to some smaller value: at L-1 only words
        differing everywhere survive, and there is no coarser non-empty block."""
        word = (0, 1, 2)
        block = block_radius(word_assignment(word), 99, uid=0)
        assert block.radius == 2
        assert admitted(block.clause, 3) == {
            w for w in itertools.product(range(MODES), repeat=3)
            if hamming(w, word) == 3}

    def test_an_uncapped_radius_would_have_blocked_everything(self):
        """The failure the cap prevents, stated directly: `sum >= r+1` over L
        0/1 indicators is unsatisfiable once r reaches L."""
        length = 2
        indicators = [Int(f"hb$0${k}") for k in range(length)]
        bounded = [And([Geq(i, IntVal("0")), Leq(i, IntVal("1"))])
                   for i in indicators]
        total = reduce(Add, indicators)
        solver = z3.Solver()
        solver.add(z3Obj(And(bounded + [Geq(total, IntVal(str(length + 1)))])))
        assert solver.check() == z3.unsat


class TestAuxiliaryVariables:
    """The indicators belong to the query, and the caller is told which they are.

    Every model returned after a radius-r block carries them, so a caller that
    does not know their identity cannot keep them out of the pool.
    """

    def test_indicators_are_reported_and_unique_per_block(self):
        word = (0, 1, 2)
        first = block_radius(word_assignment(word), 1, uid=0)
        second = block_radius(word_assignment(word), 1, uid=1)
        assert len(first.indicators) == len(word)
        assert set(first.indicators).isdisjoint(second.indicators)

    def test_indicators_are_the_only_new_variables(self):
        from stlmc.constraints.operations import get_vars

        word = (0, 1, 2)
        block = block_radius(word_assignment(word), 1, uid=7)
        modes = set(word_assignment(word))
        assert set(get_vars(block.clause)) - modes == set(block.indicators)


class TestVerdict:
    """A verdict may never cover ground the run did not visit."""

    def test_a_non_empty_pool_is_false(self):
        assert _verdict(["a path"], False, [1, 2], 2) == ("False", None)

    def test_unresolved_beats_true(self):
        result, note = _verdict([], True, [1, 2], 2)
        assert result == "Unknown"
        assert "unresolved" in note

    def test_skipped_depths_forbid_true(self):
        result, note = _verdict([], False, [1], 5)
        assert result == "Unknown"
        assert "2/3/4/5" in note, "the note must name what was skipped"

    def test_an_empty_target_set_forbids_true(self):
        """[gen] depths is clamped to 0..bound, so it can select nothing at all.
        Examining no depth is the extreme case of examining a subset."""
        result, note = _verdict([], False, [], 3)
        assert result == "Unknown"
        assert "1/2/3" in note

    def test_exhaustive_absence_is_true(self):
        assert _verdict([], False, [0, 1, 2, 3], 3) == ("True", None)

    def test_skipping_depth_zero_forbids_true(self):
        """Depth 0 is the unrolling with no jump. It is part of the bound,
        so deciding every other depth is still not absence up to it."""
        result, note = _verdict([], False, [1, 2, 3], 3)
        assert result == "Unknown"
        assert "0" in note.split("depth(s) ")[2], note

    def test_a_depth_left_open_by_a_budget_does_not_count_as_decided(self):
        """The caller passes the depths it settled, not the ones it targeted.
        A budget of zero visits every depth and settles none, which must not
        read as absence."""
        result, note = _verdict([], False, [], 2)
        assert result == "Unknown"
        assert "1/2" in note

    def test_a_depth_zero_word_forces_radius_zero(self):
        """A word at depth n has n+1 positions, so at depth 0 it has one and
        the cap min(r, n) leaves no room for a ball: any radius encodes the
        single-word block, with no indicator variables."""
        block = block_radius(word_assignment([2]), 3, uid=0)
        assert block.radius == 0
        assert block.indicators == ()
        assert admitted(block.clause, 1) == {w for w in ALL_WORDS_1 if w != (2,)}

    def test_the_note_names_the_strategy(self):
        _, note = _verdict([], False, [1], 2)
        assert note.startswith("[kappa_path]")


class TestWordIntegrity:
    """The delta backend reports every model value as an interval midpoint. A
    block built from a non-integral mode value misses the word the solver
    satisfied: at radius 0 the same model can be returned forever, at radius
    >= 1 the ball is centred off-word. The run guards the word before pooling
    or blocking; these tests pin the guard's parts."""

    def test_integral_decimals_are_canonicalised_in_the_block(self):
        """dReal formats an integral value as e.g. "1.000000"; the block must
        exclude the Hamming ball around (1, 2) exactly as if the values were
        exact. The enumeration here is over Real-typed mode variables, as on
        the delta backend (the shared `admitted` helper enumerates Int-typed
        ones, which z3 keeps distinct from same-named Reals)."""
        from stlmc.constraints.constraints import Real, RealVal

        assignment = {Real("currentMode_0"): RealVal("1.000000"),
                      Real("currentMode_1"): RealVal("2.000000")}
        for radius in (0, 1):
            block = block_radius(assignment, radius, uid=radius)
            solver = z3.Solver()
            solver.add(z3Obj(block.clause))
            still = set()
            for candidate in itertools.product(range(MODES), repeat=2):
                solver.push()
                for k, value in enumerate(candidate):
                    solver.add(z3.Real(f"currentMode_{k}") == value)
                if solver.check() == z3.sat:
                    still.add(candidate)
                solver.pop()
            expected = {w for w in itertools.product(range(MODES), repeat=2)
                        if hamming(w, (1, 2)) > radius}
            assert still == expected, radius

    def test_off_lattice_positions_are_detected(self):
        from stlmc.constraints.constraints import Real, RealVal
        from stlmc.generation.pathenum import _location_word, _off_lattice

        assignment = {Real("currentMode_0"): RealVal("1.500000"),
                      Real("currentMode_1"): RealVal("2.000000")}
        bad = _off_lattice(_location_word(assignment))
        assert bad == ["currentMode_0=1.500000"]

    def test_an_integral_word_is_not_flagged(self):
        from stlmc.constraints.constraints import Real, RealVal
        from stlmc.generation.pathenum import _location_word, _off_lattice

        assignment = {Real("currentMode_0"): RealVal("1.000000"),
                      Real("currentMode_1"): RealVal("2")}
        assert _off_lattice(_location_word(assignment)) == []