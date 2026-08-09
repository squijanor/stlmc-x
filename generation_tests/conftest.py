"""Shared fixtures for the generation-strategy tests.

Import order matters. ``import stlmc.generation.box`` on a cold interpreter
raises ``NameError: name 'Algorithm' is not defined``, because
``objects.algorithm`` does ``from ..encoding.enumerate import *`` while
``encoding.enumerate`` subclasses ``Algorithm``: whichever is imported first
sees a half-initialised module. Importing ``stlmc.cli.mc`` first resolves the
cycle, as every entry point does. The shim below makes that explicit so a
failure here is not reported as an unrelated import error.
"""

import os
import sys
from fractions import Fraction

import pytest

# Prefer this checkout over any installed stlmc. With `pip install -e .` the two
# are the same and this is a no-op; without it, an installed upstream copy would
# shadow the fork and the generation package would not exist at all.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(os.path.join(_SRC, "stlmc", "generation")):
    sys.path.insert(0, _SRC)

import stlmc.cli.mc  # noqa: E402,F401  -- import-order shim, see module docstring

from stlmc.constraints.constraints import (
    And, BoolVal, Constant, Geq, Gt, Leq, Lt, Real, RealVal, Variable,
)
from stlmc.generation.oracle import SAT, UNKNOWN, UNSAT


class FakeOracle:
    """A GrowthOracle stand-in whose falsifying set is known exactly.

    The searches under test are geometric: they ask "is there a falsifying point
    at or beyond x", bisect on the answers, and label the result. Testing them
    against a real solver measures the solver; testing them against a set whose
    answers are computable measures the search. This oracle answers from an
    explicit falsifying box, so every expected value in a test is derived by
    hand rather than recorded from a run.

    ``undecided`` is a predicate over the queried window (``{var: (lo, hi)}``,
    either bound possibly None) returning True when ``check`` should answer
    UNKNOWN however satisfiable the query is -- the delta-backend behaviour the
    detour logic exists to survive. It is a predicate rather than a list of
    points because "the solver cannot decide THIS query" is a property of the
    query, not of a point: a probe at a threshold and a narrow window around the
    same value are different questions. Use :func:`probe_at` and :func:`covers`
    rather than writing the dictionary shape out by hand.
    """

    is_exact = False

    def __init__(self, falsifying, undecided=None, tolerance=Fraction(1, 1000)):
        #: {Variable: (lo, hi)} -- the set of falsifying points
        self.falsifying = falsifying
        #: callable(window) -> bool
        self.undecided = undecided
        self._tolerance = tolerance
        self._stack = [[]]
        self._model = None
        self.calls = 0

    # -- GrowthOracle surface -------------------------------------------
    @property
    def tolerance(self):
        return self._tolerance

    def rv(self, f):
        return RealVal(str(f))

    def push(self):
        self._stack.append([])

    def pop(self):
        self._stack.pop()

    def assert_(self, formula):
        self._stack[-1].append(formula)

    def check_with(self, formula):
        self.push()
        try:
            self.assert_(formula)
            return self.check()
        finally:
            self.pop()

    def check(self):
        self.calls += 1
        window = self._window()
        if self.undecided is not None and self.undecided(window):
            self._model = None
            return UNKNOWN
        point = {}
        for var, (flo, fhi) in self.falsifying.items():
            lo, hi = window.get(var, (None, None))
            lo = flo if lo is None else max(lo, flo)
            hi = fhi if hi is None else min(hi, fhi)
            if lo > hi:
                self._model = None
                return UNSAT
            point[var] = (lo + hi) / 2
        self._model = point
        return SAT

    def model(self):
        assert self._model is not None, "model() after a non-SAT check"
        return {var: RealVal(str(value)) for var, value in self._model.items()}

    # -- helpers ---------------------------------------------------------
    def _window(self):
        """Interval implied by the asserted constraints, per variable."""
        window = {}

        def visit(node):
            if isinstance(node, BoolVal):
                return
            if isinstance(node, And):
                for child in node.children:
                    visit(child)
                return
            left = getattr(node, "left", None)
            right = getattr(node, "right", None)
            if not isinstance(left, Variable) or not isinstance(right, Constant):
                return
            bound = Fraction(str(right.value))
            lo, hi = window.get(left, (None, None))
            if isinstance(node, (Geq, Gt)):
                lo = bound if lo is None else max(lo, bound)
            elif isinstance(node, (Leq, Lt)):
                hi = bound if hi is None else min(hi, bound)
            window[left] = (lo, hi)

        for frame in self._stack:
            for formula in frame:
                visit(formula)
        return window


def probe_at(var, *values):
    """Undecided predicate: a one-sided probe whose finite endpoint is one of
    ``values``. This is the shape `_search_face` issues -- "is there a
    falsifying point at or beyond m" -- so it names a single hard query."""
    wanted = [Fraction(str(v)) for v in values]

    def undecided(window):
        lo, hi = window.get(var, (None, None))
        ends = [b for b in (lo, hi) if b is not None]
        return len(ends) == 1 and ends[0] in wanted

    return undecided


def covers(var, *values):
    """Undecided predicate: any query whose window contains one of ``values``.
    This is the shape `_harvest` issues -- a bounded window around a point."""
    wanted = [Fraction(str(v)) for v in values]

    def undecided(window):
        lo, hi = window.get(var, (None, None))
        return any((lo is None or lo <= v) and (hi is None or v <= hi)
                   for v in wanted)

    return undecided


def beyond(var, threshold):
    """Undecided predicate: every probe reaching at or past ``threshold``."""
    threshold = Fraction(str(threshold))

    def undecided(window):
        lo, _ = window.get(var, (None, None))
        return lo is not None and lo >= threshold

    return undecided


@pytest.fixture
def x():
    return Real("x_0_0")


@pytest.fixture
def y():
    return Real("y_0_0")