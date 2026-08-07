"""Incremental SMT oracle for the custom generation strategies.

Provides a uniform incremental interface --- ``push / assert_ / check / model /
pop`` --- over a pluggable SMT backend, so the generation algorithms do not
depend on the underlying solver.

Backends
--------
* :class:`Z3IncrementalOracle` -- a raw ``z3.Solver`` with native push/pop, in
  process, for linear models (``QF_LRA``). Push/pop retain learned clauses
  across the repeated trial solves of a box-growth loop.
* :class:`DrealReSolveOracle` -- a stateless backend for nonlinear models. dReal
  exposes no incremental interface in STLMC (each solve is a fresh subprocess),
  so push/pop are emulated by replaying an assertion stack and re-solving on
  every ``check()``.

Verdict convention
------------------
STLMC's wrapped solvers report a satisfiable query as ``"False"`` and an
unsatisfiable one as ``"True"``. This oracle uses native verdicts instead:
:data:`SAT` / :data:`UNSAT` / :data:`UNKNOWN`. :class:`Z3IncrementalOracle`
talks to a raw ``z3.Solver`` and needs no translation.
:class:`DrealReSolveOracle` goes through the wrapped ``dRealSolver`` and performs
the translation in :meth:`DrealReSolveOracle.check`. dReal is a delta-decision
procedure: a :data:`SAT` verdict is delta-sat, so the assignment it returns
(the midpoint of dReal's delta-box) satisfies a delta-relaxation of the
constraints rather than the exact constraints.

Import paths assume this module is ``stlmc/generation/oracle.py`` with package
root ``stlmc``.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Tuple

import z3

from ..solver.z3 import z3Obj, Z3Assignment
from ..constraints.constraints import (
    And,
    Constant,
    Formula,
    Geq,
    Leq,
    Variable,
)

# Native SMT verdicts. Strings (not an Enum) so they are picklable across worker
# processes.
SAT = "sat"
UNSAT = "unsat"
UNKNOWN = "unknown"

# An initial-condition box: variable -> (lower, upper) as Python floats.
BoxBounds = Dict[Variable, Tuple[float, float]]


# =========================================================================== #
#  Oracle interface
# =========================================================================== #
class GrowthOracle(abc.ABC):
    """Uniform incremental SMT interface.

    Contract:
      * ``assert_`` adds a constraint to the current (top) frame.
      * ``push`` / ``pop`` open / discard a frame (LIFO); ``pop`` removes every
        assertion added since the matching ``push``.
      * ``check`` returns :data:`SAT`, :data:`UNSAT`, or :data:`UNKNOWN`.
      * ``model`` is valid only immediately after a :data:`SAT` ``check`` and
        returns an assignment dict ``{Variable: Constant}``.
    """

    @abc.abstractmethod
    def assert_(self, formula: Formula) -> None: ...

    @abc.abstractmethod
    def push(self) -> None: ...

    @abc.abstractmethod
    def pop(self) -> None: ...

    @abc.abstractmethod
    def check(self) -> str: ...

    @abc.abstractmethod
    def model(self) -> Dict[Variable, Constant]: ...

    def check_with(self, formula: Formula) -> str:
        """SAT-check the current stack conjoined with ``formula``, leaving the
        stack unchanged."""
        self.push()
        try:
            self.assert_(formula)
            return self.check()
        finally:
            self.pop()


# =========================================================================== #
#  z3 backend (linear, native incremental)
# =========================================================================== #
class Z3IncrementalOracle(GrowthOracle):
    """Raw ``z3.Solver`` with native push/pop for linear models."""

    def __init__(self, logic: str = "QF_LRA", seed: int | None = None) -> None:
        # z3 logic name ("QF_LRA" / "QF_NRA"). SolverFor enables the incremental
        # theory solver for that logic.
        self._solver = z3.SolverFor(logic)
        if seed is not None:
            self._solver.set("random_seed", int(seed))
        self._sat_seen = False

    def assert_(self, formula: Formula) -> None:
        self._solver.add(z3Obj(formula))

    def push(self) -> None:
        self._solver.push()

    def pop(self) -> None:
        self._solver.pop()
        self._sat_seen = False

    def check(self) -> str:
        r = self._solver.check()
        if r == z3.sat:
            self._sat_seen = True
            return SAT
        self._sat_seen = False
        return UNSAT if r == z3.unsat else UNKNOWN

    def model(self) -> Dict[Variable, Constant]:
        if not self._sat_seen:
            raise RuntimeError("model() called without a preceding SAT check()")
        return Z3Assignment(self._solver.model()).get_assignments()


# =========================================================================== #
#  dReal backend (nonlinear, stateless)
# =========================================================================== #
class DrealReSolveOracle(GrowthOracle):
    """Stateless push/pop emulation over the wrapped ``dRealSolver``.

    dReal is subprocess-based with no incremental interface, so every ``check``
    conjoins the whole live assertion stack and re-solves. Delta-sat is reported
    as :data:`SAT`.
    """

    def __init__(self, config, logger=None, time_bound: str | None = None) -> None:
        self._config = config
        self._logger = logger
        self._time_bound = time_bound
        # A stack of frames; each frame is a list of asserted formulas.
        self._frames: List[List[Formula]] = [[]]
        self._last_model: Dict[Variable, Constant] | None = None

    def assert_(self, formula: Formula) -> None:
        self._frames[-1].append(formula)

    def push(self) -> None:
        self._frames.append([])

    def pop(self) -> None:
        if len(self._frames) == 1:
            raise RuntimeError("pop() with no matching push()")
        self._frames.pop()
        self._last_model = None

    def _all_consts(self) -> Formula:
        flat: List[Formula] = [f for frame in self._frames for f in frame]
        return And(flat)

    def check(self) -> str:
        result, model = self._solve_once(self._all_consts())
        # Translate the wrapped solver's verdict to native (see module docstring).
        if result == "False":
            self._last_model = model
            return SAT
        self._last_model = None
        return UNSAT if result == "True" else UNKNOWN

    def model(self) -> Dict[Variable, Constant]:
        if self._last_model is None:
            raise RuntimeError("model() called without a preceding SAT check()")
        return self._last_model

    def _solve_once(self, consts: Formula):
        """Solve ``consts`` with a fresh ``dRealSolver`` configured from the
        run's [dreal] section. Returns ``(result_str, assignment_or_None)`` in
        the wrapped-solver convention: ``"False"`` == (delta-)satisfiable,
        ``"True"`` == unsatisfiable, ``"Unknown"`` otherwise. The assignment is
        read only on a satisfiable result."""
        from ..solver.dreal import dRealSolver

        solver = dRealSolver()
        solver.set_config(self._config)
        if self._logger is not None:
            solver.append_logger(self._logger)
        if self._time_bound is not None:
            solver.set_time_bound(self._time_bound)
        result, _size = solver.solve(consts, None, None)
        model = solver.make_assignment().get_assignments() if result == "False" else None
        return result, model


# =========================================================================== #
#  Backend factory
# =========================================================================== #
def make_oracle(
    underlying: str,
    *,
    logic: str = "QF_LRA",
    seed: int | None = None,
    config=None,
    logger=None,
    time_bound=None,
) -> GrowthOracle:
    """A fresh :class:`GrowthOracle` for the configured backend.

    ``z3`` uses the native incremental solver (``logic`` and ``seed``); ``dreal``
    uses the stateless re-solve backend (``config``, ``logger``, ``time_bound``).
    """
    if underlying == "z3":
        return Z3IncrementalOracle(logic, seed)
    if underlying == "dreal":
        return DrealReSolveOracle(config, logger, time_bound)
    raise NotImplementedError(
        "generation supports the z3 and dreal backends; got '{}'".format(underlying)
    )


# =========================================================================== #
#  Backend-agnostic formula helpers
# =========================================================================== #
# Build Formula objects from explicit Variable objects and Python bounds; the
# caller supplies the initial-condition variables.

def box_constraint(bounds: BoxBounds) -> Formula:
    """``AND_i (lo_i <= x_i <= hi_i)`` over the given IC variables."""
    terms: List[Formula] = []
    for var, (lo, hi) in bounds.items():
        terms.append(Geq(var, _real(lo)))
        terms.append(Leq(var, _real(hi)))
    return And(terms)


def box_infty(bounds: BoxBounds, drop_var: Variable, direction: int) -> Formula:
    """The box with the face ``(drop_var, direction)`` left unbounded.

    ``direction == +1`` drops the upper bound of ``drop_var``; ``-1`` drops the
    lower bound."""
    terms: List[Formula] = []
    for var, (lo, hi) in bounds.items():
        if var == drop_var:
            if direction > 0:
                terms.append(Geq(var, _real(lo)))  # keep lower, drop upper
            else:
                terms.append(Leq(var, _real(hi)))  # keep upper, drop lower
        else:
            terms.append(Geq(var, _real(lo)))
            terms.append(Leq(var, _real(hi)))
    return And(terms)


def grow_bounds(bounds: BoxBounds, var: Variable, direction: int, delta: float) -> BoxBounds:
    """Return a copy of ``bounds`` with the face ``(var, direction)`` shifted
    outward by ``delta``. Does not mutate the input."""
    lo, hi = bounds[var]
    new = dict(bounds)
    new[var] = (lo, hi + delta) if direction > 0 else (lo - delta, hi)
    return new


def fix_modes(mode_assignment: Dict[Variable, Constant]) -> Formula:
    """``AND_k (mode_k == word[k])`` pinning the discrete path.

    ``mode_assignment`` maps each per-step mode Variable to the Constant it took
    in the pivot counterexample."""
    from ..constraints.constraints import Eq

    return And([Eq(var, val) for var, val in mode_assignment.items()])


def _real(x: float):
    """Python float -> RealVal, via string to avoid binary-float drift in the
    SMT term."""
    from ..constraints.constraints import RealVal

    return RealVal(repr(x))