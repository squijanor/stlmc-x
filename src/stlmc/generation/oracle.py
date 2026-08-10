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
import os
from fractions import Fraction
from typing import Dict, Tuple

import z3

from ..constraints.constraints import (
    And,
    Constant,
    Formula,
    Geq,
    Leq,
    RealVal,
    Variable,
)
from ..solver.z3 import Z3Assignment, z3Obj

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

    #: False for delta-decision backends: verdicts are only accurate to
    #: :attr:`tolerance`, and a returned model is a delta-sat interval midpoint
    #: rather than a certified satisfying assignment.
    is_exact: bool = True

    @property
    def tolerance(self) -> Fraction:
        """Finest meaningful resolution for a frontier on this backend."""
        return Fraction(0)

    def rv(self, f: Fraction) -> RealVal:
        """Render an exact rational as a constant this backend can parse.
        Default is the exact rational string, which z3 parses natively."""
        return RealVal(str(f))

    @abc.abstractmethod
    def model(self) -> dict[Variable, Constant]: ...

    def check_with(self, formula: Formula) -> str:
        """SAT-check the current stack conjoined with ``formula``, leaving the
        stack unchanged."""
        self.push()
        try:
            self.assert_(formula)  # noqa: UP005  -- see assert_ above, not unittest
            return self.check()
        finally:
            self.pop()


# =========================================================================== #
#  z3 backend (linear, native incremental)
# =========================================================================== #
class Z3IncrementalOracle(GrowthOracle):
    """Raw ``z3.Solver`` with native push/pop for linear models."""

    def __init__(self, logic: str = "QF_LRA", seed: int | None = None,
                 general: bool = False) -> None:
        # z3 logic name ("QF_LRA" / "QF_NRA"). SolverFor enables the incremental
        # theory solver for that logic, but it also skips most preprocessing --
        # which is fine for the arithmetic-heavy growth queries and very bad for
        # the two-step skeleton, which is mostly Boolean structure. general=True
        # asks for the default solver (full preprocessing, adaptive tactics).
        self._solver = z3.Solver() if general else z3.SolverFor(logic)
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

    def model(self) -> dict[Variable, Constant]:
        if not self._sat_seen:
            raise RuntimeError("model() called without a preceding SAT check()")
        return Z3Assignment(self._solver.model()).get_assignments()


# =========================================================================== #
#  dReal backend (nonlinear, stateless)
# =========================================================================== #
def _exact_decimal(f: Fraction) -> str:
    """Finite decimal expansion of ``f``, or raise.

    Exact iff the denominator is 2^a * 5^b. On the dReal path this always holds:
    values arrive from DrealAssignment already formatted as decimals, theta comes
    from config as a decimal, and bisection only multiplies a denominator by 2.
    The raise exists so a future change that breaks the invariant fails loudly
    instead of rounding silently."""
    d, twos, fives = f.denominator, 0, 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        raise ValueError(
            f"no finite decimal expansion for {f}; dReal's SMT2 parser has no p/q "
            "rational literal")
    scale = max(twos, fives)
    if scale == 0:
        return str(f.numerator)
    scaled = f.numerator * 10 ** scale // f.denominator
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(scale + 1, "0")
    return f"{sign}{digits[:-scale]}.{digits[-scale:]}"


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
        self._frames: list[list[Formula]] = [[]]
        self._last_model: dict[Variable, Constant] | None = None

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
        flat: list[Formula] = [f for frame in self._frames for f in frame]
        return And(flat)

    def check(self) -> str:
        result, model = self._solve_once(self._all_consts())
        # Translate the wrapped solver's verdict to native (see module docstring).
        if result == "False":
            self._last_model = model
            return SAT
        self._last_model = None
        return UNSAT if result == "True" else UNKNOWN

    def model(self) -> dict[Variable, Constant]:
        if self._last_model is None:
            raise RuntimeError("model() called without a preceding SAT check()")
        return self._last_model

    is_exact = False
    timeouts = 0
    _smt2_seq = 0

    @property
    def tolerance(self) -> Fraction:
        """dReal's delta precision: no frontier is meaningful below it."""
        try:
            return Fraction(self._config.get_section("dreal").get_value("precision"))
        except Exception:
            return Fraction("0.001")

    def rv(self, f: Fraction) -> RealVal:
        # dReal3's SMT2 parser has no p/q literal: str(Fraction("3.95")) is
        # "79/20", which is emitted verbatim, fails to parse, exits 1, and is
        # reported by _drealcheckSat as "Unknown".
        return RealVal(_exact_decimal(f))

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
        budget = self._query_budget()
        if budget is None:
            result, _size = solver.solve(consts, None, None)
            model = (solver.make_assignment().get_assignments()
                     if result == "False" else None)
            return result, model

        # Budgeted solve. Uses dRealSolver.process(), which hands back the Popen
        # (so it can be killed) and classifies on returncode rather than by
        # string-matching the model text. Without a budget an undecidable probe
        # hangs the whole run: upstream's only timeout is 1e8 seconds.
        import queue as _q
        import shutil as _sh
        import threading as _th

        # process() writes its SMT2 under ./dreal_log/ and NEVER removes it.
        # Upstream's sync path removes the file; the parallel path keeps it
        # sized for a handful of calls per run. kappa_box issues thousands, so
        # without cleanup this grows without bound (tens of thousands of files
        # in one session). Give each call its own subdirectory and drop it after.
        # [gen] keep-smt2 = 1 retains them for diagnosis.
        DrealReSolveOracle._smt2_seq += 1
        token = f"kbox_{os.getpid()}_{DrealReSolveOracle._smt2_seq}"
        solver.set_file_name(token)
        try:
            keep = str(self._config.get_section("gen").get_value("keep-smt2")) == "1"
        except Exception:
            keep = False

        def _drop():
            if not keep:
                _sh.rmtree(os.path.join("./dreal_log", token), ignore_errors=True)

        main_queue: _q.Queue = _q.Queue()
        sema = _th.Semaphore(0)
        proc = solver.process(main_queue, sema, consts)
        try:
            msg = main_queue.get(timeout=budget)
            # base commit puts (result, assignment, id(proc)); later upstream
            # revisions append elapsed and an error message.
            result, assignment = msg[0], msg[1]
        except _q.Empty:
            try:
                proc.kill()
            except Exception:
                pass
            self.timeouts += 1
            _drop()
            return "Unknown", None
        model = assignment.get_assignments() if result == "False" else None
        _drop()
        return result, model

    def set_budget(self, seconds):
        """Override the per-call budget (None = unbudgeted) until reset."""
        self._budget_override = seconds

    #: Default seconds per solver call. Finite deliberately: dReal can fail to
    #: terminate on a single probe, and STLMC's only timeout is asyncio's 1e8 s,
    #: so an unbounded default leaves a run with no way to make progress.
    #: Set [gen] query-timeout = 0 to opt out.
    DEFAULT_QUERY_TIMEOUT = 60.0

    def _query_budget(self):
        """Seconds allowed per solver call.

        ``[gen] query-timeout`` overrides; 0 / off / none disables the budget."""
        ov = getattr(self, "_budget_override", "unset")
        if ov != "unset":
            return ov
        try:
            sec = self._config.get_section("gen").get_value("query-timeout")
        except Exception:
            sec = None
        if sec in (None, ""):
            return self.DEFAULT_QUERY_TIMEOUT
        if str(sec).strip() in ("0", "off", "none"):
            return None
        return float(sec)


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
        f"generation supports the z3 and dreal backends; got '{underlying}'"
    )


# =========================================================================== #
#  Backend-agnostic formula helpers
# =========================================================================== #
# Build Formula objects from explicit Variable objects and Python bounds; the
# caller supplies the initial-condition variables.

def box_constraint(bounds: BoxBounds) -> Formula:
    """``AND_i (lo_i <= x_i <= hi_i)`` over the given IC variables."""
    terms: list[Formula] = []
    for var, (lo, hi) in bounds.items():
        terms.append(Geq(var, _real(lo)))
        terms.append(Leq(var, _real(hi)))
    return And(terms)


def box_infty(bounds: BoxBounds, drop_var: Variable, direction: int) -> Formula:
    """The box with the face ``(drop_var, direction)`` left unbounded.

    ``direction == +1`` drops the upper bound of ``drop_var``; ``-1`` drops the
    lower bound."""
    terms: list[Formula] = []
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


def grow_bounds(bounds: BoxBounds, var: Variable, direction: int,
                delta: float) -> BoxBounds:
    """Return a copy of ``bounds`` with the face ``(var, direction)`` shifted
    outward by ``delta``. Does not mutate the input."""
    lo, hi = bounds[var]
    new = dict(bounds)
    new[var] = (lo, hi + delta) if direction > 0 else (lo - delta, hi)
    return new


def fix_modes(mode_assignment: dict[Variable, Constant]) -> Formula:
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