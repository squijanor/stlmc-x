"""Linear feasibility test for a location word.

Builds a QF_LRA projection of a model's execution along a fixed location word and
reports whether that projection is unsatisfiable. Only variables whose flow
derivative is a numeric constant in every step of the word are modelled; for those
the projection asserts the constant-rate integrals, the mode invariants at both
step endpoints, the disjunction of a mode's outgoing guards at a real transition,
identity resets across stays and self-resetting jumps, the declared ranges, the
initial condition, and the dwell timeline (each dwell non-negative, bounded by the
time horizon, summing to the time bound). Constraints not linear over the modelled
variables are omitted. The projection is therefore weaker than the full model:
an unsatisfiable result means the word admits no run, any other result is
inconclusive.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import z3

from ..constraints.constraints import (
    Add,
    And,
    Div,
    Eq,
    Geq,
    Gt,
    Int,
    IntVal,
    Leq,
    Lt,
    Mul,
    Neg,
    Real,
    RealVal,
    Sub,
)
from ..constraints.operations import get_vars


class _Unsupported(Exception):
    """The expression is not linear over the modelled variables."""


def _const(expr) -> Fraction:
    """Exact value of a variable-free arithmetic expression, else raise
    _Unsupported. Kept as a Fraction so no rounding can make the projection
    stronger than the model."""
    if get_vars(expr):
        raise _Unsupported
    if isinstance(expr, (RealVal, IntVal)):
        try:
            return Fraction(str(expr.value))
        except (ValueError, ZeroDivisionError):
            raise _Unsupported from None
    if isinstance(expr, Add):
        return _const(expr.left) + _const(expr.right)
    if isinstance(expr, Sub):
        return _const(expr.left) - _const(expr.right)
    if isinstance(expr, Mul):
        return _const(expr.left) * _const(expr.right)
    if isinstance(expr, Div):
        d = _const(expr.right)
        if d == 0:
            raise _Unsupported
        return _const(expr.left) / d
    if isinstance(expr, Neg):
        return -_const(expr.child)
    raise _Unsupported


def _rat(value) -> z3.ArithRef:
    """Exact z3 rational from a Fraction (never via float)."""
    return z3.RealVal(str(Fraction(value)))


def _lin(expr, zmap: dict[str, z3.ArithRef]):
    """Translate an arithmetic node to a z3 expression over ``zmap`` (variable id
    -> z3 Real). Raise _Unsupported for a variable outside ``zmap`` or a nonlinear
    combination."""
    if isinstance(expr, (RealVal, IntVal)):
        return z3.RealVal(str(expr.value))
    if isinstance(expr, (Real, Int)):
        if expr.id not in zmap:
            raise _Unsupported
        return zmap[expr.id]
    if isinstance(expr, Add):
        return _lin(expr.left, zmap) + _lin(expr.right, zmap)
    if isinstance(expr, Sub):
        return _lin(expr.left, zmap) - _lin(expr.right, zmap)
    if isinstance(expr, Neg):
        return -_lin(expr.child, zmap)
    if isinstance(expr, Mul):
        try:
            c = _const(expr.left)
        except _Unsupported:
            c = None
        if c is not None:
            return _rat(c) * _lin(expr.right, zmap)
        c = _const(expr.right)
        return _lin(expr.left, zmap) * _rat(c)
    if isinstance(expr, Div):
        c = _const(expr.right)
        if c == 0:
            raise _Unsupported
        return _lin(expr.left, zmap) / _rat(c)
    raise _Unsupported


_CMP = {Leq: lambda a, b: a <= b, Geq: lambda a, b: a >= b,
        Lt: lambda a, b: a < b, Gt: lambda a, b: a > b,
        Eq: lambda a, b: a == b}


def _atom(atom, zmap: dict[str, z3.ArithRef]):
    """Translate a comparison atom to z3, else raise _Unsupported."""
    for cls, op in _CMP.items():
        if isinstance(atom, cls):
            return op(_lin(atom.left, zmap), _lin(atom.right, zmap))
    raise _Unsupported


def _conjuncts(formula) -> list:
    """The conjuncts of a formula; And is flattened, other shapes returned whole."""
    if isinstance(formula, And):
        out: list = []
        for c in formula.children:
            out.extend(_conjuncts(c))
        return out
    return [formula]


def _same_atom(a, b) -> bool:
    """Whether ``a`` and ``b`` are ``Eq`` of variables with matching ids."""
    if isinstance(a, Eq) and isinstance(b, Eq):
        la, ra, lb, rb = a.left, a.right, b.left, b.right
        return (isinstance(la, (Real, Int)) and isinstance(lb, (Real, Int))
                and isinstance(ra, (Real, Int)) and isinstance(rb, (Real, Int))
                and la.id == lb.id and ra.id == rb.id)
    return False


def _add_conjuncts(s: z3.Solver, formula, zmap: dict[str, z3.ArithRef]) -> None:
    """Assert every conjunct of ``formula`` that translates over ``zmap``."""
    if formula is None:
        return
    for atom in _conjuncts(formula):
        try:
            s.add(_atom(atom, zmap))
        except _Unsupported:
            pass


class LinearWordFeasibilityFilter:
    """Unsatisfiability test for a location word over the model's linear timeline,
    with a cache keyed by ``(bound, word)``."""

    def __init__(self, model, time_bound, time_horizon=None,
                 cache: dict[tuple[int, tuple[int, ...]], bool] | None = None):
        self.model = model
        self.time_bound = float(time_bound)
        self.time_horizon = None if time_horizon is None else float(time_horizon)
        self.next_str = getattr(model, "next_str", "'")
        # range_dict maps a var to (left_closed, left, right, right_closed).
        self.range_dict_ids = {v.id for v in getattr(model, "range_dict", {})}
        self.range_dict_by_id: dict[str, tuple[float | None, float | None]] = {}
        for v, rng in getattr(model, "range_dict", {}).items():
            try:
                _, left, right, _ = rng
                lo = float(left) if left > -float("inf") else None
                hi = float(right) if right < float("inf") else None
                self.range_dict_by_id[v.id] = (lo, hi)
            except Exception:
                pass
        self.cache: dict[tuple[int, tuple[int, ...]], bool] = (
            cache if cache is not None else {})

    def word_is_infeasible(self, bound: int, word: Sequence[int]) -> bool:
        """True iff the word admits no run on the linear projection; False
        (inconclusive) on a satisfiable, unknown, or unrepresentable projection,
        or on error. Result cached per ``(bound, word)``."""
        try:
            key = (int(bound), tuple(int(m) for m in word))
        except Exception:
            return False
        if key in self.cache:
            return self.cache[key]
        try:
            infeasible = self._check(key[1])
        except Exception:
            return False
        self.cache[key] = infeasible
        return infeasible

    def _rates(self, word: tuple[int, ...]) -> dict[str, list[Fraction]]:
        """Per-step rate of each variable whose flow derivative is a numeric
        constant in every step of the word; empty if there is none."""
        modules = self.model.modules
        common = None
        per_step: list[dict[str, Fraction]] = []
        for mi in word:
            flow = modules[mi]["flow"]
            here: dict[str, Fraction] = {}
            for v, rhs in zip(flow.vars, flow.exps):
                try:
                    here[v.id] = _const(rhs)
                except _Unsupported:
                    pass
            per_step.append(here)
            ids = set(here)
            common = ids if common is None else (common & ids)
        if not common:
            return {}
        return {vid: [per_step[k][vid] for k in range(len(word))]
                for vid in common}

    def _identity_reset(self, vid: str, mode: int, nxt: int) -> bool:
        """Whether the variable's value carries across the transition: a stay, or
        a real transition whose every outgoing guard resets it to itself."""
        if nxt == mode:
            return vid in self.range_dict_ids
        jumps = self.model.modules[mode]["jump"]
        if not jumps:
            return False
        want = Eq(Real(vid + self.next_str), Real(vid))
        for guard in jumps:
            post = jumps[guard]
            if not any(_same_atom(a, want) for a in _conjuncts(post)):
                return False
        return True

    def _guard_disjunction(self, mode: int, zmap: dict[str, z3.ArithRef]):
        """Disjunction of the mode's outgoing guards over their linear atoms, or
        None if any guard contributes no atom."""
        jumps = self.model.modules[mode]["jump"]
        if not jumps:
            return None
        disjuncts = []
        for guard in jumps:
            atoms = []
            for a in _conjuncts(guard):
                try:
                    atoms.append(_atom(a, zmap))
                except _Unsupported:
                    pass
            if not atoms:
                return None
            disjuncts.append(z3.And(atoms))
        if not disjuncts:
            return None
        return z3.Or(disjuncts)

    def _check(self, word: tuple[int, ...]) -> bool:
        rates = self._rates(word)
        if not rates:
            return False
        n = len(word)
        modules = self.model.modules
        s = z3.SolverFor("QF_LRA")

        start = {vid: [z3.Real(f"{vid}#s{k}") for k in range(n)] for vid in rates}
        end = {vid: [z3.Real(f"{vid}#e{k}") for k in range(n)] for vid in rates}
        dwell = [z3.Real(f"dwell#{k}") for k in range(n)]

        # dwell timeline: non-negative, bounded by the horizon, summing to the bound
        for d in dwell:
            s.add(d >= 0)
            if self.time_horizon is not None:
                s.add(d <= z3.RealVal(str(self.time_horizon)))
        s.add(z3.Sum(dwell) == z3.RealVal(str(self.time_bound)))

        # constant-rate integral per step, and value continuity across resets
        for vid, rlist in rates.items():
            for k in range(n):
                s.add(end[vid][k] == start[vid][k] + _rat(rlist[k]) * dwell[k])
            for k in range(n - 1):
                if self._identity_reset(vid, word[k], word[k + 1]):
                    s.add(start[vid][k + 1] == end[vid][k])

        # initial condition at the first step's start
        zmap0 = {vid: start[vid][0] for vid in rates}
        _add_conjuncts(s, getattr(self.model, "init", None), zmap0)

        # declared ranges at every endpoint
        for vid in rates:
            rng = self.range_dict_by_id.get(vid)
            if rng is None:
                continue
            lo, hi = rng
            for k in range(n):
                if lo is not None:
                    s.add(start[vid][k] >= z3.RealVal(str(lo)))
                    s.add(end[vid][k] >= z3.RealVal(str(lo)))
                if hi is not None:
                    s.add(start[vid][k] <= z3.RealVal(str(hi)))
                    s.add(end[vid][k] <= z3.RealVal(str(hi)))

        # invariant at both endpoints; outgoing-guard disjunction at a real transition
        for k, mi in enumerate(word):
            zs = {vid: start[vid][k] for vid in rates}
            ze = {vid: end[vid][k] for vid in rates}
            inv = modules[mi].get("inv")
            _add_conjuncts(s, inv, zs)
            _add_conjuncts(s, inv, ze)
            if k + 1 < n and word[k + 1] != mi:
                g = self._guard_disjunction(mi, ze)
                if g is not None:
                    s.add(g)

        return s.check() == z3.unsat