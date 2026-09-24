"""Delta-witness validation: does a generated counterexample actually falsify?

A witness produced against an exact backend is a model: the assignment
satisfies the encoding, so the trace it denotes falsifies the property. A
witness produced against a delta-decision backend is not. `delta-sat` means
the encoding is satisfiable up to a numerical relaxation delta, so the point
the backend hands back is a CANDIDATE: it may falsify, or it may sit inside
the delta slack and satisfy the property after all. Nothing in the generation
pipeline re-checks this, so a pool written on the delta path is a pool of
candidates until something simulates them.

For each counterexample in a pool this module reconstructs the trace,
integrating the flow of each mode in turn from the assignment's own initial
values and dwell times, evaluates the STL robustness of the goal formula at
time 0, and classifies the result.

The STL semantics are not reimplemented here: `robustness` is imported from the
visualizer, so a verdict uses the same semantics as the tool's robustness plots.
What this module owns is the numeric evaluation of the flow during integration.
The visualizer's evaluator (`infix_float`) implements Add, Sub, Mul, Div, Pow
and Neg and nothing else, so it raises on any flow containing a square root or a
trigonometric function; `_num` below covers the whole expression grammar. Atoms
inside the formula are unaffected, as `robustness` evaluates those through sympy.

A JUMP INSTANT BELONGS TO THE POST-JUMP SEGMENT

Adjacent segments meet at a variable point tau: the flow of segment i ends at
tau and the flow of segment i+1 begins at tau, and at a jump the two states
differ -- always in the discrete variables, which change across every jump, and
in a continuous variable wherever a reset is not the identity. A single instant
cannot hold both. The reconstruction gives each variable point to the segment
that begins at it -- the post-jump state -- so a jump at time t is evaluated in
the mode entered at t. Concretely, every segment but the last contributes the
half-open interval [tau_i, tau_{i+1}) by dropping its shared right endpoint; the
last segment keeps its closed right endpoint, the end of the trace. A
zero-duration segment therefore contributes no sample and cedes its instant to
the next segment, and rho(0) is read at the first segment that carries a sample,
so a jump at time 0 is read post-jump too.

TWO THRESHOLDS, NOT ONE

A verdict needs both of the tool's numbers, and they are different things.
`tau` is the run's `threshold` -- the ninth payload element, which
`base_driver` fills from `common.threshold`. `backend_delta` is the relaxation
the backend answered under, and it is 0 for an exact backend. Both matter: tau
says what the search was asked to find, backend_delta says how much of the
answer could be numerical slack.

Both are read from the pool. A pool carries tau on its ninth element and
backend_delta on its eleventh, so neither has to be supplied and a delta pool
cannot be validated as an exact one by omitting a flag. A pool written before
the eleventh element existed records no relaxation and is read as exact, which
is how it was read before.

READING A VERDICT

    falsifier       rho(0) < -backend_delta. The reconstruction violates the
                    property, by more than the backend's own slack. Reportable
                    as a counterexample without qualification.
    marginal        -backend_delta <= rho(0) < 0. Violates, but by less than
                    the backend's precision, so the violation and the slack are
                    the same size; reportable only as violating up to delta.
    threshold-only  0 <= rho(0) <= tau. Does not violate the property, but lies
                    within the run's threshold of doing so. Witnesses in this
                    band occur on the exact backend as well, and their number
                    depends on tau. The band is CLOSED at tau because tau is
                    closed in the encoding: the threshold is applied by relaxing
                    the negated goal, so the search was asked for rho(0) <= tau
                    and a witness at exactly tau is what it was told to find.
    unverified      rho(0) > tau. Neither. The candidate is not what the
                    search was looking for -- a delta artifact, or a
                    reconstruction too coarse to see the violation (see the
                    sampling note below).
    error           the trace could not be reconstructed at all, or rho(0) is
                    not a finite number. A non-finite rho(0) -- a divergent
                    integration, or a 0/0 in a flow -- satisfies none of the
                    band comparisons and must not be read as one: NaN fails
                    every comparison and would fall through to `unverified`,
                    and -inf compares below every edge and would read as a clean
                    `falsifier`. It is routed here instead.

IS THE WITNESS A TRACE THE AUTOMATON CAN PRODUCE

Robustness answers a different question from reachability. rho(0) says the
property is violated along the reconstructed signal; it does not say the signal
is a run the automaton admits. A run has to respect the discrete structure at
every jump: the guard holds on the pre-jump state, the reset relates the pre-
and post-jump states, and a segment's dwell equals the gap between its
endpoints. A delta-sat witness is checked only against the encoding the backend
was given, and a two-step encoding that fixes the discrete skeleton on an
abstraction and refines the flow separately can hand back a point that jumps
where the guard is false. Such a point may violate the property and still be no
counterexample, because no run reaches it.

`trace` records that check, alongside and orthogonal to the robustness verdict:
a genuine counterexample is a `falsifier` or `marginal` verdict carried by a
`consistent` trace. It reads the assignment's own boundary variables -- the
values the encoding names x_k_t (segment k's exit), x_{k+1}_0 (segment k+1's
entry), tau_k and time_k -- so it tests the point the backend returned rather
than an interpolation of it. A transition is explained by a declared jump of the
pre-jump mode whose reset the two states satisfy; among the jumps that explain
it the guard is read on the pre-jump state, and the transition is
`guard-violating` only when no explaining jump has a satisfied guard. A
transition no declared jump explains is a stutter when the mode is unchanged and
the state is carried identically, and a `reset-mismatch` otherwise; a dwell that
disagrees with its endpoints is a `time-mismatch`. Every comparison is signed
and takes backend_delta as its slack, so a guard met to within the backend's own
precision is not called a violation. The check runs only on a payload that
carries mode and jump structure; a reconstruction fixture without it is left
unchecked, and `trace` is empty.

SAMPLING IS PART OF THE MEASUREMENT, IN BOTH DIRECTIONS

Robustness is computed on sampled points (the visualizer's default is 50 per
segment, from `numpy.linspace`), so a feature of the signal that lives between
two samples is invisible. Which way that biases rho(0) depends on the operator,
and only one of the two is conservative.

An Always is a minimum over the sampled times, and a minimum over a subset
over-estimates: a violation narrower than the sample spacing is missed, rho(0)
reads too high, and a genuine falsifier is demoted. An Eventually and the outer
level of an Until are maxima, and a maximum over a subset UNDER-estimates: a
peak between two samples is missed, rho(0) reads too low, and the verdict moves
DOWN the ladder -- toward `falsifier`. A Release is a minimum with a maximum
nested inside it and has no guaranteed direction at all. So a formula that is
not a minimum at every temporal node can be reported as violating the property
when it does not, which is the failure this module exists to prevent, and no
sampling density removes the possibility.

What can be measured is how far rho(0) moves when the density changes.
`validate_pool(..., refine=True)` -- the default -- evaluates every
counterexample at `samples` and at 4x `samples` and records both, their
difference (`rho0_shift`), and whether the verdict changed (`stable`).
`band_margin` records the distance from rho(0) to the nearest band edge, and
`resolved` compares the two: a verdict is resolved when rho(0) is further from
every edge than both the sampling shift and the backend's own slack. A verdict
can be stable and unresolved at once -- staying inside one band says nothing
about how close to its edge it sits -- so `resolved`, not `stable`, is what says
a number is safe to quote.

The result is written beside the pool as `<pool>.validation.csv` rather than
into the pool itself: the pickled payload format stays exactly as it is, and a
consumer joins on the counterexample index.
"""

import argparse
import contextlib
import csv
import io
import math
import os
import pickle
import time
from fractions import Fraction
from functools import singledispatch
from typing import Any, Dict, List, Tuple

import numpy
from scipy.integrate import odeint

from ..constraints.constraints import (
    Add,
    And,
    Arccos,
    Arcsin,
    Arctan,
    BoolVal,
    Cos,
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
    Neq,
    Not,
    Ode,
    Or,
    Pow,
    Real,
    RealVal,
    Sin,
    Sqrt,
    Sub,
    Tan,
    Variable,
)
from ..constraints.operations import get_vars, substitution
from ..exception.exception import NotSupportedError
from ..visualize.visualizer import (
    DiscreteSampler,
    Projector,
    SolutionPointSampler,
    Visualizer,
    robustness,
)

__all__ = ["validate_ce", "validate_pool", "write_report", "main"]

DEFAULT_SAMPLES = 50
REFINE_FACTOR = 4


# ---------------------------------------------------------------- evaluation


@singledispatch
def _num(const: Any, vec, var_list: List[Variable]) -> float:
    raise NotSupportedError(f'cannot evaluate "{const}" numerically')


def _register(cls, fn):
    _num.register(cls)(fn)


_register(RealVal, lambda c, vec, vs: float(Fraction(str(c.value))))
_register(IntVal, lambda c, vec, vs: float(Fraction(str(c.value))))
_register(Real, lambda c, vec, vs: vec[vs.index(c)])
_register(Int, lambda c, vec, vs: vec[vs.index(c)])
_register(Neg, lambda c, vec, vs: -_num(c.child, vec, vs))
_register(Add, lambda c, vec, vs: _num(c.left, vec, vs) + _num(c.right, vec, vs))
_register(Sub, lambda c, vec, vs: _num(c.left, vec, vs) - _num(c.right, vec, vs))
_register(Mul, lambda c, vec, vs: _num(c.left, vec, vs) * _num(c.right, vec, vs))
_register(Div, lambda c, vec, vs: _num(c.left, vec, vs) / _num(c.right, vec, vs))
_register(Pow, lambda c, vec, vs: _num(c.left, vec, vs) ** _num(c.right, vec, vs))
_register(Sqrt, lambda c, vec, vs: numpy.sqrt(_num(c.child, vec, vs)))
_register(Sin, lambda c, vec, vs: numpy.sin(_num(c.child, vec, vs)))
_register(Cos, lambda c, vec, vs: numpy.cos(_num(c.child, vec, vs)))
_register(Tan, lambda c, vec, vs: numpy.tan(_num(c.child, vec, vs)))
_register(Arcsin, lambda c, vec, vs: numpy.arcsin(_num(c.child, vec, vs)))
_register(Arccos, lambda c, vec, vs: numpy.arccos(_num(c.child, vec, vs)))
_register(Arctan, lambda c, vec, vs: numpy.arctan(_num(c.child, vec, vs)))


def _ode_samples(time_samples: List[float], dynamic, initial_values: List[float]):
    """Integrate one segment. Same call as the visualizer's, complete evaluator."""
    res = odeint(
        lambda z, t: [_num(dyn, z, dynamic.vars) for dyn in dynamic.exps],
        initial_values,
        time_samples,
    )
    return {v: [row[dynamic.vars.index(v)] for row in res] for v in dynamic.vars}


def _time_samples(projector: Projector, samples: int) -> List[List[float]]:
    """The visualizer's time sampling, with the density exposed.

    Upstream calls numpy.linspace without `num`, fixing it at 50; the segment
    endpoints are the tau values of the assignment, parsed as fractions
    because a solver may report 79/20 rather than 3.95.
    """
    points = projector.get_variable_points()
    assignment = projector.assignment
    out: List[List[float]] = []
    for i in range(len(points) - 1):
        lo = float(Fraction(str(assignment[points[i]].value)))
        hi = float(Fraction(str(assignment[points[i + 1]].value)))
        out.append([t.item() for t in numpy.linspace(lo, hi, num=samples)])
    return out


def _own_post_jump(
    times: List[List[float]],
    point_samples: Dict[Variable, List[List[float]]],
    discrete_samples: Dict[Variable, List[List[float]]],
) -> None:
    """Give each variable point to the segment that begins at it.

    Every segment but the last is restricted to the half-open interval
    [tau_i, tau_{i+1}) by dropping the samples at or beyond its shared right
    boundary, so the boundary instant survives only in the next (post-jump)
    segment. The robustness lookup returns the first sample matching a time, so
    with the pre-jump copy gone it reads the post-jump state there. The last
    segment keeps its closed right endpoint, the end of the trace. A
    zero-duration segment keeps no sample and cedes its instant to the next one.
    `times`, `point_samples` and `discrete_samples` are edited in place, in
    lockstep by segment index.
    """
    boundaries = [
        times[i + 1][0] if times[i + 1] else None for i in range(len(times) - 1)
    ]
    for i, boundary in enumerate(boundaries):
        if boundary is None:
            continue
        keep = [k for k, t in enumerate(times[i]) if t < boundary]
        times[i] = [times[i][k] for k in keep]
        for series in point_samples.values():
            series[i] = [series[i][k] for k in keep]
        for series in discrete_samples.values():
            series[i] = [series[i][k] for k in keep]


# ---------------------------------------------------------------- validation


def _reconstruct(assn, rest, samples: int):
    """Trace reconstruction, mirroring Visualizer.generate_data."""
    modules, mode_var_dict, propositions, cont_var_dict, prop_dict = rest[0:5]
    projector = Projector(assn, mode_var_dict, propositions, modules)
    max_bound = projector.get_max_bound()
    times = _time_samples(projector, samples)
    c_val = Visualizer.make_cont_values(assn, cont_var_dict, max_bound)

    point_samples: Dict[Variable, List[List[float]]] = {}
    discrete_samples: Dict[Variable, List[List[float]]] = {}
    sp = SolutionPointSampler()
    ds = DiscreteSampler(projector)

    for index, module in enumerate(projector.get_ordered_modules()):
        dynamics = module["flow"]
        initial = [c_val[v.id][index][0] for v in dynamics.vars]
        if isinstance(dynamics, Ode):
            gen = _ode_samples(times[index], dynamics, initial)
        else:
            gen = sp.generate_samples(times[index], dynamics, initial)
        for v in gen:
            point_samples.setdefault(v, []).append(list(gen[v]))
        for v, vals in ds.generate_samples_at(times[index], index).items():
            discrete_samples.setdefault(v, []).append(vals)

    _own_post_jump(times, point_samples, discrete_samples)
    return point_samples, discrete_samples, times


def _classify(rho0: float, tau: float, backend_delta: float) -> str:
    # A non-finite rho(0) -- a divergent integration, or a 0/0 in a flow -- is
    # not a band: every comparison against NaN is False, so it would fall
    # through to `unverified`, and -inf compares below every edge and would read
    # as a clean `falsifier`. Route it out of the ordinary bands.
    if not math.isfinite(rho0):
        return "error"
    if rho0 < -backend_delta:
        return "falsifier"
    if rho0 < 0.0:
        return "marginal"
    # Closed at tau, because tau is closed in the encoding: the threshold is
    # applied by relaxing the negated goal, so what the search was asked for is
    # rho(0) <= tau. A witness at exactly tau is what it was told to find, and
    # a strict comparison here left rounding in the reconstruction to decide
    # which side of the band such a witness landed on.
    if rho0 <= tau:
        return "threshold-only"
    return "unverified"


def _band_margin(rho0: float, tau: float, backend_delta: float) -> float:
    """Distance from rho(0) to the nearest band edge.

    A verdict is three comparisons, so how far the value sits from all three is
    what says whether the verdict survives the error in measuring it. Reported
    per counterexample so an edge case identifies itself in the report rather
    than being noticed later.
    """
    return min(abs(rho0 - edge) for edge in (-backend_delta, 0.0, tau))


# ------------------------------------------------------------- trace check

# A guard or reset met to within the backend's own precision is not a fault, so
# every comparison takes backend_delta as slack; this is the floor under it, for
# an exact pool that records none. A dwell is an equality the encoding builds
# from a clock integral, so it holds to integration precision and only a gross
# disagreement -- larger than any rounding -- is a structural fault.
_TRACE_EPS = 1e-9
_DWELL_TOL = 1e-2


def _sat_margin(const: Any) -> float:
    """Signed satisfaction of a variable-free constraint: >= 0 means satisfied.

    The atom is evaluated after its variables have been replaced by their
    assignment values, so `_num` -- the complete evaluator the flow already uses
    -- sees only constants. An Eq is satisfied at equality, so its margin is the
    negated distance between the sides; a conjunction is as satisfied as its
    weakest term, a disjunction as its strongest. A mode reset compares two
    booleans, evaluated as equality of truth values.
    """
    if isinstance(const, (Geq, Gt)):
        return _num(const.left, [], []) - _num(const.right, [], [])
    if isinstance(const, (Leq, Lt)):
        return _num(const.right, [], []) - _num(const.left, [], [])
    if isinstance(const, Eq):
        if isinstance(const.left, BoolVal) or isinstance(const.right, BoolVal):
            return 1.0 if _bool(const.left) == _bool(const.right) else -1.0
        return -abs(_num(const.left, [], []) - _num(const.right, [], []))
    if isinstance(const, Neq):
        if isinstance(const.left, BoolVal) or isinstance(const.right, BoolVal):
            return 1.0 if _bool(const.left) != _bool(const.right) else -1.0
        return abs(_num(const.left, [], []) - _num(const.right, [], []))
    if isinstance(const, And):
        return min(_sat_margin(c) for c in const.children)
    if isinstance(const, Or):
        return max(_sat_margin(c) for c in const.children)
    if isinstance(const, Not):
        return -_sat_margin(const.child)
    if isinstance(const, BoolVal):
        return 1.0 if const.value == "True" else -1.0
    raise NotSupportedError(f'cannot evaluate "{const}" as a predicate')


def _bool(const: Any) -> bool:
    """Truth value of a boolean constant."""
    return getattr(const, "value", None) == "True"


def _assn_raw(assn, var_id: str):
    """The assignment's value object for a variable id, or None if it holds
    none. Real, Int and Bool valuations share one namespace, so this reads a
    value of any type; a numeric caller filters through `_assn_val`."""
    return assn.get(Real(var_id))


def _assn_val(assn, var_id: str):
    """The assignment's numeric value for a variable id, or None if it holds
    none or a non-numeric (a boolean mode encoding)."""
    value = _assn_raw(assn, var_id)
    if value is None:
        return None
    try:
        return float(Fraction(str(value.value)))
    except ValueError:
        return None


def _boundary_valuation(const, assn, bound: int, base_names, mode_ids):
    """Map every variable of a jump to its value at the bound -> bound+1 edge.

    A variable that is one of the model's own names is the pre-jump state, read
    at x_k_t (a mode at m_k); one carrying the primed suffix the encoding adds
    for a post-state is read at x_{k+1}_0 (a mode at m_{k+1}). Longest name
    first, so a name that is a prefix of another does not capture it. Each value
    is bound as its own object, so a boolean mode reset is compared as a boolean
    and a numeric one as a number. Returns None when the assignment holds none
    of a value the jump needs, so a jump that cannot be evaluated is skipped
    rather than guessed.
    """
    valuation: Dict[Variable, Any] = {}
    for var in get_vars(const):
        base = next(
            (name for name in base_names if var.id == name or var.id.startswith(name)),
            None,
        )
        is_post = base is not None and var.id != base
        name = base if base is not None else var.id
        if name in mode_ids:
            value = _assn_raw(
                assn, f"{name}_{bound + 1}" if is_post else f"{name}_{bound}"
            )
        else:
            value = _assn_raw(
                assn, f"{name}_{bound + 1}_0" if is_post else f"{name}_{bound}_t"
            )
        if value is None:
            return None
        valuation[var] = value
    return valuation


def _mode_at(assn, mode_name: str, bound: int):
    """The mode index of a segment, from the currentMode counter the encoding
    writes for every model; the mode variable is a fallback for a payload that
    carries no counter, and is read only when it is numeric."""
    value = _assn_val(assn, f"currentMode_{bound}")
    if value is None:
        value = _assn_val(assn, f"{mode_name}_{bound}")
    return None if value is None else int(round(value))


def _trace_faults(assn, modules, mode_var_dict, range_dict, backend_delta):
    """Check guard, reset and dwell at every jump of the assignment's trace.

    Returns (trace, guard_margin, dwell_slack). `trace` is "consistent",
    "guard-violating", "reset-mismatch", "time-mismatch", or "" when the payload
    carries no mode or jump structure to check against. `guard_margin` is the
    smallest (most violated) guard margin over the jumps and `dwell_slack` the
    largest dwell disagreement; both are None when nothing was checked.
    """
    mode_ids = {var.id for var in mode_var_dict.values()}
    if not mode_ids or not any(module.get("jump") for module in modules):
        return "", None, None
    base_names = sorted(
        {var.id for var in range_dict} | mode_ids, key=len, reverse=True
    )
    mode_name = next(iter(mode_ids))
    slack = max(backend_delta, _TRACE_EPS)

    segments = 0
    while _mode_at(assn, mode_name, segments) is not None:
        segments += 1

    trace = "consistent"
    worst_guard = None
    worst_dwell = None
    for bound in range(segments - 1):
        pre = _mode_at(assn, mode_name, bound)
        if pre is None or pre >= len(modules):
            continue

        explaining_guards = []
        for guard, reset in modules[pre].get("jump", {}).items():
            valuation = _boundary_valuation(
                And([guard, reset]), assn, bound, base_names, mode_ids
            )
            if valuation is None:
                continue
            if _sat_margin(substitution(reset, valuation)) >= -slack:
                explaining_guards.append(_sat_margin(substitution(guard, valuation)))

        dwell = _assn_val(assn, f"time_{bound}")
        tau_k = _assn_val(assn, f"tau_{bound}")
        tau_next = _assn_val(assn, f"tau_{bound + 1}")
        if dwell is not None and tau_k is not None and tau_next is not None:
            gap = dwell - (tau_next - tau_k)
            if worst_dwell is None or abs(gap) > abs(worst_dwell):
                worst_dwell = gap
            if abs(gap) > _DWELL_TOL and trace == "consistent":
                trace = "time-mismatch"

        if explaining_guards:
            best = max(explaining_guards)
            if worst_guard is None or best < worst_guard:
                worst_guard = best
            if best < -slack and trace in ("consistent", "time-mismatch"):
                trace = "guard-violating"
        elif not _is_stutter(assn, modules, range_dict, mode_name, bound, slack):
            if trace in ("consistent", "time-mismatch"):
                trace = "reset-mismatch"

    return trace, worst_guard, worst_dwell


def _is_stutter(assn, modules, range_dict, mode_name, bound, slack) -> bool:
    """A transition no declared jump explains is legitimate only as a stutter:
    the mode is unchanged and every continuous variable is carried identically
    across the instant (the steady self-loop the encoding adds)."""
    pre = _mode_at(assn, mode_name, bound)
    post = _mode_at(assn, mode_name, bound + 1)
    if pre is None or post is None or pre != post:
        return False
    for var in range_dict:
        entry = _assn_val(assn, f"{var.id}_{bound + 1}_0")
        exit_ = _assn_val(assn, f"{var.id}_{bound}_t")
        if entry is None or exit_ is None or abs(entry - exit_) > slack:
            return False
    return True


def _append_note(record: Dict[str, Any], message: str) -> None:
    record["note"] = f"{record['note']}; {message}" if record["note"] else message


def validate_ce(
    assn,
    rest,
    tau: float,
    backend_delta: float,
    formula,
    samples: int = DEFAULT_SAMPLES,
):
    """Validate one counterexample. Returns (verdict, rho0, rho_min, rho_max)."""
    point_samples, discrete_samples, times = _reconstruct(assn, rest, samples)
    time_max = max(t for seg in times for t in seg)
    dp: Dict[Tuple[Any, float], float] = {}
    series = [
        [
            robustness(formula, point_samples, discrete_samples, t, times, time_max, dp)
            for t in seg
        ]
        for seg in times
    ]
    flat = [x for seg in series for x in seg]
    # rho(0) at the trace's initial instant, which post-jump ownership assigns
    # to the first segment that carries a sample: a zero-duration leading
    # segment holds none, so a jump at time 0 is read in the mode entered at 0.
    first = next((i for i, seg in enumerate(series) if seg), None)
    if first is None:
        raise NotSupportedError("trace reconstruction produced no samples")
    rho0 = series[first][0]
    return _classify(rho0, tau, backend_delta), rho0, min(flat), max(flat)


def pool_backend_delta(payload) -> float:
    """The relaxation the pool was generated under, read from the pool.

    The eleventh payload element, written on the pool path by ``base_driver``.
    A pool without it predates the element and is read as exact, which is what
    it was read as before the element existed.
    """
    return float(payload[10]) if len(payload) > 10 else 0.0


def validate_pool(
    payload,
    samples: int = DEFAULT_SAMPLES,
    refine: bool = True,
    tau: float = None,
    backend_delta: float = None,
    progress=None,
) -> List[Dict[str, Any]]:
    """Validate every counterexample in a pool payload.

    `payload` is the unpickled `.counterexamples` tuple. Returns one record
    per counterexample, in pool order, so a consumer can join on `index`.
    `tau` defaults to the payload's threshold and `backend_delta` to the
    relaxation the pool records, so a delta pool cannot be validated as an
    exact one by omission. `refine` is on by default: the shift in rho(0)
    between two sampling densities is the only estimate of the sampling error
    this module can make, and every verdict is a comparison that error can
    cross. See the module docstring.
    """
    assns = payload[0]
    rest = list(payload[1:9])
    tau = float(payload[8]) if tau is None else float(tau)
    backend_delta = (
        pool_backend_delta(payload) if backend_delta is None else float(backend_delta)
    )
    labels = payload[9] if len(payload) > 9 else [""] * len(assns)
    formula = substitution(payload[6], payload[5])

    records: List[Dict[str, Any]] = []
    for i, assn in enumerate(assns):
        started = time.time()
        rec: Dict[str, Any] = {
            "index": i,
            "label": labels[i] if i < len(labels) else "",
            "verdict": "",
            "rho0": "",
            "rho_min": "",
            "rho_max": "",
            "tau": tau,
            "backend_delta": backend_delta,
            "samples": samples,
            "band_margin": "",
            "refined_verdict": "",
            "rho0_refined": "",
            "rho0_shift": "",
            "stable": "",
            "resolved": "",
            "trace": "",
            "guard_margin": "",
            "dwell_slack": "",
            "seconds": "",
            "note": "",
        }
        try:
            # the visualizer prints progress from inside robustness
            with contextlib.redirect_stdout(io.StringIO()):
                verdict, rho0, lo, hi = validate_ce(
                    assn, rest, tau, backend_delta, formula, samples
                )
                if not math.isfinite(rho0):
                    # A non-finite rho(0) is not a band; `_classify` routes it
                    # to `error`. Record the value and skip the refinement,
                    # whose band comparisons a non-finite number cannot inform.
                    rec.update(
                        verdict="error",
                        rho0=f"{rho0:.12g}",
                        note=f"non-finite rho(0): {rho0:.12g}",
                    )
                else:
                    margin = _band_margin(rho0, tau, backend_delta)
                    rec.update(
                        verdict=verdict,
                        rho0=f"{rho0:.12g}",
                        rho_min=f"{lo:.12g}",
                        rho_max=f"{hi:.12g}",
                        band_margin=f"{margin:.12g}",
                    )
                    if refine:
                        fine, rho0f, _, _ = validate_ce(
                            assn,
                            rest,
                            tau,
                            backend_delta,
                            formula,
                            samples * REFINE_FACTOR,
                        )
                        shift = abs(rho0f - rho0)
                        rec["refined_verdict"] = fine
                        rec["rho0_refined"] = f"{rho0f:.12g}"
                        rec["rho0_shift"] = f"{shift:.12g}"
                        rec["stable"] = "yes" if fine == verdict else "no"
                        # The verdict is resolved when rho(0) is further from
                        # every band edge than the two errors that could move
                        # it: the backend's own slack, and how far the value
                        # moved when the sampling was refined. A verdict can be
                        # `stable` and unresolved at once -- staying inside one
                        # band says nothing about how close to its edge it sits.
                        rec["resolved"] = (
                            "yes" if margin > max(backend_delta, shift) else "no"
                        )
                        if fine != verdict:
                            rec["note"] = (
                                f"sampling-sensitive: rho(0) "
                                f"{rho0f:.12g} at {REFINE_FACTOR}x"
                            )
                        elif rec["resolved"] == "no":
                            rec["note"] = (
                                f"edge-resident: rho(0) is {margin:.3g} "
                                f"from a band edge, against a sampling "
                                f"shift of {shift:.3g} and a backend "
                                f"delta of {backend_delta:.3g}"
                            )
        except Exception as exc:  # a reconstruction failure is a result
            rec["verdict"] = "error"
            rec["note"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        # The trace check is orthogonal to the robustness verdict and reads the
        # assignment directly, so it runs whether or not reconstruction
        # succeeded. On a payload without mode or jump structure it returns "".
        try:
            trace, guard_margin, dwell_slack = _trace_faults(
                assn, rest[0], rest[1], rest[3], backend_delta
            )
        except Exception as exc:
            trace, guard_margin, dwell_slack = "error", None, None
            _append_note(rec, f"trace check: {type(exc).__name__}")
        rec["trace"] = trace
        rec["guard_margin"] = "" if guard_margin is None else f"{guard_margin:.12g}"
        rec["dwell_slack"] = "" if dwell_slack is None else f"{dwell_slack:.12g}"
        if trace == "guard-violating":
            _append_note(
                rec,
                f"guard violated by {abs(guard_margin):.3g}: the "
                "witness is not a trace the automaton can produce",
            )
        elif trace == "reset-mismatch":
            _append_note(rec, "a jump's reset matches no declared edge")
        elif trace == "time-mismatch":
            _append_note(
                rec, f"a dwell disagrees with its endpoints by {abs(dwell_slack):.3g}"
            )
        rec["seconds"] = "%.2f" % (time.time() - started)
        records.append(rec)
        if progress is not None:
            progress(i + 1, len(assns))
    return records


def write_report(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def summarize(records: List[Dict[str, Any]]) -> str:
    from collections import Counter

    counts = Counter(r["verdict"] for r in records)
    by_label = Counter((r["label"], r["verdict"]) for r in records)
    lines = ["", f"{len(records)} counterexample(s)"]
    for verdict in ("falsifier", "marginal", "threshold-only", "unverified", "error"):
        if counts[verdict]:
            lines.append(f"  {verdict:<12} {counts[verdict]:>4}")
    unstable = [r for r in records if r["stable"] == "no"]
    if unstable:
        lines.append(
            "  {:<12} {:>4}  (verdict changes under {}x sampling)".format(
                "unstable", len(unstable), REFINE_FACTOR
            )
        )
    unresolved = [r for r in records if r["resolved"] == "no"]
    if unresolved:
        lines.append(
            "  {:<12} {:>4}  (rho(0) within the measurement error of "
            "a band edge)".format("edge", len(unresolved))
        )
    traces = Counter(r["trace"] for r in records if r["trace"])
    for trace in ("guard-violating", "reset-mismatch", "time-mismatch"):
        if traces[trace]:
            lines.append(
                f"  {trace:<14} {traces[trace]:>4}  (witness is not a "
                "trace the automaton can produce)"
            )
    lines.append("  by label:")
    for (label, verdict), n in sorted(by_label.items()):
        lines.append("    {:<10} {:<12} {:>4}".format(label or "-", verdict, n))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stlmc.generation.validate",
        description="Validate the counterexamples in a pool by simulation.",
    )
    parser.add_argument("pool", help="a .counterexamples or .counterexample file")
    parser.add_argument(
        "-samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"time samples per segment (default {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "-no-refine",
        dest="refine",
        action="store_false",
        help=f"skip the {REFINE_FACTOR}x re-validation, which "
        f"is what measures the sampling error",
    )
    parser.add_argument(
        "-tau",
        type=float,
        default=None,
        help="robustness threshold (default: the pool's own)",
    )
    parser.add_argument(
        "-delta",
        type=float,
        default=None,
        help="backend precision (default: the relaxation the "
        "pool records, 0 for a pool that records none)",
    )
    parser.add_argument(
        "-out", default=None, help="report path (default: <pool>.validation.csv)"
    )
    args = parser.parse_args(argv)

    with open(args.pool, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload[0], list):
        payload = ([payload[0]],) + tuple(payload[1:])  # single-CE file

    out = args.out or f"{args.pool}.validation.csv"
    total = len(payload[0])
    started = time.time()

    def progress(done, n):
        if done % 10 == 0 or done == n:
            print(f"  validated {done}/{n}  ({time.time() - started:.0f}s)", flush=True)

    print(f"validating {total} counterexample(s) in {os.path.basename(args.pool)}")
    records = validate_pool(
        payload,
        samples=args.samples,
        refine=args.refine,
        tau=args.tau,
        backend_delta=args.delta,
        progress=progress,
    )
    write_report(records, out)
    print(summarize(records))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
