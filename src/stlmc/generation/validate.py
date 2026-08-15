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
    error           the trace could not be reconstructed at all.

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
    Arccos,
    Arcsin,
    Arctan,
    Cos,
    Div,
    Int,
    IntVal,
    Mul,
    Neg,
    Ode,
    Pow,
    Real,
    RealVal,
    Sin,
    Sqrt,
    Sub,
    Tan,
    Variable,
)
from ..constraints.operations import substitution
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
    raise NotSupportedError(f"cannot evaluate \"{const}\" numerically")


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
    res = odeint(lambda z, t: [_num(dyn, z, dynamic.vars) for dyn in dynamic.exps],
                 initial_values, time_samples)
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

    return point_samples, discrete_samples, times


def _classify(rho0: float, tau: float, backend_delta: float) -> str:
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


def validate_ce(assn, rest, tau: float, backend_delta: float, formula,
                samples: int = DEFAULT_SAMPLES):
    """Validate one counterexample. Returns (verdict, rho0, rho_min, rho_max)."""
    point_samples, discrete_samples, times = _reconstruct(assn, rest, samples)
    time_max = max(t for seg in times for t in seg)
    dp: Dict[Tuple[Any, float], float] = {}
    series = [[robustness(formula, point_samples, discrete_samples, t, times,
                          time_max, dp)
               for t in seg] for seg in times]
    flat = [x for seg in series for x in seg]
    rho0 = series[0][0]
    return _classify(rho0, tau, backend_delta), rho0, min(flat), max(flat)


def pool_backend_delta(payload) -> float:
    """The relaxation the pool was generated under, read from the pool.

    The eleventh payload element, written on the pool path by ``base_driver``.
    A pool without it predates the element and is read as exact, which is what
    it was read as before the element existed.
    """
    return float(payload[10]) if len(payload) > 10 else 0.0


def validate_pool(payload, samples: int = DEFAULT_SAMPLES, refine: bool = True,
                  tau: float = None, backend_delta: float = None,
                  progress=None) -> List[Dict[str, Any]]:
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
    backend_delta = (pool_backend_delta(payload) if backend_delta is None
                     else float(backend_delta))
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
            "seconds": "",
            "note": "",
        }
        try:
            # the visualizer prints progress from inside robustness
            with contextlib.redirect_stdout(io.StringIO()):
                verdict, rho0, lo, hi = validate_ce(assn, rest, tau,
                                                    backend_delta, formula,
                                                    samples)
                margin = _band_margin(rho0, tau, backend_delta)
                rec.update(verdict=verdict, rho0=f"{rho0:.12g}",
                           rho_min=f"{lo:.12g}", rho_max=f"{hi:.12g}",
                           band_margin=f"{margin:.12g}")
                if refine:
                    fine, rho0f, _, _ = validate_ce(assn, rest, tau, backend_delta,
                                                    formula, samples * REFINE_FACTOR)
                    shift = abs(rho0f - rho0)
                    rec["refined_verdict"] = fine
                    rec["rho0_refined"] = f"{rho0f:.12g}"
                    rec["rho0_shift"] = f"{shift:.12g}"
                    rec["stable"] = "yes" if fine == verdict else "no"
                    # The verdict is resolved when rho(0) is further from every
                    # band edge than the two errors that could move it: the
                    # backend's own slack, and how far the value moved when the
                    # sampling was refined. A verdict can be `stable` and
                    # unresolved at once -- staying inside one band says nothing
                    # about how close to its edge it sits.
                    rec["resolved"] = ("yes" if margin > max(backend_delta, shift)
                                       else "no")
                    if fine != verdict:
                        rec["note"] = (f"sampling-sensitive: rho(0) "
                                       f"{rho0f:.12g} at {REFINE_FACTOR}x")
                    elif rec["resolved"] == "no":
                        rec["note"] = (f"edge-resident: rho(0) is {margin:.3g} "
                                       f"from a band edge, against a sampling "
                                       f"shift of {shift:.3g} and a backend "
                                       f"delta of {backend_delta:.3g}")
        except Exception as exc:            # a reconstruction failure is a result
            rec["verdict"] = "error"
            rec["note"] = f"{type(exc).__name__}: {str(exc)[:120]}"
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
        lines.append("  {:<12} {:>4}  (verdict changes under {}x sampling)".format(
            "unstable", len(unstable), REFINE_FACTOR))
    unresolved = [r for r in records if r["resolved"] == "no"]
    if unresolved:
        lines.append("  {:<12} {:>4}  (rho(0) within the measurement error of "
                     "a band edge)".format("edge", len(unresolved)))
    lines.append("  by label:")
    for (label, verdict), n in sorted(by_label.items()):
        lines.append("    {:<10} {:<12} {:>4}".format(label or "-", verdict, n))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stlmc.generation.validate",
        description="Validate the counterexamples in a pool by simulation.")
    parser.add_argument("pool", help="a .counterexamples or .counterexample file")
    parser.add_argument("-samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"time samples per segment (default {DEFAULT_SAMPLES})")
    parser.add_argument("-no-refine", dest="refine", action="store_false",
                        help=f"skip the {REFINE_FACTOR}x re-validation, which "
                             f"is what measures the sampling error")
    parser.add_argument("-tau", type=float, default=None,
                        help="robustness threshold (default: the pool's own)")
    parser.add_argument("-delta", type=float, default=None,
                        help="backend precision (default: the relaxation the "
                             "pool records, 0 for a pool that records none)")
    parser.add_argument("-out", default=None,
                        help="report path (default: <pool>.validation.csv)")
    args = parser.parse_args(argv)

    with open(args.pool, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload[0], list):
        payload = ([payload[0]],) + tuple(payload[1:])   # single-CE file

    out = args.out or f"{args.pool}.validation.csv"
    total = len(payload[0])
    started = time.time()

    def progress(done, n):
        if done % 10 == 0 or done == n:
            print(f"  validated {done}/{n}  ({time.time() - started:.0f}s)",
                  flush=True)

    print(f"validating {total} counterexample(s) in {os.path.basename(args.pool)}")
    records = validate_pool(payload, samples=args.samples, refine=args.refine,
                            tau=args.tau, backend_delta=args.delta,
                            progress=progress)
    write_report(records, out)
    print(summarize(records))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())