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
`base_driver` fills from `common.threshold`. `backend_delta` is the solver's
own precision, the separate `[dreal] precision` setting, and it is 0 for an
exact backend. Both matter: tau says what the search was asked to find,
backend_delta says how much of the answer could be numerical slack.

READING A VERDICT

    falsifier       rho(0) < -backend_delta. The reconstruction violates the
                    property, by more than the backend's own slack. Reportable
                    as a counterexample without qualification.
    marginal        -backend_delta <= rho(0) < 0. Violates, but by less than
                    the backend's precision, so the violation and the slack are
                    the same size; reportable only as violating up to delta.
    threshold-only  0 <= rho(0) < tau. Does not violate the property, but lies
                    within the run's threshold of doing so. Witnesses in this
                    band occur on the exact backend as well, and their number
                    depends on tau.
    unverified      rho(0) >= tau. Neither. The candidate is not what the
                    search was looking for -- a delta artifact, or a
                    reconstruction too coarse to see the violation (see the
                    sampling note below).
    error           the trace could not be reconstructed at all.

SAMPLING IS PART OF THE MEASUREMENT

Robustness is computed on sampled points (the visualizer's default is 50 per
segment, from `numpy.linspace`), so a violation confined to a short window
between two samples is invisible, and rho(0) is an approximation from above in
that case. `samples` sets the density; `validate_pool(..., refine=True)` runs
every counterexample twice, at `samples` and at 4x `samples`, and flags any
whose verdict is not stable across the two as `sampling-sensitive`, so a pool
can be reported as independent of the sampling resolution or not at all.

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
    if rho0 < tau:
        return "threshold-only"
    return "unverified"


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


def validate_pool(payload, samples: int = DEFAULT_SAMPLES, refine: bool = False,
                  tau: float = None, backend_delta: float = 0.0,
                  progress=None) -> List[Dict[str, Any]]:
    """Validate every counterexample in a pool payload.

    `payload` is the unpickled `.counterexamples` tuple. Returns one record
    per counterexample, in pool order, so a consumer can join on `index`.
    `tau` defaults to the payload's threshold; `backend_delta` is the solver
    precision and defaults to 0, which collapses the `marginal` band. See the
    module docstring.
    """
    assns = payload[0]
    rest = list(payload[1:9])
    tau = float(payload[8]) if tau is None else float(tau)
    backend_delta = float(backend_delta)
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
            "refined_verdict": "",
            "stable": "",
            "seconds": "",
            "note": "",
        }
        try:
            # the visualizer prints progress from inside robustness
            with contextlib.redirect_stdout(io.StringIO()):
                verdict, rho0, lo, hi = validate_ce(assn, rest, tau,
                                                    backend_delta, formula,
                                                    samples)
                rec.update(verdict=verdict, rho0=f"{rho0:.12g}",
                           rho_min=f"{lo:.12g}", rho_max=f"{hi:.12g}")
                if refine:
                    fine, rho0f, _, _ = validate_ce(assn, rest, tau, backend_delta,
                                                    formula, samples * REFINE_FACTOR)
                    rec["refined_verdict"] = fine
                    rec["stable"] = "yes" if fine == verdict else "no"
                    if fine != verdict:
                        rec["note"] = (f"sampling-sensitive: rho(0) "
                                       f"{rho0f:.12g} at {REFINE_FACTOR}x")
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
    parser.add_argument("-refine", action="store_true",
                        help=f"re-validate at {REFINE_FACTOR}x sampling and "
                             f"flag unstable verdicts")
    parser.add_argument("-tau", type=float, default=None,
                        help="robustness threshold (default: the pool's own)")
    parser.add_argument("-delta", type=float, default=0.0,
                        help="backend precision, e.g. the [dreal] precision "
                             "value (default 0)")
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