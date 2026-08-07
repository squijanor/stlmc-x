"""Region box discovery (kappa_box): single-box frontier growth, no certificate.

Around a falsifying pivot on a fixed mode path, grow an axis-aligned box of
initial conditions and collect a falsifying witness per expansion. A face grows
by a minimum separation theta at a time: it expands only while a falsifying
initial condition exists at least theta beyond the current bound, and stops when
the falsifying region ends within theta (NEG-only). theta both guarantees
termination over a continuous region and spaces the witnesses.

The pivot is taken at the shallowest depth that admits a counterexample; the box
is grown there, which keeps the encoding small relative to the maximum depth.

This module collects the witnesses of a single box. Face labeling (deep vs
boundary, positive encoding), region blocking / re-pivoting, and cross-box
thinning are not implemented here; the pool is the collected witnesses of one box.

Initial-condition variables are the step-0 state copies ``<name>_0_0`` for each
state variable named by ``range_dict``; mode variables are ``currentMode_k``.
Box bounds are exact rationals so the degenerate pivot box is represented
exactly.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction
from typing import Dict, List, Tuple

from ..constraints.constraints import (
    And,
    BoolVal,
    Constant,
    Eq,
    Formula,
    Geq,
    Leq,
    RealVal,
    Variable,
)
from ..objects.algorithm import Algorithm
from .encode import Encoder
from .oracle import SAT, Z3IncrementalOracle
from .pathenum import _gen_int, _resolve_seed, _z3_logic

_MODE_RE = re.compile(r"^currentMode_(\d+)$")

# Bound the coordinate-descent passes and the steps per face; both are safety
# nets -- growth terminates naturally once faces reach the region boundary.
_MAX_PASSES = 16
_MAX_STEPS = 10000


def _frac(value) -> Fraction:
    """Constant value string (decimal or p/q) -> exact Fraction."""
    return Fraction(str(value))


def _rv(f: Fraction) -> RealVal:
    return RealVal(str(f))


def _value_of(assn: Dict[Variable, Constant], var: Variable) -> Fraction:
    """Value of ``var`` (matched by id) in an assignment, as a Fraction."""
    for v, c in assn.items():
        if v.id == var.id:
            return _frac(c.value)
    raise KeyError(var.id)


def _mode_fix(assn: Dict[Variable, Constant]) -> Formula:
    """AND_k (currentMode_k == value) pinning the pivot's path."""
    terms = [Eq(v, c) for v, c in assn.items() if _MODE_RE.match(v.id)]
    return And(terms) if terms else BoolVal("True")


def _ic_pivots(assn: Dict[Variable, Constant], range_dict) -> Dict[Variable, Fraction]:
    """Initial-condition variables (<name>_0_0) and their pivot values."""
    ic_ids = {"{}_0_0".format(k.id) for k in range_dict}
    return {v: _frac(c.value) for v, c in assn.items() if v.id in ic_ids}


def _gen_frac(config, key: str, default: str) -> Fraction:
    if config.is_section_in("gen"):
        section = config.get_section("gen")
        if section.is_argument_in(key):
            return Fraction(section.get_value(key))
    return Fraction(default)


def _box_of(box: Dict[Variable, List[Fraction]], skip: Variable) -> Formula:
    """AND over dimensions (except ``skip``) of lo <= x <= hi."""
    terms: List[Formula] = []
    for var, (lo, hi) in box.items():
        if var is skip:
            continue
        terms.append(Geq(var, _rv(lo)))
        terms.append(Leq(var, _rv(hi)))
    return And(terms) if terms else BoolVal("True")


class RegionBoxDiscovery(Algorithm):
    """kappa_box: grow one witness box around a falsifying pivot."""

    def __init__(self) -> None:
        self.debug_name = ""

    def set_debug(self, msg: str) -> None:
        self.debug_name = msg

    def _find_pivot(
        self, encoder: Encoder, max_depth: int, logic: str, seed: int, printer
    ):
        """Sweep depths 1..N; return (depth, oracle, pivot model, encoding) for
        the first satisfiable depth, or (None, None, None, None)."""
        for depth in range(1, max_depth + 1):
            encoding = encoder.encode_at(depth)
            oracle = Z3IncrementalOracle(logic, seed)
            oracle.assert_(encoding.consts)
            if oracle.check() == SAT:
                printer.print_verbose("[kappa_box] pivot at depth {}".format(depth))
                return depth, oracle, oracle.model(), encoding
            encoder.reset()
        return None, None, None, None

    def run(self, model, goal, prop_dict, config, solver, logger, printer):
        common = config.get_section("common")
        max_depth = int(common.get_value("bound"))
        tau_max = float(common.get_value("time-bound"))
        delta = float(common.get_value("threshold"))
        underlying = common.get_value("solver")
        if underlying != "z3":
            raise NotImplementedError(
                "kappa_box currently supports the z3 backend; got '{}'".format(underlying)
            )

        logic = _z3_logic(config)
        seed = _resolve_seed(config)
        theta = _gen_frac(config, "epsilon", "0.01")  # min IC separation / granularity

        hash_seed = os.environ.get("PYTHONHASHSEED")
        if hash_seed is None or not hash_seed.isdigit():
            printer.print_normal(
                "warning: PYTHONHASHSEED is not fixed; constraint ordering is not pinned, "
                "so results may vary run to run. Set PYTHONHASHSEED for reproducibility."
            )

        encoder = Encoder(model, goal, prop_dict, delta, tau_max)
        depth, oracle, pivot, encoding = self._find_pivot(
            encoder, max_depth, logic, seed, printer
        )
        if pivot is None:
            return "True", 0.0, max_depth, []

        oracle.assert_(_mode_fix(pivot))  # pin the path for the rest of the run

        ic = _ic_pivots(pivot, encoding.range_dict)
        if not ic:
            raise RuntimeError(
                "no initial-condition variables (<name>_0_0) found; cannot grow an IC box"
            )

        box: Dict[Variable, List[Fraction]] = {v: [p, p] for v, p in ic.items()}
        witnesses: List[Dict[Variable, Constant]] = [pivot]
        printer.print_verbose(
            "[kappa_box] depth {}: growing IC box over {} dim(s)".format(depth, len(box))
        )

        changed = True
        passes = 0
        while changed and passes < _MAX_PASSES:
            changed = False
            passes += 1
            for var in box:
                others = _box_of(box, skip=var)
                for direction in (+1, -1):
                    for _ in range(_MAX_STEPS):
                        lo, hi = box[var]
                        # A falsifying IC at least theta beyond the current bound,
                        # within a theta-wide window (keeps witnesses local/spread).
                        if direction > 0:
                            edge = And([Geq(var, _rv(hi + theta)), Leq(var, _rv(hi + 2 * theta))])
                        else:
                            edge = And([Leq(var, _rv(lo - theta)), Geq(var, _rv(lo - 2 * theta))])

                        oracle.push()
                        oracle.assert_(And([others, edge]))
                        if oracle.check() != SAT:
                            oracle.pop()
                            break
                        witness = oracle.model()
                        oracle.pop()

                        witnesses.append(witness)
                        wv = _value_of(witness, var)
                        if direction > 0:
                            box[var][1] = wv
                        else:
                            box[var][0] = wv
                        changed = True
                        printer.print_verbose(
                            "[kappa_box] depth {}: {} witness(es) (pass {})".format(
                                depth, len(witnesses), passes
                            )
                        )

        result = "False" if witnesses else "True"
        return result, 0.0, depth, witnesses