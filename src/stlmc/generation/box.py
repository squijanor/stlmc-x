"""Region box discovery (kappa_box): multi-box frontier growth, no certificate.

Around a falsifying pivot on a fixed mode path, grow an axis-aligned box of
initial conditions and collect a falsifying witness per expansion. A face grows
by a minimum separation theta at a time: it expands only while a falsifying
initial condition exists at least theta beyond the current bound, and stops when
the falsifying region ends within theta (NEG-only). theta both guarantees
termination over a continuous region and spaces the witnesses.

Each witness is labeled by where it sits in the falsifying set. The interior
witnesses collected during growth are ``deep``. After a box's growth converges,
each face is examined once: if the falsifying set reaches the variable's declared
range edge, the face's marker is ``domain`` (the extent is the model domain, not
the falsifying frontier); otherwise the frontier is strictly interior and is
located by bisecting the theta-gap between the last falsifying bound and the
first non-falsifying point, giving a ``boundary`` marker at the frontier. On a
fixed mode path with linear dynamics the falsifying initial-condition set is a
convex polytope, so this bisection locates the exact frontier without a positive
(quantified) encoding.

This strategy requires the z3 backend. Growth and labeling pin the mode path and
constrain the initial condition to a box, and dreal returns ``unknown`` for such
mode-pinned initial-condition queries over nonlinear ODE dynamics, so no box
grows; a nonlinear (interval / free-mode) formulation is future work. dreal runs
use kappa_path, which poses no such queries.

Depth traversal. The strategy visits a set of target depths -- ``[gen] depths``
as a slash-separated list (e.g. ``"7/8"``), or every depth 1..N by default -- and
explores each independently. At a target depth it grows a box, labels it, blocks
its region (the box extended by theta on every face, so no falsifying sliver is
left between the theta-quantized box and the true frontier), and re-pivots
outside the blocked regions of that depth, up to the per-depth budget ``[gen]
k-ic``; absent, the depth is explored to exhaustion. The blocks are per depth:
each target depth discovers its own falsifying regions, so a region falsifying at
several depths contributes a counterexample (with a different mode path) at each,
making depth a first-class diversity axis rather than an incidental by-product of
re-pivoting. The returned pool is the union over target depths.

Thinning (``[gen] thin-ic``) drops a new deep witness that lies within thin-ic
(per axis) of a witness already in the pool -- across depths as well as within a
depth -- while boundary and domain markers are exempt and always kept, since they
mark the frontier. The default of 0 leaves thinning off.

Initial-condition variables are the step-0 state copies ``<name>_0_0`` for each
state variable named by ``range_dict``; mode variables are ``currentMode_k``.
Box bounds are exact rationals so the degenerate pivot box is represented
exactly.

The pool is returned to the driver for serialization; the per-counterexample
labels are exposed on the ``ce_labels`` attribute, aligned to the returned pool,
for the driver to append to the payload.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from ..constraints.constraints import (
    And,
    BoolVal,
    Constant,
    Eq,
    Formula,
    Geq,
    Gt,
    Leq,
    Lt,
    Or,
    RealVal,
    Variable,
)
from ..objects.algorithm import Algorithm
from .encode import Encoder
from .oracle import SAT, GrowthOracle, Z3IncrementalOracle
from .pathenum import _gen_depths, _gen_int, _resolve_seed, _z3_logic

_MODE_RE = re.compile(r"^currentMode_(\d+)$")

# Bound the coordinate-descent passes and the steps per face; both are safety
# nets -- growth terminates naturally once faces reach the region boundary.
_MAX_PASSES = 16
_MAX_STEPS = 10000

# Frontier bisection steps when [gen] bisect-iters is not set; the located
# frontier is within theta * 2**-_BISECT_ITERS of the true crossing.
_BISECT_ITERS = 20

# Per-counterexample labels.
_DEEP = "deep"
_BOUNDARY = "boundary"
_DOMAIN = "domain"


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


def _ic_ranges(
    box: Dict[Variable, List[Fraction]], range_dict
) -> Dict[Variable, Tuple[Fraction, Fraction]]:
    """Declared (lo, hi) for each IC variable in ``box``.

    ``range_dict`` maps a state Variable to ``(lo_incl, lo, hi, hi_incl)``; the
    IC variable ``<name>_0_0`` inherits the ``(lo, hi)`` of its state variable
    ``<name>``.
    """
    by_id: Dict[str, Tuple[Fraction, Fraction]] = {}
    for state_var, bounds in range_dict.items():
        by_id["{}_0_0".format(state_var.id)] = (_frac(bounds[1]), _frac(bounds[2]))
    return {var: by_id[var.id] for var in box if var.id in by_id}


def _point_witness(
    oracle: GrowthOracle, var: Variable, others: Formula, value: Fraction
) -> Optional[Dict[Variable, Constant]]:
    """A falsifying assignment with ``var == value`` inside ``others``, or None
    if none exists. Leaves the oracle stack unchanged."""
    oracle.push()
    try:
        oracle.assert_(And([others, Eq(var, _rv(value))]))
        return oracle.model() if oracle.check() == SAT else None
    finally:
        oracle.pop()


def _bisect_frontier(
    oracle: GrowthOracle,
    var: Variable,
    others: Formula,
    sat: Fraction,
    unsat: Fraction,
    iters: int,
) -> Fraction:
    """Locate the falsifying frontier between ``sat`` (falsifying) and ``unsat``
    (non-falsifying) by bisection; return the falsifying-side bound after
    ``iters`` steps. Direction-agnostic: ``sat`` and ``unsat`` may be in either
    order."""
    for _ in range(iters):
        mid = (sat + unsat) / 2
        if oracle.check_with(And([others, Eq(var, _rv(mid))])) == SAT:
            sat = mid
        else:
            unsat = mid
    return sat


def _block_box(bounds: Dict[Variable, Tuple[Fraction, Fraction]]) -> Formula:
    """Negation of a box: ``OR_i (x_i < lo_i OR x_i > hi_i)``.

    Asserted for later pivot searches so no subsequent pivot falls inside this
    box's region."""
    terms: List[Formula] = []
    for var, (lo, hi) in bounds.items():
        terms.append(Lt(var, _rv(lo)))
        terms.append(Gt(var, _rv(hi)))
    return Or(terms) if terms else BoolVal("False")


def _too_close(
    witness: Dict[Variable, Constant],
    pool: List[Dict[Variable, Constant]],
    ic_vars,
    thin: Fraction,
) -> bool:
    """True if ``witness`` lies within ``thin`` of some pooled witness on every
    initial-condition axis (an L-infinity ball of radius ``thin``)."""
    for other in pool:
        if all(abs(_value_of(witness, v) - _value_of(other, v)) < thin for v in ic_vars):
            return True
    return False


class RegionBoxDiscovery(Algorithm):
    """kappa_box: grow labeled witness boxes around falsifying pivots, blocking
    each discovered region and re-pivoting under a per-depth box budget over a
    set of target depths."""

    def __init__(self) -> None:
        self.debug_name = ""
        # Per-counterexample labels aligned to the returned pool; read by the
        # driver to append the optional labels element to the payload.
        self.ce_labels: Optional[List[str]] = None

    def set_debug(self, msg: str) -> None:
        self.debug_name = msg

    def _pivot_at(
        self,
        encoder: Encoder,
        depth: int,
        logic: str,
        seed: int,
        blocks: List[Formula],
    ):
        """Encode at ``depth`` and return (oracle, pivot model, encoding) for a
        falsifying assignment outside every blocked region, or (None, None, None).
        The returned oracle carries the encoding and the blocks, ready for growth;
        the caller resets the encoder after using the returned encoding."""
        encoding = encoder.encode_at(depth)
        oracle = Z3IncrementalOracle(logic, seed)
        oracle.assert_(encoding.consts)
        for block in blocks:
            oracle.assert_(block)
        if oracle.check() == SAT:
            return oracle, oracle.model(), encoding
        return None, None, None

    def _label_faces(
        self,
        oracle: GrowthOracle,
        box: Dict[Variable, List[Fraction]],
        ranges: Dict[Variable, Tuple[Fraction, Fraction]],
        theta: Fraction,
        iters: int,
        printer,
    ) -> Tuple[List[Dict[Variable, Constant]], List[str]]:
        """One marker witness per face of the converged box.

        For each ``(var, direction)`` face at its final bound ``b``: if the
        falsifying set reaches the declared range edge, emit a ``domain`` marker
        there; otherwise bisect the theta-gap between ``b`` and the first
        non-falsifying point and emit a ``boundary`` marker at the frontier.
        """
        markers: List[Dict[Variable, Constant]] = []
        labels: List[str] = []
        for var in box:
            lo_dom, hi_dom = ranges.get(var, (None, None))
            others = _box_of(box, skip=var)
            for direction in (+1, -1):
                b = box[var][1] if direction > 0 else box[var][0]
                wall = hi_dom if direction > 0 else lo_dom

                # The falsifying set reaches the declared edge: the observable
                # extent is the domain wall, not a falsifying frontier.
                if wall is not None:
                    at_wall = _point_witness(oracle, var, others, wall)
                    if at_wall is not None:
                        markers.append(at_wall)
                        labels.append(_DOMAIN)
                        printer.print_verbose(
                            "[kappa_box] face {}{}: domain".format(
                                var.id, "+" if direction > 0 else "-"
                            )
                        )
                        continue

                # Frontier is strictly interior. The first non-falsifying point
                # is b + direction*theta (growth stopped there), clamped to the
                # wall when the wall is nearer.
                unsat = b + (theta if direction > 0 else -theta)
                if wall is not None:
                    unsat = min(unsat, wall) if direction > 0 else max(unsat, wall)
                if oracle.check_with(And([others, Eq(var, _rv(unsat))])) == SAT:
                    # No non-falsifying point within reach; nothing to locate.
                    continue
                frontier = _bisect_frontier(oracle, var, others, b, unsat, iters)
                at_frontier = _point_witness(oracle, var, others, frontier)
                if at_frontier is not None:
                    markers.append(at_frontier)
                    labels.append(_BOUNDARY)
                    printer.print_verbose(
                        "[kappa_box] face {}{}: boundary at {}".format(
                            var.id, "+" if direction > 0 else "-", frontier
                        )
                    )
        return markers, labels

    def _grow_box(
        self,
        oracle: GrowthOracle,
        pivot: Dict[Variable, Constant],
        encoding,
        theta: Fraction,
        iters: int,
        depth: int,
        printer,
    ) -> Tuple[List[Dict[Variable, Constant]], List[str], Dict[Variable, List[Fraction]]]:
        """Grow and label one box around ``pivot`` on ``oracle`` (which already
        carries the encoding, the blocks, and the pinned mode path). Returns the
        box's witnesses, their labels, and the final box bounds."""
        ic = _ic_pivots(pivot, encoding.range_dict)
        if not ic:
            raise RuntimeError(
                "no initial-condition variables (<name>_0_0) found; cannot grow an IC box"
            )

        box: Dict[Variable, List[Fraction]] = {v: [p, p] for v, p in ic.items()}
        witnesses: List[Dict[Variable, Constant]] = [pivot]
        labels: List[str] = [_DEEP]

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
                        labels.append(_DEEP)
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

        ranges = _ic_ranges(box, encoding.range_dict)
        markers, marker_labels = self._label_faces(
            oracle, box, ranges, theta, iters, printer
        )
        witnesses.extend(markers)
        labels.extend(marker_labels)
        return witnesses, labels, box

    def run(self, model, goal, prop_dict, config, solver, logger, printer):
        common = config.get_section("common")
        max_depth = int(common.get_value("bound"))
        tau_max = float(common.get_value("time-bound"))
        delta = float(common.get_value("threshold"))
        underlying = common.get_value("solver")
        if underlying != "z3":
            raise NotImplementedError(
                "kappa_box requires the z3 backend: growth and labeling pin the mode "
                "path and box the initial condition, and dreal returns 'unknown' for "
                "such queries over nonlinear ODE dynamics. Use kappa_path on dreal, or "
                "z3 for kappa_box."
            )

        logic = _z3_logic(config)
        seed = _resolve_seed(config)
        theta = _gen_frac(config, "epsilon", "0.01")  # min IC separation / granularity
        bisect_iters = _gen_int(config, "bisect-iters") or _BISECT_ITERS
        per_depth_boxes = _gen_int(config, "k-ic")  # per target depth; None -> exhaust
        thin = _gen_frac(config, "thin-ic", "0")  # 0 -> no thinning
        target_depths = _gen_depths(config, max_depth)

        hash_seed = os.environ.get("PYTHONHASHSEED")
        if hash_seed is None or not hash_seed.isdigit():
            printer.print_normal(
                "warning: PYTHONHASHSEED is not fixed; constraint ordering is not pinned, "
                "so results may vary run to run. Set PYTHONHASHSEED for reproducibility."
            )

        encoder = Encoder(model, goal, prop_dict, delta, tau_max)
        pool: List[Dict[Variable, Constant]] = []
        labels: List[str] = []
        first_depth = max_depth
        total_boxes = 0

        for depth in target_depths:
            blocks: List[Formula] = []  # per depth: independent region discovery
            boxes_here = 0
            while per_depth_boxes is None or boxes_here < per_depth_boxes:
                oracle, pivot, encoding = self._pivot_at(
                    encoder, depth, logic, seed, blocks
                )
                if pivot is None:
                    encoder.reset()
                    break

                oracle.assert_(_mode_fix(pivot))  # pin this box's path
                witnesses, box_labels, box = self._grow_box(
                    oracle, pivot, encoding, theta, bisect_iters, depth, printer
                )
                boxes_here += 1
                total_boxes += 1
                if total_boxes == 1:
                    first_depth = depth

                kept = 0
                for witness, label in zip(witnesses, box_labels):
                    # Thin only deep (interior) witnesses; boundary and domain
                    # markers mark the frontier and are always kept.
                    if thin > 0 and label == _DEEP and _too_close(
                        witness, pool, box.keys(), thin
                    ):
                        continue
                    pool.append(witness)
                    labels.append(label)
                    kept += 1

                # Block the box extended by theta on every face. At convergence
                # the point theta beyond each face is non-falsifying, so extending
                # the block to it leaves no falsifying sliver between the
                # theta-quantized box and the true frontier for a re-pivot.
                block_bounds = {v: (lo - theta, hi + theta) for v, (lo, hi) in box.items()}
                blocks.append(_block_box(block_bounds))
                encoder.reset()  # clean slate before the next pivot search
                printer.print_verbose(
                    "[kappa_box] depth {}: box {} here, kept {}/{}; pool {}".format(
                        depth, boxes_here, kept, len(witnesses), len(pool)
                    )
                )

        self.ce_labels = labels
        printer.print_verbose(
            "[kappa_box] {} box(es) over {} target depth(s), {} witnesses: "
            "{} deep, {} boundary, {} domain".format(
                total_boxes,
                len(target_depths),
                len(pool),
                labels.count(_DEEP),
                labels.count(_BOUNDARY),
                labels.count(_DOMAIN),
            )
        )

        result = "False" if pool else "True"
        return result, 0.0, first_depth, pool