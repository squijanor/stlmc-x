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

Both backends are supported. On an exact backend a face advances by theta-wide
steps and the frontier is located by bisection. On a delta-decision backend the
face is located by a logarithmic search to tolerance ``max(delta, theta/8)`` and
witnesses are harvested on a lattice over the converged box, so the number of
solver calls is set by an explicit budget rather than by the box extent.

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

import time as _time
from collections import Counter as _Counter
from fractions import Fraction

from ..constraints.constraints import (
    And,
    Bool,
    BoolVal,
    Constant,
    Eq,
    Forall,
    Formula,
    Geq,
    Gt,
    Implies,
    Leq,
    Lt,
    Neq,
    Not,
    Or,
    Real,
    RealVal,
    Sub,
    Variable,
)
from ..constraints.operations import substitution_zero2t
from ..objects.algorithm import Algorithm
from .common import (
    MODE_RE,
    gen_depths,
    gen_float,
    gen_frac,
    gen_int,
    gen_str,
    resolve_seed,
    scoped_verdict,
    warn_unpinned_hashseed,
    z3_logic,
)
from .encode import Encoder
from .oracle import SAT, UNKNOWN, UNSAT, Z3IncrementalOracle, make_oracle

# Face-search precision on an exact oracle when [gen] bisect-iters is not set:
# the located frontier is within theta * 2**-_BISECT_ITERS of the true crossing.
_BISECT_ITERS = 20

# Default for [gen] pivot-timeout, which bounds ONE solver call inside the pivot
# search. One call decides one candidate out of many and the search as a whole is
# bounded by [gen] pivot-budget, so a per-call value near that budget would let a
# single candidate consume all of it.
_DEFAULT_CANDIDATE_TIMEOUT = 45.0

# Ceiling on the number of lattice cells harvested from one box. The lattice
# grows as the product over IC axes -- four variables at eight cells per axis is
# 4096 solver calls -- so the product is capped; the per-axis cell counts are
# then reduced proportionally and the reduction is reported.
_LATTICE_MAX = 200

# Per-counterexample labels.
_DEEP = "deep"
_BOUNDARY = "boundary"
_DOMAIN = "domain"


def _frac(value) -> Fraction:
    """Constant value string (decimal or p/q) -> exact Fraction."""
    return Fraction(str(value))


_GRID = Fraction(10) ** 12


def _grid(f: Fraction) -> Fraction:
    """Snap to a 1e-12 decimal grid.

    Harvest points are lo + span*(2i+1)/(2n); for n not of the form 2^a*5^b that
    denominator has other prime factors and has no finite decimal expansion,
    which dReal's parser cannot express. 1e-12 is far below any tolerance in
    play, so this does not move the point meaningfully."""
    return Fraction(round(f * _GRID), _GRID)


def _rv(f: Fraction) -> RealVal:
    return RealVal(str(f))


def _value_of(assn: dict[Variable, Constant], var: Variable) -> Fraction:
    """Value of ``var`` (matched by id) in an assignment, as a Fraction."""
    for v, c in assn.items():
        if v.id == var.id:
            return _frac(c.value)
    raise KeyError(var.id)


_FA_PREFIX = "__fa_"


def _abstract_foralls(formula, memo=None, counter=None):
    """Replace every ``Forall`` subformula with a fresh Bool, structurally.

    z3 cannot translate ``forall_t`` nodes, so the skeleton cannot be handed to
    it directly. Abstracting them preserves the Boolean structure exactly -- an
    Or stays an Or -- which a clause-level filter does not. Identical Forall
    nodes share one Bool, so z3 cannot pick contradictory truth values for the
    same condition.

    Returns ``(rewritten_formula, {fresh_Bool: original_Forall})``."""
    if memo is None:
        memo, counter = {}, [0]

    def go(f):
        if isinstance(f, Forall):
            # Forall.__hash__ is hash("(forall <mode> . <const>)") and omits the
            # segment bounds, so two forall_t over DIFFERENT segments could share
            # one Bool and be forced to the same truth value. Key on the full
            # segment identity instead.
            key = (str(f.current_mode_number), str(f.start_tau),
                   str(f.end_tau), str(f.const))
            if key not in memo:
                b = Bool(f"{_FA_PREFIX}{counter[0]}")
                counter[0] += 1
                memo[key] = (b, f)
            return memo[key][0]
        if isinstance(f, And):
            return And([go(c) for c in f.children])
        if isinstance(f, Or):
            return Or([go(c) for c in f.children])
        if isinstance(f, Not):
            return Not(go(f.child))
        if isinstance(f, Implies):
            return Implies(go(f.left), go(f.right))
        # forall_t also hides inside the abstraction definitions, which are
        # Eq(<abstraction Bool>, <formula containing forall_t>). Descend into
        # Eq/Neq operands too; non-Boolean operands come back unchanged.
        if isinstance(f, Eq):
            return Eq(go(f.left), go(f.right))
        if isinstance(f, Neq):
            return Neq(go(f.left), go(f.right))
        return f

    out = go(formula)
    return out, {b: orig for (b, orig) in memo.values()}


def _mode_fix(assn: dict[Variable, Constant]) -> Formula:
    """AND_k (currentMode_k == value) pinning the pivot's path."""
    terms = [Eq(v, c) for v, c in assn.items() if MODE_RE.match(v.id)]
    return And(terms) if terms else BoolVal("True")


def _skeleton_fix(assn: dict[Variable, Constant]) -> Formula:
    """Pin the pivot's whole propositional skeleton, not just its mode word.

    Psi_n is a large disjunctive structure over Boolean-abstraction variables
    (chi^..., T1^..., invAtomicID_..., newIntegral_...). Handing that to dReal
    unresolved forces ICP to branch over the Boolean structure *while* doing
    interval ODE integration, which is why constrained-IC probes stop
    terminating as depth grows. STLMC's own pipeline never does this: two-step
    resolves the abstraction first and asks dReal only for ODE feasibility on a
    concrete skeleton.

    kappa_box already fixes the mode path, so fixing the rest of the skeleton it
    came from is consistent with what the box means: the IC region that
    falsifies *via this execution structure*."""
    terms: list[Formula] = []
    for v, c in assn.items():
        if MODE_RE.match(v.id):
            terms.append(Eq(v, c))
        elif isinstance(v, Bool):
            terms.append(v if str(c.value) == "True" else Not(v))
    return And(terms) if terms else BoolVal("True")


def _ic_pivots(assn: dict[Variable, Constant], range_dict) -> dict[Variable, Fraction]:
    """Initial-condition variables (<name>_0_0) and their pivot values."""
    ic_ids = {f"{k.id}_0_0" for k in range_dict}
    return {v: _frac(c.value) for v, c in assn.items() if v.id in ic_ids}


def _box_of(box: dict[Variable, list[Fraction]], skip: Variable, rv=_rv) -> Formula:
    """AND over dimensions (except ``skip``) of lo <= x <= hi."""
    terms: list[Formula] = []
    for var, (lo, hi) in box.items():
        if var is skip:
            continue
        terms.append(Geq(var, rv(lo)))
        terms.append(Leq(var, rv(hi)))
    return And(terms) if terms else BoolVal("True")


def _ic_ranges(
    box: dict[Variable, list[Fraction]], range_dict
) -> dict[Variable, tuple[Fraction, Fraction]]:
    """Declared (lo, hi) for each IC variable in ``box``.

    ``range_dict`` maps a state Variable to ``(lo_incl, lo, hi, hi_incl)``; the
    IC variable ``<name>_0_0`` inherits the ``(lo, hi)`` of its state variable
    ``<name>``.
    """
    by_id: dict[str, tuple[Fraction, Fraction]] = {}
    for state_var, bounds in range_dict.items():
        by_id[f"{state_var.id}_0_0"] = (_frac(bounds[1]), _frac(bounds[2]))
    return {var: by_id[var.id] for var in box if var.id in by_id}


def _block_box(bounds: dict[Variable, tuple[Fraction, Fraction]], rv=_rv) -> Formula:
    """Negation of a box: ``OR_i (x_i < lo_i OR x_i > hi_i)``.

    Asserted for later pivot searches so no subsequent pivot falls inside this
    box's region."""
    terms: list[Formula] = []
    for var, (lo, hi) in bounds.items():
        terms.append(Lt(var, rv(lo)))
        terms.append(Gt(var, rv(hi)))
    return Or(terms) if terms else BoolVal("False")


def _too_close(
    witness: dict[Variable, Constant],
    pool: list[dict[Variable, Constant]],
    ic_vars,
    thin: Fraction,
) -> bool:
    """True if ``witness`` lies within ``thin`` of some pooled witness on every
    initial-condition axis (an L-infinity ball of radius ``thin``)."""
    for other in pool:
        if all(abs(_value_of(witness, v) - _value_of(other, v)) < thin
               for v in ic_vars):
            return True
    return False


class _Theta:
    """The minimum initial-condition separation, per axis.

    theta is the strategy's granularity: how far apart witnesses must be, how
    far a face must reach to count as growth, how wide a blocked region is
    extended.

    A single absolute theta fixes that granularity in model units, so it is a
    different fraction of each axis whose declared range differs.
    ``[gen] epsilon-relative`` instead sets theta as a fraction of each declared
    range, making the granularity uniform relative to the space each variable
    ranges over. ``[gen] epsilon`` remains the absolute default and the fallback
    for an axis whose range is undeclared or degenerate.
    """

    def __init__(self, absolute: Fraction, relative: Fraction | None = None,
                 range_dict=None) -> None:
        self.absolute = absolute
        self.relative = relative
        self._by_id: dict[str, Fraction] = {}
        if relative is not None and range_dict is not None:
            for state_var, bounds in range_dict.items():
                width = _frac(bounds[2]) - _frac(bounds[1])
                if width > 0:
                    self._by_id[f"{state_var.id}_0_0"] = relative * width

    def of(self, var) -> Fraction:
        """theta for one IC variable."""
        return self._by_id.get(getattr(var, "id", var), self.absolute)

    def describe(self) -> str:
        if not self._by_id:
            return f"theta={float(self.absolute)} (absolute, every axis)"
        per_axis = ", ".join(
            f"{vid}={float(value)}"
            for vid, value in sorted(self._by_id.items()))
        return (f"theta={float(self.relative)} x declared range -> {per_axis}"
                f" (fallback {float(self.absolute)})")


def _verdict(pool, any_unresolved, visited, max_depth):
    """kappa_box's wording for the shared depth-scoping rule."""
    return scoped_verdict(pool, any_unresolved, visited, max_depth,
                          tag="kappa_box", nothing_found="no box found",
                          unresolved_source="at least one pivot search")


def _merge_markers(witnesses, labels, markers, marker_labels, ic_vars, tol,
                   printer) -> int:
    """Append face markers, merging any that land on a witness already collected.

    A marker coincides with an existing witness whenever the pivot itself sits
    on a frontier: the face search confirms a bound the pivot already occupies.
    Appending it would put two entries in the pool for one initial condition.

    The merge keeps the more informative label: the two are the same point to
    within ``tol``, so the existing witness is relabeled in place and the
    frontier annotation is preserved. ``tol`` is below the granularity theta
    declares, so only points the search cannot tell apart are merged.

    ``tol`` is a callable ``var -> Fraction``, since the tolerance follows theta
    and theta may differ per axis.

    Mutates ``witnesses`` and ``labels``; returns how many were merged."""
    merged = 0
    for marker, marker_label in zip(markers, marker_labels):
        hit = None
        for index, witness in enumerate(witnesses):
            if all(abs(_value_of(marker, v) - _value_of(witness, v)) <= tol(v)
                   for v in ic_vars):
                hit = index
                break
        if hit is None:
            witnesses.append(marker)
            labels.append(marker_label)
        else:
            merged += 1
            if labels[hit] == _DEEP:
                labels[hit] = marker_label
    if merged and printer is not None:
        printer.print_verbose(
            "[kappa_box] {} marker(s) coincided with an existing witness "
            "(within {}) and were merged into it".format(
                merged, ", ".join(f"{v.id}={float(tol(v))}"
                                  for v in ic_vars)))
    return merged


class RegionBoxDiscovery(Algorithm):
    """kappa_box: grow labeled witness boxes around falsifying pivots, blocking
    each discovered region and re-pivoting under a per-depth box budget over a
    set of target depths."""

    def __init__(self) -> None:
        self.debug_name = ""
        # Per-counterexample labels aligned to the returned pool; read by the
        # driver to append the optional labels element to the payload.
        self.ce_labels: list[str] | None = None

    def set_debug(self, msg: str) -> None:
        self.debug_name = msg

    def _pivot_two_step(self, encoding, logic, seed, blocks):
        """Find a pivot in two steps, as STLMC's own checking does.

        A monolithic pivot query hands dReal the whole boolean abstraction and
        every ODE at once, so interval propagation branches over boolean
        structure while integrating, and the query is undecided at all but the
        shallowest depths. Instead:

          1. z3 solves ``encoding.skeleton`` -- the propositional/arithmetic
             structure with every ODE and forall_t still behind its abstraction
             Bool. Nonlinear terms never reach z3, so this is fast.
          2. dReal is asked only whether that ONE concrete skeleton is
             ODE-feasible, with the whole boolean assignment pinned. That is a
             pure feasibility question and is the shape dReal is good at.
          3. On unsat, the skeleton is blocked in z3 and step 1 repeats.
        """
        printer = self._printer
        from ..tree.operations import size_of_tree as _sz
        _t = _time.monotonic()
        abstracted, fa_map = _abstract_foralls(encoding.skeleton)
        self._printer.print_verbose(
            f"[diag] skeleton size={_sz(encoding.skeleton)} "
            f"consts size={_sz(encoding.consts)} "
            f"abstract_time={_time.monotonic() - _t:.1f}s")
        z3o = Z3IncrementalOracle(logic, seed, general=True)
        z3o.assert_(abstracted)
        for block in blocks:
            z3o.assert_(block)

        # Endpoint implications.
        # forall_t phi over [tau_i, tau_i+1] implies phi at ANY time in the
        # interval, in particular at both endpoints. phi is written over the
        # segment's START copy; substitution_zero2t gives the END copy. This is
        # a necessary condition, so it can only remove candidates that dReal
        # would have rejected anyway.
        #
        # Timing facts z3 cannot see.
        # Two constraints exist ONLY on the dReal side and are invisible to the
        # z3 step, so z3 happily proposes skeletons dReal must reject:
        #   (a) segment durations are declared [0, time-horizon] in the SMT2
        #       declarations, not asserted anywhere in `consts`;
        #   (b) time_k = tau_{k+1} - tau_k holds only via the clock ODE
        #       (d/dt[g@clock] = 1), which is hidden behind an abstraction Bool.
        # Both must therefore be asserted here. With time-horizon equal to
        # time-bound (a) is nearly vacuous; under a tighter horizon it rejects
        # every candidate that ignores it.
        horizon = getattr(self, "_time_horizon", None)
        n_seg = encoding.bound + 1
        timing = []
        for k in range(n_seg):
            t_k = Real(f"time_{k}")
            timing.append(Geq(t_k, RealVal("0")))
            if horizon is not None:
                timing.append(Leq(t_k, RealVal(repr(horizon))))
            timing.append(Eq(t_k, Sub(Real(f"tau_{k + 1}"),
                                      Real(f"tau_{k}"))))
        for t in timing:
            z3o.assert_(t)

        for b, fa in fa_map.items():
            endpoints = And([fa.const, substitution_zero2t(fa.const)])
            z3o.assert_(Implies(b, endpoints))
        self._printer.print_verbose(
            f"[kappa_box/two-step] z3 skeleton: {len(fa_map)} forall_t "
            f"abstracted, {len(timing)} timing facts, endpoint implications on")

        # Word rotation. z3 exhausts the forall assignments of one mode word
        # before trying another, so a rejected word can absorb the whole budget.
        # After N consecutive rejections under the same word the word is blocked
        # outright, at no solver cost. A heuristic: it can block a word that
        # would have been feasible under a later assignment, trading
        # completeness for coverage, which is why an exhausted search is not
        # reported as absence once it has fired. 0 disables.
        rotate = gen_int(self._config, "word-rotate")
        rotate = 30 if rotate is None else rotate
        streak_word, streak = None, 0
        rotated = 0    # heuristic drops; an UNSAT verdict is then inconclusive

        every = gen_int(self._config, "log-every") or 25
        budget = gen_float(self._config, "pivot-budget") or 120.0
        deadline = _time.monotonic() + budget
        attempt = 0
        while _time.monotonic() < deadline:
            attempt += 1
            self._metrics["candidates"] += 1
            _tz = _time.monotonic()
            _v0 = z3o.check()
            if attempt == 1:
                self._printer.print_verbose(
                    f"[diag] first z3 check: {_v0} in {_time.monotonic() - _tz:.1f}s")
            if _v0 != SAT:
                self._last_pivot_verdict = UNSAT
                return None, None, None
            candidate = z3o.model()
            word = "".join(
                str(int(round(float(c.value))))
                for v, c in sorted(candidate.items(), key=lambda kv: kv[0].id)
                if MODE_RE.match(v.id))
            mode_only = And([Eq(v, c) for v, c in candidate.items()
                             if MODE_RE.match(v.id)]) if word else BoolVal("True")

            # Pin only what exists in the real encoding; the fresh forall_t Bools
            # do not, so their z3 truth values are replayed as the ORIGINAL
            # forall_t formula (or its negation) for dReal to check.
            real = {v: c for v, c in candidate.items()
                    if not v.id.startswith(_FA_PREFIX)}
            fa_terms = []
            for b, orig in fa_map.items():
                for v, c in candidate.items():
                    if v.id == b.id:
                        fa_terms.append(orig if str(c.value) == "True" else Not(orig))
                        break
            pinned = And([_skeleton_fix(real)] + fa_terms) if fa_terms \
                else _skeleton_fix(real)
            # Block exactly what was tested: the candidate is checked with the
            # whole propositional assignment pinned, so a rejection refutes that
            # assignment and nothing more. Blocking the (mode word, forall) class
            # instead would exclude assignments never tried, and the number
            # discarded per rejection grows with depth.
            #
            # This is the z3-expressible counterpart of `pinned`, including the
            # forall_t abstraction Bools whose truth values selected the formulas
            # pinned for dReal; `pinned` itself carries real forall_t nodes and
            # cannot go back into z3.
            block_this = _skeleton_fix(candidate)

            d = make_oracle(self._underlying, logic=logic, seed=seed,
                            config=self._config, logger=self._logger,
                            time_bound=self._tau_max)
            d.set_budget(gen_float(self._config, "pivot-timeout")
                         or _DEFAULT_CANDIDATE_TIMEOUT)
            d.assert_(encoding.consts)
            d.assert_(pinned)
            _t0 = _time.monotonic()
            v = d.check()
            _el = _time.monotonic() - _t0
            d.set_budget("unset")
            if v == SAT or attempt % every == 0 or attempt == 1:
                printer.print_verbose(
                    "[kappa_box/two-step] candidate {} (word {}): dreal says {} "
                    "({:.1f}s)".format(attempt, word or "-", v, _el))
            if v == SAT:
                self._metrics["accepted"] += 1
                model = d.model()
                d.assert_(pinned)
                return d, model, encoding
            # unsat or unknown: this skeleton is not usable, try another
            z3o.assert_(Not(block_this))
            if rotate and word:
                streak = streak + 1 if word == streak_word else 1
                streak_word = word
                if streak >= rotate:
                    z3o.assert_(Not(mode_only))
                    rotated += 1
                    self._rotated_words += 1
                    printer.print_verbose(
                        f"[kappa_box/two-step] rotating off word {word} after {streak} "
                        "consecutive rejections")
                    streak_word, streak = None, 0

        printer.print_normal(
            f"[kappa_box/two-step] gave up after {attempt} candidates "
            f"({rotated} mode word(s) rotated off)")
        self._last_pivot_verdict = UNKNOWN
        return None, None, None

    def _pivot_at(
        self,
        encoder: Encoder,
        depth: int,
        logic: str,
        seed: int,
        blocks: list[Formula],
    ):
        """Encode at ``depth`` and return (oracle, pivot model, encoding) for a
        falsifying assignment outside every blocked region, or (None, None, None).
        The returned oracle carries the encoding and the blocks, ready for growth;
        the caller resets the encoder after using the returned encoding."""
        encoding = encoder.encode_at(depth)
        if getattr(self, "_underlying", "z3") != "z3":
            return self._pivot_two_step(encoding, logic, seed, blocks)
        # Only the exact backend reaches here; a delta backend always takes the
        # two-step path above.
        oracle = Z3IncrementalOracle(logic, seed)
        oracle.assert_(encoding.consts)
        for block in blocks:
            oracle.assert_(block)
        v = oracle.check()
        if v == SAT:
            return oracle, oracle.model(), encoding
        # UNSAT means the region is exhausted; UNKNOWN means the solver gave up.
        # Collapsing them reports "no more boxes" for a resource failure.
        self._last_pivot_verdict = v
        return None, None, None

    def _window_witness(self, oracle, var, others, centre, half):
        """A falsifying model with ``var`` within ``half`` of ``centre``.

        Uses a window rather than an equality: under delta semantics an exact
        point need not be falsifying even when its neighbourhood is."""
        oracle.push()
        try:
            oracle.assert_(And([others,
                                Geq(var, oracle.rv(centre - half)),
                                Leq(var, oracle.rv(centre + half))]))
            return oracle.model() if oracle.check() == SAT else None
        finally:
            oracle.pop()

    def _search_face(self, oracle, var, others, start, wall, tol):
        """Largest falsifying value of ``var`` in ``[start, wall]`` (or smallest,
        when ``wall < start``), to within ``tol``.

        Logarithmic in the extent instead of linear in extent/theta, and it never
        reads UNKNOWN as a frontier. Returns ``(bound, status, calls)`` with
        status in {'domain', 'frontier', 'partial', 'unresolved'}.

        An UNKNOWN answer is local: failing to decide whether a falsifying IC
        exists beyond one point says nothing about the points on either side, so
        the search retries nearer the confirmed side rather than abandoning the
        face. ``lo`` is always a value the solver confirmed falsifying, so a
        face that stops early yields a smaller box, never a wrong one:

            domain      the falsifying set reaches the declared range edge.
            frontier    bracketed between a confirmed falsifying value and a
                        confirmed non-falsifying one, to within tol.
            partial     grew beyond the pivot but never bracketed. The bound is
                        a LOWER BOUND on the extent, not the frontier.
            unresolved  no probe beyond the pivot was decided. No growth.
        """
        up = wall > start
        calls = 0
        # Probing nearer the confirmed side after an UNKNOWN. Two fractions is a
        # deliberate cap: this runs per bisection step, and a face that needs
        # many detours is telling you the query is too hard at this precision.
        detours = (Fraction(1, 4), Fraction(1, 8))

        def falsifying_beyond(m):
            """Is there a falsifying IC at or beyond ``m`` (towards the wall)?

            Counts its own solver call, so no probe can be added without being
            accounted for."""
            nonlocal calls
            calls += 1
            side = Geq(var, oracle.rv(m)) if up else Leq(var, oracle.rv(m))
            return oracle.check_with(And([others, side]))

        v = falsifying_beyond(wall)
        if v == SAT:
            return wall, "domain", calls
        # UNSAT: `wall` brackets the frontier from above. UNKNOWN: it does not,
        # but it still bounds the search interval, so the search continues with
        # `wall` as a limit it cannot rely on.
        bracketed = v == UNSAT

        lo, hi = start, wall              # lo falsifying; hi not, if bracketed
        while abs(hi - lo) > tol:
            mid = (lo + hi) / 2
            v = falsifying_beyond(mid)
            if v == UNKNOWN:
                for share in detours:
                    mid = lo + (hi - lo) * share
                    v = falsifying_beyond(mid)
                    if v != UNKNOWN:
                        break
            if v == SAT:
                lo = mid
            elif v == UNSAT:
                hi = mid
                bracketed = True
            else:
                break                     # every detour undecided: stop here
        if bracketed and abs(hi - lo) <= tol:
            return lo, "frontier", calls
        return lo, ("partial" if lo != start else "unresolved"), calls

    def _harvest(self, oracle, var, others, lo, hi, theta, budget, half):
        """Up to ``budget`` deep witnesses spread over ``[lo, hi]``.

        Decoupled from growth: on an exact backend witness count is forced to
        extent/theta, which on a delta backend costs one 27 s solve each.

        Returns ``(witnesses, requested, undecided)``. The undecided count is
        reported because a query the backend does not decide within its budget
        produces nothing, exactly like a grid point with no falsifying model:
        without the count a short pool cannot be attributed to the region or to
        the solver."""
        out = []
        span = hi - lo
        if span <= 0 or budget <= 0:
            return out, 0, 0
        n = min(budget, max(1, int(span / theta)))
        undecided = 0
        for i in range(n):
            p = _grid(lo + span * Fraction(2 * i + 1, 2 * n))
            w = And([others, Geq(var, oracle.rv(p - half)),
                     Leq(var, oracle.rv(p + half))])
            oracle.push()
            try:
                oracle.assert_(w)
                verdict = oracle.check()
                if verdict == SAT:
                    out.append(oracle.model())
                elif verdict == UNKNOWN:
                    undecided += 1
            finally:
                oracle.pop()
        return out, n, undecided

    def _harvest_lattice(self, oracle, box, theta: _Theta, budget, cap):
        """Deep witnesses on a lattice over the whole box.

        The per-axis sweep (:meth:`_harvest`) constrains the swept axis to a
        grid cell and the others to the whole box, so the solver fixes them
        freely: spacing along those axes is not controlled by theta, and the
        witness count grows as the sum over axes.

        The lattice pins every axis to its own cell, so a returned point lies in
        a known cell, theta has the same meaning on every axis, and the witness
        count is the product of the per-axis cell counts. In one dimension the
        two are equivalent: same cell count, same midpoints.

        ``cap`` bounds the product, which grows exponentially in the number of IC
        variables. When the full lattice exceeds it, the per-axis cell counts are
        reduced proportionally and the reduction is reported.

        Returns ``(witnesses, requested, undecided)``."""
        axes = list(box)
        cells, half = {}, {}
        for var in axes:
            lo, hi = box[var]
            span = hi - lo
            th = theta.of(var)
            cells[var] = min(budget, max(1, int(span / th))) if span > 0 else 0
            half[var] = th / 2
        if any(n == 0 for n in cells.values()):
            return [], 0, 0

        total = 1
        for var in axes:
            total *= cells[var]
        if cap and total > cap:
            # Shrink every axis by the same factor, so the lattice keeps its
            # shape instead of collapsing whichever axis happens to be last.
            import math
            shrink = (cap / total) ** (1.0 / len(axes))
            for var in axes:
                cells[var] = max(1, int(math.floor(cells[var] * shrink)))
            total = 1
            for var in axes:
                total *= cells[var]

        def centres(var, index):
            lo, hi = box[var]
            return _grid(lo + (hi - lo) * Fraction(2 * index + 1, 2 * cells[var]))

        out, undecided, requested = [], 0, 0
        indices = [0] * len(axes)
        while True:
            terms = []
            for position, var in enumerate(axes):
                centre = centres(var, indices[position])
                terms.append(Geq(var, oracle.rv(centre - half[var])))
                terms.append(Leq(var, oracle.rv(centre + half[var])))
            oracle.push()
            try:
                oracle.assert_(And(terms))
                requested += 1
                verdict = oracle.check()
                if verdict == SAT:
                    out.append(oracle.model())
                elif verdict == UNKNOWN:
                    undecided += 1
            finally:
                oracle.pop()
            # odometer over the cell indices
            position = len(axes) - 1
            while position >= 0:
                indices[position] += 1
                if indices[position] < cells[axes[position]]:
                    break
                indices[position] = 0
                position -= 1
            if position < 0:
                break
        return out, requested, undecided

    def _grow_box(self, oracle, pivot, encoding, theta: _Theta, iters, depth,
                  budget, printer):
        """Grow, label and sample one box around ``pivot``.

        One procedure for both kinds of oracle. Per face, a domain probe and a
        logarithmic search for the bound; then a lattice of witnesses over the
        converged box. What the oracle changes is the precision: the search runs
        to ``theta / 2**iters`` on an exact oracle and to ``theta / 8`` on a
        partial one, in both cases floored by the oracle's own tolerance, since
        no query distinguishes points closer than that."""
        ic = _ic_pivots(pivot, encoding.range_dict)
        if not ic:
            raise RuntimeError("no initial-condition variables (<name>_0_0) found")
        box = {v: [p, p] for v, p in ic.items()}
        ranges = _ic_ranges(box, encoding.range_dict)
        exact = getattr(oracle, "is_exact", True)

        def tol_of(v):
            """Face precision for one axis, floored by the oracle's tolerance."""
            target = theta.of(v) / (2 ** iters) if exact else theta.of(v) / 8
            return max(oracle.tolerance, target)
        witnesses, labels = [pivot], [_DEEP]
        markers, marker_labels, calls, unresolved = [], [], 0, False

        for var in box:
            tol = tol_of(var)
            others = _box_of(box, var, oracle.rv)
            lo_dom, hi_dom = ranges.get(var, (None, None))
            for direction in (+1, -1):
                wall = hi_dom if direction > 0 else lo_dom
                if wall is None:
                    continue
                start = box[var][1] if direction > 0 else box[var][0]
                bound, status, c = self._search_face(
                    oracle, var, others, start, wall, tol)
                calls += c
                if direction > 0:
                    box[var][1] = bound
                else:
                    box[var][0] = bound
                printer.print_verbose(
                    "[kappa_box] face {}{}: {} at {} ({} calls)".format(
                        var.id, "+" if direction > 0 else "-", status,
                        float(bound), c))
                if status in ("unresolved", "partial"):
                    # Not a reason to discard the box: every witness inside it
                    # was confirmed by the solver, so what an undecided face
                    # costs is EXTENT, not validity. The box is kept, the extent
                    # is reported as a lower bound, and no frontier marker is
                    # emitted because no frontier was located.
                    unresolved = True
                    printer.print_normal(
                        "[kappa_box] face {}{}: {} -- frontier not bracketed "
                        "within the query budget; extent {} is a LOWER BOUND{}".format(
                            var.id, "+" if direction > 0 else "-", status.upper(),
                            float(bound),
                            ", no growth on this face" if status == "unresolved"
                            else ""))
                    if status == "partial":
                        # The deepest confirmed point is still a counterexample;
                        # it is labeled deep rather than boundary precisely
                        # because it is not known to be on the frontier.
                        m = self._window_witness(oracle, var, others, bound, tol)
                        calls += 1
                        if m is not None:
                            markers.append(m)
                            marker_labels.append(_DEEP)
                    continue
                m = self._window_witness(oracle, var, others, bound, tol)
                calls += 1
                if m is not None:
                    markers.append(m)
                    marker_labels.append(_DOMAIN if status == "domain" else _BOUNDARY)

        collapsed = 0
        harvest_undecided = 0
        # Lattice by default: it is the only mode under which theta means the
        # same thing on every axis (see _harvest_lattice). `harvest = sweep`
        # restores the per-axis sweep. In one dimension the two are identical,
        # so this changes nothing for single-variable models.
        lattice = str(gen_str(self._config, "harvest") or "lattice") != "sweep"
        if lattice:
            sweeps = [(None, self._harvest_lattice(
                oracle, box, theta, budget,
                _LATTICE_MAX))]
        else:
            sweeps = [(var, self._harvest(
                oracle, var, _box_of(box, var, oracle.rv), box[var][0],
                box[var][1], theta.of(var), budget, theta.of(var) / 2))
                for var in box]
        for var, (got, requested, undecided) in sweeps:
            calls += requested
            if undecided:
                harvest_undecided += undecided
                printer.print_normal(
                    "[kappa_box] harvest{}: {} of {} point(s) UNDECIDED "
                    "within [gen] query-timeout -- the pool is short by that much "
                    "for a solver reason, not a geometric one".format(
                        f" on {var.id}" if var is not None else " (lattice)",
                        undecided, requested))
            # A harvest point is requested inside a window of +/- theta/2 and
            # the solver may return any falsifying model in it, so a witness can
            # sit away from the grid position asked for; neighbouring windows
            # overlap once the spacing is under theta, and all of them overlap
            # the pivot's when the box is narrow.
            #
            # theta is the minimum IC separation, so it is enforced here: a deep
            # witness within theta of one already collected is dropped. Frontier
            # markers are exempt, since their position is the information they
            # carry, and may sit closer than theta to a deep witness.
            for witness in got:
                if any(all(abs(_value_of(witness, v) - _value_of(other, v))
                           < theta.of(v)
                           for v in box) for other in witnesses):
                    collapsed += 1
                    continue
                witnesses.append(witness)
                labels.append(_DEEP)
        if collapsed:
            printer.print_verbose(
                "[kappa_box] {} harvested witness(es) landed within theta "
                "({}) of an existing witness and were dropped".format(
                    collapsed, ", ".join(f"{v.id}={float(theta.of(v))}"
                                         for v in box)))

        _merge_markers(witnesses, labels, markers, marker_labels, box, tol_of,
                       printer)
        printer.print_verbose(
            "[kappa_box] box done: {} solver calls, {} witnesses{}{}".format(
                calls, len(witnesses),
                ", extent is a LOWER BOUND (undecided face)" if unresolved else "",
                f", {harvest_undecided} harvest point(s) undecided"
                if harvest_undecided else ""))
        return witnesses, labels, box

    def run(self, model, goal, prop_dict, config, solver, logger, printer):
        common = config.get_section("common")
        max_depth = int(common.get_value("bound"))
        tau_max = float(common.get_value("time-bound"))
        delta = float(common.get_value("threshold"))
        underlying = common.get_value("solver")
        # [SANDBOX RECONSTRUCTION] guard dropped so the dreal path can be exercised.
        self._underlying = underlying
        self._printer = printer
        _th = common.get_value("time-horizon") if common.is_argument_in(
            "time-horizon") else "time-bound"
        self._time_horizon = float(tau_max) if str(_th) == "time-bound" else float(_th)
        self._config = config
        self._logger = logger
        self._tau_max = common.get_value("time-bound")

        logic = z3_logic(config)
        seed = resolve_seed(config)
        # theta: absolute by default, per-axis when [gen] epsilon-relative is set.
        theta = _Theta(gen_frac(config, "epsilon", "0.01"),
                       gen_frac(config, "epsilon-relative", "0") or None,
                       model.range_dict)
        bisect_iters = gen_int(config, "bisect-iters") or _BISECT_ITERS
        per_depth_boxes = gen_int(config, "k-ic")  # per target depth; None -> exhaust
        thin = gen_frac(config, "thin-ic", "0")  # 0 -> no thinning
        target_depths = gen_depths(config, max_depth)
        # A per-axis theta changes witness spacing on every axis, so the
        # resolved values are printed with the run.
        if theta.relative is not None:
            printer.print_normal(f"[kappa_box] {theta.describe()}")
        else:
            printer.print_verbose(f"[kappa_box] {theta.describe()}")

        warn_unpinned_hashseed(printer)

        self._metrics = _Counter()
        self._any_unresolved = False
        self._rotated_words = 0
        if underlying != "z3":
            # A configuration still setting a removed key would otherwise get
            # silence, which reads as the key being in effect.
            for gone in ("word-pruning", "word-check-timeout",
                         "assume-monotone-flows", "block-class",
                         "two-step-pivot", "warm-start", "lattice-max"):
                if gen_int(config, gone) is not None:
                    printer.print_normal(
                        f"warning: [gen] {gone} no longer exists and is "
                        "ignored")

            printer.print_normal(
                "[kappa_box] word-rotate={}".format(
                    30 if gen_int(config, "word-rotate") is None
                    else (gen_int(config, "word-rotate") or "off")))
            pivot_timeout = (gen_float(config, "pivot-timeout")
                             or _DEFAULT_CANDIDATE_TIMEOUT)
            printer.print_normal(
                "[kappa_box] solver budgets: query-timeout={}s, "
                "pivot-timeout={}s per candidate, pivot-budget={}s per depth "
                "([gen] query-timeout = 0 disables)".format(
                    gen_float(config, "query-timeout") or 60.0,
                    pivot_timeout,
                    gen_float(config, "pivot-budget") or 120.0))

        encoder = Encoder(model, goal, prop_dict, delta, tau_max)
        pool: list[dict[Variable, Constant]] = []
        labels: list[str] = []
        first_depth = max_depth
        total_boxes = 0

        for depth in target_depths:
            blocks: list[Formula] = []  # per depth: independent region discovery
            boxes_here = 0
            self._rotated_words = 0     # heuristic pruning is scoped to a depth
            while per_depth_boxes is None or boxes_here < per_depth_boxes:
                oracle, pivot, encoding = self._pivot_at(
                    encoder, depth, logic, seed, blocks
                )
                if pivot is None:
                    if getattr(self, "_last_pivot_verdict", None) == UNKNOWN:
                        self._any_unresolved = True
                        printer.print_normal(
                            f"[kappa_box] depth {depth}: pivot search "
                            "UNRESOLVED (solver budget exhausted) -- the region "
                            "is NOT proven empty; raise [gen] pivot-timeout to "
                            "search further")
                    else:
                        # UNSAT: the structure space is exhausted. What that is
                        # worth depends on what was pruned -- word rotation
                        # removes structures the solver never tried, so under it
                        # this means no further structure is reachable, not that
                        # none exists.
                        heuristic = []
                        if self._rotated_words:
                            heuristic.append(
                                f"{self._rotated_words} word(s) rotated off")
                        scope = ("no counterexample at this depth"
                                 if not blocks else
                                 f"no counterexample outside the "
                                 f"{boxes_here} box(es) already found")
                        if heuristic:
                            printer.print_normal(
                                "[kappa_box] depth {}: structure space exhausted, "
                                "but heuristic pruning was active ({}) -- absence is "
                                "NOT established; re-run with word-rotate = 0 "
                                "to make it conclusive".format(
                                    depth, ", ".join(heuristic)))
                        else:
                            printer.print_normal(
                                f"[kappa_box] depth {depth}: structure space "
                                f"exhausted -- {scope} (absence, established by "
                                f"exhaustion)")
                    encoder.reset()
                    break

                oracle.assert_(_skeleton_fix(pivot))  # pin path + Boolean skeleton
                witnesses, box_labels, box = self._grow_box(
                    oracle, pivot, encoding, theta, bisect_iters, depth,
                    gen_int(config, "k-witness") or 8, printer
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
                block_bounds = {v: (lo - theta.of(v), hi + theta.of(v))
                                for v, (lo, hi) in box.items()}
                blocks.append(_block_box(block_bounds, oracle.rv))
                encoder.reset()  # clean slate before the next pivot search
                printer.print_normal(
                    "[kappa_box/metrics] depth {}: candidates={} accepted={} "
                    "accept-rate={:.0%}".format(
                        depth, self._metrics["candidates"], self._metrics["accepted"],
                        (self._metrics["accepted"] / self._metrics["candidates"])
                        if self._metrics["candidates"] else 0.0))
                printer.print_verbose(
                    f"[kappa_box] depth {depth}: box {boxes_here} here, "
                    f"kept {kept}/{len(witnesses)}; pool {len(pool)}"
                )

        self.ce_labels = labels
        printer.print_verbose(
            f"[kappa_box] {total_boxes} box(es) over {len(target_depths)} "
            f"target depth(s), {len(pool)} witnesses: "
            f"{labels.count(_DEEP)} deep, {labels.count(_BOUNDARY)} boundary, "
            f"{labels.count(_DOMAIN)} domain"
        )

        result, note = _verdict(pool, getattr(self, "_any_unresolved", False),
                                target_depths, max_depth)
        if note:
            printer.print_normal(note)
        return result, 0.0, first_depth, pool