"""Region box discovery (kappa_box): multi-box frontier growth, no certificate.

Around a falsifying pivot on a fixed mode path, grow an axis-aligned box of
initial conditions and collect a falsifying witness per expansion. A face grows
by a minimum separation theta at a time: it expands only while a falsifying
initial condition exists at least theta beyond the current bound, and stops when
the falsifying region ends within theta (NEG-only). theta both guarantees
termination over a continuous region and spaces the witnesses.

Each face search asks a directional question -- "is there a falsifying initial
condition at or beyond m on this axis" -- so a located bound is a lower bound on
the directional supremum of the falsifying extent, not a certified coordinate,
and the grown rectangle is a coordinate-wise search-and-harvest envelope, not a
subset of the falsifying set. Witnesses are labeled by how their face resolved.
The interior witnesses collected during growth are ``deep``. After a box's growth
converges, each face is examined once against the effective initial-condition
domain -- the region the initial condition confines the ICs to, which the
declared variable range may be far wider than. If a falsifying initial condition
reaches that IC-domain edge, the face's marker is ``ic-domain`` (the observable
extent is the initial set's own edge, not a falsifying frontier); when that edge
is only a per-axis projection of an initial set that is not an axis-aligned box,
the marker is ``projected-domain`` instead. Otherwise the search brackets the
directional bound strictly inside the IC domain, between a confirmed falsifying
value and a confirmed non-falsifying one, giving a ``structure-frontier`` marker
there. Bracketing locates where directional support ends, which coincides with
the falsifying frontier only when the falsifying set is convex along the ray;
convexity is an assumption, not a consequence of pinning the structure.

The effective IC domain is resolved once per run (:func:`_ic_plan`). Explicit
``[gen] target-axes`` (with optional ``[gen] target-bounds``) name the axes and
their edges authoritatively; otherwise the initial condition is read as a
conjunction of independent per-axis bounds where it is one, and projected onto
each axis where it is not. An axis the initial condition pins to a point is not
a diversity axis and is dropped rather than grown as a degenerate face.

Both backends are supported. On an exact backend a face advances by theta-wide
steps and the frontier is located by bisection. On a delta-decision backend the
face is located by a logarithmic search to tolerance ``max(delta, theta/8)`` and
witnesses are harvested on a lattice over the converged box, so the number of
solver calls is set by an explicit budget rather than by the box extent.

Depth traversal. The strategy visits a set of target depths -- ``[gen] depths``
as a slash-separated list (e.g. ``"7/8"``), or every depth 1..N by default -- and
explores each independently. At a target depth it grows a box, labels it, blocks
the envelope (the box extended by theta on every face), and re-pivots outside the
blocked envelopes of that depth, up to the per-depth budget ``[gen] k-ic``;
absent or 0, the depth is explored to exhaustion. Blocking the envelope is a
coverage heuristic: the envelope is not a subset of the falsifying set and may
bridge several falsifying components, so re-pivoting is not guaranteed to recover
a disconnected region as a separate box. The blocks are per depth: each target
depth discovers its own regions, so a region falsifying at several depths
contributes a counterexample (with a different mode path) at each, making depth a
first-class diversity axis rather than an incidental by-product of re-pivoting.
The returned pool is the union over target depths.

Separation is per depth. theta is the minimum separation of the witnesses a
depth contributes, not merely of the witnesses one box contributes: growth runs
unmasked, so a later box may regrow across an earlier one, and a deep witness
within theta of one an earlier box at the same depth contributed is dropped.
Across depths no separation is imposed, because one initial condition falsifying
at several depths is the depth axis rather than redundancy. Thinning
(``[gen] thin-ic``) applies a coarser radius under the same scope, so a value at
or below theta is a no-op. Structure-frontier and domain markers (``ic-domain``,
``projected-domain``) are exempt from both and always kept, since their position
is the information they carry. The thinning default of 0 leaves it off.

Initial-condition variables are the step-0 state copies ``<name>_0_0`` of the
axes the IC plan targets (a subset of ``range_dict``); mode variables are
``currentMode_k``. Box bounds are exact rationals so the degenerate pivot box is
represented exactly.

The pool is returned to the driver for serialization; the per-counterexample
labels are exposed on the ``ce_labels`` attribute, aligned to the returned pool,
for the driver to append to the payload.
"""

from __future__ import annotations

import re
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
    Neg,
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
    backend_precision,
    gen_depths,
    gen_float,
    gen_frac,
    gen_int,
    gen_present,
    gen_str,
    resolve_seed,
    scoped_verdict,
    validate_gen,
    warn_unpinned_hashseed,
    z3_logic,
)
from .encode import Encoder
from .oracle import (
    SAT,
    UNKNOWN,
    UNSAT,
    Z3IncrementalOracle,
    make_oracle,
    query_timeout,
)

# Face-search precision on an exact oracle when [gen] bisect-iters is not set:
# the located frontier is within theta * 2**-_BISECT_ITERS of the true crossing.
_BISECT_ITERS = 20

# Default for [gen] pivot-timeout, which bounds ONE solver call inside the pivot
# search. One call decides one candidate out of many and the search as a whole is
# bounded by [gen] pivot-budget, so a per-call value near that budget would let a
# single candidate consume all of it.
_DEFAULT_CANDIDATE_TIMEOUT = 45.0

# Default for [gen] pivot-budget (seconds for one whole candidate search) and
# [gen] k-witness (per-axis cell budget of the lattice harvest). Named so the
# banner and the consumers resolve the same value: the `x or DEFAULT` idiom
# they replaced also folded a configured 0 into the default and then printed
# the default back as if it had been set (0 is now rejected by validate_gen
# for these keys, or meaningful and honored where it has a meaning).
_DEFAULT_PIVOT_BUDGET = 120.0
_DEFAULT_K_WITNESS = 8
_DEFAULT_LOG_EVERY = 25

# Ceiling on the number of lattice cells harvested from one box. The lattice
# grows as the product over IC axes -- four variables at eight cells per axis is
# 4096 solver calls -- so the product is capped; cells are then removed from the
# widest axis one at a time until the product fits, and the reduction is
# reported.
_LATTICE_MAX = 200

# Per-counterexample labels.
_DEEP = "deep"
_STRUCTURE_FRONTIER = "structure-frontier"
# A face that reaches the effective initial-condition domain edge, rather than a
# structure-relative frontier interior to it. ``ic-domain`` when that edge is
# authoritative -- named by [gen] target-axes/target-bounds, or read off an
# ``init`` that is a conjunction of independent per-axis bounds. ``projected-
# domain`` when ``init`` is relational, non-convex or disjunctive: the per-axis
# window is then the projection of the initial set onto the axis, not a certified
# box face, so the label does not claim the observed extent is the whole domain.
_IC_DOMAIN = "ic-domain"
_PROJECTED_DOMAIN = "projected-domain"


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


def _ic_pivots(
    assn: dict[Variable, Constant], range_dict, keep_ids=None
) -> dict[Variable, Fraction]:
    """Initial-condition variables (<name>_0_0) and their pivot values.

    Sorted by variable id. The assignment dict arrives in the solver's model
    order (z3's ``decls()`` order, dReal's print order), which no seed pins --
    and :meth:`RegionBoxDiscovery._grow_box` grows greedily in one pass, so
    the converged geometry depends on this iteration order. Sorting makes the
    box a function of the configuration and the solver build alone.

    ``keep_ids`` restricts the axes to a chosen subset (the IC plan's target
    axes); ``None`` keeps every step-0 copy of a ``range_dict`` variable. A
    variable the initial condition pins to a point, or one the configuration
    does not target, is dropped here rather than grown as a degenerate axis."""
    ic_ids = {f"{k.id}_0_0" for k in range_dict}
    if keep_ids is not None:
        ic_ids &= set(keep_ids)
    found = {v: _frac(c.value) for v, c in assn.items() if v.id in ic_ids}
    return {v: found[v] for v in sorted(found, key=lambda var: var.id)}


def _const_value(node):
    """The rational value of a constant expression, or ``None``.

    Handles the shapes the model parser produces for an initial-condition bound:
    a ``RealVal``/``IntVal`` leaf and a negated one (``- 0.2`` parses to
    ``Neg(RealVal("0.2"))``). Anything else -- a variable, an arithmetic term,
    a Boolean constant -- is not a numeric constant and returns ``None``."""
    if isinstance(node, Neg):
        inner = _const_value(node.child)
        return None if inner is None else -inner
    if isinstance(node, Constant):
        try:
            return Fraction(str(node.value))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _axis_bound(node, state_ids):
    """One ``init`` conjunct read as a per-axis bound.

    Returns ``(state_id, lo, hi)`` for a comparison between a declared state
    variable and a constant, with the unconstrained side ``None`` (an equality
    fixes both), or ``None`` when the conjunct is not a per-axis bound: a
    Boolean literal or its negation, a disjunction, a relational term coupling
    two variables, or a bound on a variable that is not a declared IC axis (a
    mode index)."""
    if not isinstance(node, (Geq, Gt, Leq, Lt, Eq)):
        return None
    left, right = node.left, node.right
    left_c, right_c = _const_value(left), _const_value(right)
    if isinstance(left, Variable) and left.id in state_ids and right_c is not None:
        state_id, value, var_on_left = left.id, right_c, True
    elif isinstance(right, Variable) and right.id in state_ids and left_c is not None:
        state_id, value, var_on_left = right.id, left_c, False
    else:
        return None
    if isinstance(node, Eq):
        return (state_id, value, value)
    lower = (isinstance(node, (Geq, Gt)) if var_on_left
             else isinstance(node, (Leq, Lt)))
    return (state_id, value, None) if lower else (state_id, None, value)


def _init_projection(init, range_dict):
    """Read ``init`` as per-axis interval bounds on the IC axes.

    Returns ``(per_axis, is_box)``. ``per_axis`` maps ``<name>_0_0`` to
    ``[lo, hi]`` (a side ``init`` does not bound stays ``None``; an equality
    makes ``lo == hi``). ``is_box`` is True only when *every* conjunct of the
    flattened top-level conjunction is a per-axis bound on a declared variable:
    then the projection is the initial set exactly, an axis-aligned box. One
    conjunct of any other shape -- a Boolean, a negation, a disjunction, a
    relational term, a bound on a non-declared variable -- clears it, and the
    per-axis windows are a projection of a set that is not a box."""
    state_ids = {sv.id for sv in range_dict}
    per_axis: dict[str, list] = {}
    is_box = True

    def fold(node):
        nonlocal is_box
        if isinstance(node, And):
            for child in node.children:
                fold(child)
            return
        info = _axis_bound(node, state_ids)
        if info is None:
            is_box = False
            return
        state_id, lo, hi = info
        cur = per_axis.setdefault(f"{state_id}_0_0", [None, None])
        if lo is not None:
            cur[0] = lo if cur[0] is None else max(cur[0], lo)
        if hi is not None:
            cur[1] = hi if cur[1] is None else min(cur[1], hi)

    if init is not None:
        fold(init)
    return per_axis, is_box


def _parse_target_bounds(raw):
    """``[gen] target-bounds`` -> ``{<name>_0_0: (lo, hi)}``.

    Slash-separated ``name/lo/hi`` triples: the config grammar lexes an
    identifier and a signed decimal to one VALUE token only without other
    punctuation, so a colon or comma inside the value cannot be used. Empty
    -> ``{}``. Malformed input is rejected up front by ``validate_gen``."""
    if not raw:
        return {}
    toks = [t.strip() for t in re.split(r"[,/]", raw) if t.strip() != ""]
    out = {}
    for i in range(0, len(toks) - 2, 3):
        out[f"{toks[i]}_0_0"] = (Fraction(toks[i + 1]), Fraction(toks[i + 2]))
    return out


def _edge_of(ic_id, projection, declared):
    """Effective ``(lo, hi)`` for one axis: the ``init`` projection where it
    bounds the axis, the declared range where it does not. Both finite."""
    lo_d, hi_d = declared[ic_id]
    lo, hi = projection.get(ic_id, (None, None))
    return (lo if lo is not None else lo_d, hi if hi is not None else hi_d)


def _ic_plan(config, model):
    """Resolve the IC axes, their effective domain edges, and the label a face
    reaching an edge earns.

    Precedence:
      1. ``[gen] target-axes`` (authoritative): the named axes exactly, dropping
         every declared variable not listed -- a coordinate the initial
         condition pins, say. ``[gen] target-bounds`` overrides an axis's edges;
         an axis it omits takes its ``init`` projection, then its declared range.
         Reaching such an edge is ``ic-domain``: the configuration asserts the
         box.
      2. Otherwise the ``init`` box-shape detector: every declared axis the
         initial condition does not pin to a point, edged by ``init`` where it
         bounds the axis and by the declared range where it does not. The label
         is ``ic-domain`` when ``init`` is a certified per-axis box and
         ``projected-domain`` when it is not.

    Returns ``(axis_ids, edges, label)``: ``axis_ids`` an ordered list of
    ``<name>_0_0``, ``edges`` mapping each to a finite ``(lo, hi)``, ``label``
    one of :data:`_IC_DOMAIN` / :data:`_PROJECTED_DOMAIN`."""
    range_dict = model.range_dict
    declared = {f"{sv.id}_0_0": (_frac(b[1]), _frac(b[2]))
                for sv, b in range_dict.items()}
    projection, is_box = _init_projection(getattr(model, "init", None), range_dict)

    cfg_axes = gen_str(config, "target-axes")
    cfg_bounds = _parse_target_bounds(gen_str(config, "target-bounds"))
    if cfg_axes:
        names = [t.strip() for t in re.split(r"[,/]", cfg_axes) if t.strip()]
        axis_ids = sorted({f"{n}_0_0" for n in names} & set(declared))
        edges = {i: cfg_bounds.get(i) or _edge_of(i, projection, declared)
                 for i in axis_ids}
        return axis_ids, edges, _IC_DOMAIN

    axis_ids, edges = [], {}
    for ic_id in declared:
        lo, hi = projection.get(ic_id, (None, None))
        if lo is not None and hi is not None and lo == hi:
            continue                       # init pins this axis: not a diversity axis
        axis_ids.append(ic_id)
        edges[ic_id] = _edge_of(ic_id, projection, declared)
    label = _IC_DOMAIN if is_box else _PROJECTED_DOMAIN
    return sorted(axis_ids), edges, label


def _box_of(box: dict[Variable, list[Fraction]], skip: Variable, rv=_rv) -> Formula:
    """AND over dimensions (except ``skip``) of lo <= x <= hi."""
    terms: list[Formula] = []
    for var, (lo, hi) in box.items():
        if var is skip:
            continue
        terms.append(Geq(var, rv(lo)))
        terms.append(Leq(var, rv(hi)))
    return And(terms) if terms else BoolVal("True")


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


def _within_theta(
    witness: dict[Variable, Constant],
    pool: list[dict[Variable, Constant]],
    ic_vars,
    theta: _Theta,
) -> bool:
    """True if ``witness`` lies within theta of some pooled witness.

    The per-axis counterpart of :func:`_too_close`. Separation is an L-infinity
    condition, so two witnesses are too close only when they are within theta on
    *every* axis; one axis at or beyond its own theta separates them.
    """
    for other in pool:
        if all(abs(_value_of(witness, v) - _value_of(other, v)) < theta.of(v)
               for v in ic_vars):
            return True
    return False


class _Theta:
    """The minimum initial-condition separation, per axis.

    theta is the strategy's granularity: how far apart witnesses must be, how
    far a face must reach to count as growth, how wide a blocked region is
    extended.

    A single absolute theta fixes that granularity in model units, so it is a
    different fraction of each axis whose initial-condition domain differs.
    ``[gen] epsilon-relative`` instead sets theta as a fraction of each axis's
    effective IC-domain width, making the granularity uniform relative to the
    space each variable actually ranges over initially -- not the declared
    range, which the initial condition may be far narrower than. ``[gen]
    epsilon`` remains the absolute default and the fallback for an axis whose
    IC-domain width is unknown or degenerate.
    """

    def __init__(self, absolute: Fraction, relative: Fraction | None = None,
                 widths=None) -> None:
        self.absolute = absolute
        self.relative = relative
        self._by_id: dict[str, Fraction] = {}
        if relative is not None and widths:
            for ic_id, width in widths.items():
                if width > 0:
                    self._by_id[ic_id] = relative * width

    def of(self, var) -> Fraction:
        """theta for one IC variable."""
        return self._by_id.get(getattr(var, "id", var), self.absolute)

    def describe(self) -> str:
        if not self._by_id:
            return f"theta={float(self.absolute)} (absolute, every axis)"
        per_axis = ", ".join(
            f"{vid}={float(value)}"
            for vid, value in sorted(self._by_id.items()))
        return (f"theta={float(self.relative)} x IC-domain width -> {per_axis}"
                f" (fallback {float(self.absolute)})")


class _WordRotation:
    """The ``word-rotate`` coverage heuristic, as a policy over verdicts.

    A mode word is dropped after ``limit`` consecutive **refutations** of
    candidates carrying it. Only UNSAT is one. An UNKNOWN carries no information
    about the query that produced it -- it says the candidate was expensive, not
    that its word is infeasible -- so an expired per-candidate bound may neither
    advance the streak nor reset it: undecided candidates are stepped over and
    the run of refutations continues across them.

    Counting an expiry as a refutation discards a word for a solver reason. It
    was measured doing exactly that: a depth-3 word whose candidates each hit a
    60 s ``pivot-timeout`` was blocked after 30 of them, on a goal whose
    published counterexample lives at that depth.

    ``limit = 0`` disables the heuristic, and a candidate with no mode word is
    never counted.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.dropped = 0        # words blocked by the heuristic
        self.undecided = 0      # candidates the oracle did not decide
        self._word = None
        self._streak = 0

    @property
    def streak(self) -> int:
        """Consecutive refutations of the word currently under the counter."""
        return self._streak

    def refuted(self, word) -> bool:
        """Record a refutation of a candidate carrying ``word``.

        Returns True when that word has reached the limit and must be blocked;
        the counter is reset then, so one word is reported once per run of
        refutations."""
        if not self.limit or not word:
            return False
        self._streak = self._streak + 1 if word == self._word else 1
        self._word = word
        if self._streak >= self.limit:
            self._word, self._streak = None, 0
            self.dropped += 1
            return True
        return False

    def undecided_candidate(self) -> None:
        """Record a candidate the oracle left undecided. Deliberately does not
        touch the streak: an UNKNOWN is evidence in neither direction."""
        self.undecided += 1


def _binding_bound(attempts, refuted, undecided, undecided_seconds, elapsed):
    """Which ``[gen]`` key bounded a candidate search that gave up, and why.

    The loop always ends on the ``pivot-budget`` deadline, so the exit condition
    by itself names nothing. What names a knob is where the budget went. Time
    spent on candidates that expired undecided is time a larger ``pivot-budget``
    would only buy more expiries of, so there the bound that bit is
    ``pivot-timeout``; a search whose candidates were decided, and quickly, was
    bounded by the budget itself, and raising the per-call bound would buy it
    strictly fewer candidates. Both directions were measured on space-ode: f2@3
    expiring at ``pivot-timeout``, f1@2 refuting 972 candidates at ~0.1 s each
    until ``pivot-budget`` ran out.

    Returns ``(key, explanation)``.
    """
    if undecided and 2 * undecided_seconds >= elapsed:
        return ("pivot-timeout",
                f"{undecided} of {attempts} candidate(s) expired undecided, "
                f"consuming {undecided_seconds:.0f}s of the "
                f"{elapsed:.0f}s search")
    return ("pivot-budget",
            f"{attempts} candidate(s) in {elapsed:.0f}s: {refuted} refuted, "
            f"{undecided} undecided")


def _box_budget(config):
    """``[gen] k-ic`` resolved to a per-depth box budget.

    ``None`` and ``0`` both mean no budget (the depth is explored to
    exhaustion): 0 is the section's spelling for "off" (``word-rotate``,
    ``thin-ic``, ``query-timeout``), and a budget of zero boxes would end every
    depth after zero solver calls and let an empty run report True. A negative
    value is a configuration error, for the same reason."""
    budget = gen_int(config, "k-ic")
    if budget is None or budget == 0:
        return None
    if budget < 0:
        raise ValueError(
            f"[gen] k-ic must be >= 0 (0 = no budget, explore the depth to "
            f"exhaustion); got {budget}")
    return budget


def _verdict(pool, any_unresolved, settled, max_depth):
    """kappa_box's wording for the shared depth-scoping rule.

    ``settled`` is the set of depths the run actually decided: a depth counts
    only when its structure space was exhausted by refutation alone. A depth
    left by an undecided pivot, by the box budget, or by an exhaustion a
    coverage heuristic took part in is visited but not settled, and must not
    ground a True."""
    return scoped_verdict(pool, any_unresolved, settled, max_depth,
                          tag="kappa_box", nothing_found="no box found",
                          unresolved_source="at least one pivot search or "
                                            "heuristic-assisted exhaustion")


def _face_tol(oracle, theta: _Theta, var: Variable, iters: int) -> Fraction:
    """Face precision for one axis, floored by the oracle's tolerance.

    Two points closer than this are points the search cannot tell apart, so it
    is also the radius at which a face marker is merged into a witness rather
    than admitted beside it. Both uses resolve here so they cannot diverge.
    """
    target = (theta.of(var) / (2 ** iters)
              if getattr(oracle, "is_exact", True) else theta.of(var) / 8)
    return max(oracle.tolerance, target)


def _coincides_with(marker, pool, ic_vars, tol) -> int | None:
    """Index of the first pooled witness within ``tol`` of ``marker``, or None.

    The cross-box half of the one-entry-per-initial-condition rule.
    :func:`_merge_markers` applies it inside a box; this applies it against the
    witnesses earlier boxes at the same depth contributed, which that call
    cannot see. ``tol`` is a callable ``var -> Fraction``.
    """
    for index, witness in enumerate(pool):
        if all(abs(_value_of(marker, v) - _value_of(witness, v)) <= tol(v)
               for v in ic_vars):
            return index
    return None


# Label priority for a point on more than one face. A ``structure-frontier``
# reports a falsifying frontier interior to the IC domain -- the substantive
# claim -- so it outranks a domain-edge touch, which only records that the box
# met the initial set's own boundary there; either outranks a plain interior
# ``deep`` witness. A single initial condition can sit on faces of different
# kinds (an adverse corner is on the domain edge of the pinned axes and on an
# interior frontier of another), and the pool keeps the most informative of
# them so the interior demonstration is not masked by a coincident edge.
_LABEL_RANK = {_DEEP: 0, _IC_DOMAIN: 1, _PROJECTED_DOMAIN: 1,
               _STRUCTURE_FRONTIER: 2}


def _prefer_label(current: str, candidate: str) -> str:
    """The more informative of two labels for one initial condition."""
    return candidate if _LABEL_RANK.get(candidate, 0) > _LABEL_RANK.get(
        current, 0) else current


def _merge_markers(witnesses, labels, markers, marker_labels, ic_vars, tol,
                   printer) -> int:
    """Append face markers, merging any that land on a witness already collected.

    A marker coincides with an existing witness whenever the pivot itself sits
    on a frontier: the face search confirms a bound the pivot already occupies.
    Appending it would put two entries in the pool for one initial condition.

    The merge keeps the more informative label (:data:`_LABEL_RANK`): the two
    are the same point to within ``tol``, so the existing witness is relabeled
    in place when the marker carries a stronger claim, and the frontier
    annotation is preserved. ``tol`` is below the granularity theta declares, so
    only points the search cannot tell apart are merged.

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
            labels[hit] = _prefer_label(labels[hit], marker_label)
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
        # Scoped to one depth by `run`; initialised here so a caller that drives
        # a pivot search on its own does not have to.
        self._rotated_words = 0
        self._undecided_candidates = 0
        self._pivot_giveup = None
        self._metrics = _Counter()

    def set_debug(self, msg: str) -> None:
        self.debug_name = msg

    def _skeleton_oracle(self, logic, seed):
        """The outer, purely propositional solver of the two-step pivot search.

        Built directly rather than through make_oracle, which cannot express
        general=True, so the configured bound has to be resolved and passed
        here; otherwise the skeleton solve silently keeps the constructor's
        default. It is a method rather than an inline construction so a test can
        substitute an oracle whose answers are given explicitly and exercise the
        candidate loop without a solver."""
        return Z3IncrementalOracle(logic, seed, general=True,
                                   timeout=query_timeout(self._config))

    def _candidate_oracle(self, logic, seed):
        """The inner ODE-feasibility oracle, one per candidate. This site is on
        the delta path too, since a delta backend always takes the two-step
        route. Substitutable for the same reason as :meth:`_skeleton_oracle`."""
        return make_oracle(self._underlying, logic=logic, seed=seed,
                           config=self._config, logger=self._logger,
                           time_bound=self._tau_max)

    def _exact_oracle(self, logic, seed):
        """The single-query pivot oracle of the exact path. A method for the
        same reason as :meth:`_skeleton_oracle`: a test substitutes it to
        exercise the block-frame discipline without a solver."""
        return Z3IncrementalOracle(logic, seed,
                                   timeout=query_timeout(self._config))

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
        self._pivot_giveup = None
        from ..tree.operations import size_of_tree as _sz
        _t = _time.monotonic()
        abstracted, fa_map = _abstract_foralls(encoding.skeleton)
        self._printer.print_verbose(
            f"[diag] skeleton size={_sz(encoding.skeleton)} "
            f"consts size={_sz(encoding.consts)} "
            f"abstract_time={_time.monotonic() - _t:.1f}s")
        z3o = self._skeleton_oracle(logic, seed)
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
        # before trying another, so a refuted word can absorb the whole budget.
        # After N consecutive refutations under the same word the word is
        # blocked outright, at no solver cost. A heuristic: it can block a word
        # that would have been feasible under a later assignment, trading
        # completeness for coverage, which is why an exhausted search is not
        # reported as absence once it has fired. 0 disables. Only a refutation
        # counts -- see _WordRotation.
        rotate = gen_int(self._config, "word-rotate")
        rotation = _WordRotation(30 if rotate is None else rotate)

        every = gen_int(self._config, "log-every")
        every = _DEFAULT_LOG_EVERY if every is None else every
        budget = gen_float(self._config, "pivot-budget")
        budget = _DEFAULT_PIVOT_BUDGET if budget is None else budget
        candidate_bound = gen_float(self._config, "pivot-timeout")
        if candidate_bound is None:
            candidate_bound = _DEFAULT_CANDIDATE_TIMEOUT
        started = _time.monotonic()
        deadline = started + budget
        attempt, refuted, undecided_seconds = 0, 0, 0.0
        while _time.monotonic() < deadline:
            attempt += 1
            self._metrics["candidates"] += 1
            _tz = _time.monotonic()
            _v0 = z3o.check()
            if attempt == 1:
                self._printer.print_verbose(
                    f"[diag] first z3 check: {_v0} in {_time.monotonic() - _tz:.1f}s")
            if _v0 != SAT:
                # UNSAT exhausts the structure space; UNKNOWN means the
                # skeleton solver gave up within its bound. Collapsing them
                # would report "no more boxes" for a resource failure -- the
                # same rule the exact path applies in _pivot_at.
                self._last_pivot_verdict = _v0
                return None, None, None
            candidate = z3o.model()
            # Positional: sorted by the STEP INDEX and joined with a
            # separator. The lexicographic digit join sorted currentMode_10
            # before currentMode_2 and rendered (1,12) and (11,2) identically
            # ("112"), so at >= 10 modes or depth >= 10 two different paths
            # could share one rotation streak -- and a feasible word could be
            # rotated off on another word's refutations.
            word = ".".join(
                str(int(round(float(c.value))))
                for v, c in sorted(
                    ((v, c) for v, c in candidate.items()
                     if MODE_RE.match(v.id)),
                    key=lambda kv: int(MODE_RE.match(kv[0].id).group(1))))
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

            d = self._candidate_oracle(logic, seed)
            # Clamped to what is left of the search budget, so one candidate
            # cannot overrun the deadline by a whole pivot-timeout. (The z3
            # skeleton call above is bounded by query-timeout at construction,
            # so the residual overrun of one iteration is at most that.)
            d.set_budget(min(candidate_bound,
                             max(deadline - _time.monotonic(), 0.1)))
            d.assert_(encoding.consts)
            d.assert_(pinned)
            # The region blocks must bind THIS oracle too. They are asserted
            # into z3o above, but z3's candidate only pins modes and Bools:
            # the pivot the run uses is this oracle's model, whose reals are
            # otherwise free to sit inside an already-blocked box -- and this
            # oracle is also what grows the box, so unblocked it regrows the
            # same region. Framed so growth still runs on Enc_n[w] alone,
            # mirroring _pivot_at (including the no-empty-frame rule there).
            if blocks:
                d.push()
                for block in blocks:
                    d.assert_(block)
            _t0 = _time.monotonic()
            v = d.check()
            _el = _time.monotonic() - _t0
            d.set_budget("unset")
            # An expiry is always printed: it costs the whole per-candidate
            # bound, so it is both rare and the thing a short search has to be
            # read against.
            if v != UNSAT or attempt % every == 0 or attempt == 1:
                printer.print_verbose(
                    "[kappa_box/two-step] candidate {} (word {}): dreal says {} "
                    "({:.1f}s)".format(attempt, word or "-", v, _el))
            if v == SAT:
                self._metrics["accepted"] += 1
                model = d.model()  # before pop(): pop clears the model state
                if blocks:
                    d.pop()
                return d, model, encoding
            # Not usable *as tested*: block exactly the assignment that was
            # checked so the search moves on. What that block is worth differs
            # by verdict -- UNSAT refutes the assignment, UNKNOWN only records
            # that it was tried -- and only the refutation is evidence about the
            # word it carries.
            z3o.assert_(Not(block_this))
            if v == UNSAT:
                refuted += 1
                if rotation.refuted(word):
                    z3o.assert_(Not(mode_only))
                    self._rotated_words += 1
                    printer.print_verbose(
                        f"[kappa_box/two-step] rotating off word {word} after "
                        f"{rotation.limit} consecutive refutations")
            else:
                rotation.undecided_candidate()
                undecided_seconds += _el
                self._undecided_candidates += 1

        elapsed = _time.monotonic() - started
        self._pivot_giveup = _binding_bound(
            attempt, refuted, rotation.undecided, undecided_seconds, elapsed)
        printer.print_normal(
            f"[kappa_box/two-step] gave up after {attempt} candidates: "
            f"{refuted} refuted, {rotation.undecided} undecided at "
            f"[gen] pivot-timeout = {candidate_bound}s, "
            f"{rotation.dropped} mode word(s) rotated off")
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
        underlying = getattr(self, "_underlying", "z3")
        if underlying != "z3":
            # The two-step search asserts facts that are necessary conditions of
            # the delta query rather than of the encoding -- the segment-duration
            # timing identities and the endpoint instantiation of quantified
            # subformulas -- and these hold only under a backend that realizes a
            # closed segment interval and the injected clock dynamics. Guard that
            # capability explicitly here, rather than letting the soundness of
            # the two-step arm rest on the dispatch happening to route every
            # non-exact backend to it.
            if underlying != "dreal":
                raise ValueError(
                    "the two-step pivot search is sound only under the dreal "
                    "backend, whose closed-interval and clock realization the "
                    "asserted timing and endpoint facts depend on; backend "
                    f"'{underlying}' cannot take it")
            return self._pivot_two_step(encoding, logic, seed, blocks)
        # Only the exact backend reaches here; a delta backend always takes the
        # two-step path above.
        oracle = self._exact_oracle(logic, seed)
        # One query decides one pivot here, so the pair reads 1/1 per box on
        # this backend; without it the metrics line printed 0/0 on every
        # exact run while the same line fed pivot-budget sizing on the other.
        self._metrics["candidates"] += 1
        oracle.assert_(encoding.consts)
        # The blocks bind the PIVOT, never growth (Alg. 2: Pivot solves
        # Enc_n conjoined with the negated blocks, while every growth query
        # runs on Enc_n[w] alone). Asserted permanently they also wall in the
        # box growth: _search_face reads UNSAT at a block wall as a bracketed
        # frontier and emits a `structure-frontier` marker at a purely
        # algorithmic wall.
        # So they live in a frame that is popped once the pivot is extracted.
        # No frame when there is nothing to put in it: an empty push/pop still
        # perturbs the solver's model choice, which would change the pool of
        # every existing single-box configuration for no semantic reason.
        if blocks:
            oracle.push()
            for block in blocks:
                oracle.assert_(block)
        v = oracle.check()
        if v == SAT:
            self._metrics["accepted"] += 1
            pivot = oracle.model()  # before pop(): pop clears the model state
            if blocks:
                oracle.pop()
            return oracle, pivot, encoding
        if blocks:
            oracle.pop()
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
        """Directional support bound for ``var`` towards ``wall``: the largest
        ``m`` for which a falsifying initial condition was exhibited at or beyond
        ``m`` (or the smallest, when ``wall < start``), to within ``tol``.

        The probe (:func:`falsifying_beyond`) is existential-beyond -- it asks
        whether a falsifying IC exists with the coordinate at or past ``m``, not
        whether ``m`` itself falsifies -- so the returned bound is a LOWER BOUND
        on the directional supremum of the falsifying extent, never a certified
        coordinate. The rectangle grown from it is a coordinate-wise search-and-
        harvest envelope, not a subset of the falsifying set: harvest re-queries
        each cell before keeping a witness, so individual witnesses stay valid,
        but the region between them is not asserted to falsify.

        Logarithmic in the extent instead of linear in extent/theta, and it never
        reads UNKNOWN as a frontier. Returns ``(bound, status, calls)`` with
        status in {'domain', 'frontier', 'partial', 'unresolved'}.

        An UNKNOWN answer is local: failing to decide whether a falsifying IC
        exists beyond one point says nothing about the points on either side, so
        the search retries nearer the confirmed side rather than abandoning the
        face. ``lo`` always satisfies the existential-beyond probe, so a face that
        stops early yields a smaller envelope, never a wrong witness:

            domain      a falsifying IC is exhibited at the declared range edge.
            frontier    the directional bound is bracketed between a confirmed
                        beyond-falsifying value and a confirmed non-falsifying
                        one, to within tol; the falsifying frontier itself only
                        under convexity along the ray.
            partial     grew beyond the pivot but never bracketed. The bound is
                        a LOWER BOUND on the directional support, not a frontier.
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
        # Def. harvest: the sampling window has width >= delta (the oracle's
        # tolerance); half is that width / 2, so floor it at delta / 2.
        half = max(half, oracle.tolerance / 2)
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
        variables. When the full lattice exceeds it, cells are removed from the
        widest axis one at a time until the product fits, and the reduction is
        reported.

        Returns ``(witnesses, requested, undecided)``."""
        axes = list(box)
        cells, half = {}, {}
        for var in axes:
            lo, hi = box[var]
            span = hi - lo
            th = theta.of(var)
            # Def. 7: c_j = min(k, max(1, floor(span/theta))), so c_j >= 1
            # always. An axis narrower than theta -- a face that never grew, or
            # an IC the model pins -- contributes ONE cell at its pivot value,
            # never none: zeroing it silently discarded the whole harvest of
            # every other axis, and a box was returned as pivot + markers with
            # k-witness inert.
            cells[var] = min(budget, max(1, int(span / th)))
            # Def. harvest requires the sampling window w_j >= delta (the
            # oracle's tolerance), theta by default; half is w_j / 2. A window
            # narrower than delta forfeits the claim that a returned point lies
            # in the queried cell, which is what pinning every axis buys.
            half[var] = max(th, oracle.tolerance) / 2

        total = 1
        for var in axes:
            total *= cells[var]
        uncapped = total
        if cap and total > cap:
            # One cell off the widest axis at a time, until the product fits.
            # The earlier uniform shrink floored every axis by the same factor
            # and the flooring compounded: five axes at 3 cells (243) under a
            # 200 cap collapsed to 2^5 = 32 rather than stopping at 162.
            while total > cap and any(n > 1 for n in cells.values()):
                widest = max(axes, key=lambda v: cells[v])
                total //= cells[widest]
                cells[widest] -= 1
                total *= cells[widest]
            printer = getattr(self, "_printer", None)
            if printer is not None:
                printer.print_normal(
                    "[kappa_box] lattice capped: {} cell(s) requested, "
                    "reduced to {} (cap {}): {}".format(
                        uncapped, total, cap,
                        ", ".join(f"{v.id}={cells[v]}" for v in axes)))

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
                  budget, printer, axis_ids=None, ic_edges=None,
                  domain_label=_IC_DOMAIN):
        """Grow, label and sample one box around ``pivot``.

        One procedure for both kinds of oracle. Per face, a search toward the
        IC-domain edge for the directional bound; then a lattice of witnesses
        over the converged box. What the oracle changes is the precision: the
        search runs to ``theta / 2**iters`` on an exact oracle and to
        ``theta / 8`` on a partial one, in both cases floored by the oracle's
        own tolerance, since no query distinguishes points closer than that.

        ``axis_ids`` and ``ic_edges`` are the IC plan (:func:`_ic_plan`): which
        step-0 variables are grown, and the effective ``(lo, hi)`` domain edge
        each face searches toward. Absent, every ``range_dict`` axis is grown to
        its declared range -- the standalone default. ``domain_label`` is the
        marker a face reaching an IC-domain edge earns (``ic-domain`` or
        ``projected-domain``); a bracketed crossing interior to the edge is
        always ``structure-frontier``.

        Growth is greedy and single-pass: each axis is searched against the
        already-grown extents of the axes before it and the pivot values of
        the axes after it, in the (sorted) order ``_ic_pivots`` fixes. The
        converged box is therefore a function of that order -- a second pass
        could grow it further -- and is a bounding box of confirmed slices,
        not a maximal box of the falsifying set."""
        if ic_edges is None:
            ic_edges = {f"{sv.id}_0_0": (_frac(b[1]), _frac(b[2]))
                        for sv, b in encoding.range_dict.items()}
        if axis_ids is None:
            axis_ids = sorted(ic_edges)
        ic = _ic_pivots(pivot, encoding.range_dict, keep_ids=axis_ids)
        if not ic:
            raise RuntimeError("no initial-condition variables (<name>_0_0) found")
        box = {v: [p, p] for v, p in ic.items()}

        def tol_of(v):
            return _face_tol(oracle, theta, v, iters)
        self._tol_of = tol_of      # the same tolerance the caller merges against
        witnesses, labels = [pivot], [_DEEP]
        markers, marker_labels, calls, unresolved = [], [], 0, False
        # A partial face contributes a deep witness (no frontier was located),
        # collected here and held to theta separation like any deep witness --
        # not routed through the marker merge, which is exempt from separation.
        deep_markers: list[dict[Variable, Constant]] = []

        for var in box:
            tol = tol_of(var)
            others = _box_of(box, var, oracle.rv)
            lo_dom, hi_dom = ic_edges.get(var.id, (None, None))
            for direction in (+1, -1):
                wall = hi_dom if direction > 0 else lo_dom
                if wall is None:
                    continue
                start = box[var][1] if direction > 0 else box[var][0]
                # The pivot already sits at (or past) the IC-domain edge on this
                # face: there is no interior to search, and the edge itself is
                # where a witness sits. Emit the domain marker at the pivot.
                if (direction > 0 and wall <= start) or (
                        direction < 0 and wall >= start):
                    m = self._window_witness(oracle, var, others, start, tol)
                    calls += 1
                    if m is not None:
                        markers.append(m)
                        marker_labels.append(domain_label)
                    continue
                bound, status, c = self._search_face(
                    oracle, var, others, start, wall, tol)
                calls += c
                # The wall is the IC-domain edge, so a bound that reaches it --
                # a satisfying probe AT the edge, or a bracket that lands within
                # tolerance of it, whether the edge is open or closed in the
                # encoding -- is a domain face. A bracket that stops strictly
                # inside is a structure-relative crossing.
                at_edge = (status == "domain"
                           or (status == "frontier"
                               and abs(wall - bound) <= 2 * tol))
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
                        # it is labeled deep rather than structure-frontier
                        # precisely because it is not known to be on the
                        # frontier. As a
                        # deep witness it is subject to theta separation, so it
                        # goes to deep_markers rather than the exempt marker list.
                        m = self._window_witness(oracle, var, others, bound, tol)
                        calls += 1
                        if m is not None:
                            deep_markers.append(m)
                    continue
                m = self._window_witness(oracle, var, others, bound, tol)
                calls += 1
                if m is not None:
                    markers.append(m)
                    marker_labels.append(
                        domain_label if at_edge else _STRUCTURE_FRONTIER)

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

        # Partial-face deep witnesses are separated exactly like harvested deep
        # witnesses: one within theta of a witness already collected carries
        # nothing new. Kept out of the marker merge below, which is exempt from
        # separation because a frontier marker's position is its information.
        dropped_partial = 0
        for marker in deep_markers:
            if any(all(abs(_value_of(marker, v) - _value_of(other, v))
                       < theta.of(v)
                       for v in box) for other in witnesses):
                dropped_partial += 1
                continue
            witnesses.append(marker)
            labels.append(_DEEP)
        if dropped_partial:
            printer.print_verbose(
                "[kappa_box] {} partial-face witness(es) within theta ({}) of "
                "an existing witness were dropped".format(
                    dropped_partial, ", ".join(f"{v.id}={float(theta.of(v))}"
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
        self._underlying = underlying
        self._printer = printer
        _th = common.get_value("time-horizon") if common.is_argument_in(
            "time-horizon") else "time-bound"
        self._time_horizon = float(tau_max) if str(_th) == "time-bound" else float(_th)
        self._config = config
        self._logger = logger
        self._tau_max = common.get_value("time-bound")

        # Every [gen] value the strategy reads is range-checked here, before
        # the first solver call: out-of-range values otherwise fail late (a
        # non-terminating bisection, a crash after a paid pivot search) or
        # silently (an unbounded solver call).
        validate_gen(config)

        logic = z3_logic(config)
        seed = resolve_seed(config, printer)
        # Which step-0 variables are grown, the effective IC-domain edge each
        # face searches toward, and whether reaching an edge is `ic-domain` or a
        # `projected-domain` per-axis projection. Resolved once, from the model's
        # initial condition and [gen] target-axes/target-bounds.
        axis_ids, ic_edges, domain_label = _ic_plan(config, model)
        ic_widths = {i: hi - lo for i, (lo, hi) in ic_edges.items() if hi > lo}
        printer.print_normal(
            "[kappa_box] IC domain ({}): {}".format(
                domain_label,
                ", ".join(f"{i}=[{float(lo)},{float(hi)}]"
                          for i, (lo, hi) in sorted(ic_edges.items()))
                or "declared ranges"))
        # theta: absolute by default, per-axis (fraction of the IC-domain width)
        # when [gen] epsilon-relative is set.
        theta = _Theta(gen_frac(config, "epsilon", "0.01"),
                       gen_frac(config, "epsilon-relative", "0") or None,
                       ic_widths)
        bisect_iters = gen_int(config, "bisect-iters")
        # 0 is a meaningful setting (face precision theta itself), so no `or`.
        bisect_iters = _BISECT_ITERS if bisect_iters is None else bisect_iters
        per_depth_boxes = _box_budget(config)  # per target depth; None -> exhaust
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
        self._undecided_candidates = 0
        self._pivot_giveup = None
        # A configuration still setting a removed key would otherwise get
        # silence, which reads as the key being in effect. Presence-checked
        # rather than parsed: several of these were boolean-shaped, and
        # int("true") would abort the run before the warning printed. Checked
        # on BOTH backends -- a stale key is stale regardless of the solver.
        for gone in ("word-pruning", "word-check-timeout",
                     "assume-monotone-flows", "block-class",
                     "two-step-pivot", "warm-start", "lattice-max"):
            if gen_present(config, gone):
                printer.print_normal(
                    f"warning: [gen] {gone} no longer exists and is "
                    "ignored")

        if underlying != "z3":
            printer.print_normal(
                "[kappa_box] word-rotate={}".format(
                    30 if gen_int(config, "word-rotate") is None
                    else (gen_int(config, "word-rotate") or "off")))
            pivot_timeout = gen_float(config, "pivot-timeout")
            if pivot_timeout is None:
                pivot_timeout = _DEFAULT_CANDIDATE_TIMEOUT
            pivot_budget = gen_float(config, "pivot-budget")
            if pivot_budget is None:
                pivot_budget = _DEFAULT_PIVOT_BUDGET
            # "per pivot search", not "per depth": the budget is enforced per
            # call of the candidate search, and a depth under k-ic = n runs up
            # to n+1 of them.
            printer.print_normal(
                f"[kappa_box] pivot budgets: pivot-timeout={pivot_timeout}s "
                f"per candidate, pivot-budget={pivot_budget}s per pivot search")

        # Reported on either backend, because it bounds every solver call the
        # strategy makes on either. Resolved through `query_timeout` rather
        # than read a second time here, so the banner and the oracles cannot
        # disagree about the words ("off"/"none") that disable the bound.
        bound = query_timeout(config)
        printer.print_normal(
            "[kappa_box] query-timeout={} per solver call "
            "([gen] query-timeout = 0 disables)".format(
                "off" if bound is None else f"{bound}s"))

        # The delta the backend answers under, echoed on the delta path
        # because it now reaches the binary: a run's frontiers, its pool's
        # recorded relaxation and its solver calls are all this one value, and
        # a configured value that fails to arrive is otherwise invisible.
        if underlying == "dreal":
            printer.print_normal(
                f"[kappa_box] backend precision="
                f"{float(backend_precision(config, underlying))} "
                f"per solver call ([dreal] precision)")

        encoder = Encoder(model, goal, prop_dict, delta, tau_max)
        pool: list[dict[Variable, Constant]] = []
        labels: list[str] = []
        first_depth = max_depth
        total_boxes = 0
        # Depths the run DECIDED, not merely targeted: a depth is settled only
        # by an exhaustion in which every candidate was refuted. scoped_verdict
        # is contracted on this set; passing the target set instead let a depth
        # that was skipped (or exhausted under a heuristic) ground a True.
        settled: set[int] = set()

        for depth in target_depths:
            blocks: list[Formula] = []  # per depth: independent region discovery
            boxes_here = 0
            # The witnesses admitted at this depth, and the scope of both
            # admission rules below. theta is the pool's minimum separation, and
            # a depth is where that has to hold: growth runs unmasked, so a
            # later box may regrow across an earlier one and place a lattice
            # centre arbitrarily close to a witness the earlier box contributed.
            # Across depths the opposite holds -- one initial condition
            # falsifying at several depths is the depth axis, not redundancy --
            # so neither rule reaches beyond the depth it is applied in.
            depth_pool: list[dict[Variable, Constant]] = []
            # depth_pool and pool are appended in lockstep from here, so index
            # i of the former is index depth_offset + i of the latter -- which
            # is what lets a marker relabel an entry an earlier box contributed.
            depth_offset = len(pool)
            # Both caveats on an exhaustion claim are scoped to a depth, since
            # the structure space and its blocks are -- and so is the metrics
            # line, which is printed under a per-depth label and previously
            # accumulated over the whole run.
            self._rotated_words = 0
            self._undecided_candidates = 0
            self._metrics = _Counter()
            while per_depth_boxes is None or boxes_here < per_depth_boxes:
                oracle, pivot, encoding = self._pivot_at(
                    encoder, depth, logic, seed, blocks
                )
                if pivot is None:
                    if getattr(self, "_last_pivot_verdict", None) == UNKNOWN:
                        self._any_unresolved = True
                        # Name the bound that actually bit. `pivot-timeout` and
                        # `pivot-budget` are read by the two-step search alone,
                        # so on the exact backend -- one query, no candidate
                        # loop -- neither is the knob: `query-timeout` is.
                        if self._pivot_giveup is None:
                            key, why = ("query-timeout",
                                        "the pivot query was left undecided")
                        else:
                            key, why = self._pivot_giveup
                        printer.print_normal(
                            f"[kappa_box] depth {depth}: pivot search UNRESOLVED "
                            f"({why}) -- the region is NOT proven empty; raise "
                            f"[gen] {key} to search further")
                    else:
                        # UNSAT: the structure space is exhausted. What that is
                        # worth depends on what left it. Two things remove
                        # candidates the oracle never refuted: word rotation
                        # blocks structures outright, and a candidate that
                        # expired undecided is blocked so the search can make
                        # progress, on no evidence about it. Under either,
                        # exhaustion means no further structure was reachable,
                        # not that none exists.
                        heuristic, remedies = [], []
                        if self._rotated_words:
                            heuristic.append(
                                f"{self._rotated_words} word(s) rotated off")
                            remedies.append("word-rotate = 0")
                        if self._undecided_candidates:
                            heuristic.append(
                                f"{self._undecided_candidates} candidate(s) "
                                "blocked undecided")
                            remedies.append("a larger [gen] pivot-timeout")
                        scope = ("no counterexample at this depth"
                                 if not blocks else
                                 f"no counterexample outside the "
                                 f"{boxes_here} box(es) already found")
                        if heuristic:
                            # An exhaustion a heuristic took part in does not
                            # settle the depth: candidates were removed that no
                            # oracle refuted (Def. of the pruning policy), so
                            # the verdict must not read this depth as decided.
                            self._any_unresolved = True
                            printer.print_normal(
                                "[kappa_box] depth {}: structure space exhausted, "
                                "but not every candidate was refuted ({}) -- "
                                "absence is NOT established; re-run with {} to "
                                "make it conclusive".format(
                                    depth, ", ".join(heuristic),
                                    " and ".join(remedies)))
                        else:
                            settled.add(depth)
                            printer.print_normal(
                                f"[kappa_box] depth {depth}: structure space "
                                f"exhausted -- {scope} (absence, established by "
                                f"exhaustion)")
                    encoder.reset()
                    break

                oracle.assert_(_skeleton_fix(pivot))  # pin path + Boolean skeleton
                k_witness = gen_int(config, "k-witness")
                witnesses, box_labels, box = self._grow_box(
                    oracle, pivot, encoding, theta, bisect_iters, depth,
                    _DEFAULT_K_WITNESS if k_witness is None else k_witness,
                    printer, axis_ids, ic_edges, domain_label
                )
                boxes_here += 1
                total_boxes += 1
                if total_boxes == 1:
                    first_depth = depth

                kept = 0
                collapsed_across = 0
                merged_across = 0
                # Earlier boxes at this depth only. _grow_box already applies
                # both admission rules within a box, so checking against a
                # snapshot rather than the live list leaves the single-box case
                # exactly as it was and adds only the cross-box half.
                prior_here = list(depth_pool)
                tol_of = self._tol_of
                for witness, label in zip(witnesses, box_labels):
                    if label == _DEEP:
                        # Separation: a deep witness within theta of one an
                        # earlier box at this depth contributed carries nothing
                        # the pool does not already have.
                        if _within_theta(witness, prior_here, box.keys(), theta):
                            collapsed_across += 1
                            continue
                        if thin > 0 and _too_close(
                            witness, depth_pool, box.keys(), thin
                        ):
                            continue
                    else:
                        # One entry per initial condition: a marker landing on a
                        # witness an earlier box already contributed relabels it
                        # rather than adding a second entry for the same point.
                        # Markers are exempt from separation, not from this --
                        # two boxes at one depth can converge on the same face,
                        # and without the check the pool carries the same
                        # initial condition twice, at distance zero on every
                        # axis the metrics measure.
                        hit = _coincides_with(
                            witness, prior_here, box.keys(), tol_of)
                        if hit is not None:
                            merged_across += 1
                            at = depth_offset + hit
                            labels[at] = _prefer_label(labels[at], label)
                            continue
                    depth_pool.append(witness)
                    pool.append(witness)
                    labels.append(label)
                    kept += 1
                if collapsed_across:
                    printer.print_verbose(
                        f"[kappa_box] {collapsed_across} deep witness(es) of "
                        f"this box landed within theta of a witness an earlier "
                        f"box at depth {depth} contributed and were dropped")
                if merged_across:
                    printer.print_verbose(
                        f"[kappa_box] {merged_across} marker(s) coincided with a "
                        f"witness an earlier box at depth {depth} contributed "
                        f"and were merged into it")

                # Block the envelope: the box extended by theta on every face.
                # This is a coverage heuristic, not an exact exclusion of a
                # falsifying region -- the envelope is a search-and-harvest
                # rectangle, not a subset of the falsifying set, so it can also
                # exclude falsifying initial conditions it bridges. A re-pivot
                # must escape the envelope on some axis and is not guaranteed to
                # recover a disconnected component as a separate box.
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
            f"{labels.count(_DEEP)} deep, "
            f"{labels.count(_STRUCTURE_FRONTIER)} structure-frontier, "
            f"{labels.count(_IC_DOMAIN)} ic-domain, "
            f"{labels.count(_PROJECTED_DOMAIN)} projected-domain"
        )

        result, note = _verdict(pool, getattr(self, "_any_unresolved", False),
                                settled, max_depth)
        if note:
            printer.print_normal(note)
        return result, 0.0, first_depth, pool