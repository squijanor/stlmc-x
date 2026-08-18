"""Discrete path enumeration (kappa_path).

Enumerates structurally distinct counterexample paths by blocking each found
location word and re-solving. The traversal visits a set of target depths --
``[gen] depths`` as a slash-separated list (e.g. ``"8/9/10/11"``), or every depth
0..N by default -- and at each target depth it solves the falsification encoding,
records the counterexample, excludes a Hamming-ball of radius r around its
location word, and re-solves until that depth is exhausted or its per-depth
budget is reached.

The budget ``[gen] k-paths`` is per target depth, not global: each targeted depth
contributes up to that many paths, so the pool is depth-uniform (every targeted
depth is filled to the same density) rather than front-loaded onto the shallowest
depths. Absent, a depth is enumerated to exhaustion.

The blocking predicate is a Boolean clause over the per-step mode variables:
radius 0 excludes exactly one location word; radius r excludes every word within
Hamming distance r of it. What a depth contributes is the set of *falsifying*
words at that depth, a subset of the path lattice that depends on the property as
much as on the model. A radius above 0 is a **coarsening**: it excludes words
that were never exhibited as counterexamples, so an UNSAT verdict under it means
no counterexample outside the union of the balls, not that the path lattice is
exhausted. Each depth reports which of the two it reached. The verdict itself is
unaffected -- an empty pool means no counterexample was found at any depth, hence
no block was ever asserted, hence every UNSAT was decided on the bare encoding.

At depth n a location word has n+1 positions, so a radius above n encodes an
unsatisfiable block and would report the depth as exhausted after a single path.
The radius is capped at one below the word length -- the coarsest ball that
still admits a word -- and the cap is reported.

The pool carries counterexample assignments only: the auxiliary indicator
variables a radius-r block introduces are part of the query, not of any
counterexample, and are removed before an assignment is pooled.

Every solver call is timed against the number of blocking clauses already in
the query, since what re-solving costs under accumulated blocks cannot be
recovered from a run's total afterwards. Per call the cost is reported
verbosely; per depth it is reported always.

Reproducibility: the generation seed is -gen-seed if given (>= 0), otherwise the
PYTHONHASHSEED value; one of the two must be present. For the z3 backend the
seed is passed as its random_seed. Constraint ordering is fixed by
PYTHONHASHSEED, which must be set for a run to be reproducible; z3's random_seed
alone does not pin it.
"""

from __future__ import annotations

import time
from functools import reduce
from typing import NamedTuple

from ..constraints.constraints import (
    Add,
    And,
    Constant,
    Eq,
    Formula,
    Geq,
    Implies,
    Int,
    IntVal,
    Neq,
    Or,
    Variable,
)
from ..objects.algorithm import Algorithm
from .common import (
    MODE_RE,
    backend_precision,
    gen_depths,
    gen_int,
    resolve_seed,
    scoped_verdict,
    validate_gen,
    warn_unpinned_hashseed,
    z3_logic,
)
from .encode import Encoder
from .oracle import SAT, UNKNOWN, make_oracle


class PathBlock(NamedTuple):
    """A blocking clause together with what a caller must know about it.

    ``radius`` is the radius the clause actually encodes, which is the requested
    one capped at the word length. ``indicators`` are the auxiliary variables the
    clause introduces; they belong to the query and never to a counterexample.
    """

    clause: Formula
    radius: int
    indicators: tuple[Variable, ...]


def _location_word(assn: dict[Variable, Constant]) -> list[tuple[Variable, Constant]]:
    """The (currentMode_k, value) pairs of an assignment, ordered by step k."""
    steps: list[tuple[int, Variable, Constant]] = []
    for var, val in assn.items():
        m = MODE_RE.match(var.id)
        if m is not None:
            steps.append((int(m.group(1)), var, val))
    steps.sort(key=lambda t: t[0])
    return [(var, val) for _, var, val in steps]


def _canon_mode_val(val: Constant) -> Constant:
    """Canonical constant for an integral mode value: ``1.000000`` -> ``1``.

    The delta backend formats every model value as a fixed-point decimal; the
    block compares against it, so normalise to the integer spelling both
    solvers parse canonically. A non-numeric or non-integral value is returned
    unchanged -- detecting those is the caller's job (see ``_off_lattice``)."""
    try:
        f = float(val.value)
    except (TypeError, ValueError):
        return val
    if f.is_integer():
        return type(val)(str(int(f)))
    return val


def _off_lattice(word: list[tuple[Variable, Constant]]) -> list[str]:
    """The word positions whose mode value is not an integer, rendered for a
    log line; empty when the word is sound.

    A delta backend reports each model value as the midpoint of an interval.
    If that midpoint is not integral for a *mode* variable, the word it spells
    is not the word the solver satisfied, and a block built from it misses:
    at radius 0 the solver can return the same model forever, at radius >= 1
    the ball is centred off-word. Such a model cannot be blocked or pooled."""
    bad = []
    for var, val in word:
        try:
            f = float(val.value)
        except (TypeError, ValueError):
            bad.append(f"{var.id}={val.value}")
            continue
        if not f.is_integer():
            bad.append(f"{var.id}={val.value}")
    return bad


def _missing_modes(word: list[tuple[Variable, Constant]], depth: int) -> list[str]:
    """The mode variables ``currentMode_k`` for k in 0..depth that the model
    omits, rendered for a log line; empty when the word has its full arity.

    A depth-n word has one position per step, k = 0..n. A backend can return a
    model that pins only some of them, and a block built from a short word
    misses: a radius-0 clause ``OR_k (mode_k != w_k)`` over the positions
    present excludes every word that agrees on them -- a disjunction coarser
    than the single word the solver satisfied -- so it drops falsifying words
    never exhibited, exactly as a radius-r ball does. Such a model cannot be
    blocked or pooled."""
    present = {int(MODE_RE.match(var.id).group(1)) for var, _ in word}
    return [f"currentMode_{k}" for k in range(depth + 1) if k not in present]


def block_radius(assn: dict[Variable, Constant], radius: int, uid: int) -> PathBlock:
    """Clause excluding every location word within Hamming distance ``radius`` of
    ``assn``'s word. ``uid`` makes the radius>=1 indicator variables unique.

    radius 0: ``OR_k (mode_k != w_k)`` -- excludes exactly the one word.
    radius r: ``sum_k [mode_k != w_k] >= r+1`` via 0/1 indicators.

    A word of L positions has at most L differing positions, so ``sum >= r+1`` is
    unsatisfiable for r >= L and blocks every word rather than a ball. The radius
    is therefore capped at L-1, the coarsest ball that still admits a word, and
    the cap is returned so the caller can report it. A negative radius is 0.
    """
    word = _location_word(assn)
    if not word:
        raise RuntimeError(
            "no currentMode_k variables in assignment; model has no discrete "
            "modes to enumerate"
        )

    word = [(var, _canon_mode_val(val)) for var, val in word]
    effective = max(0, min(radius, len(word) - 1))
    if effective == 0:
        return PathBlock(Or([Neq(var, val) for var, val in word]), 0, ())

    consts: list[Formula] = []
    indicators: list[Variable] = []
    for k, (var, val) in enumerate(word):
        ind = Int(f"hb${uid}${k}")
        indicators.append(ind)
        consts.append(Or([Eq(ind, IntVal("0")), Eq(ind, IntVal("1"))]))
        consts.append(Implies(Neq(var, val), Eq(ind, IntVal("1"))))
        consts.append(Implies(Eq(var, val), Eq(ind, IntVal("0"))))
    consts.append(Geq(reduce(Add, indicators), IntVal(str(effective + 1))))
    return PathBlock(And(consts), effective, tuple(indicators))


def _verdict(pool, any_unresolved, visited, max_depth):
    """kappa_path's wording for the shared depth-scoping rule."""
    return scoped_verdict(pool, any_unresolved, visited, max_depth,
                          tag="kappa_path",
                          nothing_found="no counterexample found",
                          unresolved_source="at least one depth")


def _exhaustion_note(depth: int, found: int, coarsened: bool) -> str:
    """What an UNSAT at ``depth`` establishes, given what was blocked to get it.

    Three different claims share one verdict, and collapsing them turns a
    coarsening heuristic or a budget into a statement about the model. Note what
    even the strongest of the three says: the blocks exclude the words already
    exhibited, so an UNSAT means no *falsifying* word remains, not that the depth's
    path lattice has been enumerated. The lattice is an upper bound on the pool
    and the gap between them is a property of the goal, not only of the model.
    """
    if found == 0:
        return (f"[kappa_path] depth {depth}: no counterexample at this depth "
                "(absence, established by exhaustion)")
    if coarsened:
        return (f"[kappa_path] depth {depth}: search exhausted after {found} "
                "path(s), but radius-r blocking was active -- a radius-r block "
                "also excludes words never exhibited, so absence of further "
                "paths is NOT established; re-run with radius = 0 to make it "
                "conclusive")
    return (f"[kappa_path] depth {depth}: falsifying words exhausted after "
            f"{found} path(s) -- no location word outside the pool falsifies at "
            "this depth")


class DiscretePathEnum(Algorithm):
    """kappa_path: enumerate distinct paths over the target depths under a budget."""

    def __init__(self) -> None:
        self.debug_name = ""

    def set_debug(self, msg: str) -> None:
        self.debug_name = msg

    def run(self, model, goal, prop_dict, config, solver, logger, printer):
        common = config.get_section("common")
        max_depth = int(common.get_value("bound"))
        tau_max = float(common.get_value("time-bound"))
        delta = float(common.get_value("threshold"))
        underlying = common.get_value("solver")

        # Fail fast on malformed [gen] values, before any solver work, so a
        # typo dies as a configuration error naming the key rather than as a
        # bare ValueError somewhere below.
        validate_gen(config)

        per_depth = gen_int(config, "k-paths")  # per target depth; None -> exhaust
        target_depths = gen_depths(config, max_depth)
        radius = gen_int(config, "radius")
        logic = z3_logic(config)
        seed = resolve_seed(config, printer)

        # Fold out-of-range values here, each with a notice, so the banner below
        # states what the run actually uses rather than what was written.
        if radius is not None and radius < 0:
            printer.print_normal(
                f"[kappa_path] [gen] radius {radius} is negative; using 0")
        radius = max(0, radius or 0)
        if per_depth is not None and per_depth < 0:
            printer.print_normal(
                f"[kappa_path] [gen] k-paths {per_depth} is negative; using 0 "
                "(every depth is visited, no query is posed, nothing is decided)")
            per_depth = 0

        # The resolved parameters, not the configured ones: depths are clamped to
        # 0..bound, negatives fold to 0 above, and an unrecognised key reads as
        # absent, so a log that does not state them cannot be checked against
        # what was intended.
        printer.print_normal(
            "[kappa_path] radius={}, k-paths={}, target depths={}".format(
                radius,
                "exhaust" if per_depth is None else per_depth,
                "/".join(str(d) for d in target_depths) or "none"))
        if not target_depths:
            printer.print_normal(
                f"[kappa_path] [gen] depths selected no depth in 0..{max_depth}, "
                "so no depth is examined and nothing can be concluded")

        warn_unpinned_hashseed(printer)

        # z3 uses logic/seed; dreal uses config/logger/time-bound.
        def new_oracle():
            return make_oracle(
                underlying,
                logic=logic,
                seed=seed,
                config=config,
                logger=logger,
                time_bound=tau_max,
            )

        # The delta the backend answers under, echoed on the delta path
        # because it now reaches the binary: a run's frontiers, its pool's
        # recorded relaxation and its solver calls are all this one value, and
        # a configured value that fails to arrive is otherwise invisible.
        if underlying == "dreal":
            printer.print_normal(
                f"[kappa_path] backend precision="
                f"{float(backend_precision(config, underlying))} "
                f"per solver call ([dreal] precision)")

        encoder = Encoder(model, goal, prop_dict, delta, tau_max)
        pool: list[dict[Variable, Constant]] = []
        # Auxiliary variables introduced by the blocks. They appear in every
        # model the solver returns afterwards, so they are stripped before an
        # assignment is pooled.
        auxiliary: set[Variable] = set()
        block_id = 0
        unresolved = False
        first_depth = max_depth
        # Depths settled by an UNSAT, which is not the same as the depths
        # targeted: a budget can stop a depth before it is decided, and a budget
        # of zero stops every depth before a single query is posed.
        decided: list[int] = []

        for depth in target_depths:
            encoding = encoder.encode_at(depth)
            oracle = new_oracle()
            oracle.assert_(encoding.consts)

            found = 0
            coarsened = False   # a radius>=1 block was asserted at this depth
            capped = False      # the cap has been reported at this depth
            blocks = 0          # blocking clauses in the query at this depth
            calls: list[float] = []   # seconds taken by each solver call
            while per_depth is None or found < per_depth:
                started = time.perf_counter()
                verdict = oracle.check()
                calls.append(time.perf_counter() - started)
                # Cost against the number of blocks already asserted. A run's
                # total does not decompose into these afterwards, and on a
                # backend that re-solves the whole stack per call the two are
                # the quantities a re-solve strategy is judged on.
                printer.print_verbose(
                    f"[kappa_path] depth {depth}: call {len(calls)} over "
                    f"{blocks} block(s): {verdict} in {calls[-1]:.3f}s")
                if verdict != SAT:
                    # UNSAT and UNKNOWN are different results. UNKNOWN means the
                    # backend gave up, which is NOT evidence of absence;
                    # collapsing the two lets an unresolved search be reported as
                    # "no counterexample". What an UNSAT establishes depends on
                    # what was blocked to reach it (see _exhaustion_note).
                    if verdict == UNKNOWN:
                        unresolved = True
                        why = oracle.unknown_reason() or "backend did not decide"
                        printer.print_normal(
                            f"[kappa_path] depth {depth}: search UNRESOLVED "
                            f"({why}) -- the path space is NOT proven exhausted")
                    else:
                        decided.append(depth)
                        printer.print_normal(
                            _exhaustion_note(depth, found, coarsened))
                    break

                assn = oracle.model()
                # Guard the word before pooling or blocking. A model cannot be
                # excluded when its mode values do not spell an integral word,
                # or when it omits a step's mode variable so the word is short
                # of its depth+1 arity: either way the block misses the word the
                # solver satisfied -- a radius-0 clause over the positions
                # present excludes every word that agrees on them, not the one
                # word -- and the next call may return the same model. A mode
                # word is also exactly what this strategy pools. So the model is
                # dropped, the depth stops UNRESOLVED, and the run continues on
                # the remaining depths.
                word = _location_word(assn)
                bad = _off_lattice(word)
                missing = _missing_modes(word, depth)
                if not word or bad or missing:
                    unresolved = True
                    if not word:
                        what = "no currentMode_k variables in the model"
                    elif bad:
                        what = ", ".join(bad)
                    else:
                        what = "missing " + ", ".join(missing)
                    printer.print_normal(
                        f"[kappa_path] depth {depth}: cannot block this model "
                        f"({what}); the model is not pooled and the depth "
                        "stops UNRESOLVED -- the path space is NOT proven "
                        "exhausted")
                    break

                if not pool:
                    first_depth = depth
                pool.append({v: c for v, c in assn.items() if v not in auxiliary})

                block = block_radius(assn, radius, block_id)
                if block.radius < radius and not capped:
                    capped = True
                    printer.print_normal(
                        f"[kappa_path] depth {depth}: [gen] radius {radius} "
                        f"exceeds the {depth + 1} positions of a word at this "
                        f"depth; capped to {block.radius}")
                coarsened = coarsened or block.radius >= 1
                auxiliary.update(block.indicators)
                oracle.assert_(block.clause)
                blocks += 1
                block_id += 1
                found += 1
                printer.print_verbose(
                    f"[kappa_path] depth {depth}: {found} path(s) here, "
                    f"{len(pool)} total"
                )
            else:
                printer.print_normal(
                    f"[kappa_path] depth {depth}: stopped at the [gen] k-paths "
                    f"budget after {found} path(s) -- the path space at this "
                    "depth is NOT known to be exhausted")

            # A depth that posed no query (a zero budget) has nothing to report.
            if calls:
                printer.print_normal(
                    f"[kappa_path] depth {depth}: {len(calls)} solver call(s), "
                    f"{sum(calls):.2f}s total, slowest {max(calls):.2f}s")

            encoder.reset()

        # The verdict speaks about `decided`, not about what was targeted. A
        # coarsening radius cannot corrupt it either: an empty pool means no
        # counterexample was found at any depth, so no block was ever asserted
        # and every UNSAT was decided on the bare encoding.
        result, note = _verdict(pool, unresolved, decided, max_depth)
        if note:
            printer.print_normal(note)
        # The driver prints this as the bound a counterexample was found at, so
        # it is the shallowest depth that yielded one, not the bound searched.
        return result, 0.0, first_depth, pool