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

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    Not,
    Or,
    Variable,
)
from ..objects.algorithm import Algorithm
from .common import (
    MODE_RE,
    backend_precision,
    gen_depths,
    gen_float,
    gen_int,
    ode_settings,
    resolve_seed,
    scoped_verdict,
    validate_gen,
    warn_unpinned_hashseed,
    z3_logic,
)
from .encode import Encoder
from .feasibility import LinearWordFeasibilityFilter
from .oracle import SAT, UNKNOWN, UNSAT, fix_modes, make_oracle, query_timeout
from .reduced import ReducedPivotSearch

# On the delta backend a depth is decided by many cheap verifications (the
# scenario over-approximation proposes far more structures than there are
# falsifying words). Logging every refutation verbosely buries the ones that
# matter, so the per-candidate line is throttled: a satisfiable or undecided
# result always prints, and refutations print at this cadence. [gen] log-every
# overrides it.
_LOG_EVERY_DEFAULT = 25

# Floor for a budget-clamped verification call, matching the box strategy's
# per-candidate floor: the last call under a tight [gen] pivot-budget still gets
# a real, if short, attempt rather than a zero-length one.
_MIN_CALL_BUDGET = 0.1

# Outstanding-check window as a multiple of the worker count on the parallel
# delta path: the proposing thread keeps up to this many dReal checks in flight
# so a slow in-order commit does not idle the pool.
_WINDOW_FACTOR = 2


class _LiveVerifiers:
    """The verifiers whose dReal check is in flight, so the proposing thread can
    kill them when it stops a depth or ends the run rather than leave the pool
    blocked until each per-call budget expires (Future.cancel does not stop a
    running check). Thread-safe: workers add and discard, the proposing thread
    terminates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: set = set()

    def add(self, verifier) -> None:
        with self._lock:
            self._live.add(verifier)

    def discard(self, verifier) -> None:
        with self._lock:
            self._live.discard(verifier)

    def terminate_all(self) -> None:
        with self._lock:
            live = list(self._live)
        for verifier in live:
            try:
                verifier.terminate()
            except Exception:
                pass


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
    return scoped_verdict(
        pool,
        any_unresolved,
        visited,
        max_depth,
        tag="kappa_path",
        nothing_found="no counterexample found",
        unresolved_source="at least one depth",
    )


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
        return (
            f"[kappa_path] depth {depth}: no counterexample at this depth "
            "(absence, established by exhaustion)"
        )
    if coarsened:
        return (
            f"[kappa_path] depth {depth}: search exhausted after {found} "
            "path(s), but radius-r blocking was active -- a radius-r block "
            "also excludes words never exhibited, so absence of further "
            "paths is NOT established; re-run with radius = 0 to make it "
            "conclusive"
        )
    return (
        f"[kappa_path] depth {depth}: falsifying words exhausted after "
        f"{found} path(s) -- no location word outside the pool falsifies at "
        "this depth"
    )


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
        _th = (
            common.get_value("time-horizon")
            if common.is_argument_in("time-horizon")
            else "time-bound"
        )
        self._time_horizon = tau_max if str(_th) == "time-bound" else float(_th)

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
                f"[kappa_path] [gen] radius {radius} is negative; using 0"
            )
        radius = max(0, radius or 0)
        if per_depth is not None and per_depth < 0:
            printer.print_normal(
                f"[kappa_path] [gen] k-paths {per_depth} is negative; using 0 "
                "(every depth is visited, no query is posed, nothing is decided)"
            )
            per_depth = 0

        # The resolved parameters, not the configured ones: depths are clamped to
        # 0..bound, negatives fold to 0 above, and an unrecognised key reads as
        # absent, so a log that does not state them cannot be checked against
        # what was intended.
        printer.print_normal(
            "[kappa_path] radius={}, k-paths={}, target depths={}".format(
                radius,
                "exhaust" if per_depth is None else per_depth,
                "/".join(str(d) for d in target_depths) or "none",
            )
        )
        if not target_depths:
            printer.print_normal(
                f"[kappa_path] [gen] depths selected no depth in 0..{max_depth}, "
                "so no depth is examined and nothing can be concluded"
            )

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
                f"per solver call ([dreal] precision)"
            )
            order, step = ode_settings(config)
            printer.print_normal(
                f"[kappa_path] dReal ODE integration: "
                f"order={'auto' if order is None else order}, "
                f"step={'auto' if step is None else step} "
                f"([dreal] ode-order / ode-step)"
            )

        encoder = Encoder(model, goal, prop_dict, delta, tau_max)

        # The delta backend takes a two-step reduced query, not the monolithic
        # encoding.consts. Handing dReal consts keeps every quantified subformula
        # and leaves the location word un-pinned, so dReal integrates the whole
        # path lattice per solve, and each re-solve over the accumulated blocks
        # pays that cost again. Instead this enumerates falsifying structures on
        # the reconstructed reduced query -- only the core-selected property
        # forall_t plus the full model execution (generation.reduced) -- and
        # verifies each on dReal. The exact backend is unaffected and keeps the
        # single incremental query below.
        if underlying == "dreal":
            return self._run_reduced(
                encoder,
                target_depths,
                per_depth,
                radius,
                seed,
                logic,
                config,
                logger,
                printer,
                max_depth,
                tau_max,
            )

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
            coarsened = False  # a radius>=1 block was asserted at this depth
            capped = False  # the cap has been reported at this depth
            blocks = 0  # blocking clauses in the query at this depth
            calls: list[float] = []  # seconds taken by each solver call
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
                    f"{blocks} block(s): {verdict} in {calls[-1]:.3f}s"
                )
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
                            f"({why}) -- the path space is NOT proven exhausted"
                        )
                    else:
                        decided.append(depth)
                        printer.print_normal(_exhaustion_note(depth, found, coarsened))
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
                        "exhausted"
                    )
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
                        f"depth; capped to {block.radius}"
                    )
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
                    "depth is NOT known to be exhausted"
                )

            # A depth that posed no query (a zero budget) has nothing to report.
            if calls:
                printer.print_normal(
                    f"[kappa_path] depth {depth}: {len(calls)} solver call(s), "
                    f"{sum(calls):.2f}s total, slowest {max(calls):.2f}s"
                )

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

    def _path_verify_workers(self, config) -> int:
        """Candidate-verification pool size for the delta path. A pool only when
        ``[common]`` parallel is enabled. A dReal ODE check has a large memory
        footprint, so running one per core oversubscribes memory and every solve
        thrashes.

        ``[common]`` parallel-core is read as a request: a value from 1 up to the
        core count is honored as-is, at the user's own risk (it may exceed the
        memory-safe default). A value above the core count -- the generous
        configuration default -- resolves to a memory-safe fraction of the cores:
        a quarter when the count is a multiple of four, half otherwise. Disabled,
        absent or unreadable gives 1, which verifies one at a time in the calling
        thread."""
        try:
            common = config.get_section("common")
            if str(common.get_value("parallel")) != "true":
                return 1
            core = int(common.get_value("parallel-core"))
        except (AttributeError, KeyError, TypeError, ValueError):
            return 1
        cpus = os.cpu_count() or core
        if 1 <= core <= cpus:
            return core
        return max(1, cpus // 4 if cpus % 4 == 0 else cpus // 2)

    def _run_reduced(
        self,
        encoder,
        target_depths,
        per_depth,
        radius,
        seed,
        logic,
        config,
        logger,
        printer,
        max_depth,
        tau_max,
    ):
        """Delta path: enumerate falsifying words on the base checker's reduced
        query and verify each on dReal.

        For each target depth the reduced search proposes falsifying structures
        over the abstracted skeleton: each yields a reduced dReal query
        (``total_const``) carrying only the core-selected property forall_t plus
        the full model execution, and its z3 twin ``path_const``. The structure's
        location word is pinned onto ``total_const`` -- the reduction may leave a
        mode free, so without the pin a delta-sat witness could spell a sibling
        word -- and dReal decides it:

        * sat   -- a genuine delta-falsifier: the witness is pooled and the word
                   is excluded by a radius block, so later structures move to a
                   different word (word-level de-duplication).
        * unsat -- skeleton-feasible but not delta-realizable: a spurious pairing.
                   Only ``path_const AND word`` is excluded, not ``path_const``
                   alone, so a sibling word that shares this reduced path stays
                   available.

        The per-candidate dReal check is the expensive step and each check is an
        independent subprocess, so when ``[common]`` parallel is enabled they are
        fanned across a worker pool while proposal, the linear feasibility screen
        and block bookkeeping stay on the calling thread (all z3). Candidates are
        committed in proposal order. Every result is sound. A depth that pools
        nothing (a safety property, the exhaustion-dominated case) is decided
        identically to the serial reference and only faster, since every block is
        then the same regardless of a verdict. A depth that pools a word commits
        in fan-out order rather than serial proposal order, so its greedy
        radius-separated pool may differ from the serial one while remaining a
        valid pool: the verdict is the same, the radius separation is enforced at
        commit so a wider pool cannot pool a word inside an earlier word's ball,
        and -- when the depth runs to natural exhaustion (no ``[gen] k-paths`` cap
        and no binding ``[gen] pivot-budget``) -- at radius 0 the word set and
        count match serial. Under a finite ``k-paths`` cap even the radius-0 pool
        can be a different valid subset, because speculation after a SAT changes
        which later word reaches the cap. The verdict match likewise holds only
        absent a binding ``pivot-budget``: under one the pool advances faster, so
        this run may reach a counterexample or exhaust a depth within the budget
        where the serial run reports ``Unknown`` -- both sound, not identical.

        A search ``unknown`` (scenario solver or reduced-query minimizer), a
        verification ``unknown``, or hitting the ``[gen] pivot-budget`` elapsed
        bound leaves the depth unresolved without corrupting the verdict, so an
        honest ``Unknown`` is reported rather than relying on an external timeout.

        ``_make_reduced_search`` is a seam so a test can substitute a scripted
        search. The word block, the radius cap, the depth-scoping verdict and the
        exhaustion wording are the same as the exact path.
        """
        acc = _PathReducedAcc(max_depth)
        feasibility = LinearWordFeasibilityFilter(
            encoder.model, tau_max, getattr(self, "_time_horizon", None)
        )

        qsec = query_timeout(config)
        timeout_ms = None if qsec is None else max(1, int(qsec * 1000))
        every = gen_int(config, "log-every")
        every = _LOG_EVERY_DEFAULT if every is None or every < 1 else every
        # A per-depth elapsed budget. Absent, a depth runs until its skeleton
        # space is exhausted -- sound, but the over-approximation can propose far
        # more structures than there are words, so a bound turns a runaway depth
        # into an honest Unknown instead of leaving it to the external timeout.
        depth_budget = gen_float(config, "pivot-budget")
        workers = self._path_verify_workers(config)

        # One run-scoped pool for the whole discovery, released under try/finally
        # so an exception cannot leak live threads. Proposal, the linear screen
        # and block bookkeeping run on this thread; the workers only run dReal.
        # ``live`` tracks the in-flight checks so they are killed at shutdown
        # rather than waited out.
        pool_exec = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        live = _LiveVerifiers()
        try:
            for depth in target_depths:
                components = encoder.enumerate_components_at(depth)
                search = self._make_reduced_search(
                    components, encoder.model, seed, timeout_ms
                )
                if pool_exec is None:
                    self._reduced_depth_serial(
                        acc,
                        depth,
                        search,
                        per_depth,
                        radius,
                        feasibility,
                        qsec,
                        every,
                        depth_budget,
                        config,
                        logger,
                        seed,
                        logic,
                        tau_max,
                        printer,
                    )
                else:
                    self._reduced_depth_parallel(
                        acc,
                        depth,
                        search,
                        per_depth,
                        radius,
                        feasibility,
                        qsec,
                        every,
                        depth_budget,
                        config,
                        logger,
                        seed,
                        logic,
                        tau_max,
                        printer,
                        pool_exec,
                        workers,
                        live,
                    )
                encoder.reset()
        finally:
            if pool_exec is not None:
                # Kill any check still running before releasing the pool, so the
                # process is not held open until the per-call budgets expire.
                live.terminate_all()
                pool_exec.shutdown(wait=False, cancel_futures=True)

        printer.print_normal(
            f"[kappa_path] pooled {len(acc.pool)} counterexample word(s) over "
            f"{len(target_depths)} target depth(s)"
        )

        result, note = _verdict(acc.pool, acc.unresolved, acc.decided, max_depth)
        if note:
            printer.print_normal(note)
        return result, 0.0, acc.first_depth, acc.pool

    def _reduced_depth_serial(
        self,
        acc,
        depth,
        search,
        per_depth,
        radius,
        feasibility,
        qsec,
        every,
        depth_budget,
        config,
        logger,
        seed,
        logic,
        tau_max,
        printer,
    ):
        """One target depth, one candidate at a time (parallel-core = 1 or
        parallel off)."""
        dctx = _PathDepthAcc(depth, radius, per_depth)
        candidates = 0
        calls: list[float] = []
        depth_start = time.perf_counter()
        while per_depth is None or dctx.found < per_depth:
            if (
                depth_budget is not None
                and time.perf_counter() - depth_start > depth_budget
            ):
                acc.unresolved = True
                dctx.undecided = True
                printer.print_normal(
                    f"[kappa_path] depth {depth}: stopped at the [gen] "
                    f"pivot-budget ({depth_budget}s) after {candidates} "
                    "candidate structure(s) -- the path space is NOT proven "
                    "exhausted"
                )
                break

            res = search.propose()
            if res is None:
                self._reduced_exhaustion(
                    acc, dctx, depth, search.last_verdict(), printer
                )
                break

            candidates += 1
            raw_word = _location_word(res[2])
            guard = _guard_reason(raw_word, depth)
            if guard is not None:
                acc.unresolved = True
                printer.print_normal(
                    f"[kappa_path] depth {depth}: cannot block this model "
                    f"({guard}); the model is not pooled and the depth stops "
                    "UNRESOLVED -- the path space is NOT proven exhausted"
                )
                break

            cand = _reduced_candidate(res, depth)
            if feasibility.word_is_infeasible(depth, cand.mode_seq):
                if candidates == 1 or candidates % every == 0:
                    printer.print_verbose(
                        f"[kappa_path] depth {depth}: candidate {candidates} "
                        f"(word {cand.word_str}) infeasible on the linear "
                        "timeline -- skipped"
                    )
                search.add_block(Not(cand.word_pin))
                continue

            verifier = make_oracle(
                underlying="dreal",
                logic=logic,
                seed=seed,
                config=config,
                logger=logger,
                time_bound=tau_max,
            )
            verifier.assert_(cand.total_const)
            verifier.assert_(cand.word_pin)
            if depth_budget is not None:
                remaining = depth_budget - (time.perf_counter() - depth_start)
                call_budget = max(remaining, _MIN_CALL_BUDGET)
                if qsec is not None:
                    call_budget = min(call_budget, qsec)
                verifier.set_budget(call_budget)
            started = time.perf_counter()
            verdict = verifier.check()
            calls.append(time.perf_counter() - started)
            model = dict(verifier.model()) if verdict == SAT else None
            why = verifier.unknown_reason() if verdict == UNKNOWN else None
            _kind, block = _reduced_commit(
                acc,
                dctx,
                cand,
                verdict,
                model,
                why,
                candidates,
                calls[-1],
                every,
                printer,
            )
            search.add_block(block)
        else:
            printer.print_normal(
                f"[kappa_path] depth {depth}: stopped at the [gen] k-paths "
                f"budget after {dctx.found} word(s) -- the path space at this "
                "depth is NOT known to be exhausted"
            )

        if calls:
            printer.print_normal(
                f"[kappa_path] depth {depth}: pooled {dctx.found} word(s) from "
                f"{candidates} candidate structure(s) over {len(calls)} dReal "
                f"call(s), {sum(calls):.2f}s total, slowest {max(calls):.2f}s"
            )

    def _reduced_depth_parallel(
        self,
        acc,
        depth,
        search,
        per_depth,
        radius,
        feasibility,
        qsec,
        every,
        depth_budget,
        config,
        logger,
        seed,
        logic,
        tau_max,
        printer,
        pool_exec,
        workers,
        live,
    ):
        """One target depth with the per-candidate dReal checks fanned across the
        pool. Proposal, the linear screen and block bookkeeping stay on this
        thread; feasible candidates are proposed ahead under a provisional pair
        block (which advances the search and is the block a refuted candidate
        keeps), and their checks run on the pool. Results commit in proposal
        order: a pooled word adds its radius block, and the radius separation is
        re-checked at commit so a candidate proposed before an earlier word's
        block landed cannot pool a word inside that word's ball."""
        dctx = _PathDepthAcc(depth, radius, per_depth)
        candidates = 0
        calls: list[float] = []
        depth_start = time.perf_counter()
        deadline = None if depth_budget is None else depth_start + depth_budget
        window = _WINDOW_FACTOR * workers

        inflight: dict = {}  # pos -> (cand, future)
        ready: dict = {}  # pos -> ("skip", cand) | ("job", cand, v, m, el, why)
        next_pos = 0
        commit_pos = 1
        spent = None  # None | UNSAT | UNKNOWN | "guard"
        guard_msg = None
        capped = False  # the [gen] k-paths budget was reached

        def _verify(cand):
            # Read the budget when the worker starts (not when queued): a
            # candidate that waited behind draining work still respects what is
            # left of the depth budget. None left: UNKNOWN without launching
            # dReal, matching the serial per-call clamp.
            call_budget = None
            if deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return (
                        UNKNOWN,
                        None,
                        0.0,
                        "the per-depth [gen] pivot-budget elapsed before "
                        "this candidate started",
                    )
                call_budget = max(remaining, _MIN_CALL_BUDGET)
                if qsec is not None:
                    call_budget = min(call_budget, qsec)
            verifier = make_oracle(
                underlying="dreal",
                logic=logic,
                seed=seed,
                config=config,
                logger=logger,
                time_bound=tau_max,
            )
            # Registered so a cap or the run's end can kill this check instead of
            # waiting out its budget; discarded when it returns.
            live.add(verifier)
            try:
                verifier.assert_(cand.total_const)
                verifier.assert_(cand.word_pin)
                if call_budget is not None:
                    verifier.set_budget(call_budget)
                t0 = time.perf_counter()
                v = verifier.check()
                el = time.perf_counter() - t0
                m = dict(verifier.model()) if v == SAT else None
                why = verifier.unknown_reason() if v == UNKNOWN else None
                return (v, m, el, why)
            finally:
                live.discard(verifier)

        while True:
            while (
                spent is None
                and len(inflight) < window
                and (deadline is None or time.perf_counter() < deadline)
                and (per_depth is None or dctx.found < per_depth)
            ):
                res = search.propose()
                if res is None:
                    spent = UNSAT if search.last_verdict() == UNSAT else UNKNOWN
                    break
                raw_word = _location_word(res[2])
                guard = _guard_reason(raw_word, depth)
                if guard is not None:
                    guard_msg = (
                        f"[kappa_path] depth {depth}: cannot block this model "
                        f"({guard}); the model is not pooled and the depth "
                        "stops UNRESOLVED -- the path space is NOT proven "
                        "exhausted"
                    )
                    spent = "guard"
                    break
                next_pos += 1
                pos = next_pos
                cand = _reduced_candidate(res, depth)
                if feasibility.word_is_infeasible(depth, cand.mode_seq):
                    search.add_block(Not(cand.word_pin))
                    ready[pos] = ("skip", cand)
                    continue
                search.add_block(cand.pair_block)
                inflight[pos] = (cand, pool_exec.submit(_verify, cand))

            for pos in [p for p, (c, f) in inflight.items() if f.done()]:
                cand, fut = inflight.pop(pos)
                v, m, el, why = fut.result()
                ready[pos] = ("job", cand, v, m, el, why)

            while commit_pos in ready:
                item = ready.pop(commit_pos)
                if item[0] == "skip":
                    cand = item[1]
                    candidates += 1
                    if candidates == 1 or candidates % every == 0:
                        printer.print_verbose(
                            f"[kappa_path] depth {depth}: candidate "
                            f"{candidates} (word {cand.word_str}) infeasible on "
                            "the linear timeline -- skipped"
                        )
                    commit_pos += 1
                    continue
                _tag, cand, v, m, el, why = item
                candidates += 1
                calls.append(el)
                kind, block = _reduced_commit(
                    acc, dctx, cand, v, m, why, candidates, el, every, printer
                )
                if kind == "pool":
                    # The provisional pair block is already installed; the radius
                    # block (which subsumes it) excludes this word's ball from
                    # every later proposal.
                    search.add_block(block)
                commit_pos += 1
                if per_depth is not None and dctx.found >= per_depth:
                    capped = True
                    break

            # A path budget already met with nothing left to commit or verify
            # (normally reached in the commit loop above; also the degenerate
            # k-paths = 0, which proposes no query at all).
            if (
                not capped
                and per_depth is not None
                and dctx.found >= per_depth
                and not inflight
                and commit_pos not in ready
            ):
                capped = True

            if capped:
                for _c, f in inflight.values():
                    f.cancel()
                # Kill the speculative checks the cap makes stale so the pool is
                # not held blocked until their budgets expire.
                live.terminate_all()
                inflight = {}
                ready = {}
                printer.print_normal(
                    f"[kappa_path] depth {depth}: stopped at the [gen] k-paths "
                    f"budget after {dctx.found} word(s) -- the path space at "
                    "this depth is NOT known to be exhausted"
                )
                break

            past_deadline = deadline is not None and time.perf_counter() >= deadline
            if not inflight and (spent is not None or past_deadline):
                break

            can_refill = (
                spent is None
                and len(inflight) < window
                and (deadline is None or time.perf_counter() < deadline)
                and (per_depth is None or dctx.found < per_depth)
            )
            if inflight and not can_refill and commit_pos not in ready:
                pending = [f for _c, f in inflight.values()]
                timeout = (
                    None
                    if deadline is None
                    else max(0.0, deadline - time.perf_counter())
                )
                done_set, _ = wait(
                    pending, timeout=timeout, return_when=FIRST_COMPLETED
                )
                if (
                    not done_set
                    and deadline is not None
                    and time.perf_counter() >= deadline
                ):
                    for _c, f in inflight.values():
                        f.cancel()
                    # Kill checks the deadline makes stale so the depth is not
                    # held open until their per-call budgets expire.
                    live.terminate_all()
                    inflight = {}

        if spent == "guard":
            acc.unresolved = True
            if guard_msg:
                printer.print_normal(guard_msg)
        elif spent in (UNSAT, UNKNOWN):
            self._reduced_exhaustion(acc, dctx, depth, spent, printer)
        elif not capped and deadline is not None and (time.perf_counter() >= deadline):
            acc.unresolved = True
            dctx.undecided = True
            printer.print_normal(
                f"[kappa_path] depth {depth}: stopped at the [gen] pivot-budget "
                f"({depth_budget}s) after {candidates} candidate structure(s) "
                "-- the path space is NOT proven exhausted"
            )

        if calls:
            printer.print_normal(
                f"[kappa_path] depth {depth}: pooled {dctx.found} word(s) from "
                f"{candidates} candidate structure(s) over {len(calls)} dReal "
                f"call(s), {sum(calls):.2f}s total, slowest {max(calls):.2f}s"
            )

    def _reduced_exhaustion(self, acc, dctx, depth, last_verdict, printer):
        """Depth-scoping at proposal exhaustion, shared by both paths. An
        exhausted skeleton space settles the depth only when every structure was
        decided: a verify or minimizer UNKNOWN leaves a structure of unknown
        delta-feasibility, so absence is not established despite the search
        running dry."""
        if last_verdict == UNSAT and not dctx.undecided:
            acc.decided.append(depth)
            printer.print_normal(_exhaustion_note(depth, dctx.found, dctx.coarsened))
        elif last_verdict == UNSAT:
            printer.print_normal(
                f"[kappa_path] depth {depth}: the skeleton space is enumerated "
                "but at least one structure was undecided, so absence is NOT "
                "established at this depth"
            )
        else:
            acc.unresolved = True
            printer.print_normal(
                f"[kappa_path] depth {depth}: structure search UNRESOLVED "
                "(scenario solver or reduced-query minimizer did not decide) -- "
                "the path space is NOT proven exhausted"
            )

    def _make_reduced_search(self, components, model, seed, timeout_ms):
        """Construct the reduced-query structure search. A seam so a test can
        substitute a scripted search that returns proposals sharing a
        ``path_const`` -- the case the word-aware block exists for."""
        return ReducedPivotSearch(components, model, seed=seed, timeout_ms=timeout_ms)


class _PathReducedAcc:
    """Run-level accumulators of the delta (reduced two-step) enumeration,
    mutated across target depths."""

    def __init__(self, max_depth: int) -> None:
        self.pool: list[dict[Variable, Constant]] = []
        self.block_id = 0
        self.unresolved = False
        self.first_depth = max_depth
        self.decided: list[int] = []


class _PathDepthAcc:
    """Per-depth accumulators of the delta enumeration."""

    def __init__(self, depth: int, radius: int, per_depth) -> None:
        self.depth = depth
        self.radius = radius
        # The radius the block actually encodes: capped at the word length
        # (depth positions), so the commit-time separation check matches the
        # ball the block excludes rather than the uncapped configured radius.
        self.eff_radius = max(0, min(radius, depth))
        self.per_depth = per_depth
        self.found = 0
        self.coarsened = False  # a radius>=1 block was asserted at this depth
        self.capped = False  # the cap has been reported at this depth
        self.pooled_words: list[tuple] = []
        self.undecided = False  # a verify/minimizer returned UNKNOWN here


class _ReducedCand(NamedTuple):
    """A proposed structure with everything the commit needs precomputed."""

    total_const: Formula
    path_const: Formula
    assn: dict
    canon: list
    word_str: str
    word_pin: Formula
    pair_block: Formula
    mode_seq: list


def _guard_reason(raw_word, depth: int):
    """Why a proposed word cannot be pinned or blocked, or None when it is
    sound: an empty word, an off-lattice mode value, or a word short of its
    depth+1 arity. Either way a block built from it would miss the word the
    solver satisfied, so the caller drops the model and stops the depth."""
    if not raw_word:
        return "no currentMode_k variables in the model"
    bad = _off_lattice(raw_word)
    if bad:
        return ", ".join(bad)
    missing = _missing_modes(raw_word, depth)
    if missing:
        return "missing " + ", ".join(missing)
    return None


def _reduced_candidate(res, depth: int) -> _ReducedCand:
    """Precompute a proposed structure's canonical word, its pin, its pair block
    and its mode sequence. Assumes ``_guard_reason(word, depth)`` is None."""
    total_const, path_const, assn = res
    canon = [(var, _canon_mode_val(val)) for var, val in _location_word(assn)]
    word_str = ".".join(str(val.value) for _, val in canon)
    word_pin = fix_modes(dict(canon))
    # The word-aware exclusion for a pair that is not pooled: remove (this
    # reduced path AND this word), leaving the path available to a sibling word.
    pair_block = Not(And([path_const, word_pin]))
    mode_seq = [int(round(float(val.value))) for _, val in canon]
    return _ReducedCand(
        total_const, path_const, assn, canon, word_str, word_pin, pair_block, mode_seq
    )


def _word_key(canon) -> tuple:
    """A hashable, position-ordered key of a canonical word for de-duplication
    and Hamming comparison."""
    return tuple((var.id, val.value) for var, val in canon)


def _within_ball(key, pooled, radius: int) -> bool:
    """Whether ``key`` is within Hamming distance ``radius`` of any pooled word
    (words at a depth share their positions, so equal-length comparison is
    exact). At radius 0 this is exact de-duplication."""
    for other in pooled:
        if sum(1 for a, b in zip(key, other) if a[1] != b[1]) <= radius:
            return True
    return False


def _reduced_commit(
    acc, dctx, cand, verdict, model, why, candidate_no, elapsed, every, printer
):
    """Apply one verified candidate's verdict: pool a genuine delta-falsifier,
    or exclude a refuted/undecided pair. Mutates ``acc`` (pool, block id, first
    depth, unresolved) and ``dctx`` (found, coarsening, cap, pooled words,
    undecided), and returns ``(kind, block)`` where ``kind`` is ``"pool"`` for a
    pooled word (its radius block) or ``"block"`` otherwise (the pair block)."""
    depth = dctx.depth
    # A refutation is the common, uninteresting case; throttle it so the
    # satisfiable and undecided results stay legible.
    if verdict != UNSAT or candidate_no == 1 or candidate_no % every == 0:
        printer.print_verbose(
            f"[kappa_path] depth {depth}: candidate {candidate_no} "
            f"(word {cand.word_str}) over {dctx.found} pooled word(s): "
            f"{verdict} in {elapsed:.3f}s"
        )

    if verdict == UNKNOWN:
        acc.unresolved = True
        dctx.undecided = True
        why_txt = why or "backend did not decide"
        printer.print_normal(
            f"[kappa_path] depth {depth}: candidate {candidate_no} "
            f"(word {cand.word_str}) UNRESOLVED ({why_txt}) -- this structure "
            "is not counted toward exhaustion"
        )
        return "block", cand.pair_block

    if verdict == UNSAT:
        return "block", cand.pair_block

    mw = _location_word(model)
    if _missing_modes(mw, depth) or _off_lattice(mw):
        acc.unresolved = True
        dctx.undecided = True
        printer.print_normal(
            f"[kappa_path] depth {depth}: candidate {candidate_no} "
            f"(word {cand.word_str}) is satisfiable but its witness omits part "
            "of the location word; not pooled, depth left unresolved"
        )
        return "block", cand.pair_block

    key = _word_key(cand.canon)
    # The witness must spell the pinned word. A delta-sat model is a relaxed
    # witness, so it can carry a complete integral word other than the pinned
    # one; pooling it while blocking the pinned word would leave the pool entry
    # and its exclusion disagreeing. Reject like the schema-incomplete case:
    # exclude the pinned pair and leave the depth unresolved.
    if _word_key([(var, _canon_mode_val(val)) for var, val in mw]) != key:
        acc.unresolved = True
        dctx.undecided = True
        printer.print_normal(
            f"[kappa_path] depth {depth}: candidate {candidate_no} "
            f"(word {cand.word_str}) is satisfiable but its witness spells a "
            "different word; not pooled, depth left unresolved"
        )
        return "block", cand.pair_block
    # Enforce radius separation at commit: a word inside an already-pooled word's
    # ball is excluded even if it was proposed before that word's block landed.
    # The ball uses the effective (capped) radius, so at radius 0 -- and at a
    # radius capped to 0 by the word length, e.g. depth 0 -- this is exact
    # de-duplication and never drops a distinct word.
    if _within_ball(key, dctx.pooled_words, dctx.eff_radius):
        return "block", cand.pair_block

    if not acc.pool:
        acc.first_depth = depth
    acc.pool.append(dict(model))
    dctx.pooled_words.append(key)

    block = block_radius(cand.assn, dctx.radius, acc.block_id)
    if block.radius < dctx.radius and not dctx.capped:
        dctx.capped = True
        printer.print_normal(
            f"[kappa_path] depth {depth}: [gen] radius {dctx.radius} exceeds "
            f"the {depth + 1} positions of a word at this depth; capped to "
            f"{block.radius}"
        )
    dctx.coarsened = dctx.coarsened or block.radius >= 1
    acc.block_id += 1
    dctx.found += 1
    printer.print_verbose(
        f"[kappa_path] depth {depth}: {dctx.found} word(s) here, {len(acc.pool)} total"
    )
    return "pool", block.clause
