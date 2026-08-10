"""Discrete path enumeration (kappa_path).

Enumerates structurally distinct counterexample paths by blocking each found
location word and re-solving. The traversal visits a set of target depths --
``[gen] depths`` as a slash-separated list (e.g. ``"8/9/10/11"``), or every depth
1..N by default -- and at each
target depth it solves the falsification encoding, records the counterexample,
excludes a Hamming-ball of radius r around its location word, and re-solves until
that depth is exhausted or its per-depth budget is reached.

The budget ``[gen] k-paths`` is per target depth, not global: each targeted depth
contributes up to that many paths, so the pool is depth-uniform (every targeted
depth is filled to the same density) rather than front-loaded onto the shallowest
depths. Absent, a depth is enumerated to exhaustion.

The blocking predicate is a Boolean clause over the per-step mode variables:
radius 0 excludes exactly one location word; radius r excludes every word within
Hamming distance r of it.

Reproducibility: the generation seed is -gen-seed if given (>= 0), otherwise the
PYTHONHASHSEED value; one of the two must be present. For the z3 backend the
seed is passed as its random_seed. Constraint ordering is fixed by
PYTHONHASHSEED, which must be set for a run to be reproducible; z3's random_seed
alone does not pin it.
"""

from __future__ import annotations

from functools import reduce

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
    gen_depths,
    gen_int,
    resolve_seed,
    warn_unpinned_hashseed,
    z3_logic,
)
from .encode import Encoder
from .oracle import SAT, UNKNOWN, make_oracle


def _location_word(assn: dict[Variable, Constant]) -> list[tuple[Variable, Constant]]:
    """The (currentMode_k, value) pairs of an assignment, ordered by step k."""
    steps: list[tuple[int, Variable, Constant]] = []
    for var, val in assn.items():
        m = MODE_RE.match(var.id)
        if m is not None:
            steps.append((int(m.group(1)), var, val))
    steps.sort(key=lambda t: t[0])
    return [(var, val) for _, var, val in steps]


def block_radius(assn: dict[Variable, Constant], radius: int, uid: int) -> Formula:
    """Clause excluding every location word within Hamming distance ``radius`` of
    ``assn``'s word. ``uid`` makes the radius>=1 indicator variables unique.

    radius 0: ``OR_k (mode_k != w_k)`` -- excludes exactly the one word.
    radius r: ``sum_k [mode_k != w_k] >= r+1`` via 0/1 indicators.
    """
    word = _location_word(assn)
    if not word:
        raise RuntimeError(
            "no currentMode_k variables in assignment; model has no discrete "
            "modes to enumerate"
        )

    if radius <= 0:
        return Or([Neq(var, val) for var, val in word])

    consts: list[Formula] = []
    indicators: list[Variable] = []
    for k, (var, val) in enumerate(word):
        ind = Int(f"hb${uid}${k}")
        indicators.append(ind)
        consts.append(Or([Eq(ind, IntVal("0")), Eq(ind, IntVal("1"))]))
        consts.append(Implies(Neq(var, val), Eq(ind, IntVal("1"))))
        consts.append(Implies(Eq(var, val), Eq(ind, IntVal("0"))))
    consts.append(Geq(reduce(Add, indicators), IntVal(str(radius + 1))))
    return And(consts)


class DiscretePathEnum(Algorithm):
    """kappa_path: enumerate distinct paths over depths 1..N under a pool budget."""

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

        per_depth = gen_int(config, "k-paths")  # per target depth; None -> exhaust
        target_depths = gen_depths(config, max_depth)
        radius = gen_int(config, "radius") or 0
        logic = z3_logic(config)
        seed = resolve_seed(config)

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

        encoder = Encoder(model, goal, prop_dict, delta, tau_max)
        pool: list[dict[Variable, Constant]] = []
        block_id = 0
        unresolved = False

        for depth in target_depths:
            encoding = encoder.encode_at(depth)
            oracle = new_oracle()
            oracle.assert_(encoding.consts)

            found = 0
            while per_depth is None or found < per_depth:
                verdict = oracle.check()
                if verdict != SAT:
                    # UNSAT means the path lattice at this depth is exhausted.
                    # UNKNOWN means the backend gave up, which is NOT evidence of
                    # absence; collapsing the two lets an unresolved search be
                    # reported as "no counterexample".
                    if verdict == UNKNOWN:
                        unresolved = True
                        printer.print_normal(
                            f"[kappa_path] depth {depth}: search UNRESOLVED "
                            "(backend did not decide) -- the path space is NOT "
                            "proven exhausted")
                    break
                assn = oracle.model()
                pool.append(assn)
                oracle.assert_(block_radius(assn, radius, block_id))
                block_id += 1
                found += 1
                printer.print_verbose(
                    f"[kappa_path] depth {depth}: {found} path(s) here, "
                    f"{len(pool)} total"
                )

            encoder.reset()

        # "True" claims no counterexample exists up to the bound. That is only
        # justified when every depth was actually decided.
        # "True" claims no counterexample exists up to the bound. That is only
        # justified when every depth was actually decided.
        if pool:
            result = "False"
        elif unresolved:
            result = "Unknown"
            printer.print_normal(
                "[kappa_path] no counterexample found, but at least one depth was "
                "unresolved: reporting Unknown, not True")
        else:
            result = "True"
        return result, 0.0, max_depth, pool