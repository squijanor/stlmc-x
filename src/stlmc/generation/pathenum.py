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

import os
import re
from functools import reduce
from typing import Dict, List, Tuple

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
from .encode import Encoder
from .oracle import SAT, make_oracle

# Per-step mode index variable produced by the encoding: currentMode_<step>.
_MODE_RE = re.compile(r"^currentMode_(\d+)$")

# Config z3 logic -> the logic name the base Z3Solver passes to z3.SolverFor.
_Z3_LOGIC = {"QF_LRA": "LRA", "QF_NRA": "NRA"}


def _location_word(assn: Dict[Variable, Constant]) -> List[Tuple[Variable, Constant]]:
    """The (currentMode_k, value) pairs of an assignment, ordered by step k."""
    steps: List[Tuple[int, Variable, Constant]] = []
    for var, val in assn.items():
        m = _MODE_RE.match(var.id)
        if m is not None:
            steps.append((int(m.group(1)), var, val))
    steps.sort(key=lambda t: t[0])
    return [(var, val) for _, var, val in steps]


def block_radius(assn: Dict[Variable, Constant], radius: int, uid: int) -> Formula:
    """Clause excluding every location word within Hamming distance ``radius`` of
    ``assn``'s word. ``uid`` makes the radius>=1 indicator variables unique.

    radius 0: ``OR_k (mode_k != w_k)`` -- excludes exactly the one word.
    radius r: ``sum_k [mode_k != w_k] >= r+1`` via 0/1 indicators.
    """
    word = _location_word(assn)
    if not word:
        raise RuntimeError(
            "no currentMode_k variables in assignment; model has no discrete modes to enumerate"
        )

    if radius <= 0:
        return Or([Neq(var, val) for var, val in word])

    consts: List[Formula] = []
    indicators: List[Variable] = []
    for k, (var, val) in enumerate(word):
        ind = Int("hb${}${}".format(uid, k))
        indicators.append(ind)
        consts.append(Or([Eq(ind, IntVal("0")), Eq(ind, IntVal("1"))]))
        consts.append(Implies(Neq(var, val), Eq(ind, IntVal("1"))))
        consts.append(Implies(Eq(var, val), Eq(ind, IntVal("0"))))
    consts.append(Geq(reduce(Add, indicators), IntVal(str(radius + 1))))
    return And(consts)


def _gen_int(config, key: str):
    """Read an integer from the optional [gen] section, or None if absent."""
    if config.is_section_in("gen"):
        section = config.get_section("gen")
        if section.is_argument_in(key):
            return int(section.get_value(key))
    return None


def _gen_depths(config, max_depth: int) -> List[int]:
    """Target depths: the ``[gen] depths`` list clamped to 1..max_depth, or every
    depth 1..max_depth when absent.

    The list is slash-separated (e.g. ``"8/9/10/11"``): the config grammar lexes a
    bare number as a NUMBER token and accepts only a single VALUE token inside
    quotes, and a VALUE may contain ``/`` -- so ``"8/9/10/11"`` is one token while
    ``"8,9,10,11"`` does not parse. Commas are still tolerated if they get through.
    """
    if config.is_section_in("gen"):
        section = config.get_section("gen")
        if section.is_argument_in("depths"):
            raw = str(section.get_value("depths"))
            picked = sorted({int(tok) for tok in re.split(r"[,/]", raw) if tok.strip()})
            return [d for d in picked if 1 <= d <= max_depth]
    return list(range(1, max_depth + 1))


def _z3_logic(config) -> str:
    if config.is_section_in("z3"):
        z3_section = config.get_section("z3")
        if z3_section.is_argument_in("logic"):
            return _Z3_LOGIC.get(z3_section.get_value("logic"), "LRA")
    return "LRA"


def _resolve_seed(config) -> int:
    """The generation seed: -gen-seed if given (>= 0), else PYTHONHASHSEED.

    Raises if neither is present, so a run is never silently non-reproducible.
    """
    common = config.get_section("common")
    if common.is_argument_in("gen-seed"):
        value = int(common.get_value("gen-seed"))
        if value >= 0:
            return value
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is not None and hash_seed.isdigit():
        return int(hash_seed)
    raise ValueError(
        "no generation seed: pass -gen-seed <n> or set PYTHONHASHSEED to a non-negative integer"
    )


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

        per_depth = _gen_int(config, "k-paths")  # per target depth; None -> exhaust
        target_depths = _gen_depths(config, max_depth)
        radius = _gen_int(config, "radius") or 0
        logic = _z3_logic(config)
        seed = _resolve_seed(config)

        hash_seed = os.environ.get("PYTHONHASHSEED")
        if hash_seed is None or not hash_seed.isdigit():
            printer.print_normal(
                "warning: PYTHONHASHSEED is not fixed; constraint ordering is not pinned, "
                "so the pool may vary run to run despite -gen-seed. Set PYTHONHASHSEED for reproducibility."
            )

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
        pool: List[Dict[Variable, Constant]] = []
        block_id = 0

        for depth in target_depths:
            encoding = encoder.encode_at(depth)
            oracle = new_oracle()
            oracle.assert_(encoding.consts)

            found = 0
            while per_depth is None or found < per_depth:
                if oracle.check() != SAT:
                    break
                assn = oracle.model()
                pool.append(assn)
                oracle.assert_(block_radius(assn, radius, block_id))
                block_id += 1
                found += 1
                printer.print_verbose(
                    "[kappa_path] depth {}: {} path(s) here, {} total".format(
                        depth, found, len(pool)
                    )
                )

            encoder.reset()

        result = "False" if pool else "True"
        return result, 0.0, max_depth, pool