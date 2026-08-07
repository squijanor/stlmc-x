"""Discrete path enumeration (kappa_path).

Enumerates structurally distinct counterexample paths by blocking each found
location word and re-solving. The sweep runs depths 1..N; at each depth it
solves the falsification encoding, records the counterexample, excludes a
Hamming-ball of radius r around its location word, and re-solves until the depth
is exhausted or the pool budget is reached.

The blocking predicate is a Boolean clause over the per-step mode variables:
radius 0 excludes exactly one location word; radius r excludes every word within
Hamming distance r of it.
"""

from __future__ import annotations

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
from .oracle import SAT, Z3IncrementalOracle

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


def _z3_logic(config) -> str:
    if config.is_section_in("z3"):
        z3_section = config.get_section("z3")
        if z3_section.is_argument_in("logic"):
            return _Z3_LOGIC.get(z3_section.get_value("logic"), "LRA")
    return "LRA"


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
        if underlying != "z3":
            raise NotImplementedError(
                "kappa_path currently supports the z3 backend; got '{}'".format(underlying)
            )

        budget = _gen_int(config, "k-paths")  # None -> enumerate every depth to exhaustion
        radius = _gen_int(config, "radius") or 0
        logic = _z3_logic(config)

        encoder = Encoder(model, goal, prop_dict, delta, tau_max)
        pool: List[Dict[Variable, Constant]] = []
        block_id = 0

        for depth in range(1, max_depth + 1):
            if budget is not None and len(pool) >= budget:
                break

            encoding = encoder.encode_at(depth)
            oracle = Z3IncrementalOracle(logic)
            oracle.assert_(encoding.consts)

            while budget is None or len(pool) < budget:
                if oracle.check() != SAT:
                    break
                assn = oracle.model()
                pool.append(assn)
                oracle.assert_(block_radius(assn, radius, block_id))
                block_id += 1
                printer.print_verbose(
                    "[kappa_path] depth {}: {} path(s)".format(depth, len(pool))
                )

            encoder.reset()

        result = "False" if pool else "True"
        return result, 0.0, max_depth, pool