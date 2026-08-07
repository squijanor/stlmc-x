"""Discrete path enumeration (kappa_path), radius 0, single depth.

Enumerates structurally distinct counterexample paths at a fixed depth: solve
the falsification encoding, record the counterexample, exclude its location word
with a Boolean clause over the per-step mode variables, and re-solve until the
encoding is unsatisfiable or the budget is reached. The radius-0 form excludes
exactly one location word per found counterexample.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..constraints.constraints import Constant, Formula, Neq, Or, Variable
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


def block_exact_path(assn: Dict[Variable, Constant]) -> Formula:
    """kappa_path^0: a clause excluding exactly the location word of ``assn``."""
    word = _location_word(assn)
    if not word:
        raise RuntimeError(
            "no currentMode_k variables in assignment; model has no discrete modes to enumerate"
        )
    return Or([Neq(var, val) for var, val in word])


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
    """kappa_path at radius 0, enumerating distinct paths at the configured depth."""

    def __init__(self) -> None:
        self.debug_name = ""

    def set_debug(self, msg: str) -> None:
        self.debug_name = msg

    def run(self, model, goal, prop_dict, config, solver, logger, printer):
        common = config.get_section("common")
        bound = int(common.get_value("bound"))
        tau_max = float(common.get_value("time-bound"))
        delta = float(common.get_value("threshold"))
        underlying = common.get_value("solver")
        if underlying != "z3":
            raise NotImplementedError(
                "kappa_path currently supports the z3 backend; got '{}'".format(underlying)
            )

        budget = _gen_int(config, "k-paths")  # None -> enumerate to exhaustion

        encoder = Encoder(model, goal, prop_dict, delta, tau_max)
        encoding = encoder.encode_at(bound)

        oracle = Z3IncrementalOracle(_z3_logic(config))
        oracle.assert_(encoding.consts)

        pool: List[Dict[Variable, Constant]] = []
        while budget is None or len(pool) < budget:
            if oracle.check() != SAT:
                break
            assn = oracle.model()
            pool.append(assn)
            oracle.assert_(block_exact_path(assn))
            printer.print_verbose("[kappa_path] depth {}: {} path(s)".format(bound, len(pool)))

        result = "False" if pool else "True"
        return result, 0.0, bound, pool