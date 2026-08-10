"""Configuration reading and verdict scoping shared by the generation strategies.

The strategies are independent in mechanism: they block different objects, pose
different queries, and neither imports the other. What they do share is the way
a run is *parameterised* and the way its result is *scoped*, and both of those
are properties of the driver contract rather than of either algorithm:

* every strategy reads its parameters from the optional ``[gen]`` section, the
  ``[z3]`` logic, and the generation seed;
* every strategy visits a set of target depths and must not let its verdict
  speak about depths it never examined.

Keeping those here is what lets a strategy be rewritten without disturbing the
other. Nothing about an algorithm belongs in this module: no geometry, no
blocking predicate, no oracle policy.

``[gen]`` accessors return ``None`` (or the supplied default) for an absent key,
so a caller distinguishes "not configured" from a configured value and can apply
its own default.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction

# Per-step mode index variable produced by the encoding: currentMode_<step>.
MODE_RE = re.compile(r"^currentMode_(\d+)$")

# Config z3 logic -> the logic name the base Z3Solver passes to z3.SolverFor.
_Z3_LOGIC = {"QF_LRA": "LRA", "QF_NRA": "NRA"}


def _gen_value(config, key: str):
    """The raw ``[gen]`` value for ``key``, or None when the section or the key
    is absent. The single point at which the section's optionality is handled."""
    if config is None or not config.is_section_in("gen"):
        return None
    section = config.get_section("gen")
    if not section.is_argument_in(key):
        return None
    return section.get_value(key)


def gen_int(config, key: str):
    """An integer ``[gen]`` value, or None when absent."""
    value = _gen_value(config, key)
    return None if value is None else int(value)


def gen_float(config, key: str):
    """A float ``[gen]`` value, or None when absent, empty or zero.

    Zero folds into None because every caller uses this for a positive budget
    whose "off" setting is 0.
    """
    value = _gen_value(config, key)
    if value in (None, "", "0"):
        return None
    return float(value)


def gen_frac(config, key: str, default: str) -> Fraction:
    """An exact-rational ``[gen]`` value, or ``default`` when absent."""
    value = _gen_value(config, key)
    return Fraction(default if value is None else value)


def gen_str(config, key: str):
    """A string ``[gen]`` value, or None when absent or empty."""
    value = _gen_value(config, key)
    if value in (None, ""):
        return None
    return str(value).strip().strip('"')


def gen_depths(config, max_depth: int) -> list[int]:
    """Target depths: the ``[gen] depths`` list clamped to 1..max_depth, or every
    depth 1..max_depth when absent.

    The list is slash-separated (e.g. ``"8/9/10/11"``): the config grammar lexes a
    bare number as a NUMBER token and accepts only a single VALUE token inside
    quotes, and a VALUE may contain ``/`` -- so ``"8/9/10/11"`` is one token while
    ``"8,9,10,11"`` does not parse. Commas are still tolerated if they get through.
    """
    raw = _gen_value(config, "depths")
    if raw is None:
        return list(range(1, max_depth + 1))
    picked = sorted({int(tok) for tok in re.split(r"[,/]", str(raw)) if tok.strip()})
    return [d for d in picked if 1 <= d <= max_depth]


def z3_logic(config) -> str:
    """The z3 logic name for ``[z3] logic``, defaulting to linear arithmetic."""
    if config is not None and config.is_section_in("z3"):
        z3_section = config.get_section("z3")
        if z3_section.is_argument_in("logic"):
            return _Z3_LOGIC.get(z3_section.get_value("logic"), "LRA")
    return "LRA"


def resolve_seed(config) -> int:
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
        "no generation seed: pass -gen-seed <n> or set PYTHONHASHSEED to a "
        "non-negative integer"
    )


def warn_unpinned_hashseed(printer) -> None:
    """Warn when PYTHONHASHSEED is not fixed.

    A seed passed with -gen-seed reaches the solver but not the order in which
    constraints are built, so it alone does not make a run reproducible.
    """
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is None or not hash_seed.isdigit():
        printer.print_normal(
            "warning: PYTHONHASHSEED is not fixed; constraint ordering is not "
            "pinned, so the pool may vary run to run despite -gen-seed. Set "
            "PYTHONHASHSEED for reproducibility."
        )


def scoped_verdict(pool, any_unresolved, visited, max_depth, *,
                   tag: str, nothing_found: str, unresolved_source: str):
    """The run's verdict, and the line explaining it (or None).

    A verdict may only speak about the depths that were **decided**. The driver
    prints "up to bound N" from the bound rather than from the explored depths,
    so a run that settled a subset must not report True: absence over a subset of
    depths is not absence up to the bound, and is reported as Unknown. A caller
    passes the depths it actually settled, which is not always the set it
    targeted -- a depth can be visited and left open by a budget or a bound.

    ``tag``, ``nothing_found`` and ``unresolved_source`` are the caller's own
    wording: the rule is shared, the vocabulary for what a strategy looks for is
    not.
    """
    if pool:
        return "False", None
    if any_unresolved:
        return "Unknown", (f"[{tag}] {nothing_found}, but {unresolved_source} "
                           "was unresolved: reporting Unknown, not True")
    skipped = set(range(1, max_depth + 1)) - set(visited)
    if skipped:
        seen = "/".join(str(d) for d in sorted(set(visited))) or "none"
        missed = "/".join(str(d) for d in sorted(skipped))
        return "Unknown", (
            f"[{tag}] {nothing_found}, decided depth(s) {seen}, but depth(s) "
            f"{missed} of 1..{max_depth} were not decided -- reporting Unknown, "
            "not True: absence over a subset of depths is not absence up to the "
            "bound")
    return "True", None