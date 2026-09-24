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

Depth and bound
---------------
They are one quantity under two names. A strategy's *depth* is the argument to
``Model.make_consts`` and to ``k_size_stl_formula``, which is exactly the
unrolling index upstream sweeps from ``[common] bound``; the driver prints it
back as "bound". "Depth" is kept because these strategies visit a *set* of them
rather than sweeping upward, not because it denotes anything else.

At depth *n* the encoding is one initial state, *n* steps each carrying a jump,
and one final flow segment: **n+1 mode segments and n jumps**, hence a location
word of *n+1* positions and *n+1* segment durations. Depth **0** is therefore
well formed and not empty -- one segment, no jump, a word of one position -- and
it is a depth at which a counterexample can exist and at which no other can.
Upstream is not consistent about it: ``EnumerateAlgorithm`` sweeps ``0..N`` while
``SmtAlgorithm`` sweeps ``1..N``, so the same model and property are reported one
bound apart by the two. The strategies here follow the former and cover ``0..N``.

Note this is *not* the quantity the STLMC papers call the bound: there it counts
the variable points of the symbolic discrete signal (the size of the phi-refinement),
which the encoding derives from this one as ``2*(depth+1)`` STL intervals. The two
are locked together by the configuration but are not the same number.
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
    """A float ``[gen]`` value, or None when absent or empty.

    A configured 0 is returned as 0.0, NOT folded into None: with the fold,
    five keys silently replaced a configured 0 with their default and the run
    banner printed the default back as if it had been set. What a zero means
    (off, or a configuration error) is the caller's decision -- see
    :func:`validate_gen`.
    """
    value = _gen_value(config, key)
    if value in (None, ""):
        return None
    return float(value)


def gen_present(config, key: str) -> bool:
    """Whether ``key`` is set in ``[gen]`` at all, without parsing its value.

    The presence check for removed/renamed keys: parsing would crash on a
    boolean-shaped stale value (``int("true")``) before the warning prints."""
    return _gen_value(config, key) is not None


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
    """Target depths: the ``[gen] depths`` list clamped to 0..max_depth, or every
    depth 0..max_depth when absent.

    The range starts at 0, not 1: depth 0 is the unrolling with no jump, which is
    a trajectory like any other and is where upstream's own default algorithm
    reports a large share of its counterexamples (see the module docstring).

    The list is slash-separated (e.g. ``"8/9/10/11"``): the config grammar lexes a
    bare number as a NUMBER token and accepts only a single VALUE token inside
    quotes, and a VALUE may contain ``/`` -- so ``"8/9/10/11"`` is one token while
    ``"8,9,10,11"`` does not parse. Commas are still tolerated if they get through.
    """
    raw = _gen_value(config, "depths")
    if raw is None:
        return list(range(0, max_depth + 1))
    picked = sorted({int(tok) for tok in re.split(r"[,/]", str(raw)) if tok.strip()})
    return [d for d in picked if 0 <= d <= max_depth]


def _check_int(key: str, raw: str) -> None:
    try:
        int(raw)
    except ValueError:
        raise ValueError(f'[gen] {key} = "{raw}": an integer is required') from None


def _check_depths(key: str, raw: str) -> None:
    for tok in re.split(r"[,/]", str(raw)):
        if tok.strip():
            try:
                int(tok)
            except ValueError:
                raise ValueError(
                    f'[gen] {key} = "{raw}": "{tok.strip()}" is not an '
                    'integer; the list is slash-separated, e.g. "8/9/10/11"'
                ) from None


def _check_query_timeout(key: str, raw: str) -> None:
    # Section.get_value strips quotes from real configurations; the extra
    # strip here keeps the check usable on raw test harness values too.
    if str(raw).strip().strip('"') in ("", "0", "off", "none"):
        return
    try:
        sec = float(raw)
    except ValueError:
        raise ValueError(
            f'[gen] {key} = "{raw}": a number of seconds is required '
            '(0, "off" or "none" disables the per-call bound)'
        ) from None
    if sec != sec or sec in (float("inf"), float("-inf")) or sec < 0:
        raise ValueError(
            f'[gen] {key} = "{raw}": a finite number of seconds >= 0 is '
            'required (0, "off" or "none" disables the per-call bound)'
        )


def _check_flag01(key: str, raw: str) -> None:
    if str(raw).strip() not in ("0", "1"):
        raise ValueError(f'[gen] {key} = "{raw}": 0 or 1 is required')


def _check_frac(key: str, raw: str) -> Fraction:
    try:
        return Fraction(str(raw).strip().strip('"'))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f'[gen] {key} = "{raw}": a number is required') from None


def _check_positive_theta(key: str, raw: str) -> None:
    if _check_frac(key, raw) <= 0:
        raise ValueError(
            f'[gen] {key} = "{raw}": must be > 0 -- theta is the growth '
            "granularity, and at 0 the exact face search never terminates"
        )


def _check_unit_frac(key: str, raw: str) -> None:
    value = _check_frac(key, raw)
    if not (0 <= value < 1):
        raise ValueError(f'[gen] {key} = "{raw}": must be in [0, 1) (0 = off)')


def _check_nonneg_frac(key: str, raw: str) -> None:
    if _check_frac(key, raw) < 0:
        raise ValueError(f'[gen] {key} = "{raw}": must be >= 0 (0 = off)')


def _check_int_at_least(floor: int, note: str = ""):
    def check(key: str, raw: str) -> None:
        _check_int(key, raw)
        if int(raw) < floor:
            raise ValueError(f'[gen] {key} = "{raw}": must be >= {floor}{note}')

    return check


def _check_positive_seconds(key: str, raw: str) -> None:
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f'[gen] {key} = "{raw}": a number of seconds is required'
        ) from None
    if not value > 0 or value != value or value == float("inf"):
        raise ValueError(
            f'[gen] {key} = "{raw}": must be a finite number of seconds > 0 '
            '-- these are the search budgets, and neither has an "off" '
            "spelling"
        )


_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def _check_target_axes(key: str, raw: str) -> None:
    toks = [
        t.strip() for t in re.split(r"[,/]", str(raw).strip().strip('"')) if t.strip()
    ]
    if not toks:
        raise ValueError(
            f'[gen] {key} = "{raw}": at least one axis name is required; the '
            'list is slash-separated, e.g. "y/psi/r/Vc"'
        )
    for tok in toks:
        if not _NAME_RE.match(tok):
            raise ValueError(
                f'[gen] {key} = "{raw}": "{tok}" is not a state-variable name'
            )


def _check_target_bounds(key: str, raw: str) -> None:
    toks = [
        t.strip() for t in re.split(r"[,/]", str(raw).strip().strip('"')) if t.strip()
    ]
    if not toks:
        return
    if len(toks) % 3 != 0:
        raise ValueError(
            f'[gen] {key} = "{raw}": expected name/lo/hi triples (a multiple of '
            'three slash-separated tokens), e.g. "y/0.5/2.5/psi/0.1/0.7"'
        )
    for i in range(0, len(toks), 3):
        name, lo_raw, hi_raw = toks[i], toks[i + 1], toks[i + 2]
        if not _NAME_RE.match(name):
            raise ValueError(
                f'[gen] {key} = "{raw}": "{name}" is not a state-variable name'
            )
        try:
            lo, hi = Fraction(lo_raw), Fraction(hi_raw)
        except (ValueError, ZeroDivisionError):
            raise ValueError(
                f'[gen] {key} = "{raw}": {name} edges "{lo_raw}/{hi_raw}" must '
                "both be numbers"
            ) from None
        if lo > hi:
            raise ValueError(
                f'[gen] {key} = "{raw}": {name} lower edge {lo_raw} exceeds '
                f"upper edge {hi_raw}"
            )


# Key -> check. Entries cover the keys kappa_path reads and the keys the
# backends share; the sibling strategy registers its own keys here as they
# gain validation. A key with no entry passes unchecked.
_GEN_KEY_CHECKS = {
    "radius": _check_int,
    "k-paths": _check_int,
    "depths": _check_depths,
    "query-timeout": _check_query_timeout,
    "keep-smt2": _check_flag01,
    # kappa_box keys carry range checks as well as parse checks, because an
    # out-of-range value has NO defined semantics there (unlike a negative
    # radius, which folds with a notice): theta = 0 never terminates, a
    # zero-box budget would decide a depth in zero solver calls, a negative
    # word-rotate blocks a word on its first refutation.
    "epsilon": _check_positive_theta,
    "epsilon-relative": _check_unit_frac,
    "thin-ic": _check_nonneg_frac,
    "bisect-iters": _check_int_at_least(0, " (0 = face precision theta)"),
    "k-witness": _check_int_at_least(
        1, " (the per-axis cell budget of the lattice harvest, Def. 7: k >= 1)"
    ),
    "k-ic": _check_int_at_least(
        0, " (0 = no budget, explore each depth to exhaustion)"
    ),
    "word-rotate": _check_int_at_least(0, " (0 = off)"),
    "log-every": _check_int_at_least(1),
    "pivot-budget": _check_positive_seconds,
    "pivot-timeout": _check_positive_seconds,
    # The authoritative IC-domain axes and edges: a slash-separated list of
    # axis names, and name/lo/hi triples. A malformed value here would either
    # drop an intended axis silently or fail deep inside the plan.
    "target-axes": _check_target_axes,
    "target-bounds": _check_target_bounds,
}


def validate_gen(config, keys=None) -> None:
    """Fail fast on malformed ``[gen]`` values.

    Run at strategy start, before any solver work, so a bad value dies as a
    configuration error naming the key and the section rather than as a bare
    ``ValueError`` deep inside a run (where the driver's blanket handler
    reduces it to an unattributed one-liner). ``keys`` restricts the check to
    the keys a caller reads; by default every key with an entry in
    ``_GEN_KEY_CHECKS`` that is present in the configuration is checked.
    Parse-level validation only: range folds that have defined semantics
    (a negative radius, say) stay in the strategy, next to their notices.
    """
    if config is None or not config.is_section_in("gen"):
        return
    section = config.get_section("gen")
    for key, check in _GEN_KEY_CHECKS.items():
        if keys is not None and key not in keys:
            continue
        if section.is_argument_in(key):
            check(key, section.get_value(key))


def z3_logic(config) -> str:
    """The z3 logic name for ``[z3] logic``, defaulting to linear arithmetic.

    An absent ``[z3]`` section or an absent ``logic`` key resolves to linear
    arithmetic. A *present* ``logic`` whose value is not one of the recognised
    names is a configuration error, not a silent downgrade: falling back to
    linear arithmetic on an unrecognised value hands a nonlinear model a
    linear-arithmetic solver, whose UNKNOWN answers then read as solver
    give-ups rather than as the misspelling that produced them.
    """
    if config is not None and config.is_section_in("z3"):
        z3_section = config.get_section("z3")
        if z3_section.is_argument_in("logic"):
            raw = z3_section.get_value("logic")
            try:
                return _Z3_LOGIC[raw]
            except KeyError:
                raise ValueError(
                    f'[z3] logic = "{raw}": unrecognised value; expected one '
                    f"of {', '.join(sorted(_Z3_LOGIC))}"
                ) from None
    return "LRA"


#: The delta a delta-decision backend runs at when nothing sets it. dReal3's
#: own compiled default, so an absent ``[dreal] precision`` and this value name
#: the same run.
DEFAULT_BACKEND_PRECISION = Fraction("1/1000")


def backend_precision(config, solver: str) -> Fraction:
    """The relaxation the backend answers under, as a non-negative Fraction.

    Zero for an exact backend, where a satisfying answer is a model. On a
    delta-decision backend it is ``[dreal] precision``, and it is three things
    at once: the value passed to the solver, the floor under any frontier the
    geometry locates, and what makes a pool self-describing, since a witness is
    a point in a region accepted up to this value. Resolved in one place so
    those three cannot name different runs.

    An absent key is the backend's own default, so an unset configuration and
    the default describe the same run. A present one must be a positive finite
    number: zero or a negative relaxation has no reading, and silently
    substituting the default for one would report a run that did not happen.
    """
    if str(solver).strip().lower() != "dreal":
        return Fraction(0)
    if config is None or not config.is_section_in("dreal"):
        return DEFAULT_BACKEND_PRECISION
    section = config.get_section("dreal")
    if not section.is_argument_in("precision"):
        return DEFAULT_BACKEND_PRECISION
    raw = str(section.get_value("precision")).strip().strip('"')
    try:
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        raise ValueError(f'[dreal] precision = "{raw}" is not a number') from exc
    if value <= 0:
        raise ValueError(
            f"[dreal] precision = {raw} must be positive; it is the "
            f"relaxation the backend answers under"
        )
    return value


def ode_settings(config) -> tuple[int | None, float | None]:
    """The dReal ODE integration flags to emit, as ``(order, step)``.

    Read here alongside :func:`backend_precision` so a configured value reaches
    the command line rather than only describing the run. ``[dreal] ode-order``
    and ``ode-step`` are mandatory floats in the configuration, so both are
    normally present and are passed straight through; only a genuinely absent
    section or key (a caller that built its own configuration) leaves the flag
    off. A present value must be positive: a zero or negative integration
    setting has no reading. ``order`` is dReal's Taylor order (a whole number);
    ``step`` is the integration step size.
    """
    return (
        _ode_setting(config, "ode-order", integral=True),
        _ode_setting(config, "ode-step", integral=False),
    )


def _ode_setting(config, key: str, *, integral: bool):
    if config is None or not config.is_section_in("dreal"):
        return None
    section = config.get_section("dreal")
    if not section.is_argument_in(key):
        return None
    raw = str(section.get_value(key)).strip().strip('"')
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f'[dreal] {key} = "{raw}" is not a number') from exc
    if value != value or value in (float("inf"), float("-inf")) or value <= 0:
        raise ValueError(
            f"[dreal] {key} = {raw} must be a positive number; it is a dReal "
            "ODE integration setting"
        )
    if integral:
        if value != int(value):
            raise ValueError(
                f"[dreal] {key} = {raw} must be a whole number (dReal's Taylor order)"
            )
        return int(value)
    return value


def resolve_seed(config, printer=None) -> int:
    """The generation seed: -gen-seed if given (>= 0), else PYTHONHASHSEED.

    A negative -gen-seed is ignored by design (the fallthrough to
    PYTHONHASHSEED), but never silently: the notice below is what tells a
    reader of the log which seed source the run actually used. Raises if
    neither source is present, so a run is never silently non-reproducible.
    """
    common = config.get_section("common")
    if common.is_argument_in("gen-seed"):
        value = int(common.get_value("gen-seed"))
        if value >= 0:
            return value
        if printer is not None:
            printer.print_normal(
                f"warning: -gen-seed {value} is negative and is ignored; "
                "falling back to PYTHONHASHSEED"
            )
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


def scoped_verdict(
    pool,
    any_unresolved,
    decided_depths,
    max_depth,
    *,
    tag: str,
    nothing_found: str,
    unresolved_source: str,
):
    """The run's verdict, and the line explaining it (or None).

    A verdict may only speak about the depths that were **decided**. The driver
    prints "up to bound N" from the bound rather than from the explored depths,
    so a run that settled a subset must not report True: absence over a subset of
    depths is not absence up to the bound, and is reported as Unknown. A caller
    passes the depths it actually settled, which is not always the set it
    targeted -- a depth can be visited and left open by a budget or a bound.

    The bound covers ``0..max_depth``. Depth 0 is part of it because it is a
    reachable unrolling that carries no jump, so a run that leaves it out has not
    established absence even after deciding every other depth.

    ``tag``, ``nothing_found`` and ``unresolved_source`` are the caller's own
    wording: the rule is shared, the vocabulary for what a strategy looks for is
    not.
    """
    if pool:
        return "False", None
    if any_unresolved:
        return "Unknown", (
            f"[{tag}] {nothing_found}, but {unresolved_source} "
            "was unresolved: reporting Unknown, not True"
        )
    skipped = set(range(0, max_depth + 1)) - set(decided_depths)
    if skipped:
        seen = "/".join(str(d) for d in sorted(set(decided_depths))) or "none"
        missed = "/".join(str(d) for d in sorted(skipped))
        return "Unknown", (
            f"[{tag}] {nothing_found}, decided depth(s) {seen}, but depth(s) "
            f"{missed} of 0..{max_depth} were not decided -- reporting Unknown, "
            "not True: absence over a subset of depths is not absence up to the "
            "bound"
        )
    return "True", None
