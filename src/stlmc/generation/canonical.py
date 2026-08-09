"""Canonical ordering for the pickled pool payload.

Python dictionaries preserve insertion order; the solver assignment dicts are
built by iterating containers whose iteration order is identity-based, and
identity is not stable across processes (PYTHONHASHSEED does not pin it, since
the keys are objects without a value-based __hash__). Two runs agreeing in every
counterexample therefore write files whose key sequences differ, so a checksum
comparison of the pool fails while its content is identical. A consumer that
looks assignments up by key is unaffected.

`canonicalize` rebuilds the payload with every dict re-inserted in a
deterministic key order, making the written bytes a function of the content
alone.

Scope and limits, both deliberate:

* Only dict / list / tuple are rebuilt. Arbitrary objects pass through
  untouched: canonicalizing them would mean mutating objects the driver is still
  using, since the model's range_dict is iterated again right after the write to
  build the .cfg companion. The payload holds no set or frozenset at any depth;
  one added later would be a residual source of instability.
* Object sharing is preserved through a memo on id(), so a dict referenced
  twice is still one object afterwards and pickle records it once; the memo also
  makes cyclic structures safe.
* The rebuild is non-destructive: the caller's containers are not modified.

Key order is "natural": digit runs compare numerically, so currentMode_2
precedes currentMode_10 rather than following it. That is a readability
choice for anyone diffing two pools by eye; any total order would do for
reproducibility.
"""

import re
from typing import Any, Dict, List, Tuple

__all__ = ["canonicalize", "sort_key"]

_DIGITS = re.compile(r"(\d+)")


def _natural(text: str) -> Tuple[Tuple[int, int, str], ...]:
    """Split a string into comparable chunks, digit runs compared as numbers."""
    out: List[Tuple[int, int, str]] = []
    for i, part in enumerate(_DIGITS.split(text)):
        if i % 2:
            out.append((0, int(part), ""))
        else:
            out.append((1, 0, part))
    return tuple(out)


def sort_key(key: Any) -> Tuple[Any, ...]:
    """A total, process-independent order over the key types a payload holds.

    Variables (Bool, Real, ...) carry a string ``id`` and sort by it, with the
    class name as tiebreak so a Bool and a Real of the same name keep a fixed
    relative position. Everything else -- plain strings, and the formula
    objects used as keys in the proposition dictionary -- sorts by its string
    form. The leading rank keeps the three families apart, so the comparison
    never has to order an id against a formula.
    """
    ident = getattr(key, "id", None)
    if isinstance(ident, str):
        return (0, _natural(ident), type(key).__name__)
    if isinstance(key, str):
        return (1, _natural(key), "")
    try:
        return (2, _natural(str(key)), type(key).__name__)
    except Exception:
        return (3, _natural(repr(type(key))), "")


def canonicalize(obj: Any, _memo: Dict[int, Any] = None) -> Any:
    """Return ``obj`` with every reachable dict re-inserted in ``sort_key`` order.

    Lists and tuples keep their order -- it is meaningful (the pool is a
    sequence, and its labels are positional) and it is already deterministic.
    """
    if _memo is None:
        _memo = {}
    marker = id(obj)
    if marker in _memo:
        return _memo[marker]

    if isinstance(obj, dict):
        out: Dict[Any, Any] = {}
        _memo[marker] = out
        for key in sorted(obj.keys(), key=sort_key):
            out[key] = canonicalize(obj[key], _memo)
        return out

    if isinstance(obj, list):
        out_list: List[Any] = []
        _memo[marker] = out_list
        for item in obj:
            out_list.append(canonicalize(item, _memo))
        return out_list

    if isinstance(obj, tuple):
        built = tuple(canonicalize(item, _memo) for item in obj)
        _memo[marker] = built
        return built

    return obj