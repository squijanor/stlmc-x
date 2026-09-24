"""Tests for LinearWordFeasibilityFilter on models built from the constraint AST."""

from types import SimpleNamespace

from stlmc.constraints.constraints import (
    And,
    Div,
    Eq,
    Geq,
    Leq,
    Mul,
    Ode,
    Real,
    RealVal,
)
from stlmc.generation.feasibility import LinearWordFeasibilityFilter

NEXT = "'"


def _model(modules, init, ranges):
    """A minimal object exposing exactly the attributes the filter reads."""
    return SimpleNamespace(modules=modules, init=init, range_dict=ranges, next_str=NEXT)


def _identity_reset(*vids):
    """The reset half of a jump post: v' = v for each variable."""
    return And([Eq(Real(v + NEXT), Real(v)) for v in vids])


def _flow(**rates):
    variables, exps = [], []
    for v, c in rates.items():
        variables.append(Real(v))
        exps.append(RealVal(str(c)))
    return Ode(variables, exps)


def _rng(lo, hi):
    return (True, float(lo), float(hi), True)


def test_sound_pruning():
    # v climbs at rate 1 under invariant v <= 1, but the horizon (3) forces
    # v to 3: impossible.
    m = _model(
        modules=[
            {"flow": _flow(v=1), "inv": And([Leq(Real("v"), RealVal("1"))]), "jump": {}}
        ],
        init=And([Eq(Real("v"), RealVal("0"))]),
        ranges={Real("v"): _rng(0, 10)},
    )
    f = LinearWordFeasibilityFilter(m, time_bound=3)
    assert f.word_is_infeasible(0, [0]) is True


def test_no_false_pruning():
    # An absorbing mode 1 (dv/dt = 0) reachable from mode 0 lets the horizon be
    # filled, so the word must NOT be rejected.
    modules = [
        {
            "flow": _flow(v=1),
            "inv": And([Leq(Real("v"), RealVal("1"))]),
            "jump": {Geq(Real("v"), RealVal("0")): _identity_reset("v")},
        },
        {"flow": _flow(v=0), "inv": And([]), "jump": {}},
    ]
    m = _model(modules, And([Eq(Real("v"), RealVal("0"))]), {Real("v"): _rng(0, 10)})
    f = LinearWordFeasibilityFilter(m, time_bound=3)
    assert f.word_is_infeasible(0, [0, 1]) is False


def test_property_independence():
    import inspect

    sig = inspect.signature(LinearWordFeasibilityFilter.word_is_infeasible)
    assert all(
        "prop" not in p and "stl" not in p and "goal" not in p for p in sig.parameters
    )
    m = _model(
        modules=[
            {"flow": _flow(v=1), "inv": And([Leq(Real("v"), RealVal("1"))]), "jump": {}}
        ],
        init=And([Eq(Real("v"), RealVal("0"))]),
        ranges={Real("v"): _rng(0, 10)},
    )
    f = LinearWordFeasibilityFilter(m, time_bound=3)
    assert f.word_is_infeasible(0, [0]) == f.word_is_infeasible(0, [0]) is True
    assert list(f.cache.keys()) == [(0, (0,))]


def test_nonlinear_conservative():
    # The only variable has a nonlinear derivative (v*v), so no linear fact can
    # be extracted: the filter must defer, never reject.
    m = _model(
        modules=[
            {
                "flow": Ode([Real("v")], [Mul(Real("v"), Real("v"))]),
                "inv": And([Leq(Real("v"), RealVal("1"))]),
                "jump": {},
            }
        ],
        init=And([Eq(Real("v"), RealVal("0"))]),
        ranges={Real("v"): _rng(0, 10)},
    )
    f = LinearWordFeasibilityFilter(m, time_bound=3)
    assert f.word_is_infeasible(0, [0]) is False


def test_guard_disjunction_sound():
    # Mode 0 has two outgoing jumps with different guards to different targets.
    # A run that takes the v-guard has w below the w-guard; ANDing the guards
    # would wrongly reject it, ORing must not.
    modules = [
        {
            "flow": _flow(v=1, w=0),
            "inv": And([Leq(Real("v"), RealVal("10"))]),
            "jump": {
                Geq(Real("v"), RealVal("2")): _identity_reset("v", "w"),
                Geq(Real("w"), RealVal("5")): _identity_reset("v", "w"),
            },
        },
        {"flow": _flow(v=0, w=0), "inv": And([]), "jump": {}},
    ]
    m = _model(
        modules,
        And([Eq(Real("v"), RealVal("0")), Eq(Real("w"), RealVal("0"))]),
        {Real("v"): _rng(0, 100), Real("w"): _rng(0, 0)},
    )
    f = LinearWordFeasibilityFilter(m, time_bound=3, time_horizon=3)
    assert f.word_is_infeasible(0, [0, 1]) is False


def test_guard_prunes():
    # The only exit guard (v >= 5) is unsatisfiable under the invariant (v <= 1),
    # so no run can leave mode 0: infeasible.
    modules = [
        {
            "flow": _flow(v=1),
            "inv": And([Leq(Real("v"), RealVal("1"))]),
            "jump": {Geq(Real("v"), RealVal("5")): _identity_reset("v")},
        },
        {"flow": _flow(v=0), "inv": And([]), "jump": {}},
    ]
    m = _model(modules, And([Eq(Real("v"), RealVal("0"))]), {Real("v"): _rng(0, 100)})
    f = LinearWordFeasibilityFilter(m, time_bound=3)
    assert f.word_is_infeasible(0, [0, 1]) is True


def _three_mode_climb():
    # v climbs at rate 1; modes 0 and 1 cap v <= 2 and exit at v >= 2; mode 2 is
    # absorbing (dv/dt = 0). A word that never reaches mode 2 cannot span a
    # horizon longer than 2.
    def climb(target):
        return {
            "flow": _flow(v=1),
            "inv": And([Leq(Real("v"), RealVal("2"))]),
            "jump": {
                Geq(Real("v"), RealVal("2")): And(
                    [
                        Eq(Real("m" + NEXT), RealVal(str(target))),
                        Eq(Real("v" + NEXT), Real("v")),
                    ]
                )
            },
        }

    modules = [climb(1), climb(2), {"flow": _flow(v=0), "inv": And([]), "jump": {}}]
    return _model(modules, And([Eq(Real("v"), RealVal("0"))]), {Real("v"): _rng(0, 10)})


def test_multistep_stay_cannot_fill_horizon():
    f = LinearWordFeasibilityFilter(_three_mode_climb(), time_bound=5)
    # staying in mode 0 across three steps: capped at v <= 2, never absorbs
    assert f.word_is_infeasible(2, [0, 0, 0]) is True
    # reaching the absorbing mode: the horizon can be filled
    assert f.word_is_infeasible(2, [0, 1, 2]) is False


def test_time_horizon_prunes():
    # v climbs at 1 under a loose invariant v <= 10. One step spans the whole
    # horizon (5), so without a per-step dwell cap the word is feasible. A
    # time_horizon of 2 caps the single dwell below 5, so it cannot fill.
    m = _model(
        [{"flow": _flow(v=1), "inv": And([Leq(Real("v"), RealVal("10"))]), "jump": {}}],
        And([Eq(Real("v"), RealVal("0"))]),
        {Real("v"): _rng(0, 100)},
    )
    assert (
        LinearWordFeasibilityFilter(m, time_bound=5).word_is_infeasible(0, [0]) is False
    )
    assert (
        LinearWordFeasibilityFilter(m, time_bound=5, time_horizon=2).word_is_infeasible(
            0, [0]
        )
        is True
    )


def test_exact_rate_is_not_pruned():
    # Rate 5/3 over dwell 3 reaches exactly v = 5, feasible under v <= 5. A float
    # rate would round to just above 5 and wrongly prune, so exact rationals are
    # required.
    m = _model(
        [
            {
                "flow": Ode([Real("v")], [Div(RealVal("5"), RealVal("3"))]),
                "inv": And([Leq(Real("v"), RealVal("5"))]),
                "jump": {},
            }
        ],
        And([Eq(Real("v"), RealVal("0"))]),
        {Real("v"): _rng(0, 100)},
    )
    f = LinearWordFeasibilityFilter(m, time_bound=3)
    assert f.word_is_infeasible(0, [0]) is False
