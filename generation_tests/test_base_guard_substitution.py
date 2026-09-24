"""Encoding tests for the base checker's guard-retaining backend query.

``EnumerateAlgorithm.scenario_check`` builds the backend query from a minimized
path constraint plus the whole model execution (init, flows, invariants, guards,
resets). The abstraction Booleans that name the flow/invariant relations are
resolved in one of two ways: the reach branch substitutes each Boolean by its
defining formula, while the STL branch does the same. These tests pin the
properties that make the two resolutions interchangeable and keep the guard in
the query:

* substitution is complete -- no abstraction Boolean survives, so conjoining the
  definitions and substituting them are equisatisfiable;
* every real atom of the model execution (guards, resets, flow relations) is
  retained, so a witness is a run of the automaton and an enabled jump keeps its
  guard;
* on a decidable linear instance the substituted and the conjoined encodings
  return the same verdict, for both a satisfiable and an unsatisfiable query.

They are binary-free (no dReal, no benchmark model); the ODE timing and the
SAT/UNSAT agreement on the ODE benchmarks are the base A/B on the pinned set.
"""

import os

import pytest
import z3

from stlmc.constraints.constraints import (
    And,
    Bool,
    Eq,
    Geq,
    Leq,
    Or,
    Real,
    RealVal,
)
from stlmc.constraints.operations import get_vars, substitution
from stlmc.generation import encode as _encode  # noqa: F401  resolve import order
from stlmc.objects.object_factory import ObjectFactory
from stlmc.solver.z3 import z3Obj

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURES = os.path.join(_REPO, "generation_tests", "fixtures")
_MODELS = {
    "path_branch": os.path.join(_FIXTURES, "path_branch.model"),
    "box_region": os.path.join(_FIXTURES, "box_region.model"),
}

_BOUND = 2
_INEQ = (Geq, Leq)


# --------------------------------------------------------------------------- #
#  build the base STL backend query the way scenario_check does
# --------------------------------------------------------------------------- #
def _encode_base(model_file, bound=_BOUND):
    """Return (model_execution, model_abstract_const) for a parsed model.

    Mirrors the model side of ``EnumerateAlgorithm.run`` /
    ``scenario_check`` on the STL (non-reach) path: clear the abstraction map,
    generate the STL condition, then accumulate the per-step model constraints.
    The property/path part of the query is scenario-specific and orthogonal to
    the abstraction resolution under test, so it is left out.
    """
    om = ObjectFactory("model-with-goal-enhanced").generate_object_manager()
    model, _prop_dict, _goals, _labels = om.generate_objects(model_file)

    model.boolean_abstract.clear()
    model.gen_stl_condition()

    initial_model_f, _ = model.init_consts()
    acc_model = [model.k_step_consts(b)[0] for b in range(bound + 1)]
    model_f_k_final = model.k_step_consts(bound, is_final=True)[0]

    model_execution = And(
        [initial_model_f] + [acc_model[b] for b in range(bound)] + [model_f_k_final]
    )
    model_abstract_const = And(
        [Eq(v, model.boolean_abstract[v]) for v in model.boolean_abstract]
    )
    return model, model_execution, model_abstract_const


def _reduction_of(model_abstract_const):
    reduction = dict()
    for mac in model_abstract_const.children:
        reduction[mac.left] = mac.right
    return reduction


def _count_ineqs(const):
    total = 1 if isinstance(const, _INEQ) else 0
    for child in getattr(const, "children", []):
        total += _count_ineqs(child)
    left, right = getattr(const, "left", None), getattr(const, "right", None)
    for side in (left, right):
        if side is not None and not isinstance(const, _INEQ):
            total += _count_ineqs(side)
    child = getattr(const, "child", None)
    if child is not None:
        total += _count_ineqs(child)
    return total


# --------------------------------------------------------------------------- #
#  1. substitution is complete: no abstraction Boolean remains
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(_MODELS))
def test_substitution_leaves_no_abstraction_boolean(name):
    model, model_execution, model_abstract_const = _encode_base(_MODELS[name])
    abstraction_bools = set(model.boolean_abstract.keys())
    assert abstraction_bools, "fixture has no abstraction Booleans to resolve"

    total = substitution(model_execution, _reduction_of(model_abstract_const))

    remaining = abstraction_bools.intersection(get_vars(total))
    assert remaining == set(), (
        f"abstraction Booleans survived substitution: {remaining}"
    )


# --------------------------------------------------------------------------- #
#  2. the reduction reproduces the abstraction map exactly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(_MODELS))
def test_reduction_reproduces_the_abstraction_map(name):
    model, _me, model_abstract_const = _encode_base(_MODELS[name])
    for mac in model_abstract_const.children:
        assert isinstance(mac, Eq)
    reduction = _reduction_of(model_abstract_const)
    assert reduction == dict(model.boolean_abstract)


# --------------------------------------------------------------------------- #
#  3. guards, resets and flow relations are retained through substitution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(_MODELS))
def test_real_atoms_of_the_model_execution_are_retained(name):
    _model, model_execution, model_abstract_const = _encode_base(_MODELS[name])
    total = substitution(model_execution, _reduction_of(model_abstract_const))

    me_reals = {v for v in get_vars(model_execution) if isinstance(v, Real)}
    total_reals = {v for v in get_vars(total) if isinstance(v, Real)}
    # substitution only rewrites Boolean leaves, so every real variable of the
    # execution -- guard, reset and flow variables -- must survive.
    assert me_reals <= total_reals
    # ... and no inequality atom (a guard or invariant bound) is dropped.
    assert _count_ineqs(total) >= _count_ineqs(model_execution)


# --------------------------------------------------------------------------- #
#  4. on a decidable linear query the two encodings agree (z3 is ground truth)
#     conjoining And(b == def) and substituting b -> def are equisatisfiable
#     when the substitution is complete; z3 decides both directly.
# --------------------------------------------------------------------------- #
def _z3_sat(const):
    solver = z3.Solver()
    solver.add(z3Obj(const))
    return solver.check() == z3.sat


def _conjoin_vs_subst(query, definitions):
    conjoined = And([query] + [Eq(b, d) for b, d in definitions.items()])
    substituted = substitution(query, definitions)
    return _z3_sat(conjoined), _z3_sat(substituted)


def test_satisfiable_linear_query_agrees_across_encodings():
    x, y = Real("x"), Real("y")
    b0, b1 = Bool("b0"), Bool("b1")
    definitions = {b0: Geq(x, RealVal("1")), b1: Leq(y, RealVal("2"))}
    # an Or over the abstraction Booleans, the shape of the module disjunction
    query = And(
        [
            Or([b0, b1]),
            Geq(x, RealVal("0")),
            Leq(x, RealVal("5")),
            Geq(y, RealVal("0")),
            Leq(y, RealVal("5")),
        ]
    )

    conjoined_sat, substituted_sat = _conjoin_vs_subst(query, definitions)
    assert conjoined_sat is True
    assert substituted_sat == conjoined_sat


def test_guard_unsatisfiable_query_is_impossible_in_both_encodings():
    x = Real("x")
    b0 = Bool("b0")
    definitions = {b0: Geq(x, RealVal("1"))}
    # the jump is enabled (b0 asserted) but its guard x >= 1 cannot hold with
    # x <= 0, so the query must be unsatisfiable under either resolution.
    query = And([b0, Leq(x, RealVal("0"))])

    conjoined_sat, substituted_sat = _conjoin_vs_subst(query, definitions)
    assert conjoined_sat is False
    assert substituted_sat == conjoined_sat


def test_a_free_boolean_would_hide_the_unsat_guard():
    """Control for the test above: without the definition the enabled jump is
    satisfiable, so it is the retained guard -- not the Boolean alone -- that
    makes the infeasible jump impossible."""
    x = Real("x")
    b0 = Bool("b0")
    query = And([b0, Leq(x, RealVal("0"))])
    assert _z3_sat(query) is True
    assert _z3_sat(substitution(query, {b0: Geq(x, RealVal("1"))})) is False
