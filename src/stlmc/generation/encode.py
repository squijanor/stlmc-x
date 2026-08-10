"""Reusable per-depth STLMC falsification encoding for the generation strategies.

`Encoder` reproduces the constraint tree that `SmtAlgorithm.run`
(`encoding/monolithic.py`) builds for a single depth, without running the
single-counterexample loop, so the generation strategies and their parallel
workers build the same encoding and assert it into a `GrowthOracle`. It also
assembles the `.counterexamples` pool payload from a collected pool.

Import note: `make_boolean_abstract_consts` and `substitution` are the two
helpers `monolithic.py` pulls in through star imports; they are imported here by
name from the modules that provide them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..constraints.constraints import And, BoolVal, Constant, Formula, Variable
from ..constraints.operations import substitution
from ..encoding.enumerate import make_boolean_abstract_consts
from ..encoding.monolithic import clause, k_size_stl_formula
from ..encoding.static_learning import StaticLearner
from ..objects.goal import Goal, ReachGoal
from ..objects.model import Model


@dataclass
class Encoding:
    """One depth's falsification encoding plus the data needed to read a model.

    ``consts`` is ``model_const AND contradiction AND stl_const AND
    boolean_abstract_consts`` at ``bound`` -- the same conjunction
    ``SmtAlgorithm.run`` passes to the solver. ``boolean_abstract`` maps the
    abstraction Bools to their defining formulas (needed by non-z3 backends).
    """

    consts: Formula
    #: ``consts`` WITHOUT the boolean-abstraction definitions: the propositional
    #: and arithmetic skeleton, with every ODE / forall_t term still hidden
    #: behind its abstraction Bool, so z3 can solve it even for a nonlinear
    #: model. Required by the two-step pivot.
    skeleton: Formula
    boolean_abstract: dict
    bound: int
    range_dict: dict


class Encoder:
    """Builds the falsification encoding per depth for a fixed model and goal.

    ``gen_stl_condition`` and, when requested, static learning run once at
    construction. ``encode_at`` builds one depth. ``make_consts`` accumulates
    into ``model.boolean_abstract``, so call ``reset`` between depths, matching
    the per-bound ``model.clear`` in ``SmtAlgorithm.run``.
    """

    def __init__(
        self,
        model: Model,
        goal: Goal,
        prop_dict: dict,
        delta: float,
        tau_max: float,
        *,
        static_depth: int | None = None,
    ) -> None:
        if isinstance(goal, ReachGoal):
            raise ValueError(
                "Encoder supports STL falsification goals, not reachability")
        self.model = model
        self.goal = goal
        self.prop_dict = prop_dict
        self.delta = float(delta)
        self.tau_max = float(tau_max)

        model.gen_stl_condition()

        self._static: StaticLearner | None = None
        if static_depth is not None:
            goal_f = substitution(goal.get_formula(), prop_dict)
            self._static = StaticLearner(model, goal_f)
            self._static.generate_learned_clause(static_depth, self.delta)

    def encode_at(self, bound: int) -> Encoding:
        model_const = self.model.make_consts(bound)
        stl_const = k_size_stl_formula(
            self.model, self.goal, self.prop_dict, bound, self.delta, self.tau_max
        )
        boolean_abstract = dict(self.model.boolean_abstract)
        ba_consts = make_boolean_abstract_consts(boolean_abstract)

        if self._static is not None:
            clause_in = clause(And([model_const, stl_const, ba_consts]))
            contradiction = self._static.get_contradiction_upto(bound, clause_in)
        else:
            contradiction = BoolVal("True")

        skeleton = And([model_const, contradiction, stl_const])
        consts = And([skeleton, ba_consts])
        return Encoding(
            consts=consts,
            skeleton=skeleton,
            boolean_abstract=boolean_abstract,
            bound=bound,
            range_dict=self.model.range_dict,
        )

    def reset(self) -> None:
        self.model.clear()
        self.goal.clear()

    def counterexamples_payload(
        self, assn_dicts: list[dict[Variable, Constant]], label: str
    ) -> tuple[Any, ...]:
        """Assemble the ``.counterexamples`` pool payload for a pool.

        A pickled 9-tuple in this fixed order:
        ``(assn_dicts, modules, mode_var_dict, prop_dict, range_dict, PD,
        formula, label, delta)``, where ``assn_dicts`` is a list of one solver
        assignment dict per counterexample. The multi-CE analogue of the
        single-CE ``.counterexample`` tuple written by ``base_driver``.
        """
        return (
            list(assn_dicts),
            self.model.modules,
            self.model.mode_var_dict,
            self.model.prop_dict,
            self.model.range_dict,
            self.prop_dict,
            self.goal.get_formula(),
            label,
            float(self.delta),
        )