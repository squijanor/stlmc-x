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

import hashlib
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..constraints.constraints import (
    And,
    Bool,
    BoolVal,
    Constant,
    Formula,
    Not,
    Variable,
)
from ..constraints.operations import (
    reduce_not,
    relaxing,
    remove_binary,
    substitution,
)
from ..encoding.enumerate import (
    calc_sub_formulas,
    chi,
    k_depth_stl_consts,
    make_boolean_abstract_consts,
    time_ordering,
)
from ..encoding.monolithic import clause, k_size_stl_formula
from ..encoding.static_learning import StaticLearner
from ..objects.goal import Goal, ReachGoal
from ..objects.model import Model


@dataclass
class Encoding:
    """One depth's falsification encoding plus the data needed to read a model.

    ``consts`` is ``model_const AND contradiction AND stl_const`` at ``bound``
    with the abstraction Booleans substituted by their definitions -- the same
    query ``SmtAlgorithm.run`` passes to the solver, and what the exact path
    hands the solver directly. ``boolean_abstract`` maps the abstraction Bools to
    their defining formulas. The delta path does not read ``consts``: it
    reconstructs a reduced query from :meth:`Encoder.enumerate_components_at`
    instead.
    """

    consts: Formula
    boolean_abstract: dict
    bound: int
    range_dict: dict


@dataclass
class StlComponents:
    """The base checker's STL + model components, un-collapsed, exposed per depth.

    ``encode_at`` builds the MONOLITHIC ``stl_const`` (via ``k_size_stl_formula``),
    which reconstructs the property path over ALL of its Bools, so
    ``encoding.consts`` carries every property ``forall_t``. That is the whole
    cost the two-step pivot cannot get under while it pins onto ``consts``: no
    matter what Booleans are pinned on top, ``consts`` keeps all the quantified
    subformulas dReal must integrate.

    The base checker (``EnumerateAlgorithm``) is fast on the identical encoding
    because it does not pin onto ``consts`` -- it REPLACES it, reconstructing a
    reduced query (``assn2path`` / ``path2const`` / ``time_path2const``) that
    carries ONLY the core-selected ``forall_t``. To port that into ``box.py`` the
    two-step needs the same raw material the base checker's ``run`` holds when it
    calls ``scenario_check`` at a given bound: the accumulated per-bound STL goal
    and timing definitions, the final condition, the time order, ``sub_formulas``,
    and the model consts (both the non-final ``k_model_f`` and the final
    ``model_f_k_final``). This dataclass is exactly that state, un-collapsed.

    Every field is computed the same way ``EnumerateAlgorithm.run`` computes it
    (``encoding/enumerate.py``); the reuse is deliberate so the two cannot drift.
    Fields map to ``scenario_check``'s parameters as:

      - ``model_consts[b]``          -> ``acc_model[b]`` (``k_model_f``)
      - ``model_f_k_final``          -> ``model_f_k_final`` (final model consts)
      - ``stl_consts[b]``            -> ``acc_stl[b]`` (``k_stl_f``)
      - ``stl_time_consts[b]``       -> ``acc_stl_time[b]`` (``k_stl_time_f``)
      - ``final_f_k``                -> ``stl_final``
      - ``time_order_const``         -> ``stl_time_order``
      - ``sub_formulas``             -> ``sub_formulas``

    so a caller can build ``current_minimize_info`` (``enumerate.py:199``),
    minimize against it, and reconstruct the property path -- without re-reading
    config (``tau_max`` / ``delta`` are carried) and without touching the
    monolithic ``stl_const``.

    ``boolean_abstract`` is a snapshot of the abstraction map taken after
    building. The reconstruction resolves the ODE-integral and continuous-
    invariant Booleans against it; the guards and resets are kept by retaining
    the full model execution in the query, not by the abstraction map.
    """

    bound: int
    tau_max: float
    delta: float
    sub_formulas: set
    #: chi(1, 1, stl_formula): the initial STL choice the base checker seeds the
    #: scenario solver with, and the base of the accumulated target.
    initial_stl_f: Formula
    initial_model_f: Formula
    initial_track_const: Formula
    #: Per-bound model consts, index = bound, 0..``bound`` (non-final k_model_f).
    model_consts: list
    model_track_consts: list
    #: model.k_step_consts(bound, is_final=True): the final segment's model consts.
    model_f_k_final: Formula
    model_track_f_k_final: Formula
    #: Per-bound accumulated STL goal / timing definitions, index = bound.
    stl_consts: list
    stl_time_consts: list
    #: STL final condition at depth 2*bound+2 (the last depth's `final`).
    final_f_k: Formula
    #: time_ordering(2*bound+2, tau_max).
    time_order_const: Formula
    #: Snapshot of model.boolean_abstract after building all bounds 0..``bound``.
    boolean_abstract: dict = field(default_factory=dict)


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
                "Encoder supports STL falsification goals, not reachability"
            )
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

        # Resolve the abstraction Booleans by substitution rather than conjoining
        # their definitions, matching SmtAlgorithm and the reduced query.
        consts = And([model_const, contradiction, stl_const])
        consts = substitution(consts, boolean_abstract)
        return Encoding(
            consts=consts,
            boolean_abstract=boolean_abstract,
            bound=bound,
            range_dict=self.model.range_dict,
        )

    def enumerate_components_at(self, bound: int) -> StlComponents:
        """Expose the base checker's un-collapsed STL + model components at ``bound``.

        This reproduces the state ``EnumerateAlgorithm.run`` holds when it calls
        ``scenario_check`` for ``bound`` (``encoding/enumerate.py``), by calling
        the SAME per-depth builders (``model.k_step_consts``,
        ``k_depth_stl_consts``, ``time_ordering``, ``calc_sub_formulas``,
        ``chi``). Nothing is collapsed into a monolithic ``stl_const`` and no
        full-Bool ``path2const`` is run, so the caller keeps the freedom to
        minimize against the recursive falsification target first and reconstruct
        the property path from the CORE only.

        Side effect: like the base checker's ``run`` (which does
        ``model.boolean_abstract.clear()`` before its bound loop), this clears and
        repopulates ``model.boolean_abstract`` for bounds 0..``bound``. It is
        therefore an ALTERNATIVE encode path to ``encode_at`` on the same
        ``Encoder`` -- pick one per pivot; do not interleave their model state.
        The returned ``boolean_abstract`` is a snapshot, so the abstraction map
        the caller substitutes with is stable even if the model is later reset.
        """
        model = self.model
        # Match EnumerateAlgorithm.run: a clean abstraction map, STL condition on.
        model.boolean_abstract.clear()
        model.gen_stl_condition()

        # Same falsification target the base checker derives (enumerate.py:80-90).
        raw_stl_formula = substitution(self.goal.get_formula(), self.prop_dict)
        neg_formula = reduce_not(Not(raw_stl_formula))
        reduced_formula = remove_binary(neg_formula)
        stl_formula = relaxing(reduced_formula, self.delta)
        sub_formulas = calc_sub_formulas(stl_formula)
        initial_stl_f = chi(1, 1, stl_formula)

        initial_model_f, initial_track_const = model.init_consts()

        model_consts: list[Formula] = []
        model_track_consts: list[Formula] = []
        stl_consts: list[Formula] = []
        stl_time_consts: list[Formula] = []
        final_f_k: Formula | None = None
        time_order_const: Formula | None = None
        model_f_k_final: Formula | None = None
        model_track_f_k_final: Formula | None = None

        for b in range(0, int(bound) + 1):
            # Model consts: the non-final k_model_f and the final one, exactly as
            # run does (enumerate.py:118-121). Both mutate boolean_abstract; the
            # keys are per (module, bound) so re-assignment is idempotent.
            model_f_k, track_f_k = model.k_step_consts(b)
            model_f_k_final, model_track_f_k_final = model.k_step_consts(
                b, is_final=True
            )
            model_consts.append(model_f_k)
            model_track_consts.append(track_f_k)

            # STL goal / timing definitions accumulated over the two depths of
            # bound b (enumerate.py:130-145). final_f_k is the last depth's final.
            stl_children: list[Formula] = []
            time_children: list[Formula] = []
            for d in range(2 * b + 1, 2 * b + 3):
                stl_f_d, time_f_d, final_f_d = k_depth_stl_consts(
                    sub_formulas, d, self.tau_max
                )
                stl_children.append(stl_f_d)
                time_children.append(time_f_d)
                final_f_k = final_f_d
            time_order_const = time_ordering(2 * b + 2, self.tau_max)
            stl_consts.append(And(stl_children))
            stl_time_consts.append(And(time_children))

        assert final_f_k is not None and time_order_const is not None
        return StlComponents(
            bound=int(bound),
            tau_max=self.tau_max,
            delta=self.delta,
            sub_formulas=sub_formulas,
            initial_stl_f=initial_stl_f,
            initial_model_f=initial_model_f,
            initial_track_const=initial_track_const,
            model_consts=model_consts,
            model_track_consts=model_track_consts,
            model_f_k_final=model_f_k_final,
            model_track_f_k_final=model_track_f_k_final,
            stl_consts=stl_consts,
            stl_time_consts=stl_time_consts,
            final_f_k=final_f_k,
            time_order_const=time_order_const,
            boolean_abstract=dict(model.boolean_abstract),
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


_MODE_RE = re.compile(r"^currentMode_(\d+)$")


def _mode_word(assn: dict[Variable, Constant]) -> list[int]:
    """The ``currentMode_k`` integer values ordered by step ``k``."""
    steps: list[tuple[int, int]] = []
    for var, const in assn.items():
        m = _MODE_RE.match(var.id)
        if m is not None:
            value = int(round(float(Fraction(str(const.value)))))
            steps.append((int(m.group(1)), value))
    steps.sort(key=lambda kv: kv[0])
    return [value for _, value in steps]


def _reduce_word(raw: list[int]) -> list[int]:
    """Collapse consecutive equal modes (``0,0,1,1`` -> ``0,1``)."""
    out: list[int] = []
    for m in raw:
        if not out or m != out[-1]:
            out.append(m)
    return out


def _sigma(assn: dict[Variable, Constant]) -> list[list[Any]]:
    """The Boolean-abstraction truth assignment as id-sorted ``[id, value]`` pairs."""
    pairs = [
        [var.id, str(const.value) == "True"]
        for var, const in assn.items()
        if isinstance(var, Bool)
    ]
    pairs.sort(key=lambda pair: pair[0])
    return pairs


def _structure_id(reduced_word: list[int], sigma: list[list[Any]]) -> str:
    """A stable key over ``(reduced_word, sigma)`` for structurally equal CEs."""
    digest = hashlib.blake2b(digest_size=16)
    canonical = repr((tuple(reduced_word), tuple((i, v) for i, v in sigma)))
    digest.update(canonical.encode())
    return digest.hexdigest()


def structure_signature_records(
    assn_dicts: list[dict[Variable, Constant]],
) -> list[dict[str, Any]]:
    """One structure-signature record per pooled counterexample (payload index 11).

    Each record maps ``raw_word`` (the ``currentMode_k`` word), ``reduced_word``
    (its run-length collapse), ``sigma`` (the Boolean-abstraction assignment as
    id-sorted ``[id, value]`` pairs), and ``structure_id`` (a stable key over
    ``(reduced_word, sigma)``). Every field is read from the assignment dict;
    counterexamples with equal ``(reduced_word, sigma)`` share one ``reduced_word``
    and ``sigma`` object.
    """
    records: list[dict[str, Any]] = []
    shared: dict[str, tuple[list[int], list[list[Any]]]] = {}
    for assn in assn_dicts:
        raw = _mode_word(assn)
        reduced = _reduce_word(raw)
        sigma = _sigma(assn)
        sid = _structure_id(reduced, sigma)
        reduced, sigma = shared.setdefault(sid, (reduced, sigma))
        records.append(
            {
                "raw_word": raw,
                "reduced_word": reduced,
                "sigma": sigma,
                "structure_id": sid,
            }
        )
    return records
