"""Reduced-query reconstruction for kappa_box's two-step pivot (the delta fix).

The two-step pivot's ceiling is that it pins onto ``encoding.consts``, which keeps
every quantified subformula (60 forall_t on AUV-ode f2 depth 2 = 48 property + 12
invariant) that dReal must integrate -- the intrinsic ~615 s cost that the only
working pin (``free``) pays per query, and that ``consts`` keeps no matter what is
pinned on top.

The base checker (``EnumerateAlgorithm``) is fast on the identical encoding
because it does not pin onto ``consts`` -- it REPLACES it: from an accepted
structure it minimizes to a sufficient core against the falsification target and
reconstructs a reduced query (``assn2path`` / ``path2const`` / ``time_path2const``)
carrying ONLY the core-selected property forall_t, keeping the full model
execution. This module ports exactly that, for a single pivot depth, so
kappa_box's delta two-step can hand dReal the reduced query (and grow on it)
instead of ``consts``.

It is a faithful transcription of the pivot-relevant half of
``EnumerateAlgorithm.scenario_check`` (``encoding/enumerate.py``), specialized to
one bound and driven by the un-collapsed components that
``Encoder.enumerate_components_at`` exposes. Every reconstruction primitive
(``assn2path``, ``path2const``, ``time_path2const``, ``pick_time_and_props``,
``contradiction_gen``, ``contradiction_gen_inv``) is the base checker's own, so
the reduced query this builds is the same object the base checker would hand
dReal -- verified in the sandbox to reproduce its forall_t count.

The recursive minimize target ``current_minimize_info`` (``enumerate.py:199``) is
used here in its flattened form. Unrolling the ``unsat@k`` recursion,
``unsat@0`` is equivalent to::

    init  AND  AND_{b<N} [ k_model_f(b) AND k_stl_f(b) AND k_stl_time_f(b) ]
              AND        [ model_f_k_final(N) AND k_stl_f(N) AND k_stl_time_f(N)
                           AND time_order(N) ]

(earlier bounds use the non-final model consts and no time order; only the pivot
bound uses the final model consts with the time order -- exactly the two
``current_minimize_info`` shapes at ``enumerate.py:199`` and ``:341``). ``stl_final``
is NOT part of the minimize target (the base checker adds it to the scenario
solver and to ``total_const``, never to F). Minimizing candidate literals against
``Not(F)`` therefore gives the same small core -- and hence the same small
property path -- as the recursive solver, with no push/pop bookkeeping.
"""

from __future__ import annotations

import z3

from ..constraints.constraints import (
    And,
    Bool,
    BoolVal,
    Eq,
    Int,
    Not,
    Real,
)
from ..constraints.operations import get_vars
from ..encoding.enumerate import (
    assn2path,
    contradiction_gen,
    contradiction_gen_inv,
    path2const,
    pick_time_and_props,
    time_path2const,
)
from ..encoding.monolithic import clause
from ..solver.z3 import Z3Assignment, z3Obj

# Native verdicts, matching generation.oracle.
SAT = "sat"
UNSAT = "unsat"
UNKNOWN = "unknown"


class ReducedPivotSearch:
    """Base-checker reduced-query pivot search at one depth, for kappa_box.

    Construction sets up the base checker's scenario solver at bound ``N`` (init
    consts + the accumulated next-form paths for bounds ``< N`` + the final-form
    path at ``N`` + ``stl_final`` + the contradiction clauses) and precomputes the
    flattened minimize target ``Not(F)`` and the clause set. Each :meth:`next`
    solves for one falsifying structure, minimizes it to a sufficient core,
    reconstructs the reduced dReal query ``total_const`` and its z3-expressible
    twin ``path_const``, and blocks that structure so the following :meth:`next`
    yields a different one -- exactly the base checker's ``generalized_symbolic_path``
    enumeration.

    ``model.boolean_abstract`` must hold the abstraction map that
    ``enumerate_components_at`` populated (the constructor snapshots ``components``'
    copy and uses that), since ``path2const`` and the model-execution term read the
    ODE integrals and invariants from it.
    """

    def __init__(self, components, model, *, seed: int | None = None,
                 timeout_ms: int | None = None) -> None:
        self.model = model
        self.tau_max = components.tau_max
        self.sub_formulas = components.sub_formulas
        self.boolean_abstract = components.boolean_abstract
        self.seed = seed
        self.timeout_ms = timeout_ms
        N = int(components.bound)
        self.N = N

        init_conj = And([components.initial_model_f,
                         components.initial_stl_f,
                         components.initial_track_const])
        # next-form structural path per earlier bound (no time order, no final).
        nexts = [And([components.model_consts[b],
                      components.stl_consts[b],
                      components.stl_time_consts[b]]) for b in range(N)]
        # final-form structural path at the pivot bound (final model + time order).
        n_path_N = And([components.model_f_k_final,
                        components.stl_consts[N],
                        components.stl_time_consts[N],
                        components.time_order_const])
        self.stl_final = components.final_f_k

        # Flattened recursive falsification target F (see module docstring).
        self._not_F = Not(And([init_conj] + nexts + [n_path_N]))

        # Clause set the minimizer draws real/timing atoms from (enumerate.py:102,
        # 211-212): the clauses of the init consts and of every structural path.
        cs = set()
        cs |= clause(init_conj)
        for b in range(N):
            cs |= clause(nexts[b])
        cs |= clause(n_path_N)
        self.clause_set = cs

        # Scenario solver: the abstract structure z3 searches for a falsifying
        # skeleton over (enumerate.py:94-96, 208-222). Native z3 so the model can
        # be read back for minimize signing, like the base checker.
        s = z3.SolverFor("QF_LRA")
        if seed is not None:
            s.set("random_seed", int(seed))
        if timeout_ms:
            s.set("timeout", int(timeout_ms))
        s.add(z3Obj(init_conj))
        for b in range(N):
            s.add(z3Obj(nexts[b]))
        s.add(z3Obj(n_path_N))
        s.add(z3Obj(self.stl_final))
        contra_v, contra_e = contradiction_gen(self.clause_set, self.sub_formulas)
        contra_v_inv = contradiction_gen_inv(self.boolean_abstract)
        s.add(z3Obj(contra_v))
        s.add(z3Obj(contra_e))
        s.add(z3Obj(contra_v_inv))
        self._scenario = s
        self._last_verdict = None

    def add_block(self, formula) -> None:
        """Assert a persistent block into the scenario solver (kappa_box's IC
        region blocks, or any structure to exclude)."""
        self._scenario.add(z3Obj(formula))

    def last_verdict(self) -> str | None:
        return self._last_verdict

    def next(self):
        """Next reduced pivot, or ``None`` when the structure space is exhausted
        or the scenario solve gives up.

        Returns ``(total_const, path_const, assn)``: ``total_const`` is the reduced
        dReal query, ``path_const`` its z3 twin (block ``Not(path_const)`` on the
        candidate oracle), ``assn`` the scenario assignment. The found structure is
        blocked internally so a subsequent call yields a different one.
        """
        r = self._scenario.check()
        if r == z3.sat:
            self._last_verdict = SAT
        elif r == z3.unsat:
            self._last_verdict = UNSAT
            return None
        else:
            self._last_verdict = UNKNOWN
            return None

        m = self._scenario.model()
        assn = Z3Assignment(m).get_assignments()
        total_const, path_const = self._reconstruct(m, assn)
        # Generalize: block this structure (enumerate.py:326-327) so next() moves on.
        self._scenario.add(z3Obj(Not(path_const)))
        return total_const, path_const, assn

    def _reconstruct(self, m, assn):
        """Minimize the candidate to a sufficient core and reconstruct the reduced
        query -- a transcription of ``scenario_check`` lines 239-317 (STL branch),
        using the flattened target so no recursive solver is threaded."""
        true_ = BoolVal("True")
        false_ = BoolVal("False")

        s = z3.Solver()
        if self.seed is not None:
            s.set("random_seed", int(self.seed))
        s.set("core.minimize", True)
        if self.timeout_ms:
            s.set("timeout", int(self.timeout_ms))
        s.add(z3Obj(self._not_F))

        # Sign the candidate's Boolean literals: true ones are tracked assumptions
        # (their track id enters the core), false ones are hard. (enumerate.py:244-252)
        true_bool_ids: set[str] = set()
        real_set = set()
        for v in assn:
            val = assn[v]
            if isinstance(v, Bool) and isinstance(val, BoolVal):
                track_id = f"p@{v.id}"
                if val == true_:
                    true_bool_ids.add(track_id)
                    s.assert_and_track(z3Obj(Eq(v, true_)), track_id)
                else:
                    s.add(z3Obj(Eq(v, false_)))
            elif isinstance(v, (Real, Int)):
                real_set.add(v)

        # Sign the real/timing clauses touching those reals. (enumerate.py:257-266)
        real_dict: dict[str, object] = {}
        for c in self.clause_set:
            if get_vars(c).intersection(real_set):
                track_id = f"p@real_{id(c)}"
                if m.eval(z3Obj(c)):
                    real_dict[track_id] = c
                    s.assert_and_track(z3Obj(Eq(c, true_)), track_id)
                else:
                    s.add(z3Obj(Eq(c, false_)))

        s.check()
        cores = {str(x) for x in s.unsat_core()}
        p_reals = cores.difference(true_bool_ids)
        p_bools = cores.difference(p_reals)

        # Keep only the path-relevant time/prop Bools; drop intermediate goals.
        p_bools = pick_time_and_props(p_bools, self.sub_formulas)  # (enumerate.py:283)
        path_bool_consts = {Bool(p.replace("p@", "")) for p in p_bools}
        path_real_consts = [real_dict[p] for p in p_reals if p in real_dict]
        path_const = And(list(path_bool_consts) + list(path_real_consts))

        # Full model execution: EVERY abstraction (ODE integrals + invariants).
        model_abstract_const = And(
            [Eq(v, self.boolean_abstract[v]) for v in self.boolean_abstract])
        # Reduced property path: only the core-selected forall_t. (enumerate.py:295-298)
        extra_prop_path, extra_time_path = assn2path(
            p_bools, self.sub_formulas, self.tau_max)
        extra_prop_path_const = path2const(extra_prop_path, self.model)
        extra_time_path_const = time_path2const(extra_time_path)
        # Range consts, so the reduced query does not drop them with the core.
        range_const = And(
            [self.model.make_range_consts(d)[0] for d in range(0, self.N + 1)])

        total_const = And([path_const, extra_prop_path_const, self.stl_final,
                           extra_time_path_const, range_const,
                           model_abstract_const])
        return total_const, path_const