"""Encoder test: make_flow_consts retains the exact constant-flow equation in the
selected-mode branch of the step encoding."""
from stlmc.constraints.constraints import (
    Add,
    And,
    Eq,
    Int,
    IntVal,
    Leq,
    Mul,
    Ode,
    Real,
    RealVal,
    Sub,
)
from stlmc.objects.model import StlMC


def _one_mode_model(rate):
    mode_var = Int("m")
    modules = [{
        "mode": And([Eq(mode_var, IntVal("0"))]),
        "flow": Ode([Real("v")], [RealVal(rate)]),
        "inv": And([Leq(Real("v"), RealVal("100"))]),
        "jump": {},
    }]
    model = StlMC({"m": mode_var}, {Real("v"): (True, 0.0, 100.0, True)}, {}, {},
                  modules, And([Eq(Real("v"), RealVal("0"))]))
    model.gen_stl_condition()
    return model


def test_make_flow_consts_retains_constant_flow():
    model = _one_mode_model("2")
    _children, _integrals, linear = model.make_flow_consts(0)
    emitted = " ".join(str(c) for c in linear[0].children)
    state_eq = Eq(Real("v_0_t"),
                  Add(Real("v_0_0"), Mul(RealVal("2"), Real("time_0"))))
    assert str(state_eq) in emitted
    # the repaired step-zero duration identity: time_0 = tau_1 - tau_0
    duration_eq = Eq(Real("time_0"), Sub(Real("tau_1"), Real("tau_0")))
    assert str(duration_eq) in emitted


def test_constant_flow_is_in_the_selected_mode_branch():
    model = _one_mode_model("2")
    step = str(model.k_step_consts(0)[0])
    # the exact relation sits alongside the mode selector in the step encoding
    assert "(v_0_t = (v_0_0 + (2 * time_0)))" in step
    assert "currentMode_0" in step