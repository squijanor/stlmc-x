"""End-to-end properties of a kappa_box run, on the exact backend.

These are the claims a reader of a pool is entitled to make, checked against a
real run rather than a fake oracle: every counterexample falsifies, labels line
up with the pool, theta-separation holds, the file is a function of its content,
and a verdict never covers ground the run did not visit.

The exact backend is used on purpose. Under z3 a witness is a model of the
encoding, so any failure here is the strategy's; under a delta backend the same
assertions would be measuring dReal's timing (see the reproducibility caveat in
generation-strategies.md). Runtime is a few seconds per run.
"""

import os
import pickle
import subprocess
import sys
from fractions import Fraction

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
MODEL = os.path.join(REPO, "benchmarks", "additional", "wat-poly", "water.model")

CFG = """
common {{
    solver       = "z3"
    bound        = 2
    time-bound   = 8
    threshold    = 0.1
    generation   = "box"
    verbose      = "true"
}}
gen {{
    epsilon = {epsilon}
    k-ic    = 1
    {extra}
}}
"""

THETA = Fraction(1, 100)


def run_box(tmp_path, epsilon=THETA, extra="depths = 2", goal="f2"):
    """One kappa_box run in an isolated directory; returns (stdout, pool path)."""
    cfg = tmp_path / "box.cfg"
    cfg.write_text(CFG.format(epsilon=float(epsilon), extra=extra))
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONHASHSEED="0")
    proc = subprocess.run(
        [sys.executable, "-c",
         "from stlmc.cli.mc import main; main()",
         MODEL, "-model-cfg", str(cfg), "-goal", goal, "-gen-seed", "0"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    pools = [p for p in os.listdir(str(tmp_path)) if p.endswith(".counterexamples")]
    return proc.stdout, (str(tmp_path / pools[0]) if pools else None)


def ic_values(payload, var_id):
    out = []
    for assignment in payload[0]:
        for var, const in assignment.items():
            if getattr(var, "id", "") == var_id:
                out.append(Fraction(str(const.value)))
                break
    return out


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One shared run: these assertions are independent views of one pool."""
    tmp = tmp_path_factory.mktemp("kappa_box")
    stdout, pool = run_box(tmp)
    assert pool is not None, "a falsifying run must write a pool\n" + stdout
    with open(pool, "rb") as handle:
        payload = pickle.load(handle)
    return stdout, pool, payload


def test_run_falsifies_and_writes_a_pool(run):
    stdout, _, payload = run
    assert "result : False" in stdout
    assert isinstance(payload[0], list) and payload[0], "pool must be a non-empty list"


def test_labels_are_aligned_and_from_the_vocabulary(run):
    _, _, payload = run
    assert len(payload) == 10, "labels ride the tenth payload element"
    labels = payload[9]
    assert len(labels) == len(payload[0]), "one label per counterexample"
    assert set(labels) <= {"deep", "boundary", "domain"}


def test_every_counterexample_falsifies(run):
    """The property that makes a pool a pool. Uses the tool's own STL semantics."""
    sys.path.insert(0, SRC)
    from stlmc.generation.validate import validate_pool

    _, _, payload = run
    records = validate_pool(payload, samples=25)
    bad = [r for r in records if r["verdict"] in ("unverified", "error")]
    assert not bad, "non-falsifying entries: {}".format(
        [(r["index"], r["verdict"], r["note"]) for r in bad[:5]])


def test_deep_witnesses_are_theta_separated(run):
    """theta is the minimum IC separation, as an L-infinity ball.

    NOT per-axis: two well-separated points in 2-D may have arbitrarily close
    projections on one axis, and asserting per-axis gaps would be asserting
    something the strategy never claimed (and cannot deliver -- growth along one
    axis leaves the others free inside the box). The claim is that no two deep
    witnesses are within theta on EVERY axis at once. Markers are exempt: their
    position is the frontier, which is the information they carry.
    """
    _, _, payload = run
    labels = payload[9]
    axes = ("x1_0_0", "x2_0_0")
    columns = {v: ic_values(payload, v) for v in axes}
    columns = {v: col for v, col in columns.items() if col}
    deep = [i for i, label in enumerate(labels) if label == "deep"]
    for a_index in range(len(deep)):
        for b_index in range(a_index + 1, len(deep)):
            i, j = deep[a_index], deep[b_index]
            gap = max(abs(col[i] - col[j]) for col in columns.values())
            assert gap >= THETA, (
                f"deep witnesses {i} and {j} are {float(gap)} apart in "
                f"L-inf, under theta {float(THETA)}"
                )


def test_frontier_markers_bound_the_pool(run):
    """A boundary/domain marker sits at an extreme of the pool on some axis.

    Weak by construction, being what can be asserted without re-solving, but it
    catches a marker emitted from the interior of the box.
    """
    _, _, payload = run
    labels = payload[9]
    markers = [i for i, label in enumerate(labels) if label in ("boundary", "domain")]
    assert markers, "a converged box must be labeled"
    for index in markers:
        at_extreme = False
        for var_id in ("x1_0_0", "x2_0_0"):
            values = ic_values(payload, var_id)
            if values and values[index] in (min(values), max(values)):
                at_extreme = True
        assert at_extreme, f"marker {index} is interior on every axis"


def test_pool_has_no_duplicate_initial_conditions(run):
    """One entry per distinct IC: the marker-merge and harvest-collapse rules."""
    _, _, payload = run
    axes = [ic_values(payload, v) for v in ("x1_0_0", "x2_0_0")]
    axes = [a for a in axes if a]
    points = list(zip(*axes))
    assert len(set(points)) == len(points), "duplicate initial conditions in the pool"


def test_serialization_is_canonical(run):
    """Two writes of the same content must produce the same bytes."""
    sys.path.insert(0, SRC)
    from stlmc.generation.canonical import canonicalize

    _, pool, payload = run
    once = pickle.dumps(canonicalize(payload))
    twice = pickle.dumps(canonicalize(pickle.loads(open(pool, "rb").read())))
    assert once == twice


def test_the_printed_verdict_matches_the_pool(run):
    """False iff a pool was written. The three-way logic itself is unit-tested
    in test_box_search.py::TestVerdict, which does not depend on which depths a
    particular benchmark happens to falsify at."""
    stdout, pool, _ = run
    assert ("result : False" in stdout) == (pool is not None)


def test_a_restricted_run_never_claims_absence_up_to_the_bound(tmp_path):
    """[gen] depths restricts what the verdict has seen.

    Whatever this model does at depth 1, the run may not report True: either it
    found a counterexample (False) or it examined one depth out of two (Unknown).
    """
    stdout, _ = run_box(tmp_path, extra="depths = 1")
    assert "result : True" not in stdout, (
        "claimed absence up to the bound while skipping depth 2\n" + stdout)