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
from collections import Counter
from fractions import Fraction

import pytest
from conftest import BOX_MODEL as MODEL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

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
    k-ic    = {k_ic}
    {extra}
}}
"""

THETA = Fraction(1, 100)


def run_box(tmp_path, epsilon=THETA, extra="depths = 2", goal="f2", k_ic=1):
    """One kappa_box run in an isolated directory; returns (stdout, pool path)."""
    cfg = tmp_path / "box.cfg"
    cfg.write_text(CFG.format(epsilon=float(epsilon), extra=extra, k_ic=k_ic))
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
    assert len(payload) == 11, (
        "labels ride the tenth payload element and the backend relaxation the "
        "eleventh")
    labels = payload[9]
    assert len(labels) == len(payload[0]), "one label per counterexample"
    assert set(labels) <= {"deep", "structure-frontier", "ic-domain",
                           "projected-domain"}


@pytest.fixture(scope="module")
def validation(run):
    """One validation of the shared pool. Refinement is on by default, so this
    evaluates every counterexample at two sampling densities; the assertions
    below are independent views of that one measurement rather than three."""
    sys.path.insert(0, SRC)
    from stlmc.generation.validate import validate_pool

    _, _, payload = run
    return validate_pool(payload, samples=15)


def test_every_counterexample_falsifies(validation, run):
    """The property that makes a pool a pool. Uses the tool's own STL semantics.

    `threshold-only` passes: the encoding applies tau by relaxing the negated
    goal, so a witness within tau of violating is what the search was asked for.

    A domain / structure-frontier marker sits at the acceptance boundary by
    construction -- its robustness is ~tau -- so the tool's own solver accepts it
    while an independent re-evaluation can land it a solver-epsilon past tau. A
    marker within that epsilon of tau is the boundary, not a spurious
    non-falsifier; a deep witness, which is interior, is held to the strict bar.
    """
    _, _, payload = run
    tau = Fraction(str(payload[8]))
    boundary_eps = Fraction(1, 1_000_000)

    def spurious(record):
        if record["verdict"] not in ("unverified", "error"):
            return False
        if record["label"] != "deep" and record["verdict"] == "unverified":
            return abs(Fraction(str(record["rho0"])) - tau) > boundary_eps
        return True

    bad = [r for r in validation if spurious(r)]
    assert not bad, "non-falsifying entries: {}".format(
        [(r["index"], r["label"], r["verdict"], r["rho0"]) for r in bad[:5]])


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
    """A frontier / domain marker sits at an extreme of the pool on some axis.

    Weak by construction, being what can be asserted without re-solving, but it
    catches a marker emitted from the interior of the box. The comparison is to
    a tolerance, not exact: a marker sits at the box edge, but a harvested deep
    witness may land a sampling-window's width past it, so the pool extreme can
    be a hair beyond the marker.
    """
    _, _, payload = run
    labels = payload[9]
    markers = [
        i for i, label in enumerate(labels)
        if label in ("structure-frontier", "ic-domain", "projected-domain")
    ]
    assert markers, "a converged box must be labeled"
    tol = Fraction(1, 1000)
    for index in markers:
        at_extreme = False
        for var_id in ("x1_0_0", "x2_0_0"):
            values = ic_values(payload, var_id)
            if values and (abs(values[index] - min(values)) <= tol
                           or abs(values[index] - max(values)) <= tol):
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


def test_a_configured_query_timeout_reaches_the_pivot(tmp_path):
    """[gen] query-timeout bounds the pivot's own solver.

    The pivot builds its oracle directly rather than through ``make_oracle``, so
    a configured bound reaches it only if the site passes it explicitly. Set
    below z3's own resolution every query answers unknown, the pivot cannot be
    decided, and the run must say so. A run that ignored the key would take the
    constructor's 60 s default instead, decide the pivot, and report False: the
    assertion separates "the bound arrived" from "the bound was 60 s".
    """
    stdout, pool = run_box(tmp_path, extra="depths = 2\n    query-timeout = 0.001")
    assert "pivot search UNRESOLVED" in stdout, stdout
    assert "result : Unknown" in stdout, stdout
    assert pool is None, "an undecided pivot must not produce a pool"
    # The advice must name the bound that bit. On this backend the pivot is one
    # query and pivot-timeout is not read at all, so naming it would send a
    # reader to a key with no effect on the run in front of them.
    assert "raise [gen] query-timeout" in stdout, stdout
    assert "pivot-timeout" not in stdout, stdout


def test_the_reported_bound_is_the_resolved_one(tmp_path):
    """The banner must agree with the bound the oracles were given.

    It is reported on this backend at all because the bound governs every solver
    call the strategy makes here. Re-reading the key rather than resolving it
    disagreed with the oracles: a reader for positive budgets folds a disabled
    bound to its default, and rejects the words that disable one, so a run could
    report a bound nothing was holding to -- or fail before it started.
    """
    for spelling, reported in (("300", "query-timeout=300.0s"),
                               ("0", "query-timeout=off"),
                               ('"off"', "query-timeout=off")):
        work = tmp_path / spelling.strip('"')
        work.mkdir()
        stdout, _ = run_box(work, extra=f"depths = 2\n    query-timeout = {spelling}")
        assert reported in stdout, (spelling, stdout)


def test_a_restricted_run_never_claims_absence_up_to_the_bound(tmp_path):
    """[gen] depths restricts what the verdict has seen.

    Whatever this model does at depth 1, the run may not report True: either it
    found a counterexample (False) or it examined one depth out of two (Unknown).
    """
    stdout, _ = run_box(tmp_path, extra="depths = 1")
    assert "result : True" not in stdout, (
        "claimed absence up to the bound while skipping depth 2\n" + stdout)

def test_a_zero_box_budget_is_no_budget_not_an_instant_true(tmp_path):
    """[gen] k-ic = 0 spells "off", like the section's other keys.

    The defect: 0 made the per-depth loop guard false before the first solver
    call, no depth was decided, and the run reported True in under a
    millisecond on a goal that falsifies. 0 must mean "no budget" (explore the
    depth to exhaustion), so this run has to find the counterexample.
    """
    stdout, pool = run_box(tmp_path, extra="depths = 2\n    k-ic = 0")
    assert "result : False" in stdout, stdout
    assert pool is not None


def test_zero_epsilon_fails_fast_with_the_key_named(tmp_path):
    """[gen] epsilon = 0 must be rejected before the first solver call.

    The defect: theta = 0 made the face bisection's termination condition
    unreachable over exact rationals, so the run hung inside the first face
    search at one solver call per iteration -- a silent hang, not an error.
    """
    stdout, pool = run_box(tmp_path, epsilon=0)
    assert "[gen] epsilon" in stdout and "must be > 0" in stdout, stdout
    assert pool is None


def deep_pairs_within_theta(payload, axes=("x1_0_0", "x2_0_0"), theta=None):
    """Pairs of deep witnesses that are within ``theta`` on every axis."""
    theta = THETA if theta is None else theta
    labels = payload[9]
    columns = {v: ic_values(payload, v) for v in axes}
    columns = {v: col for v, col in columns.items() if col}
    deep = [i for i, label in enumerate(labels) if label == "deep"]
    close = []
    for a_index in range(len(deep)):
        for b_index in range(a_index + 1, len(deep)):
            i, j = deep[a_index], deep[b_index]
            gap = max(abs(col[i] - col[j]) for col in columns.values())
            if gap < theta:
                close.append((i, j, float(gap)))
    return close


@pytest.fixture(scope="module")
def multibox(tmp_path_factory):
    """One shared k-ic = 3 run whose single depth holds more than one box. The
    separation and one-entry-per-initial-condition assertions are independent
    views of it, so the multi-box run happens once."""
    tmp = tmp_path_factory.mktemp("kappa_box_multibox")
    stdout, pool = run_box(
        tmp, k_ic=3,
        extra="depths = 2\n    k-witness = 3\n    bisect-iters = 4")
    assert pool is not None, stdout
    assert stdout.count("box ") > 1, "this test needs more than one box\n" + stdout
    with open(pool, "rb") as handle:
        return stdout, pickle.load(handle)


def test_separation_holds_across_boxes_at_one_depth(multibox):
    """theta separates the witnesses of a DEPTH, not of a box.

    Growth runs unmasked, so a second box at the same depth may regrow across
    the first and place a lattice centre arbitrarily close to a witness the
    first contributed. Checking a box against itself cannot see that, which is
    why every other separation assertion here runs at k-ic = 1.
    """
    stdout, payload = multibox
    close = deep_pairs_within_theta(payload)
    assert not close, f"deep witnesses closer than theta at one depth: {close[:5]}"


def test_separation_is_not_imposed_across_depths(tmp_path):
    """...and only of a depth: an initial condition may recur at another one.

    A region falsifying at several depths contributes a box at each, which is
    the depth axis rather than redundancy. Since separation now holds within
    every depth, any pair closer than theta is necessarily a cross-depth pair,
    so their presence is what shows the axis survived.
    """
    stdout, pool = run_box(
        tmp_path, k_ic=3,
        extra='depths = "1/2"\n    k-witness = 3\n    bisect-iters = 4')
    assert pool is not None, stdout
    with open(pool, "rb") as handle:
        payload = pickle.load(handle)
    assert "over 2 target depth(s)" in stdout, stdout
    assert deep_pairs_within_theta(payload), (
        "no initial condition recurred across depths, so this run cannot "
        "distinguish per-depth separation from global separation\n" + stdout)


def test_thinning_does_not_reach_across_depths(tmp_path):
    """[gen] thin-ic is a coarser rule under the same scope as theta.

    Thinning against the whole pool removes exactly the cross-depth repetition
    depth-spreading exists to produce, so a radius above theta would silently
    cost the depth axis rather than only the intra-depth redundancy it is meant
    to control.
    """
    stdout, pool = run_box(
        tmp_path, k_ic=3,
        extra=('depths = "1/2"\n    k-witness = 3\n    bisect-iters = 4'
               '\n    thin-ic = 0.05'))
    assert pool is not None, stdout
    with open(pool, "rb") as handle:
        payload = pickle.load(handle)
    assert "over 2 target depth(s)" in stdout, stdout
    close = deep_pairs_within_theta(payload, theta=Fraction(1, 20))
    assert close, (
        "thin-ic removed every witness pair within its own radius, so it is "
        "still being applied across depths\n" + stdout)


def test_the_pool_records_the_relaxation_it_was_generated_under(run, validation):
    """The eleventh element is what lets a consumer tell a witness from a
    candidate. On the exact backend it is 0, and that 0 is a statement about
    the run rather than an unfilled default."""
    _, _, payload = run
    assert payload[10] == 0.0

    sys.path.insert(0, SRC)
    from stlmc.generation.validate import pool_backend_delta

    assert pool_backend_delta(payload) == 0.0
    # Read from the pool rather than supplied: the fixture passes no
    # backend_delta argument.
    assert validation and all(r["backend_delta"] == 0.0 for r in validation)


def test_every_record_reports_whether_its_verdict_is_resolved(validation):
    """Refinement is on by default, so every record carries the shift in
    rho(0) between two sampling densities and its distance to the nearest band
    edge. A verdict quoted without those is a verdict with no error bar."""
    for rec in validation:
        assert rec["band_margin"] != ""
        assert rec["rho0_refined"] != ""
        assert rec["rho0_shift"] != ""
        assert rec["resolved"] in ("yes", "no")


def test_one_entry_per_initial_condition_across_boxes(multibox):
    """Def. pool rule (ii) is a depth rule too.

    Two boxes at one depth can converge on the same face, and a marker is
    exempt from separation but not from this: without the cross-box check the
    pool carries the same initial condition twice, at distance zero on every
    axis the metrics measure. The single-box configuration every other
    duplicate assertion runs at cannot see it.
    """
    stdout, payload = multibox
    axes = [ic_values(payload, v) for v in ("x1_0_0", "x2_0_0")]
    axes = [a for a in axes if a]
    points = list(zip(*axes))
    duplicates = [p for p, n in Counter(points).items() if n > 1]
    assert not duplicates, (
        "the same initial condition appears more than once: "
        f"{[tuple(float(x) for x in p) for p in duplicates[:3]]}\n" + stdout)