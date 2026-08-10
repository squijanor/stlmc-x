"""End-to-end properties of a kappa_path run, on the exact backend.

These are the claims a reader of a pool and of a verdict is entitled to make,
checked against a real run rather than a constructed assignment: the pool holds
counterexamples and nothing else, distinct paths are actually distinct, and a
verdict never covers ground the run did not visit.

The exact backend is used on purpose. Under z3 a model of the encoding is a
counterexample, so any failure here is the strategy's rather than the decision
procedure's.
"""

import os
import pickle
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
MODEL = os.path.join(REPO, "benchmarks", "additional", "wat-poly", "water.model")

CFG = """
common {{
    solver       = "z3"
    bound        = {bound}
    time-bound   = 8
    threshold    = 0.1
    generation   = "pathenum"
    verbose      = "true"
}}
gen {{
{gen}
}}
"""


def run_path(tmp_path, gen, goal="f2", bound=3):
    """One kappa_path run in an isolated directory; returns (stdout, pool path)."""
    cfg = tmp_path / "path.cfg"
    cfg.write_text(CFG.format(bound=bound, gen=gen))
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONHASHSEED="0")
    proc = subprocess.run(
        [sys.executable, "-c",
         "from stlmc.cli.mc import main; main()",
         MODEL, "-model-cfg", str(cfg), "-goal", goal, "-gen-seed", "0"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    pools = [p for p in os.listdir(str(tmp_path)) if p.endswith(".counterexamples")]
    return proc.stdout, (str(tmp_path / pools[0]) if pools else None)


def words(payload):
    """The location word of every counterexample in a pool, in pool order."""
    mode_re = re.compile(r"^currentMode_(\d+)$")
    out = []
    for assignment in payload[0]:
        steps = []
        for var, const in assignment.items():
            m = mode_re.match(getattr(var, "id", ""))
            if m is not None:
                steps.append((int(m.group(1)), str(const.value)))
        out.append(tuple(v for _, v in sorted(steps)))
    return out


@pytest.fixture(scope="module")
def radius_one(tmp_path_factory):
    """One shared radius-1 run: the assertions below are views of one pool."""
    tmp = tmp_path_factory.mktemp("kappa_path_r1")
    stdout, pool = run_path(tmp, '    k-paths = 3\n    radius = 1\n    depths = "2/3"')
    assert pool is not None, "a falsifying run must write a pool\n" + stdout
    with open(pool, "rb") as fr:
        return stdout, pickle.load(fr)


# ============================================================ pool contents

def test_the_pool_holds_no_blocking_artefacts(radius_one):
    """A radius-r block introduces indicator variables into the query. They are
    in every model the solver returns afterwards, so an unfiltered pool carries
    a number of them that grows with the counterexample's position in the pool.
    They are not part of any counterexample."""
    _, payload = radius_one
    for index, assignment in enumerate(payload[0]):
        leaked = sorted(v.id for v in assignment if v.id.startswith("hb$"))
        assert not leaked, f"counterexample {index} carries {leaked}"


def test_the_pool_shape_depends_on_depth_and_not_on_position(radius_one):
    """The consequence of the above that a consumer trips over. Two
    counterexamples at the same depth are assignments over the same encoding, so
    they must carry the same variables. With the blocking artefacts present the
    key set instead grows with the counterexample's position in the pool."""
    _, payload = radius_one
    by_depth = {}
    for word, assignment in zip(words(payload), payload[0]):
        by_depth.setdefault(len(word), set()).add(
            frozenset(v.id for v in assignment))
    assert len(by_depth) > 1, "fixture must span more than one depth"
    for depth, shapes in by_depth.items():
        assert len(shapes) == 1, (
            f"depth {depth}: {len(shapes)} different key sets in one depth")


def test_paths_are_distinct(radius_one):
    """Enumeration means distinct location words, and under radius 1 they are
    additionally at Hamming distance 2 or more within a depth."""
    _, payload = radius_one
    per_depth = {}
    for word in words(payload):
        per_depth.setdefault(len(word), []).append(word)
    for depth_words in per_depth.values():
        assert len(set(depth_words)) == len(depth_words)
        for i, a in enumerate(depth_words):
            for b in depth_words[i + 1:]:
                distance = sum(1 for x, y in zip(a, b) if x != y)
                assert distance >= 2, (a, b)


def test_serialization_is_canonical(tmp_path_factory):
    """The written file is a function of its content. Assignment dicts are built
    by iterating identity-ordered containers, so two runs agreeing in every
    counterexample would otherwise write files differing in bytes."""
    gen = '    k-paths = 3\n    radius = 1\n    depths = "2/3"'
    first = run_path(tmp_path_factory.mktemp("canon_a"), gen)[1]
    second = run_path(tmp_path_factory.mktemp("canon_b"), gen)[1]
    with open(first, "rb") as fa, open(second, "rb") as fb:
        assert fa.read() == fb.read()


# ================================================================== verdicts

def test_a_restricted_run_never_claims_absence_up_to_the_bound(tmp_path):
    """[gen] depths restricts what the verdict has seen. This goal has no
    counterexample at depth 1, so before the scoping rule the run reported
    "True up to bound 3" while depths 2 and 3 were never examined."""
    stdout, pool = run_path(tmp_path, "    k-paths = 2\n    depths = 1",
                            goal="f3", bound=3)
    assert pool is None, "fixture assumes this goal does not falsify at depth 1"
    assert "result : True" not in stdout, (
        "claimed absence up to the bound while skipping depths 2 and 3\n" + stdout)
    assert "result : Unknown" in stdout


def test_full_coverage_still_reports_absence(tmp_path):
    """The scoping rule must not turn every absence into Unknown: with every
    depth up to the bound visited and decided, True is the honest verdict."""
    stdout, pool = run_path(tmp_path, "    k-paths = 2", goal="f3", bound=1)
    assert pool is None
    assert "result : True" in stdout, stdout


def test_an_empty_target_depth_set_examines_nothing(tmp_path):
    """depths is clamped to 1..bound, so it can select nothing. A run that
    examined no depth at all may not report absence."""
    stdout, pool = run_path(tmp_path, '    k-paths = 2\n    depths = "20/21"',
                            goal="f2", bound=3)
    assert pool is None
    assert "result : True" not in stdout, stdout
    assert "selected no depth" in stdout


def test_the_reported_bound_is_where_a_counterexample_was_found(tmp_path):
    """The driver prints the returned bound as the one a counterexample was
    found at, so returning the configured bound misreports a shallow find."""
    stdout, pool = run_path(tmp_path, "    k-paths = 2\n    depths = 1",
                            goal="f1", bound=3)
    assert pool is not None
    assert "result : False at bound 1" in stdout, stdout


# ================================================== what an UNSAT establishes

def test_exhaustion_under_a_coarsening_radius_is_not_absence(tmp_path):
    """A radius-r block excludes words never exhibited, so the UNSAT that ends
    the loop does not establish that the lattice is covered. The run has to say
    which of the two it reached."""
    stdout, _ = run_path(tmp_path, "    k-paths = 99\n    radius = 1\n    depths = 2")
    assert "radius-r blocking was active" in stdout, stdout
    assert "absence of further paths is NOT established" in stdout


def test_exhaustion_without_a_radius_covers_the_lattice(tmp_path):
    """At radius 0 the same UNSAT does establish it, and says so."""
    stdout, _ = run_path(tmp_path, "    k-paths = 99\n    depths = 2")
    assert "path lattice exhausted" in stdout, stdout
    assert "every location word at this depth is covered" in stdout


def test_a_budgeted_stop_is_not_reported_as_exhaustion(tmp_path):
    """Stopping at k-paths says nothing about the rest of the lattice."""
    stdout, _ = run_path(tmp_path, "    k-paths = 1\n    depths = 2")
    assert "stopped at the [gen] k-paths budget" in stdout, stdout
    assert "path lattice exhausted" not in stdout
    assert "search exhausted" not in stdout


def test_a_radius_above_the_word_length_is_capped_and_reported(tmp_path):
    """At depth 1 a word has two positions, so radius 5 would encode a block no
    word satisfies: the next solve returns UNSAT and the depth looks exhausted
    after one path. The cap keeps the block a ball, and the run reports it."""
    gen = "    k-paths = 99\n    radius = 5\n    depths = 1"
    stdout, pool = run_path(tmp_path, gen, goal="f1", bound=3)
    assert pool is not None
    assert "capped to 1" in stdout, stdout