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
from conftest import PATH_MODEL as MODEL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

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
        [
            sys.executable,
            "-c",
            "from stlmc.cli.mc import main; main()",
            MODEL,
            "-model-cfg",
            str(cfg),
            "-goal",
            goal,
            "-gen-seed",
            "0",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
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
        return stdout, pickle.load(fr), pool


# ============================================================ pool contents


def test_the_pool_holds_no_blocking_artefacts(radius_one):
    """A radius-r block introduces indicator variables into the query. They are
    in every model the solver returns afterwards, so an unfiltered pool carries
    a number of them that grows with the counterexample's position in the pool.
    They are not part of any counterexample."""
    _, payload, _ = radius_one
    for index, assignment in enumerate(payload[0]):
        leaked = sorted(v.id for v in assignment if v.id.startswith("hb$"))
        assert not leaked, f"counterexample {index} carries {leaked}"


def test_the_pool_shape_depends_on_depth_and_not_on_position(radius_one):
    """The consequence of the above that a consumer trips over. Two
    counterexamples at the same depth are assignments over the same encoding, so
    they must carry the same variables. With the blocking artefacts present the
    key set instead grows with the counterexample's position in the pool."""
    _, payload, _ = radius_one
    by_depth = {}
    for word, assignment in zip(words(payload), payload[0]):
        by_depth.setdefault(len(word), set()).add(frozenset(v.id for v in assignment))
    assert len(by_depth) > 1, "fixture must span more than one depth"
    for depth, shapes in by_depth.items():
        assert len(shapes) == 1, (
            f"depth {depth}: {len(shapes)} different key sets in one depth"
        )


def test_paths_are_distinct(radius_one):
    """Enumeration means distinct location words, and under radius 1 they are
    additionally at Hamming distance 2 or more within a depth."""
    _, payload, _ = radius_one
    per_depth = {}
    for word in words(payload):
        per_depth.setdefault(len(word), []).append(word)
    for depth_words in per_depth.values():
        assert len(set(depth_words)) == len(depth_words)
        for i, a in enumerate(depth_words):
            for b in depth_words[i + 1 :]:
                distance = sum(1 for x, y in zip(a, b) if x != y)
                assert distance >= 2, (a, b)


def test_serialization_is_canonical(radius_one, tmp_path_factory):
    """The written file is a function of its content. Assignment dicts are built
    by iterating identity-ordered containers, so two runs agreeing in every
    counterexample would otherwise write files differing in bytes. The first
    write is the shared run's pool; the second is an independent run at the same
    configuration and seed."""
    gen = '    k-paths = 3\n    radius = 1\n    depths = "2/3"'
    first = radius_one[2]
    second = run_path(tmp_path_factory.mktemp("canon_b"), gen)[1]
    with open(first, "rb") as fa, open(second, "rb") as fb:
        assert fa.read() == fb.read()


# ================================================================== verdicts


def test_a_restricted_run_never_claims_absence_up_to_the_bound(tmp_path):
    """[gen] depths restricts what the verdict has seen. This goal has no
    counterexample at depth 1, so before the scoping rule the run reported
    "True up to bound 3" while depths 2 and 3 were never examined."""
    stdout, pool = run_path(
        tmp_path, "    k-paths = 2\n    depths = 1", goal="f3", bound=3
    )
    assert pool is None, "fixture assumes this goal does not falsify at depth 1"
    assert "result : True" not in stdout, (
        "claimed absence up to the bound while skipping depths 2 and 3\n" + stdout
    )
    assert "result : Unknown" in stdout


def test_full_coverage_still_reports_absence(tmp_path):
    """The scoping rule must not turn every absence into Unknown: with every
    depth up to the bound visited and decided, True is the honest verdict."""
    stdout, pool = run_path(tmp_path, "    k-paths = 2", goal="f3", bound=1)
    assert pool is None
    assert "result : True" in stdout, stdout


def test_a_counterexample_at_depth_zero_is_found_and_reported(tmp_path):
    """Depth 0 unrolls to one mode segment and no jump, which is a trajectory
    like any other. The depth set includes 0, so a counterexample that exists
    only there is found and the run reports False rather than True. Its word has
    a single position."""
    stdout, pool = run_path(tmp_path, "    depths = 0", goal="f1", bound=3)
    assert pool is not None, "fixture assumes this goal falsifies at depth 0\n" + stdout
    assert "result : False at bound 0" in stdout, stdout
    with open(pool, "rb") as handle:
        payload = pickle.load(handle)
    assert len(payload[0]) == 1
    assert [len(w) for w in words(payload)] == [1]


def test_an_empty_target_depth_set_examines_nothing(tmp_path):
    """depths is clamped to 0..bound, so it can select nothing. A run that
    examined no depth at all may not report absence."""
    stdout, pool = run_path(
        tmp_path, '    k-paths = 2\n    depths = "20/21"', goal="f2", bound=3
    )
    assert pool is None
    assert "result : True" not in stdout, stdout
    assert "selected no depth" in stdout


def test_the_reported_bound_is_where_a_counterexample_was_found(tmp_path):
    """The driver prints the returned bound as the one a counterexample was
    found at, so returning the configured bound misreports a shallow find."""
    stdout, pool = run_path(
        tmp_path, "    k-paths = 2\n    depths = 1", goal="f1", bound=3
    )
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


def test_exhaustion_without_a_radius_is_over_the_falsifying_words(tmp_path):
    """At radius 0 the UNSAT does establish something, and the claim is bounded:
    no word outside the pool falsifies. It is not that the depth's path lattice
    was enumerated -- the falsifying subset depends on the goal, and can be a
    small fraction of the lattice."""
    stdout, _ = run_path(tmp_path, "    k-paths = 99\n    depths = 2")
    assert "falsifying words exhausted" in stdout, stdout
    assert "no location word outside the pool falsifies" in stdout


def test_a_budgeted_stop_is_not_reported_as_exhaustion(tmp_path):
    """Stopping at k-paths says nothing about the rest of the lattice."""
    stdout, _ = run_path(tmp_path, "    k-paths = 1\n    depths = 2")
    assert "stopped at the [gen] k-paths budget" in stdout, stdout
    assert "falsifying words exhausted" not in stdout
    assert "search exhausted" not in stdout


def test_a_radius_above_the_word_length_is_capped_and_reported(tmp_path):
    """At depth 1 a word has two positions, so radius 5 would encode a block no
    word satisfies: the next solve returns UNSAT and the depth looks exhausted
    after one path. The cap keeps the block a ball, and the run reports it."""
    gen = "    k-paths = 99\n    radius = 5\n    depths = 1"
    stdout, pool = run_path(tmp_path, gen, goal="f1", bound=3)
    assert pool is not None
    assert "capped to 1" in stdout, stdout


def test_an_undecided_depth_is_reported_rather_than_waited_out(tmp_path):
    """A solver call that cannot return in time is bounded per call, and the
    result is a reported UNRESOLVED depth and a verdict of Unknown -- not a hang.
    A per-call bound below the backend's own resolution makes every query answer
    unknown, so the depth is undecided by construction rather than by relying on
    a particular model being hard to decide."""
    gen = "    k-paths = 2\n    depths = 4\n    query-timeout = 0.001"
    stdout, pool = run_path(tmp_path, gen, goal="f2", bound=4)
    assert pool is None
    assert "search UNRESOLVED" in stdout, stdout
    assert "result : Unknown" in stdout, stdout


def test_a_zero_budget_settles_nothing_and_says_so(tmp_path):
    """[gen] k-paths = 0 visits every depth and poses no query at all. The
    verdict is over the depths a run decided, not the ones it targeted, so this
    reports Unknown -- it reported True, over the full bound, on no evidence."""
    stdout, pool = run_path(tmp_path, "    k-paths = 0", goal="f2", bound=2)
    assert pool is None
    assert "result : True" not in stdout, stdout
    assert "result : Unknown" in stdout
    assert "were not decided" in stdout


def test_a_pool_with_no_labels_still_records_the_relaxation(radius_one):
    """The three trailing pool elements are positional, so a strategy that has
    no per-counterexample labels contributes an empty list rather than omitting
    the slot -- otherwise the relaxation would land where a consumer reads
    labels, and the structure signatures where it reads the relaxation.
    """
    _, payload, _ = radius_one
    assert len(payload) == 12
    assert payload[9] == [], "kappa_path has no label vocabulary"
    assert payload[10] == 0.0, "the exact backend answers under no relaxation"
    assert isinstance(payload[11], list), "structure signatures ride the twelfth"


def test_structure_signatures_are_pool_aligned_and_wellformed(radius_one):
    """The twelfth pool element is one structure-signature record per
    counterexample, in pool order. Each record's raw_word is the location word
    the entry spells; structure_id is a non-empty key and is one-to-one with an
    entry's (reduced_word, sigma), so entries sharing a skeleton share the key."""
    _, payload, _ = radius_one
    records = payload[11]
    pool_words = words(payload)
    assert len(records) == len(payload[0]) == len(pool_words)
    for word, record in zip(pool_words, records):
        assert set(record) == {"raw_word", "reduced_word", "sigma", "structure_id"}
        assert tuple(str(m) for m in record["raw_word"]) == word
        assert isinstance(record["structure_id"], str) and record["structure_id"]
        assert all(len(pair) == 2 for pair in record["sigma"])
    by_structure = {}
    for record in records:
        key = (tuple(record["reduced_word"]), tuple((i, v) for i, v in record["sigma"]))
        by_structure.setdefault(key, set()).add(record["structure_id"])
    assert all(len(ids) == 1 for ids in by_structure.values()), (
        "structure_id must be one-to-one with (reduced_word, sigma)"
    )
