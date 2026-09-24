"""Unit checks for the structure-signature records at pool payload index 11
(``stlmc.generation.encode.structure_signature_records``).

The record is a function of the assignment alone, so these build assignments by
hand and check the emitted record against a known word and skeleton -- no
solver.
"""

from stlmc.constraints.constraints import Bool, BoolVal, Int, IntVal, Real, RealVal
from stlmc.generation.encode import structure_signature_records


def _assignment(word, sigma, ic=None):
    """An assignment dict: ``currentMode_k`` from ``word``, Boolean-abstraction
    variables from ``sigma`` (``{id: bool}``), and optional real initial
    conditions from ``ic`` (``{id: value}``)."""
    assn = {}
    for k, mode in enumerate(word):
        assn[Int(f"currentMode_{k}")] = IntVal(str(mode))
    for bool_id, value in sigma.items():
        assn[Bool(bool_id)] = BoolVal(value)
    for var_id, value in (ic or {}).items():
        assn[Real(var_id)] = RealVal(str(value))
    return assn


def test_raw_and_reduced_word_for_a_known_word():
    """raw_word is the currentMode word; reduced_word collapses consecutive
    equal modes and nothing else."""
    assn = _assignment([0, 1, 1, 0, 0], {"chi^{1,1}_7": True}, {"x_0_0": "1.5"})
    (record,) = structure_signature_records([assn])
    assert record["raw_word"] == [0, 1, 1, 0, 0]
    assert record["reduced_word"] == [0, 1, 0]


def test_sigma_is_every_boolean_id_sorted_with_its_truth_value():
    """sigma is the Boolean-abstraction assignment as id-sorted ``[id, value]``
    pairs; continuous and mode variables are not in it."""
    assn = _assignment(
        [0, 1],
        {"T1^{2,2}_9": False, "chi^{1,1}_9": True, "invAtomicID_3": True},
        {"x_0_0": "0.2"},
    )
    (record,) = structure_signature_records([assn])
    assert record["sigma"] == [
        ["T1^{2,2}_9", False],
        ["chi^{1,1}_9", True],
        ["invAtomicID_3", True],
    ]


def test_structure_id_is_nonempty_and_joins_on_reduced_word_and_sigma():
    """structure_id is a stable, non-empty key: equal ``(reduced_word, sigma)``
    map to one id even when the raw word differs, and a different sigma maps
    elsewhere."""
    a = _assignment([0, 0, 1], {"chi_1": True})  # reduced [0, 1]
    b = _assignment([0, 1, 1], {"chi_1": True})  # reduced [0, 1], same sigma
    c = _assignment([0, 1, 1], {"chi_1": False})  # same reduced, other sigma
    ra, rb, rc = structure_signature_records([a, b, c])
    assert isinstance(ra["structure_id"], str) and ra["structure_id"]
    assert ra["structure_id"] == rb["structure_id"]
    assert ra["structure_id"] != rc["structure_id"]


def test_records_are_pool_aligned_and_share_one_object_per_structure():
    """One record per assignment, in pool order; structurally equal records
    share one reduced_word and sigma object, while raw_word stays
    per-counterexample."""
    pool = [
        _assignment([0, 0, 1], {"chi_1": True}),
        _assignment([0, 1, 1], {"chi_1": True}),
    ]
    records = structure_signature_records(pool)
    assert len(records) == len(pool)
    assert records[0]["structure_id"] == records[1]["structure_id"]
    assert records[0]["sigma"] is records[1]["sigma"]
    assert records[0]["reduced_word"] is records[1]["reduced_word"]
    assert records[0]["raw_word"] == [0, 0, 1]
    assert records[1]["raw_word"] == [0, 1, 1]
