"""Tests for the fail-fast validation of ``[gen]`` values.

A malformed value used to die as a bare ``ValueError`` deep inside a run --
after model parsing, sometimes after solver work -- which the driver's blanket
handler reduces to an unattributed one-liner. ``validate_gen`` runs at strategy
start and names the key, the value and the expectation. The per-call bound
additionally rejects a negative ``query-timeout`` at resolution time: it used
to unbound z3 silently (the constructor's ``> 0`` guard skipped the bound with
no notice) and to crash the dReal path in ``Queue.get(timeout<0)`` after the
subprocess had already spawned.
"""

import pytest

from stlmc.generation.common import validate_gen
from stlmc.generation.oracle import DEFAULT_QUERY_TIMEOUT, query_timeout
from stlmc.objects.configuration import Configuration, Section


def config_with_gen(**values):
    section = Section()
    section.name = "gen"
    section.arguments = {k.replace("_", "-"): v for k, v in values.items()}
    config = Configuration()
    config.add_section(section)
    return config


class TestValidateGen:
    def test_a_well_formed_configuration_passes(self):
        validate_gen(config_with_gen(radius="2", k_paths="8",
                                     depths="0/2/5", query_timeout="45.5",
                                     keep_smt2="1"))

    def test_no_gen_section_passes(self):
        validate_gen(Configuration())
        validate_gen(None)

    def test_an_unknown_key_is_not_this_check_s_business(self):
        """Unrecognised keys read as absent by design; the run banner is what
        makes them visible. Validation must not reject them."""
        validate_gen(config_with_gen(radious="2"))

    @pytest.mark.parametrize("key, value", [
        ("radius", "big"),
        ("radius", "2.5"),
        ("k-paths", "2.5"),
        ("depths", "1/x/3"),
        ("depths", "8.5"),
        ("query-timeout", "-5"),
        ("query-timeout", "abc"),
        ("query-timeout", "nan"),
        ("query-timeout", "inf"),
        ("keep-smt2", "true"),
    ])
    def test_a_malformed_value_names_the_key(self, key, value):
        config = config_with_gen(**{key.replace("-", "_"): value})
        with pytest.raises(ValueError) as err:
            validate_gen(config)
        assert f"[gen] {key}" in str(err.value)
        assert value in str(err.value)

    def test_the_check_set_can_be_restricted(self):
        """A strategy may validate only the keys it reads."""
        config = config_with_gen(depths="1/x/3")
        validate_gen(config, keys={"radius"})
        with pytest.raises(ValueError):
            validate_gen(config, keys={"depths"})


class TestQueryTimeoutRange:
    """Resolution-time defence for callers that do not go through
    ``validate_gen`` (both strategies build oracles from the factory, but the
    sibling also constructs backends directly)."""

    def test_spellings_of_off_still_disable_the_bound(self):
        for spelling in ("0", "off", "none"):
            assert query_timeout(
                config_with_gen(query_timeout=spelling)) is None, spelling

    def test_absent_still_falls_back_to_the_default(self):
        assert query_timeout(Configuration()) == DEFAULT_QUERY_TIMEOUT

    @pytest.mark.parametrize("value", ["-5", "-0.1", "nan", "inf", "abc"])
    def test_a_non_bound_is_rejected_at_resolution(self, value):
        with pytest.raises(ValueError) as err:
            query_timeout(config_with_gen(query_timeout=value))
        assert "query-timeout" in str(err.value)