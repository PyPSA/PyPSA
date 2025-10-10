# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest

import pypsa
from pypsa._options import option_context
from pypsa.components.types import all_standard_attrs_set
from pypsa.network.transform import _get_potential_typos  # Replace with actual import


class TestGetPotentialTypos:
    def test_basic_functionality_and_edge_cases(self):
        """Test basic functionality, empty inputs, and exact matches."""
        # Empty inputs
        assert _get_potential_typos([], all_standard_attrs_set) == set()
        assert _get_potential_typos(["test"], set()) == set()
        assert _get_potential_typos([], set()) == set()

        # Exact matches are not typos
        custom_attrs = ["p_nom", "s_nom", "bus"]
        assert _get_potential_typos(custom_attrs, all_standard_attrs_set) == set()

        # Return type validation
        result = _get_potential_typos(["p_nok"], all_standard_attrs_set)
        assert isinstance(result, set)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    @pytest.mark.parametrize(
        ("string_a", "string_b", "should_match"),
        [
            # Valid typos (distance = 1)
            ("p_nok", "p_nom", True),  # substitution
            ("p_nomm", "p_nom", True),  # insertion
            ("p_no", "p_nom", True),  # deletion
            ("p_Nom", "p_nom", True),  # case difference
            ("activ", "active", True),  # deletion
            ("buss", "bus", True),  # insertion
            ("p-nom", "p_nom", True),  # special character
            ("p_nom", "p_nom_", True),  # trailing underscore
            ("_p_nom", "p_nom", True),  # leading underscore
            (
                "pp_nom",
                "p_nom",
                True,
            ),  # typo with different amount of characters before _
            # Should not raise typos
            ("p_nom", "s_nom", False),  # different base attributes
            ("p_nom", "p_nom", False),  # exact match
            ("bus0", "bus1", False),  # numeric suffix
            ("bus", "bus1", False),  # numeric suffix
            ("bus1", "bus10", False),  # numeric suffix
            ("bus", "bus10", False),  # numeric suffix
            ("hello_dear", "hello", False),  # distance > 1
            ("a", "ab", False),  # single char
            ("ab", "a", False),  # single char
            ("a", "b", False),  # single char
            ("typ", "type", False),  # Special case
        ],
    )
    def test_single_matches(self, string_a, string_b, should_match):
        """Test various typo detection patterns and exclusions."""
        result = _get_potential_typos([string_a], [string_b])

        if should_match:
            assert result == {(string_a, string_b)}
        else:
            assert result == set()

    def test_edge_cases_and_special_scenarios(self):
        """Test edge cases, long attributes, and special characters."""
        # Very long attributes
        long_custom = "very_long_attribute_name_that_might_cause_issues"
        long_standard = "very_long_attribute_name_that_might_cause_issue"  # missing 's'
        result = _get_potential_typos([long_custom], {long_standard})
        assert (long_custom, long_standard) in result

        # Multiple typos of same standard attr
        custom_attrs = ["p_nok", "p_nom_", "pnom"]  # All potential typos of p_nom
        result = _get_potential_typos(custom_attrs, {"p_nom"})
        assert len(result) >= 1  # Should find at least some matches

        # Underscore variations
        underscore_tests = [
            ("p_nom_", "p_nom"),  # trailing underscore
            ("_p_nom", "p_nom"),  # leading underscore
        ]
        for custom, standard in underscore_tests:
            result = _get_potential_typos([custom], {standard})
            assert (custom, standard) in result

        # No false positives among standard attrs
        standard_sample = list(all_standard_attrs_set)[:5]
        result = _get_potential_typos(standard_sample, all_standard_attrs_set)
        # This mainly ensures no crashes with real data
        assert isinstance(result, set)

    def test_unintended_attribute_warning(self, caplog):
        """Test warning for attributes that are standard for other components."""
        n = pypsa.Network()
        n.add("Bus", "test_bus")

        # Add a generator attribute 'p_nom' to a bus - this should trigger the warning
        # p_nom is a standard attribute for generators but not for buses
        n.add("Bus", "bus_with_generator_attr", p_nom=100.0)
        assert "is a standard attribute for other components" in caplog.text
        assert "p_nom" in caplog.text
        caplog.clear()

        # Add a bus attribute 'v_nom' to a generator - this should trigger the warning
        # v_nom is a standard attribute for buses but not for generators
        n.add("Generator", "gen_with_bus_attr", bus="test_bus", v_nom=380.0)
        assert "is a standard attribute for other components" in caplog.text
        assert "v_nom" in caplog.text

    def test_potential_typo_warning(self, caplog):
        """Test warning for potential typos in attribute names."""
        n = pypsa.Network()
        n.add("Bus", "test_bus")

        # Add a generator with a typo in p_nom (p_nok)
        n.add("Generator", "gen_with_typo", bus="test_bus", p_nok=100.0)
        assert "is not a standard attribute for" in caplog.text
        assert "p_nok" in caplog.text
        assert "p_nom" in caplog.text
        caplog.clear()

        # Add a bus with a typo in v_nom (v_nok)
        n.add("Bus", "bus_with_typo", v_nok=380.0)
        assert "is not a standard attribute for" in caplog.text
        assert "v_nok" in caplog.text
        assert "v_nom" in caplog.text
        caplog.clear()

        # Add a generator with trailing underscore in p_nom (p_nom_)
        n.add("Generator", "gen_with_underscore", bus="test_bus", p_nom_=100.0)
        assert "is not a standard attribute for" in caplog.text
        assert "p_nom_" in caplog.text
        assert "p_nom" in caplog.text

    def test_both_warnings_simultaneously(self, caplog):
        """Test that both warnings can be triggered in the same add operation."""
        n = pypsa.Network()
        n.add("Bus", "test_bus")

        # Add a generator with both unintended attribute and typo
        n.add(
            "Generator",
            "gen_with_both_issues",
            bus="test_bus",
            v_nom=380.0,  # Unintended attribute (bus attribute on generator)
            p_nok=100.0,
        )  # Typo (should be p_nom)

        # Check that both warnings were logged
        assert "is a standard attribute for other components" in caplog.text
        assert "is not a standard attribute for" in caplog.text
        assert "v_nom" in caplog.text
        assert "p_nok" in caplog.text

    def test_no_warning_for_valid_custom_attributes(self, caplog):
        """Test that custom attributes that are not typos or unintended don't trigger warnings."""
        n = pypsa.Network()
        n.add("Bus", "test_bus")

        # Add a generator with a clearly custom attribute that shouldn't trigger warnings
        n.add(
            "Generator",
            "gen_with_custom_attr",
            bus="test_bus",
            my_custom_attribute="custom_value",
            another_custom_123=42,
        )

        # Check that no attribute-related warnings were logged
        assert "are default attributes for other components" not in caplog.text
        assert "is likely a typo of standard attribute" not in caplog.text

    def test_typo_warning_option_disabled(self, caplog):
        """Test that typo warnings can be disabled via option while unintended warnings remain."""
        n = pypsa.Network()
        n.add("Bus", "test_bus")

        # Disable typo warnings via option
        with option_context("warnings.attribute_typos", False):
            # Add a generator with a typo - this should NOT trigger typo warning
            n.add("Generator", "gen_with_typo", bus="test_bus", p_nok=100.0)
            assert "is likely a typo of standard attribute" not in caplog.text

            # Add a generator with unintended attribute - this should still trigger warning
            n.add("Generator", "gen_with_unintended", bus="test_bus", v_nom=380.0)
            assert "is a standard attribute for other components" in caplog.text
            assert "v_nom" in caplog.text
