# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest

from pypsa.plot.statistics.base import UNSET, sanitize_mathtext
from pypsa.plot.statistics.schema import apply_parameter_schema


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("H$_2$ Storage", "H<sub>2</sub> Storage"),
        ("CO$_2$", "CO<sub>2</sub>"),
        ("x$^2$", "x<sup>2</sup>"),
        ("N$_2$O", "N<sub>2</sub>O"),
        ("plain name", "plain name"),
        (42, 42),
    ],
)
def test_sanitize_mathtext(label, expected):
    """#7: LaTeX sub-/superscripts convert to Plotly tags."""
    assert sanitize_mathtext(label) == expected


@pytest.mark.parametrize("param", ["color", "x", "storage"])
def test_schema_distinguishes_none_from_unset(param):
    """#4: UNSET on any schema param applies the default; explicit None is kept."""
    defaulted = apply_parameter_schema("installed_capacity", "bar", {param: UNSET})
    assert defaulted[param] is not UNSET

    explicit = apply_parameter_schema("installed_capacity", "bar", {param: None})
    assert explicit[param] is None


@pytest.mark.parametrize(
    ("stats_name", "allowed"),
    [("installed_capacity", True), ("energy_balance", False), ("capex", False)],
)
def test_optional_param_allowed_per_signature(stats_name, allowed):
    """#3: storage is enabled via signature introspection, not a manual list."""
    result = apply_parameter_schema(stats_name, "bar", {"storage": True})
    assert ("storage" in result) is allowed


def test_schema_raises_for_excluded_filter():
    """#3: an excluded, explicitly-set filter raises instead of being dropped."""
    with pytest.raises(ValueError, match="bus_carrier"):
        apply_parameter_schema("prices", "bar", {"carrier": "AC"})


def test_schema_drops_structural_excluded_param_silently():
    """Structurally-injected excluded params (nice_names) are dropped, not raised."""
    result = apply_parameter_schema("prices", "bar", {"nice_names": True})
    assert "nice_names" not in result


class TestSchemaContext:
    """Test that schema functions work with context parameter."""

    def test_apply_parameter_schema_with_context(self):
        """Test apply_parameter_schema accepts context parameter."""
        # Test that function accepts context parameter
        kwargs = {"x": None, "y": None, "color": None}
        context = {"index_names": ["scenario"]}

        # This should not raise an error
        result = apply_parameter_schema("installed_capacity", "bar", kwargs, context)
        assert isinstance(result, dict)

    def test_apply_parameter_schema_backward_compatible(self):
        """Test apply_parameter_schema works without context parameter."""
        from pypsa.plot.statistics.schema import apply_parameter_schema

        # Test backward compatibility
        kwargs = {"x": None, "y": None, "color": None}

        # This should not raise an error
        result = apply_parameter_schema("installed_capacity", "bar", kwargs)
        assert isinstance(result, dict)
