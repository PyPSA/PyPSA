"""Test constants module."""

import re

import pytest

from pypsa.constants import (
    DEFAULT_EPSG,
    DEFAULT_TIMESTAMP,
    RE_PORTS,
    RE_PORTS_FILTER,
    RE_PORTS_GE_2,
)


class TestConstants:
    """Test constants values."""

    def test_default_epsg(self):
        """Test DEFAULT_EPSG constant."""
        assert DEFAULT_EPSG == 4326
        assert isinstance(DEFAULT_EPSG, int)

    def test_default_timestamp(self):
        """Test DEFAULT_TIMESTAMP constant."""
        assert DEFAULT_TIMESTAMP == "now"
        assert isinstance(DEFAULT_TIMESTAMP, str)


class TestRegexPatterns:
    """Test regex pattern constants."""

    def test_re_ports_basic_matches(self):
        """Test RE_PORTS pattern matches basic bus columns."""
        assert RE_PORTS.match("bus")
        assert RE_PORTS.match("bus0")
        assert RE_PORTS.match("bus1")
        assert RE_PORTS.match("bus2")
        assert RE_PORTS.match("bus10")
        assert RE_PORTS.match("bus123")

    def test_re_ports_capture_groups(self):
        """Test RE_PORTS pattern capture groups."""
        match = RE_PORTS.match("bus")
        assert match.group(1) == ""

        match = RE_PORTS.match("bus0")
        assert match.group(1) == "0"

        match = RE_PORTS.match("bus123")
        assert match.group(1) == "123"

    def test_re_ports_non_matches(self):
        """Test RE_PORTS pattern non-matches."""
        assert not RE_PORTS.match("Bus")
        assert not RE_PORTS.match("bus_")
        assert not RE_PORTS.match("bus-1")
        assert not RE_PORTS.match("busx")
        assert not RE_PORTS.match("load")
        assert not RE_PORTS.match("generator")
        assert not RE_PORTS.match("")

    def test_re_ports_filter_matches(self):
        """Test RE_PORTS_FILTER pattern matches."""
        assert RE_PORTS_FILTER.match("bus")
        assert RE_PORTS_FILTER.match("bus0")
        assert RE_PORTS_FILTER.match("bus1")
        assert RE_PORTS_FILTER.match("bus2")
        assert RE_PORTS_FILTER.match("bus10")
        assert RE_PORTS_FILTER.match("bus123")

    def test_re_ports_filter_non_matches(self):
        """Test RE_PORTS_FILTER pattern non-matches."""
        assert not RE_PORTS_FILTER.match("Bus")
        assert not RE_PORTS_FILTER.match("bus_")
        assert not RE_PORTS_FILTER.match("bus-1")
        assert not RE_PORTS_FILTER.match("busx")
        assert not RE_PORTS_FILTER.match("load")
        assert not RE_PORTS_FILTER.match("")

    def test_re_ports_ge_2_matches(self):
        """Test RE_PORTS_GE_2 pattern matches ports >= 2."""
        assert RE_PORTS_GE_2.match("bus2")
        assert RE_PORTS_GE_2.match("bus3")
        assert RE_PORTS_GE_2.match("bus9")
        assert RE_PORTS_GE_2.match("bus10")
        assert RE_PORTS_GE_2.match("bus11")
        assert RE_PORTS_GE_2.match("bus23")
        assert RE_PORTS_GE_2.match("bus100")
        assert RE_PORTS_GE_2.match("bus999")

    def test_re_ports_ge_2_capture_groups(self):
        """Test RE_PORTS_GE_2 pattern capture groups."""
        match = RE_PORTS_GE_2.match("bus2")
        assert match.group(1) == "2"

        match = RE_PORTS_GE_2.match("bus10")
        assert match.group(1) == "10"

        match = RE_PORTS_GE_2.match("bus123")
        assert match.group(1) == "123"

    def test_re_ports_ge_2_non_matches(self):
        """Test RE_PORTS_GE_2 pattern non-matches for ports < 2."""
        assert not RE_PORTS_GE_2.match("bus")
        assert not RE_PORTS_GE_2.match("bus0")
        assert not RE_PORTS_GE_2.match("bus1")
        assert not RE_PORTS_GE_2.match("Bus2")
        assert not RE_PORTS_GE_2.match("bus_2")
        assert not RE_PORTS_GE_2.match("bus-2")
        assert not RE_PORTS_GE_2.match("busx")
        assert not RE_PORTS_GE_2.match("")

    def test_regex_pattern_types(self):
        """Test that regex patterns are compiled Pattern objects."""
        assert isinstance(RE_PORTS, re.Pattern)
        assert isinstance(RE_PORTS_FILTER, re.Pattern)
        assert isinstance(RE_PORTS_GE_2, re.Pattern)

    @pytest.mark.parametrize(
        ("pattern", "test_string", "expected"),
        [
            (RE_PORTS, "bus", True),
            (RE_PORTS, "bus0", True),
            (RE_PORTS, "bus123", True),
            (RE_PORTS, "Bus", False),
            (RE_PORTS_FILTER, "bus", True),
            (RE_PORTS_FILTER, "bus0", True),
            (RE_PORTS_FILTER, "Bus", False),
            (RE_PORTS_GE_2, "bus2", True),
            (RE_PORTS_GE_2, "bus1", False),
            (RE_PORTS_GE_2, "bus0", False),
        ],
    )
    def test_regex_patterns_parametrized(self, pattern, test_string, expected):
        """Parametrized test for regex patterns."""
        result = pattern.match(test_string) is not None
        assert result == expected

    def test_regex_patterns_edge_cases(self):
        """Test regex patterns with edge cases."""
        # Test with leading zeros - should not match for GE_2
        assert RE_PORTS.match("bus00")
        assert RE_PORTS_FILTER.match("bus00")
        assert not RE_PORTS_GE_2.match("bus00")  # 00 < 2
        assert not RE_PORTS_GE_2.match("bus01")  # 01 < 2
        assert not RE_PORTS_GE_2.match("bus001")  # Leading zeros should not match

        # Test with very long numbers
        long_num = "2" * 100  # Use 2 instead of 1 since 111...1 >= 2
        assert RE_PORTS.match(f"bus{long_num}")
        assert RE_PORTS_FILTER.match(f"bus{long_num}")
        assert RE_PORTS_GE_2.match(f"bus{long_num}")

        # Test multi-digit numbers starting with 1 that are >= 2
        assert RE_PORTS_GE_2.match("bus10")
        assert RE_PORTS_GE_2.match("bus15")
        assert RE_PORTS_GE_2.match("bus100")

        # Test partial matches (should not match)
        assert not RE_PORTS.match("bus123extra")
        assert not RE_PORTS_FILTER.match("bus123extra")
        assert not RE_PORTS_GE_2.match("bus123extra")
