# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import warnings

import pytest


def test_statistics_deprecated_kwargs(ac_dc_solved):
    """Test that old parameter names raise deprecation warnings for all statistics methods."""
    from pypsa.statistics.expressions import StatisticsAccessor

    n = ac_dc_solved

    # Test each method for deprecated 'comps' parameter
    for method_name in StatisticsAccessor._methods:
        if method_name in ["prices"]:
            continue
        with pytest.raises(DeprecationWarning) as excinfo:
            getattr(n.statistics, method_name)(comps="Generator")
        assert "`comps` is deprecated" in str(excinfo.value)
        assert "use `components` instead" in str(excinfo.value)

    # Test a few methods for deprecated 'aggregate_groups' parameter
    for method_name in ["supply", "energy_balance", "capex"]:
        with pytest.raises(DeprecationWarning) as excinfo:
            getattr(n.statistics, method_name)(aggregate_groups="sum")
        assert "`aggregate_groups` is deprecated" in str(excinfo.value)
        assert "use `groupby_method` instead" in str(excinfo.value)

    # Test a few methods for deprecated 'aggregate_time' parameter
    for method_name in ["supply", "withdrawal", "transmission"]:
        with pytest.raises(DeprecationWarning) as excinfo:
            getattr(n.statistics, method_name)(aggregate_time="mean")
        assert "`aggregate_time` is deprecated" in str(excinfo.value)
        assert "use `groupby_time` instead" in str(excinfo.value)

    # Test the __call__ method also has deprecated parameters
    with pytest.raises(DeprecationWarning) as excinfo:
        n.statistics(comps="Generator")
    assert "`comps` is deprecated" in str(excinfo.value)

    with pytest.raises(DeprecationWarning) as excinfo:
        n.statistics(aggregate_groups="sum")
    assert "`aggregate_groups` is deprecated" in str(excinfo.value)

    with pytest.raises(DeprecationWarning) as excinfo:
        n.statistics(aggregate_time="mean")
    assert "`aggregate_time` is deprecated" in str(excinfo.value)

    # Test that both old and new parameters raise error
    with pytest.raises(DeprecationWarning) as excinfo:
        n.statistics.installed_capacity(comps="Generator", components="Generator")
    assert "received both comps and components" in str(excinfo.value)

    # Test equivalence: old and new params produce same results
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_result = n.statistics.installed_capacity(comps="Generator")
    new_result = n.statistics.installed_capacity(components="Generator")
    assert old_result.equals(new_result)
