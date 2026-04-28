# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import warnings

import pytest


def test_statistics_deprecated_kwargs(ac_dc_solved):
    """Test that old parameter names raise deprecation warnings for all statistics methods."""
    from pypsa.statistics.expressions import StatisticsAccessor

    n = ac_dc_solved

    for method_name in StatisticsAccessor._methods:
        if method_name in ["prices"]:
            continue
        with pytest.warns(DeprecationWarning, match="`comps` is deprecated"):
            getattr(n.statistics, method_name)(comps="Generator")

    for method_name in ["supply", "energy_balance", "capex"]:
        with pytest.warns(DeprecationWarning, match="`aggregate_groups` is deprecated"):
            getattr(n.statistics, method_name)(aggregate_groups="sum")

    for method_name in ["supply", "withdrawal", "transmission"]:
        with pytest.warns(DeprecationWarning, match="`aggregate_time` is deprecated"):
            getattr(n.statistics, method_name)(aggregate_time="mean")

    with pytest.warns(DeprecationWarning, match="`comps` is deprecated"):
        n.statistics(comps="Generator")
    with pytest.warns(DeprecationWarning, match="`aggregate_groups` is deprecated"):
        n.statistics(aggregate_groups="sum")

    with pytest.raises(DeprecationWarning, match="received both comps and components"):
        n.statistics.installed_capacity(comps="Generator", components="Generator")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_result = n.statistics.installed_capacity(comps="Generator")
    new_result = n.statistics.installed_capacity(components="Generator")
    assert old_result.equals(new_result)
