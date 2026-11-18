# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for runtime verification guards."""

import subprocess
import sys

import pandas as pd
import pytest
import xarray as xr

import pypsa
from pypsa.common import UnexpectedError
from pypsa.guards import _assert_data_integrity, _assert_xarray_integrity


def test_log_free_import():
    """Test that importing pypsa does not print anything."""
    code = "import pypsa"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.stdout == "", "No stdout should be produced"
    assert result.stderr == "", "No stderr should be produced"


def test_runtime_verification_enabled():
    """Test that runtime verification is enabled during tests."""
    assert pypsa.options.debug.runtime_verification is True, (
        "Runtime verification should be enabled during tests"
    )


def test_runtime_verification_disabled_by_default():
    """Test that runtime verification is disabled by default without conftest."""
    code = "import pypsa; print(pypsa.options.debug.runtime_verification)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    last_line = result.stdout.strip().splitlines()[-1]
    assert last_line == "False", "Runtime verification should be False by default"


def _rename_period_levels(n):
    n.set_investment_periods([2020, 2030])
    index = n._snapshots_data.index.set_names(["a", "b"])
    index.name = "snapshot"
    n._snapshots_data.index = index


def _drop_period_years(n):
    n.set_investment_periods([2020, 2030])
    n._investment_periods_data = n._investment_periods_data.drop(columns="years")


def _rename_scenario_index(n):
    n.set_scenarios(["low", "high"])
    n._scenarios_data.index.name = "foo"


def _drop_scenario_weight(n):
    n.set_scenarios(["low", "high"])
    n._scenarios_data = n._scenarios_data.drop(columns="weight")


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (
            lambda n: setattr(n.c.generators.static.index, "name", "foo"),
            "static index name",
        ),
        (
            lambda n: n.c.generators.dynamic.update(p_max_pu=pd.Series([1.0])),
            "not a DataFrame",
        ),
        (
            lambda n: n.c.generators.dynamic.update(
                bogus=pd.DataFrame({"no_such_generator": [1.0]}, index=n.snapshots)
            ),
            "not in the static index",
        ),
        (
            lambda n: setattr(n, "_snapshots_data", n._snapshots_data.iloc[:0]),
            "must not be empty",
        ),
        (
            lambda n: setattr(n._snapshots_data.index, "name", "foo"),
            "must be named 'snapshot'",
        ),
        (
            lambda n: setattr(
                n, "_snapshots_data", n._snapshots_data.drop(columns="stores")
            ),
            "missing required columns",
        ),
        (_rename_period_levels, "must have names"),
        (_drop_period_years, "missing required columns"),
        (_rename_scenario_index, "must be named 'scenario'"),
        (_drop_scenario_weight, "'weight' column"),
    ],
)
def test_data_integrity_guard_detects_corruption(ac_dc_network, corrupt, match):
    n = ac_dc_network.copy()
    corrupt(n)
    with pytest.raises(UnexpectedError, match=match):
        _assert_data_integrity(n)


def test_xarray_integrity_guard_detects_order_mismatch(ac_dc_network):
    c = ac_dc_network.c.generators
    da = xr.DataArray(range(len(c.names)), coords={"name": c.names[::-1]})
    with pytest.raises(UnexpectedError, match="Component order mismatch"):
        _assert_xarray_integrity(c, da)


def test_guard_wraps_unexpected_exceptions():
    with pytest.raises(UnexpectedError, match="Unexpected error in guard function"):
        _assert_data_integrity(None)
