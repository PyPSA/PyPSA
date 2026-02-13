# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import xarray

from pypsa.components.array import _from_xarray


def test_as_xarray_static(ac_dc_network):
    n = ac_dc_network
    da = n.c.generators._as_xarray("bus")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert list(da.coords) == ["name"]
    assert np.array_equal(da.coords["name"], n.c.generators.static.index)

    # Check data
    assert np.array_equal(da.values, n.c.generators.static["bus"].values)


def test_as_xarray_dynamic(ac_dc_network):
    n = ac_dc_network
    da = n.c.generators._as_xarray("p_max_pu")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert list(da.coords) == ["snapshot", "name"]
    assert np.array_equal(da.snapshot, n.snapshots)
    assert np.array_equal(da.coords["name"], n.c.generators.static.index)

    # Check data
    non_dynamic_index = n.c.generators.static.index.difference(
        n.c.generators.dynamic.p_max_pu.columns
    )
    assert np.array_equal(
        da.sel(name=non_dynamic_index),
        np.ones((10, 3))
        * n.c.generators.static.loc[non_dynamic_index, "p_max_pu"].values,
    )
    assert np.array_equal(
        da.sel(name=n.c.generators.dynamic["p_max_pu"].columns),
        n.c.generators.dynamic["p_max_pu"].values,
    )


def test_as_xarray_static_with_periods(ac_dc_network):
    """
    This is the same test as test_as_xarray_static, since static data is not
    affected by periods.
    """
    n = ac_dc_network
    # Add investment periods to the network
    n.investment_periods = [2000, 2010]

    da = n.c.generators._as_xarray("bus")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert list(da.coords) == ["name"]
    assert np.array_equal(da.coords["name"], n.c.generators.static.index)

    # Check data
    assert np.array_equal(da.values, n.c.generators.static["bus"].values)


def test_as_xarray_dynamic_with_periods(ac_dc_network):
    n = ac_dc_network
    # Add investment periods to the network
    n.investment_periods = [2000, 2010]

    da = n.c.generators._as_xarray("p_max_pu")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert list(da.coords) == ["snapshot", "period", "timestep", "name"]
    assert np.array_equal(da.snapshot, n.snapshots)
    assert np.array_equal(da.period.to_index().unique(), n.periods)
    assert np.array_equal(da.timestep.to_index().unique(), n.timesteps)
    assert np.array_equal(da.coords["name"], n.c.generators.static.index)

    # Check data
    non_dynamic_index = n.c.generators.static.index.difference(
        n.c.generators.dynamic.p_max_pu.columns
    )
    assert np.array_equal(
        da.sel(name=non_dynamic_index),
        np.ones((20, 3))
        * n.c.generators.static.loc[non_dynamic_index, "p_max_pu"].values,
    )
    assert np.array_equal(
        da.sel(name=n.c.generators.dynamic["p_max_pu"].columns),
        n.c.generators.dynamic["p_max_pu"].values,
    )


def test_as_xarray_static_with_scenarios(ac_dc_network):
    n = ac_dc_network
    # Add scenarios to the network
    scenarios = ["scenario1", "scenario2"]
    n.scenarios = scenarios
    da = n.c.generators._as_xarray("bus")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert list(da.coords) == ["scenario", "name"]
    assert np.array_equal(
        da.coords["name"], n.c.generators.static.index.get_level_values("name").unique()
    )
    assert np.array_equal(da.scenario, scenarios)

    # Check data
    assert np.array_equal(da.values.flatten(), n.c.generators.static["bus"].values)


def test_as_xarray_dynamic_with_scenarios(ac_dc_network):
    n = ac_dc_network
    # Add scenarios to the network
    scenarios = ["scenario1", "scenario2"]
    n.scenarios = scenarios

    da = n.c.generators._as_xarray("p_max_pu")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert np.array_equal(da.snapshot, n.snapshots)
    # assert np.array_equal(
    #     da.coords["name"], n.c.generators.static.index.get_level_values("component").unique()
    # ) # TODO sorting
    assert np.array_equal(da.scenario, scenarios)

    # Check data
    # non_dynamic_index = n.c.generators.static.index.difference(n.c.generators.dynamic.p_max_pu.columns)
    assert np.array_equal(
        da.sel(
            scenario=scenarios[0],
            name=n.c.generators.dynamic.p_max_pu.columns.get_level_values(1).unique(),
        ).values,
        n.c.generators.dynamic.p_max_pu[scenarios[0]].values,
    )

    # TODO add test for non_dynamic_index


def test_as_xarray_scenario_dim_order(ac_dc_network):
    """Scenario dataarrays must have dims matching coords iteration order."""
    n = ac_dc_network
    n.scenarios = ["s1", "s2"]

    for attr in ["active", "p_max_pu", "bus"]:
        da = n.c.generators._as_xarray(attr)
        coord_dim_order = tuple(c for c in da.coords if c in da.dims)
        assert da.dims == coord_dim_order, (
            f"Dim order mismatch for {attr}: dims={da.dims}, coords order={coord_dim_order}"
        )


def test_ds_property_consistency(ac_dc_network):
    n = ac_dc_network
    """Test that ds property returns the same data as individual da calls."""
    ds = n.c.generators.ds

    # Test all attributes match individual _as_xarray calls
    for attr in ds.data_vars:
        da_individual = n.c.generators._as_xarray(attr)
        da_from_ds = ds[attr]
        assert set(da_individual.coords) == set(da_from_ds.coords)
        assert da_individual.equals(da_from_ds)


def test_from_xarray(ac_dc_types):
    """Test that _from_xarray works with the new signature requiring Components parameter."""
    n = ac_dc_types
    for c in n.components:
        # Test with a few representative attributes
        for attr in list(c.static.columns)[
            :2
        ]:  # Only test first 2 attributes per component
            da = c.da[attr]
            # Main test: ensure function works with Components parameter
            result = _from_xarray(da, c)

            # Basic checks
            assert isinstance(result, pd.DataFrame | pd.Series)

            # For dynamic data, check round-trip consistency
            if attr in c.dynamic and not c.dynamic[attr].empty:
                dynamic_data = c._as_dynamic(attr)
                # Check basic shape compatibility
                if isinstance(result, pd.DataFrame) and isinstance(
                    dynamic_data, pd.DataFrame
                ):
                    assert result.shape == dynamic_data.shape
                elif isinstance(result, pd.Series) and isinstance(
                    dynamic_data, pd.Series
                ):
                    assert len(result) == len(dynamic_data)


def test_from_xarray_auxiliary_dimensions():
    """Test _from_xarray with auxiliary dimensions like contingency scenarios."""
    import pandas as pd
    import xarray as xr

    from pypsa.components.array import _from_xarray

    # Create mock component for testing
    class MockComponent:
        def __init__(self):
            self.component_names = ["gen1", "gen2"]
            self.scenarios = ["s1", "s2"]
            self.has_scenarios = True

    c = MockComponent()

    # Test case: name + scenario + auxiliary dimension (3+ dimensions)
    data_with_scenario = xr.DataArray(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dims=["scenario", "name", "cycle"],
        coords={
            "scenario": ["s1", "s2"],
            "name": ["gen1", "gen2"],
            "cycle": ["c1", "c2"],
        },
    )
    result_with_scenario = _from_xarray(data_with_scenario, c)
    assert isinstance(result_with_scenario, pd.DataFrame | pd.Series)

    # Test case: name + snapshot + auxiliary dimension (3+ dimensions)
    data_with_snapshot = xr.DataArray(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dims=["snapshot", "name", "cycle"],
        coords={
            "snapshot": pd.date_range("2020-01-01", periods=2, freq="h"),
            "name": ["gen1", "gen2"],
            "cycle": ["c1", "c2"],
        },
    )
    c.has_scenarios = False
    result_with_snapshot = _from_xarray(data_with_snapshot, c)
    assert isinstance(result_with_snapshot, pd.DataFrame | pd.Series)


def test_from_xarray_edge_cases():
    """Test _from_xarray edge cases and dimension handling."""
    import pandas as pd
    import xarray as xr

    from pypsa.components.array import _from_xarray

    # Create mock component for testing
    class MockComponent:
        def __init__(self):
            self.names = ["gen1", "gen2"]
            self.scenarios = ["s1", "s2"]
            self.has_scenarios = False

    c = MockComponent()

    # Test case 1: Missing name dimension (should be expanded)
    data_no_name = xr.DataArray(
        [1, 2, 3],
        dims=["snapshot"],
        coords={"snapshot": pd.date_range("2020-01-01", periods=3, freq="h")},
    )
    result = _from_xarray(data_no_name, c)
    assert isinstance(result, pd.DataFrame | pd.Series)
    # After expansion, should have both name and snapshot dimensions
    assert result.shape[1] == 2  # Should have 2 components (gen1, gen2)

    # Test case 2: 2D case with snapshot + auxiliary dim (no name, expanded)
    data_2d_no_name = xr.DataArray(
        [[1, 2], [3, 4], [5, 6]],
        dims=["snapshot", "cycle"],
        coords={
            "snapshot": pd.date_range("2020-01-01", periods=3, freq="h"),
            "cycle": ["c1", "c2"],
        },
    )
    result_2d = _from_xarray(data_2d_no_name, c)
    assert isinstance(result_2d, pd.DataFrame)
    # After name expansion, we have 3+ dimensions, so combined index is created
    assert len(result_2d.columns) == 4  # gen1*c1, gen1*c2, gen2*c1, gen2*c2
