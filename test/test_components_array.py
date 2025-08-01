import numpy as np
import xarray

from pypsa.components.array import _from_xarray


def test_as_xarray_static(ac_dc_network):
    n = ac_dc_network
    da = n.components.generators._as_xarray("bus")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert list(da.coords) == ["name"]
    assert np.array_equal(da.coords["name"], n.generators.index)

    # Check data
    assert np.array_equal(da.values, n.generators["bus"].values)


def test_as_xarray_dynamic(ac_dc_network):
    n = ac_dc_network
    da = n.components.generators._as_xarray("p_max_pu")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert list(da.coords) == ["snapshot", "name"]
    assert np.array_equal(da.snapshot, n.snapshots)
    assert np.array_equal(da.coords["name"], n.generators.index)

    # Check data
    non_dynamic_index = n.generators.index.difference(n.generators_t.p_max_pu.columns)
    assert np.array_equal(
        da.sel(name=non_dynamic_index),
        np.ones((10, 3)) * n.generators.loc[non_dynamic_index, "p_max_pu"].values,
    )
    assert np.array_equal(
        da.sel(name=n.generators_t["p_max_pu"].columns),
        n.generators_t["p_max_pu"].values,
    )


def test_as_xarray_static_with_periods(ac_dc_network):
    """
    This is the same test as test_as_xarray_static, since static data is not
    affected by periods.
    """
    n = ac_dc_network
    # Add investment periods to the network
    n.investment_periods = [2000, 2010]

    da = n.components.generators._as_xarray("bus")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert list(da.coords) == ["name"]
    assert np.array_equal(da.coords["name"], n.generators.index)

    # Check data
    assert np.array_equal(da.values, n.generators["bus"].values)


def test_as_xarray_dynamic_with_periods(ac_dc_network):
    n = ac_dc_network
    # Add investment periods to the network
    n.investment_periods = [2000, 2010]

    da = n.components.generators._as_xarray("p_max_pu")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert list(da.coords) == ["snapshot", "period", "timestep", "name"]
    assert np.array_equal(da.snapshot, n.snapshots)
    assert np.array_equal(da.period.to_index().unique(), n.periods)
    assert np.array_equal(da.timestep.to_index().unique(), n.timesteps)
    assert np.array_equal(da.coords["name"], n.generators.index)

    # Check data
    non_dynamic_index = n.generators.index.difference(n.generators_t.p_max_pu.columns)
    assert np.array_equal(
        da.sel(name=non_dynamic_index),
        np.ones((20, 3)) * n.generators.loc[non_dynamic_index, "p_max_pu"].values,
    )
    assert np.array_equal(
        da.sel(name=n.generators_t["p_max_pu"].columns),
        n.generators_t["p_max_pu"].values,
    )


def test_as_xarray_static_with_scenarios(ac_dc_network):
    n = ac_dc_network
    # Add scenarios to the network
    scenarios = ["scenario1", "scenario2"]
    n.scenarios = scenarios
    da = n.components.generators._as_xarray("bus")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert list(da.coords) == ["scenario", "name"]
    assert np.array_equal(
        da.coords["name"], n.generators.index.get_level_values("name").unique()
    )
    assert np.array_equal(da.scenario, scenarios)

    # Check data
    assert np.array_equal(da.values.flatten(), n.generators["bus"].values)


def test_as_xarray_dynamic_with_scenarios(ac_dc_network):
    n = ac_dc_network
    # Add scenarios to the network
    scenarios = ["scenario1", "scenario2"]
    n.scenarios = scenarios

    da = n.components.generators._as_xarray("p_max_pu")

    assert isinstance(da, xarray.DataArray)

    # Check coords
    assert np.array_equal(da.snapshot, n.snapshots)
    # assert np.array_equal(
    #     da.coords["name"], n.generators.index.get_level_values("component").unique()
    # ) # TODO sorting
    assert np.array_equal(da.scenario, scenarios)

    # Check data
    # non_dynamic_index = n.generators.index.difference(n.generators_t.p_max_pu.columns)
    assert np.array_equal(
        da.sel(
            scenario=scenarios[0],
            name=n.generators_t.p_max_pu.columns.get_level_values(1).unique(),
        ).values,
        n.generators_t.p_max_pu[scenarios[0]].values,
    )

    # TODO add test for non_dynamic_index


def test_ds_property_consistency(ac_dc_network):
    n = ac_dc_network
    """Test that ds property returns the same data as individual da calls."""
    ds = n.components.generators.ds

    # Test all attributes match individual _as_xarray calls
    for attr in ds.data_vars:
        da_individual = n.components.generators._as_xarray(attr)
        da_from_ds = ds[attr]
        assert set(da_individual.coords) == set(da_from_ds.coords)
        assert da_individual.equals(da_from_ds)


def test_from_xarray(ac_dc_types):
    n = ac_dc_types
    for c in n.components:
        for attr in c.static.columns:
            da = c.da[attr]
            df = _from_xarray(da)
            if attr in c.dynamic and not c.dynamic[attr].empty:
                assert df.equals(c._as_dynamic(attr))
