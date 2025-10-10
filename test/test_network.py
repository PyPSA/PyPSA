# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import copy
import sys
import warnings

import linopy
import numpy as np
import pandas as pd
import pytest

import pypsa

rng = np.random.default_rng()


def swap_df_index(df, axis=0):
    df_swapped = df.copy()

    if axis == 0:
        # Swap rows: swap row 0 with row 1
        df_swapped.iloc[[0, 1]] = df_swapped.iloc[[1, 0]].values
    elif axis == 1:
        # Swap columns: swap column 0 with column 1
        df_swapped.iloc[:, [0, 1]] = df_swapped.iloc[:, [1, 0]].values
    else:
        raise ValueError(
            f"Invalid axis {axis}. Allowed values are 0 (rows) and 1 (columns)."
        )

    return df_swapped


@pytest.fixture
def n_5bus():
    # Set up empty network with 5 buses.
    n = pypsa.Network()
    n.add("Bus", [f"bus_{i} " for i in range(5)])
    return n


@pytest.fixture
def n_5bus_7sn():
    # Set up empty network with 5 buses and 7 snapshots.
    n = pypsa.Network()
    n_buses = 5
    n_snapshots = 7
    n.add("Bus", [f"bus_{i} " for i in range(n_buses)])
    n.set_snapshots(range(n_snapshots))
    return n


def test_remove(ac_dc_network):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    two components of Generator are removed with remove

    THEN    the generator dataframe and the time-dependent generator
    dataframe should not contain the removed elements.
    """
    n = ac_dc_network

    generators = {"Manchester Wind", "Frankfurt Wind"}

    n.remove("Generator", generators)

    assert not generators.issubset(n.c.generators.static.index)
    assert not generators.issubset(n.c.generators.dynamic.p_max_pu.columns)


def test_remove_misspelled_component(ac_dc_network):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    a misspelled component is removed with remove

    THEN    the function should not change anything in the Line
    component dataframe and an error should be logged.
    """
    n = ac_dc_network
    misspelled_component = "Liness"
    with pytest.raises(AttributeError, match=f"components '{misspelled_component}'"):
        n.remove(misspelled_component, ["0", "1"])


def test_add_misspelled_component(n_5bus):
    """
    GIVEN   an empty PyPSA network with 5 buses.

    WHEN    multiple components of a misspelled component are added

    THEN    the function should not change anything and an error should
    be logged.
    """
    misspelled_component = "Generatro"
    with pytest.raises(AttributeError, match=f"components '{misspelled_component}'"):
        n_5bus.add(
            misspelled_component,
            ["g_1", "g_2"],
            bus=["bus_1", "bus_2"],
        )


def test_add_duplicated_names(n_5bus):
    """
    GIVEN   an empty PyPSA network with 5 buses.

    WHEN    adding generators with the same name

    THEN    the function should fail and an error should be logged.
    """
    with pytest.raises(ValueError, match="must be unique"):
        n_5bus.add(
            "Generator",
            ["g_1", "g_1"],
            bus=["bus_1", "bus_2"],
        )


@pytest.mark.parametrize("slicer", [0, slice(0, 1), slice(None, None)])
def test_add_static(n_5bus, slicer):
    buses = n_5bus.c.buses.static.index[slicer]

    load_names = "load_" + buses

    n_5bus.add("Load", load_names, bus=buses, p_set=3)

    if slicer == 0:
        load_names = pd.Index([load_names])

    assert len(n_5bus.c.loads.static) == len(load_names)
    assert n_5bus.c.loads.static.index.name == "name"
    assert n_5bus.c.loads.static.index.equals(load_names)
    assert (n_5bus.c.loads.static.bus == buses).all()
    assert (n_5bus.c.loads.static.p_set == 3).all()

    if slicer == slice(None, None):
        # Test different names shape
        with pytest.raises(ValueError, match="for each component name."):
            n_5bus.add("Load", load_names[1:] + "_a", bus=buses)


@pytest.mark.parametrize("slicer", [slice(0, 1), slice(None, None)])
def test_add_static_with_index(n_5bus, slicer):
    buses = n_5bus.c.buses.static.index[slicer]

    load_names = "load_" + buses
    buses = pd.Series(buses, index=load_names)

    n_5bus.add(
        "Load",
        load_names,
        bus=buses,
        p_set=3,
    )

    assert len(n_5bus.c.loads.static) == len(load_names)
    assert n_5bus.c.loads.static.index.name == "name"
    assert n_5bus.c.loads.static.index.equals(load_names)
    assert (n_5bus.c.loads.static.bus == buses).all()
    assert (n_5bus.c.loads.static.p_set == 3).all()

    if len(buses) > 1:
        # Test unaligned names index
        with pytest.raises(ValueError, match="index which does not align"):
            n_5bus.add("Load", load_names + "_a", bus=swap_df_index(buses))


def test_add_varying_single(n_5bus_7sn):
    buses = n_5bus_7sn.c.buses.static.index

    # Add load component at every bus with time-dependent attribute p_set.
    p_set = rng.random(size=(len(n_5bus_7sn.snapshots)))
    n_5bus_7sn.add(
        "Load",
        "load_1",
        bus=buses[0],
        p_set=p_set,
    )

    assert len(n_5bus_7sn.c.loads.static) == 1
    assert n_5bus_7sn.c.loads.static.index.name == "name"
    assert (n_5bus_7sn.c.loads.static.index == "load_1").all()
    assert (n_5bus_7sn.c.loads.static.bus == buses[0]).all()
    assert (p_set == n_5bus_7sn.c.loads.dynamic.p_set.T).all().all()
    assert (
        n_5bus_7sn.c.loads.static.p_set == 0
    ).all()  # Assert that default value is set

    # Test different snapshots shape
    with pytest.raises(ValueError, match="for each snapshot"):
        n_5bus_7sn.add(
            "Load",
            "load_1_a",
            p_set=p_set[1:],
        )


@pytest.mark.parametrize("slicer", [slice(0, 1), slice(None, None)])
def test_add_varying_multiple(n_5bus_7sn, slicer):
    buses = n_5bus_7sn.c.buses.static.index[slicer]

    # Add load component at every bus with time-dependent attribute p_set.
    load_names = "load_" + buses
    p_set = rng.random(size=(len(n_5bus_7sn.snapshots), len(buses)))
    n_5bus_7sn.add(
        "Load",
        load_names,
        bus=buses,
        p_set=p_set,
    )

    assert len(n_5bus_7sn.c.loads.static) == len(load_names)
    assert n_5bus_7sn.c.loads.static.index.name == "name"
    assert n_5bus_7sn.c.loads.static.index.equals(load_names)
    assert (n_5bus_7sn.c.loads.static.bus == buses).all()
    assert (n_5bus_7sn.c.loads.dynamic.p_set == p_set).all().all()
    assert (
        n_5bus_7sn.c.loads.static.p_set == 0
    ).all()  # Assert that default value is set

    if len(buses) > 1:
        # Test different names shape
        with pytest.raises(ValueError, match="but expected"):
            n_5bus_7sn.add("Load", load_names[1:] + "_a", p_set=p_set)

    # Test different snapshots shape
    with pytest.raises(ValueError, match="but expected"):
        n_5bus_7sn.add("Load", load_names + "_c", p_set=p_set[1:])


def test_add_varying_multiple_with_index(n_5bus_7sn):
    buses = n_5bus_7sn.c.buses.static.index

    # Add load component at every bus with time-dependent attribute p_set.
    load_names = "load_" + buses
    p_set = pd.DataFrame(
        rng.random(size=(len(n_5bus_7sn.snapshots), len(buses))),
        index=n_5bus_7sn.snapshots,
        columns=load_names,
    )

    n_5bus_7sn.add(
        "Load",
        load_names,
        bus=buses,
        p_set=p_set,
    )

    assert len(n_5bus_7sn.c.loads.static) == len(load_names)
    assert n_5bus_7sn.c.loads.static.index.name == "name"
    assert n_5bus_7sn.c.loads.static.index.equals(load_names)
    assert (n_5bus_7sn.c.loads.static.bus == buses).all()
    assert (n_5bus_7sn.c.loads.dynamic.p_set == p_set).all().all()
    assert (
        n_5bus_7sn.c.loads.static.p_set == 0
    ).all()  # Assert that default value is set

    # Test different names shape
    with pytest.raises(ValueError, match="index which does not align"):
        n_5bus_7sn.add("Load", load_names[1:] + "_a", p_set=p_set)

    # Test unaligned names index
    with pytest.raises(ValueError, match="index which does not align"):
        n_5bus_7sn.add("Load", load_names + "_b", p_set=swap_df_index(p_set))

    # Test different snapshots shape
    with pytest.raises(ValueError, match="index which does not align"):
        n_5bus_7sn.add("Load", load_names + "_c", p_set=p_set[1:])

    # Test different snapshots index
    with pytest.raises(ValueError, match="index which does not align"):
        n_5bus_7sn.add("Load", load_names + "_d", p_set=swap_df_index(p_set, axis=1))


def test_add_overwrite_static(n_5bus, caplog):
    n_5bus.add("Bus", [f"bus_{i} " for i in range(6)], x=1)

    assert (n_5bus.c.buses.static.iloc[:5].x == 0).all()
    assert (n_5bus.c.buses.static.iloc[5].x == 1).all()
    assert caplog.records[-1].levelname == "WARNING"

    n_5bus.add("Bus", [f"bus_{i} " for i in range(5)], x=1, overwrite=True)
    assert (n_5bus.c.buses.static.x == 1).all()


def test_add_overwrite_varying(n_5bus_7sn, caplog):
    bus_names = [f"bus_{i} " for i in range(6)]

    n_5bus_7sn.add("Bus", bus_names, p=[1] * 6)
    assert (n_5bus_7sn.c.buses.dynamic.p.iloc[:, :5] == 0).all().all()
    assert (n_5bus_7sn.c.buses.dynamic.p.iloc[:, 5] == 1).all().all()
    assert caplog.records[-1].levelname == "WARNING"

    n_5bus_7sn.add("Bus", bus_names[:5], p=[2] * 5, overwrite=True)
    assert (n_5bus_7sn.c.buses.dynamic.p.loc[:, bus_names[:5]] == 2).all().all()
    assert (n_5bus_7sn.c.buses.dynamic.p.loc[:, bus_names[5]] == 1).all().all()

    p = rng.random(size=(7, 5))
    n_5bus_7sn.add("Bus", bus_names[:5], p=p, overwrite=False)
    assert (n_5bus_7sn.c.buses.dynamic.p.loc[:, bus_names[:5]] == 2).all().all()
    n_5bus_7sn.add("Bus", bus_names[:5], p=p, overwrite=True)
    assert (n_5bus_7sn.c.buses.dynamic.p.loc[:, bus_names[:5]] == p).all().all()


def test_add_stochastic():
    n = pypsa.Network()
    n.add("Bus", "bus_1", v_mag_pu_set=0.1)
    n.add("Bus", "bus_2", v_mag_pu_set=0.1)

    multi_indexed = pd.MultiIndex.from_product(
        [["bus_3", "bus_4"], ["scenario_1", "scenario_2"]]
    )

    with pytest.raises(TypeError, match="Component names must be a one-dimensional."):
        n.add("Bus", multi_indexed, v_mag_pu_set=0.1)

    n.set_scenarios(["scenario_1", "scenario_2"])

    with pytest.raises(
        TypeError,
        match=(
            "Component names must be a one-dimensional. For stochastic networks, they "
            "will be casted to all dimensions and data per scenario can be changed after adding them."
        ),
    ):
        n.add("Bus", multi_indexed, v_mag_pu_set=0.1)


def test_multiple_add_defaults(n_5bus):
    """
    GIVEN   an empty PyPSA network with 5 buses.

    WHEN    adding multiple components of Generator and Load with add

    THEN    the defaults should be set correctly according to
    n.default_component_attrs.
    """
    gen_names = ["g_1", "g_2"]
    n_5bus.add(
        "Generator",
        gen_names,
        bus=["bus_1", "bus_2"],
    )

    line_names = ["l_1", "l_2"]
    n_5bus.add(
        "Load",
        line_names,
        bus=["bus_1", "bus_2"],
    )

    # TODO: Improve tests since component is the same now
    assert (
        n_5bus.c.generators.static.loc[gen_names[0], "control"]
        == n_5bus.components.Generator.defaults.loc["control", "default"]
    )

    assert (
        n_5bus.c.loads.static.loc[line_names[0], "p_set"]
        == n_5bus.components.Load.defaults.loc["p_set", "default"]
    )


def test_add_return_names():
    """Test that return_names parameter controls return behavior."""
    n = pypsa.Network()

    # Default behavior - should return None
    assert n.add("Bus", "bus1") is None
    assert n.add("Bus", "bus2", return_names=False) is None

    # With return_names=True - should return Index
    result = n.add("Bus", "bus3", return_names=True)
    assert isinstance(result, pd.Index)
    assert result[0] == "bus3"

    # Multiple components
    result = n.add("Bus", ["bus4", "bus5"], return_names=True)
    assert len(result) == 2
    assert all(name in result for name in ["bus4", "bus5"])

    # Component method
    assert n.components.buses.add("bus6") is None
    result = n.components.buses.add("bus7", return_names=True)
    assert isinstance(result, pd.Index)
    assert result[0] == "bus7"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="pd.equals fails on windows (https://stackoverflow.com/questions/62128721).",
)
def test_equality_behavior(networks):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    comparing the network to itself

    THEN    the networks should be equal.
    """
    n = networks
    deep_copy = copy.deepcopy(n)
    assert n is not deep_copy
    assert n.equals(deep_copy, log_mode="strict")

    assert n == deep_copy

    # TODO: Could add more property based tests here (hypothesis)
    deep_copy.name = "new_name"
    assert n != deep_copy

    assert n != "other_type"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="pd.equals fails on windows (https://stackoverflow.com/questions/62128721).",
)
def test_copy_default_behavior(networks):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    copying the network with timestamps

    THEN    the copied network should have the same generators, loads
    and timestamps.
    """
    n = networks
    network_copy = n.copy()
    assert n == network_copy
    assert n is not network_copy


def test_copy_with_model(ac_dc_network):
    n = ac_dc_network
    n.optimize.create_model()
    n_copy = n.copy()

    assert n.equals(n_copy, log_mode="strict")
    assert isinstance(n.model, linopy.Model)
    assert isinstance(n_copy.model, linopy.Model)

    n.optimize.solve_model()
    with pytest.raises(
        ValueError,
        match="Copying a solved network with an attached solver model is not supported.",
    ):
        n_copy = n.copy()

    n.optimize()
    with pytest.raises(
        ValueError,
        match="Copying a solved network with an attached solver model is not supported.",
    ):
        n_copy = n.copy()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="pd.equals fails on windows (https://stackoverflow.com/questions/62128721).",
)
def test_copy_snapshots(networks):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    copying the network without snapshots

    THEN    the copied network should only have the current time index.
    """
    n = networks

    if n.has_scenarios:
        with pytest.raises(
            NotImplementedError,
            match="Copying a stochastic network with a selection is currently not supported.",
        ):
            n.copy(snapshots=n.snapshots[:5])

        return

    copied_n = n.copy(snapshots=[])
    assert copied_n.snapshots.size == 1

    copied_n = n.copy(snapshots=n.snapshots[:5])
    n.set_snapshots(n.snapshots[:5])
    assert copied_n == n


def test_single_add_network_static(ac_dc_network, n_5bus):
    """
    GIVEN   the AC DC exemplary pypsa network and an empty PyPSA network with 5
    buses.

    WHEN    the second network is added to the first

    THEN    the first network should now contain its original buses and
    also the buses in the second network
    """
    n = ac_dc_network.merge(n_5bus, with_time=False)
    new_buses = set(n.c.buses.static.index)
    assert new_buses.issuperset(n_5bus.c.buses.static.index)


def test_single_add_network_with_time(ac_dc_network, n_5bus):
    """
    GIVEN   the AC DC exemplary pypsa network and an empty PyPSA network with 5
    buses and the same snapshots.

    WHEN    the second network is added to the first

    THEN    the first network should now contain its original buses and
    also the buses in the second network
    """
    with pytest.raises(ValueError):
        ac_dc_network.merge(n_5bus, with_time=True)

    n_5bus.set_snapshots(ac_dc_network.snapshots)
    n = ac_dc_network.merge(n_5bus, with_time=True)
    new_buses = set(n.c.buses.static.index)
    assert new_buses.issuperset(n_5bus.c.buses.static.index)


def test_shape_reprojection(ac_dc_shapes):
    n = ac_dc_shapes

    with pytest.warns(UserWarning):  # noqa
        area_before = n.c.shapes.static.geometry.area.sum()
    x, y = n.c.buses.static.x.values, n.c.buses.static.y.values

    n.to_crs("epsg:3035")

    assert n.c.shapes.static.crs == "epsg:3035"
    assert n.crs == "epsg:3035"
    assert area_before != n.c.shapes.static.geometry.area.sum()
    assert not np.allclose(x, n.c.buses.static.x.values)
    assert not np.allclose(y, n.c.buses.static.y.values)


def test_components_referencing(ac_dc_network):
    with pypsa.option_context("api.new_components_api", False):
        ac_dc_network = pypsa.examples.ac_dc_meshed()
        assert id(ac_dc_network.buses) == id(ac_dc_network.components.buses.static)
        assert id(ac_dc_network.c.buses.dynamic) == id(
            ac_dc_network.components.buses.dynamic
        )
        assert id(ac_dc_network.components.buses) == id(ac_dc_network.components.Bus)


@pytest.mark.parametrize("use_component", [True, False])
def test_rename_component_names(use_component):
    n = pypsa.Network()
    n.snapshots = [0, 1]

    n.add("Bus", "bus1", v_mag_pu_set=[0.1, 0.2])
    n.add("Bus", "bus2", v_mag_pu_set=[0.1, 0.2])
    n.add("Line", "line1", bus0="bus1", bus1="bus2", s_max_pu=[0.1, 0.2])
    n.add("Generator", "gen1", bus="bus1", p_min_pu=[0.1, 0.2])

    if use_component:
        c = n.c.buses
        c.rename_component_names(bus1="bus3")
    else:
        n.rename_component_names("Bus", bus1="bus3")

    with pytest.raises(ValueError):
        n.rename_component_names("Bus", bus1=10)

    assert "bus1" not in n.c.buses.static.index
    assert "bus1" not in n.c.buses.dynamic.v_mag_pu_set.columns
    assert "bus2" in n.c.buses.static.index
    assert "bus2" in n.c.buses.dynamic.v_mag_pu_set.columns
    assert "bus3" in n.c.buses.static.index
    assert "bus3" in n.c.buses.dynamic.v_mag_pu_set.columns

    assert "bus1" not in n.c.lines.static.bus0.to_list()
    assert "bus1" not in n.c.lines.static.bus1.to_list()
    assert "bus3" in n.c.lines.static.bus0.to_list()
    assert "bus2" in n.c.lines.static.bus1.to_list()

    assert "bus1" not in n.c.generators.static.bus.to_list()
    assert "bus3" in n.c.generators.static.bus.to_list()


def test_components_repr(ac_dc_network):
    n = ac_dc_network

    assert repr(n).startswith("PyPSA Network 'AC-DC-Meshed'")
    assert len(repr(n)) > len(str(n))

    n = pypsa.Network()
    assert repr(n).startswith("Empty PyPSA Network 'Unnamed Network'")
    assert len(repr(n)) > len(str(n))


@pytest.mark.parametrize("new_components_api", [True, False])
def test_api_components_legacy(new_components_api):
    """
    Test the API of the components module.
    """
    with pypsa.option_context("api.new_components_api", new_components_api):
        n = pypsa.examples.ac_dc_meshed()

        if not new_components_api:
            assert n.buses is n.components.buses.static
            assert n.c.buses.dynamic is n.components.buses.dynamic
            assert n.lines is n.components.lines.static
            assert n.c.lines.dynamic is n.components.lines.dynamic
            assert n.generators is n.components.generators.static
            assert n.c.generators.dynamic is n.components.generators.dynamic
        else:
            assert n.buses is n.components.buses
            with pytest.warns(
                DeprecationWarning, match=r"Use `n\.buses\.dynamic` as a drop-in"
            ):
                assert n.buses_t is n.components.buses.dynamic
            assert n.lines is n.components.lines
            with pytest.warns(
                DeprecationWarning, match=r"Use `n\.lines\.dynamic` as a drop-in"
            ):
                assert n.lines_t is n.components.lines.dynamic
            assert n.generators is n.components.generators
            with pytest.warns(
                DeprecationWarning, match=r"Use `n\.generators\.dynamic` as a drop-in"
            ):
                assert n.generators_t is n.components.generators.dynamic


@pytest.mark.parametrize("new_components_api", [True, False])
def test_api_new_components_api(component_name, new_components_api):
    """
    Test the API of the components module.
    """
    warnings.filterwarnings(
        "ignore",
        message=".*is deprecated as of 1.0 and will be .*",
        category=DeprecationWarning,
    )
    with pypsa.option_context("api.new_components_api", new_components_api):
        n = pypsa.examples.ac_dc_meshed()
        if not new_components_api:
            with pytest.warns(
                DeprecationWarning,
                match="Use `self.components.<component>.static` instead.",
            ):
                assert n.static(component_name) is n.c[component_name].static
            with pytest.warns(
                DeprecationWarning,
                match="Use `self.components.<component>.dynamic` instead.",
            ):
                assert n.dynamic(component_name) is n.c[component_name].dynamic

            setattr(n, component_name, "test")
            with pytest.warns(
                DeprecationWarning,
                match="Use `self.components.<component>.static` instead.",
            ):
                assert n.static(component_name) == "test"
            setattr(n, f"{component_name}_t", "test")
            with pytest.warns(
                DeprecationWarning,
                match="Use `self.components.<component>.dynamic` instead.",
            ):
                assert n.dynamic(component_name) == "test"
        else:
            with pytest.warns(
                DeprecationWarning,
                match="Use `self.components.<component>.static` instead.",
            ):
                assert n.static(component_name) is n.c[component_name].static
            with pytest.warns(
                DeprecationWarning,
                match="Use `self.components.<component>.dynamic` instead.",
            ):
                assert n.dynamic(component_name) is n.c[component_name].dynamic
            with pytest.raises(AttributeError):
                setattr(n, component_name, "test")
            with pytest.warns(DeprecationWarning, match="cannot be set"):
                setattr(n, f"{component_name}_t", "test")
