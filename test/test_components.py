import copy
import sys

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

    assert not generators.issubset(n.generators.index)
    assert not generators.issubset(n.generators_t.p_max_pu.columns)


def test_remove_misspelled_component(ac_dc_network):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    a misspelled component is removed with remove

    THEN    the function should not change anything in the Line
    component dataframe and an error should be logged.
    """
    n = ac_dc_network

    with pytest.raises(ValueError):
        n.remove("Liness", ["0", "1"])


def test_add_misspelled_component(n_5bus):
    """
    GIVEN   an empty PyPSA network with 5 buses.

    WHEN    multiple components of a misspelled component are added

    THEN    the function should not change anything and an error should
    be logged.
    """
    misspelled_component = "Generatro"
    with pytest.raises(ValueError, match="not found"):
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
    buses = n_5bus.buses.index[slicer]

    load_names = "load_" + buses

    n_5bus.add("Load", load_names, bus=buses, p_set=3)

    if slicer == 0:
        load_names = pd.Index([load_names])

    assert len(n_5bus.loads) == len(load_names)
    assert n_5bus.loads.index.name == "Load"
    assert n_5bus.loads.index.equals(load_names)
    assert (n_5bus.loads.bus == buses).all()
    assert (n_5bus.loads.p_set == 3).all()

    if slicer == slice(None, None):
        # Test different names shape
        with pytest.raises(ValueError, match="for each component name."):
            n_5bus.add("Load", load_names[1:] + "_a", bus=buses)


@pytest.mark.parametrize("slicer", [slice(0, 1), slice(None, None)])
def test_add_static_with_index(n_5bus, slicer):
    buses = n_5bus.buses.index[slicer]

    load_names = "load_" + buses
    buses = pd.Series(buses, index=load_names)

    n_5bus.add(
        "Load",
        load_names,
        bus=buses,
        p_set=3,
    )

    assert len(n_5bus.loads) == len(load_names)
    assert n_5bus.loads.index.name == "Load"
    assert n_5bus.loads.index.equals(load_names)
    assert (n_5bus.loads.bus == buses).all()
    assert (n_5bus.loads.p_set == 3).all()

    if len(buses) > 1:
        # Test unaligned names index
        with pytest.raises(ValueError, match="index which does not align"):
            n_5bus.add("Load", load_names + "_a", bus=swap_df_index(buses))


def test_add_varying_single(n_5bus_7sn):
    buses = n_5bus_7sn.buses.index

    # Add load component at every bus with time-dependent attribute p_set.
    p_set = rng.random(size=(len(n_5bus_7sn.snapshots)))
    n_5bus_7sn.add(
        "Load",
        "load_1",
        bus=buses[0],
        p_set=p_set,
    )

    assert len(n_5bus_7sn.loads) == 1
    assert n_5bus_7sn.loads.index.name == "Load"
    assert (n_5bus_7sn.loads.index == "load_1").all()
    assert (n_5bus_7sn.loads.bus == buses[0]).all()
    assert (n_5bus_7sn.loads_t.p_set.T == p_set).all().all()
    assert (n_5bus_7sn.loads.p_set == 0).all()  # Assert that default value is set

    # Test different snapshots shape
    with pytest.raises(ValueError, match="for each snapshot"):
        n_5bus_7sn.add(
            "Load",
            "load_1_a",
            p_set=p_set[1:],
        )


@pytest.mark.parametrize("slicer", [slice(0, 1), slice(None, None)])
def test_add_varying_multiple(n_5bus_7sn, slicer):
    buses = n_5bus_7sn.buses.index[slicer]

    # Add load component at every bus with time-dependent attribute p_set.
    load_names = "load_" + buses
    p_set = rng.random(size=(len(n_5bus_7sn.snapshots), len(buses)))
    n_5bus_7sn.add(
        "Load",
        load_names,
        bus=buses,
        p_set=p_set,
    )

    assert len(n_5bus_7sn.loads) == len(load_names)
    assert n_5bus_7sn.loads.index.name == "Load"
    assert n_5bus_7sn.loads.index.equals(load_names)
    assert (n_5bus_7sn.loads.bus == buses).all()
    assert (n_5bus_7sn.loads_t.p_set == p_set).all().all()
    assert (n_5bus_7sn.loads.p_set == 0).all()  # Assert that default value is set

    if len(buses) > 1:
        # Test different names shape
        with pytest.raises(ValueError, match="but expected"):
            n_5bus_7sn.add("Load", load_names[1:] + "_a", p_set=p_set)

    # Test different snapshots shape
    with pytest.raises(ValueError, match="but expected"):
        n_5bus_7sn.add("Load", load_names + "_c", p_set=p_set[1:])


def test_add_varying_multiple_with_index(n_5bus_7sn):
    buses = n_5bus_7sn.buses.index

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

    assert len(n_5bus_7sn.loads) == len(load_names)
    assert n_5bus_7sn.loads.index.name == "Load"
    assert n_5bus_7sn.loads.index.equals(load_names)
    assert (n_5bus_7sn.loads.bus == buses).all()
    assert (n_5bus_7sn.loads_t.p_set == p_set).all().all()
    assert (n_5bus_7sn.loads.p_set == 0).all()  # Assert that default value is set

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

    assert (n_5bus.buses.iloc[:5].x == 0).all()
    assert (n_5bus.buses.iloc[5].x == 1).all()
    assert caplog.records[-1].levelname == "WARNING"

    n_5bus.add("Bus", [f"bus_{i} " for i in range(5)], x=1, overwrite=True)
    assert (n_5bus.buses.x == 1).all()


def test_add_overwrite_varying(n_5bus_7sn, caplog):
    bus_names = [f"bus_{i} " for i in range(6)]

    n_5bus_7sn.add("Bus", bus_names, p=[1] * 6)
    assert (n_5bus_7sn.buses_t.p.iloc[:, :5] == 0).all().all()
    assert (n_5bus_7sn.buses_t.p.iloc[:, 5] == 1).all().all()
    assert caplog.records[-1].levelname == "WARNING"

    n_5bus_7sn.add("Bus", bus_names[:5], p=[2] * 5, overwrite=True)
    assert (n_5bus_7sn.buses_t.p.loc[:, bus_names[:5]] == 2).all().all()
    assert (n_5bus_7sn.buses_t.p.loc[:, bus_names[5]] == 1).all().all()

    p = rng.random(size=(7, 5))
    n_5bus_7sn.add("Bus", bus_names[:5], p=p, overwrite=False)
    assert (n_5bus_7sn.buses_t.p.loc[:, bus_names[:5]] == 2).all().all()
    n_5bus_7sn.add("Bus", bus_names[:5], p=p, overwrite=True)
    assert (n_5bus_7sn.buses_t.p.loc[:, bus_names[:5]] == p).all().all()


def test_multiple_add_defaults(n_5bus):
    """
    GIVEN   an empty PyPSA network with 5 buses.

    WHEN    adding multiple components of Generator and Load with add

    THEN    the defaults should be set correctly according to
    n.component_attrs.
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

    assert (
        n_5bus.generators.loc[gen_names[0], "control"]
        == n_5bus.component_attrs.Generator.loc["control", "default"]
    )

    assert (
        n_5bus.loads.loc[line_names[0], "p_set"]
        == n_5bus.component_attrs.Load.loc["p_set", "default"]
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="pd.equals fails on windows (https://stackoverflow.com/questions/62128721).",
)
def test_equality_behavior(all_networks):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    comparing the network to itself

    THEN    the networks should be equal.
    """
    for n in all_networks:
        deep_copy = copy.deepcopy(n)
        assert n == deep_copy
        assert n is not deep_copy

        # TODO: Could add more property based tests here (hypothesis)
        deep_copy.name = "new_name"
        assert n != deep_copy


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="pd.equals fails on windows (https://stackoverflow.com/questions/62128721).",
)
def test_copy_default_behavior(all_networks):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    copying the network with timestamps

    THEN    the copied network should have the same generators, loads
    and timestamps.
    """
    for n in all_networks:
        network_copy = n.copy()
        assert n == network_copy
        assert n is not network_copy


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="pd.equals fails on windows (https://stackoverflow.com/questions/62128721).",
)
def test_copy_snapshots(all_networks):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    copying the network without snapshots

    THEN    the copied network should only have the current time index.
    """
    for n in all_networks:
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
    new_buses = set(n.buses.index)
    assert new_buses.issuperset(n_5bus.buses.index)


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
    new_buses = set(n.buses.index)
    assert new_buses.issuperset(n_5bus.buses.index)


def test_shape_reprojection(ac_dc_network_shapes):
    n = ac_dc_network_shapes

    with pytest.warns(UserWarning):
        area_before = n.shapes.geometry.area.sum()
    x, y = n.buses.x.values, n.buses.y.values

    n.to_crs("epsg:3035")

    assert n.shapes.crs == "epsg:3035"
    assert n.crs == "epsg:3035"
    assert area_before != n.shapes.geometry.area.sum()
    assert not np.allclose(x, n.buses.x.values)
    assert not np.allclose(y, n.buses.y.values)
