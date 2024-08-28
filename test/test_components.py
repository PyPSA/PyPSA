import copy
import sys

import numpy as np
import pytest

import pypsa


@pytest.fixture
def empty_network_5_buses():
    # Set up empty network with 5 buses.
    network = pypsa.Network()
    n_buses = 5
    for i in range(n_buses):
        network.add("Bus", f"bus_{i}")
    return network


def test_mremove(ac_dc_network):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    two components of Generator are removed with mremove

    THEN    the generator dataframe and the time-dependent generator
    dataframe should not contain the removed elements.
    """
    network = ac_dc_network

    generators = {"Manchester Wind", "Frankfurt Wind"}

    network.mremove("Generator", generators)

    assert not generators.issubset(network.generators.index)
    assert not generators.issubset(network.generators_t.p_max_pu.columns)


def test_mremove_misspelled_component(ac_dc_network, caplog):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    a misspelled component is removed with mremove

    THEN    the function should not change anything in the Line
    component dataframe and an error should be logged.
    """
    network = ac_dc_network

    len_lines = len(network.lines.index)

    network.mremove("Liness", ["0", "1"])

    assert len_lines == len(network.lines.index)
    assert caplog.records[-1].levelname == "ERROR"


def test_madd_static(empty_network_5_buses):
    """
    GIVEN   an empty PyPSA network with 5 buses.

    WHEN    multiple components of Load are added to the network with
    madd and attribute p_set

    THEN    the corresponding load components should be in the index of
    the static load dataframe. Also the column p_set should contain any
    value greater than 0.
    """
    buses = empty_network_5_buses.buses.index

    # Add load components at every bus with attribute p_set.
    load_names = "load_" + buses
    empty_network_5_buses.madd(
        "Load",
        load_names,
        bus=buses,
        p_set=3,
    )

    assert load_names.equals(empty_network_5_buses.loads.index)
    assert (empty_network_5_buses.loads.p_set == 3).all()


def test_madd_t(empty_network_5_buses):
    """
    GIVEN   an empty PyPSA network with 5 buses and 7 snapshots.

    WHEN    multiple components of Load are added to the network with
    madd and attribute p_set

    THEN    the corresponding load components should be in the columns
    of the time-dependent load_t dataframe. Also, the shape of the
    dataframe should resemble 7 snapshots x 5 buses.
    """
    # Set up empty network with 5 buses and 7 snapshots.
    snapshots = range(7)
    empty_network_5_buses.set_snapshots(snapshots)
    buses = empty_network_5_buses.buses.index

    # Add load component at every bus with time-dependent attribute p_set.
    load_names = "load_" + buses
    rng = np.random.default_rng()  # Create a random number generator
    empty_network_5_buses.madd(
        "Load",
        load_names,
        bus=buses,
        p_set=rng.random(size=(len(snapshots), len(buses))),
    )

    assert load_names.equals(empty_network_5_buses.loads_t.p_set.columns)
    assert empty_network_5_buses.loads_t.p_set.shape == (len(snapshots), len(buses))


def test_madd_misspelled_component(empty_network_5_buses, caplog):
    """
    GIVEN   an empty PyPSA network with 5 buses.

    WHEN    multiple components of a misspelled component are added

    THEN    the function should not change anything and an error should
    be logged.
    """
    misspelled_component = "Generatro"
    empty_network_5_buses.madd(
        misspelled_component,
        ["g_1", "g_2"],
        bus=["bus_1", "bus_2"],
    )

    assert empty_network_5_buses.generators.empty
    assert caplog.records[-1].levelname == "ERROR"
    assert caplog.records[-1].message == (
        f"Component class {misspelled_component} not found"
    )


def test_madd_duplicated_index(empty_network_5_buses, caplog):
    """
    GIVEN   an empty PyPSA network with 5 buses.

    WHEN    adding generators with the same name

    THEN    the function should fail and an error should be logged.
    """
    empty_network_5_buses.madd(
        "Generator",
        ["g_1", "g_1"],
        bus=["bus_1", "bus_2"],
    )

    assert caplog.records[-1].levelname == "ERROR"
    assert caplog.records[-1].message == (
        "Error, new components for Generator are not unique"
    )


def test_madd_defaults(empty_network_5_buses):
    """
    GIVEN   an empty PyPSA network with 5 buses.

    WHEN    adding multiple components of Generator and Load with madd

    THEN    the defaults should be set correctly according to
    n.component_attrs.
    """
    gen_names = ["g_1", "g_2"]
    empty_network_5_buses.madd(
        "Generator",
        gen_names,
        bus=["bus_1", "bus_2"],
    )

    line_names = ["l_1", "l_2"]
    empty_network_5_buses.madd(
        "Load",
        line_names,
        bus=["bus_1", "bus_2"],
    )

    assert (
        empty_network_5_buses.generators.loc[gen_names[0], "control"]
        == empty_network_5_buses.component_attrs.Generator.loc["control", "default"]
    )

    assert (
        empty_network_5_buses.loads.loc[line_names[0], "p_set"]
        == empty_network_5_buses.component_attrs.Load.loc["p_set", "default"]
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
    for network in all_networks:
        network_copy = network.copy()
        assert network == network_copy
        assert network is not network_copy


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
    for network in all_networks:
        copied_network = network.copy(snapshots=[])
        assert copied_network.snapshots.size == 1

        copied_network = network.copy(snapshots=network.snapshots[:5])
        network.set_snapshots(network.snapshots[:5])
        assert copied_network == network


def test_add_network_static(ac_dc_network, empty_network_5_buses):
    """
    GIVEN   the AC DC exemplary pypsa network and an empty PyPSA network with 5
    buses.

    WHEN    the second network is added to the first

    THEN    the first network should now contain its original buses and
    also the buses in the second network
    """

    n = ac_dc_network.merge(empty_network_5_buses, with_time=False)
    new_buses = set(n.buses.index)
    assert new_buses.issuperset(empty_network_5_buses.buses.index)


def test_add_network_with_time(ac_dc_network, empty_network_5_buses):
    """
    GIVEN   the AC DC exemplary pypsa network and an empty PyPSA network with 5
    buses and the same snapshots.

    WHEN    the second network is added to the first

    THEN    the first network should now contain its original buses and
    also the buses in the second network
    """
    with pytest.raises(ValueError):
        ac_dc_network.merge(empty_network_5_buses, with_time=True)

    empty_network_5_buses.set_snapshots(ac_dc_network.snapshots)
    n = ac_dc_network.merge(empty_network_5_buses, with_time=True)
    new_buses = set(n.buses.index)
    assert new_buses.issuperset(empty_network_5_buses.buses.index)


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
