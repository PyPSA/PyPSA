# -*- coding: utf-8 -*-
import geopandas
import numpy as np
import pytest
import shapely
from shapely.geometry import Point , LineString, Polygon, MultiPoint

import pypsa


@pytest.fixture
def empty_network_5_buses():
    # Set up empty network with 5 buses.
    network = pypsa.Network()
    n_buses = 5
    for i in range(n_buses):
        network.add("Bus", f"bus_{i}")
    return network


def test_geo_components_gpd_df(geo_components_network):
    """
    GIVEN   exemplary pypsa network with geo_components = {"Bus", "Line", "Link", "Transformer"}.

    THEN    these components should be of type geopandas.GeoDataFrame.
    and the geometry column should be of type geopandas.GeoSeries.
    """

    assert isinstance(geo_components_network.buses, geopandas.GeoDataFrame)
    assert isinstance(geo_components_network.lines, geopandas.GeoDataFrame)
    assert isinstance(geo_components_network.links, geopandas.GeoDataFrame)
    assert isinstance(geo_components_network.transformers, geopandas.GeoDataFrame)

    assert isinstance(geo_components_network.buses.geometry, geopandas.GeoSeries)
    assert isinstance(geo_components_network.lines.geometry, geopandas.GeoSeries)
    assert isinstance(geo_components_network.links.geometry, geopandas.GeoSeries)
    assert isinstance(geo_components_network.transformers.geometry, geopandas.GeoSeries)


def test_geo_component_add():
    """
    GIVEN   exemplary pypsa network with geo_components = {"Bus", "Line", "Link", "Transformer"}.

    WHEN    Point(0,1) and 'POINT (1 4)' (wkt_string) should be recognized as   valid and transformed into shapely.geometry objects. 
    Meanwhile, {"None", "", np.nan, "nan"} and any invalid strings should be added as None in the geometry column.
    note - invalid strings will also be added as None.

    THEN    Geometry should be added as shapely.geometry objects or None.
    """
    network = pypsa.Network()

    # {"None", "", np.nan, "nan"}
    network.add("Bus", "bus_1", geometry="")
    assert network.buses.geometry.values[0] == None

    network.add("Bus", "bus_2", geometry=Point(0, 1))
    assert isinstance(network.buses.geometry.values[1], shapely.geometry.point.Point)

    network.add("Bus", "bus_3", geometry="POINT (1 4)")
    assert isinstance(network.buses.geometry.values[2], shapely.geometry.point.Point)

    network.add("Line", "line_1",bus0="bus_1",bus1="bus_2",x=0.1,r=0.01, geometry=LineString([(0,0),(1,1)]))
    assert isinstance(network.lines.geometry.values[0], shapely.geometry.linestring.LineString)

    network.add("Link","link_1",bus0="bus_1",bus1="bus_2", geometry=Polygon([(0,0),(1,1),(1,0)]))
    assert isinstance(network.links.geometry.values[0], shapely.geometry.polygon.Polygon)

    network.add("Transformer","transformer_1",bus0="bus_1",bus1="bus_2", geometry=MultiPoint([(0,0),(1,1)]))
    assert isinstance(network.transformers.geometry.values[0], shapely.geometry.multipoint.MultiPoint)




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
    empty_network_5_buses.madd(
        "Load",
        load_names,
        bus=buses,
        p_set=np.random.rand(len(snapshots), len(buses)),
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

    assert empty_network_5_buses.generators.loc[gen_names[0], "control"] == (
        empty_network_5_buses.component_attrs.Generator.loc["control", "default"]
    )
    assert empty_network_5_buses.loads.loc[line_names[0], "p_set"] == (
        empty_network_5_buses.component_attrs.Load.loc["p_set", "default"]
    )


def test_copy_default_behavior(ac_dc_network):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    copying the network with timestamps

    THEN    the copied network should have the same generators, loads
    and timestamps.
    """
    snapshot = ac_dc_network.snapshots[2]
    copied_network = ac_dc_network.copy()

    loads = ac_dc_network.loads.index.tolist()
    generators = ac_dc_network.generators.index.tolist()
    copied_loads = copied_network.loads.index.tolist()
    copied_generators = copied_network.generators.index.tolist()

    assert loads == copied_loads
    assert generators == copied_generators
    assert not copied_network.snapshots.empty
    assert snapshot in copied_network.snapshots


def test_copy_deep_copy_behavior(ac_dc_network):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    copying the network and changing a component

    THEN    the original network should have not changed.
    """
    copied_network = ac_dc_network.copy()

    copied_network.loads.rename(index={"London": "Berlin"}, inplace=True)

    assert ac_dc_network.loads.index[0] != copied_network.loads.index[0]


def test_copy_no_snapshot(ac_dc_network):
    """
    GIVEN   the AC DC exemplary pypsa network.

    WHEN    copying the network without snapshots

    THEN    the copied network should only have the current time index.
    """
    snapshot = ac_dc_network.snapshots[2]
    copied_network = ac_dc_network.copy(with_time=False, snapshots=snapshot)

    assert copied_network.snapshots.size == 1
    assert snapshot not in copied_network.snapshots
