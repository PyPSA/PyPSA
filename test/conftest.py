#!/usr/bin/env python3
"""
Created on Mon Jan 31 18:29:48 2022.

@author: fabian
"""

import os

import geopandas as gpd
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import pytest
from shapely.geometry import Polygon

import pypsa


def pytest_addoption(parser):
    parser.addoption(
        "--test-docs",
        action="store_true",
        default=False,
        help="run sphinx build test",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "test_docs: mark test as sphinx build")


@pytest.fixture(scope="module")
def scipy_network():
    """
    scigrid network (module scope)
    """
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "scigrid-de",
        "scigrid-with-load-gen-trafos",
    )
    n = pypsa.Network(csv_folder)
    n.generators.control = "PV"
    g = n.generators[n.generators.bus == "492"]
    n.generators.loc[g.index, "control"] = "PQ"
    n.calculate_dependent_values()
    n.determine_network_topology()
    return n


@pytest.fixture(scope="function")
def scipy_network_scoped(scipy_network):
    """
    scipy network (function scope)
    """
    return scipy_network


@pytest.fixture(scope="module")
def ac_dc_network():
    """
    ac-dc-meshed network (module scope)
    """
    csv_folder = os.path.join(
        os.path.dirname(__file__), "..", "examples", "ac-dc-meshed", "ac-dc-data"
    )
    n = pypsa.Network(csv_folder)
    n.buses["country"] = ["UK", "UK", "UK", "UK", "DE", "DE", "DE", "NO", "NO"]
    n.links_t.p_set.drop(columns=n.links_t.p_set.columns, inplace=True)
    return n


@pytest.fixture(scope="function")
def ac_dc_network_scoped(ac_dc_network):
    """
    ac-dc-meshed network (function scope)
    """
    return ac_dc_network


@pytest.fixture(scope="module")
def ac_dc_network_r():
    """
    ac-dc-meshed network with results (module scope)
    """
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data",
        "results-lopf",
    )
    n = pypsa.Network(csv_folder)
    n.buses["country"] = ["UK", "UK", "UK", "UK", "DE", "DE", "DE", "NO", "NO"]
    n.links_t.p_set.drop(columns=n.links_t.p_set.columns, inplace=True)
    return n


@pytest.fixture(scope="function")
def ac_dc_network_r_scoped(ac_dc_network_r):
    """
    ac-dc-meshed network with results (function scope)
    """
    return ac_dc_network_r


@pytest.fixture(scope="module")
def ac_dc_network_mi(ac_dc_network):
    """
    ac-dc-meshed network with investment periods (module scope)
    """

    n = ac_dc_network
    n.snapshots = pd.MultiIndex.from_product([[2013], n.snapshots])
    n.investment_periods = [2013]
    gens_i = n.generators.index
    rng = np.random.default_rng()  # Create a random number generator
    n.generators_t.p[gens_i] = rng.random(size=(len(n.snapshots), len(gens_i)))
    return n


@pytest.fixture(scope="function")
def ac_dc_network_mi_scoped(ac_dc_network_mi):
    """
    ac-dc-meshed network with investment periods (function scope)
    """
    return ac_dc_network_mi


@pytest.fixture(scope="module")
def ac_dc_network_shapes(ac_dc_network):
    """
    ac-dc-meshed network with shapes (module scope)
    """
    n = ac_dc_network

    # Create bounding boxes around points
    def create_bbox(x, y, delta=0.1):
        return Polygon(
            [
                (x - delta, y - delta),
                (x - delta, y + delta),
                (x + delta, y + delta),
                (x + delta, y - delta),
            ]
        )

    bboxes = n.buses.apply(lambda row: create_bbox(row["x"], row["y"]), axis=1)

    # Convert to GeoSeries
    geo_series = gpd.GeoSeries(bboxes, crs="epsg:4326")

    n.add(
        "Shape",
        name=geo_series.index,
        geometry=geo_series,
        idx=geo_series.index,
        component="Bus",
    )

    return n


@pytest.fixture(scope="module")
def storage_hvdc_network():
    """
    storage-hvdc network (module scope)
    """
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "opf-storage-hvdc",
        "opf-storage-data",
    )
    return pypsa.Network(csv_folder)


@pytest.fixture(scope="module")
def pandapower_custom_network():
    """
    pandapower custom network (module scope)
    """
    net = pp.create_empty_network()
    bus1 = pp.create_bus(net, vn_kv=20.0, name="Bus 1")
    bus2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")
    bus3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")
    # create bus elements
    pp.create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=bus3, p_mw=0.100, q_mvar=0.05, name="Load")
    pp.create_shunt(net, bus=bus3, p_mw=0.0, q_mvar=0.0, name="Shunt")
    # create branch elements
    pp.create_transformer(
        net, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV", name="Trafo"
    )
    pp.create_line(
        net,
        from_bus=bus2,
        to_bus=bus3,
        length_km=0.1,
        std_type="NAYY 4x50 SE",
        name="Line",
    )
    return net


@pytest.fixture(scope="module")
def pandapower_cigre_network():
    """
    pandapower cigre network (module scope)
    """
    return pn.create_cigre_network_mv(with_der="all")


@pytest.fixture(
    scope="module",
    params=[
        "ac_dc_network_scoped",
        "ac_dc_network_r_scoped",
        "ac_dc_network_mi_scoped",
    ],
)
def all_networks(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "ac_dc_network_scoped",
        "ac_dc_network_r_scoped",
        "ac_dc_network_mi_scoped",
    ]
)
def all_networks_scoped(request):
    return request.getfixturevalue(request.param)
