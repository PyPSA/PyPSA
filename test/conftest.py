#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:29:48 2022.

@author: fabian
"""

import os

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import pytest

import pypsa

SUPPORTED_APIS = ["pyomo", "linopy", "native"]
SOLVER_NAME = "glpk"


def optimize(n, api, *args, **kwargs):
    if api == "linopy":
        return n.optimize(solver_name=SOLVER_NAME, *args, **kwargs)
    elif api == "pyomo":
        return n.lopf(pyomo=True, solver_name=SOLVER_NAME, *args, **kwargs)
    elif api == "native":
        return n.lopf(pyomo=False, solver_name=SOLVER_NAME, *args, **kwargs)
    else:
        raise ValueError(f"api must be one of {SUPPORTED_APIS}")


@pytest.fixture(scope="function")
def scipy_network():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "scigrid-de",
        "scigrid-with-load-gen-trafos",
    )

    return pypsa.Network(csv_folder)


@pytest.fixture(scope="module")
def ac_dc_network():
    csv_folder = os.path.join(
        os.path.dirname(__file__), "..", "examples", "ac-dc-meshed", "ac-dc-data"
    )
    n = pypsa.Network(csv_folder)
    n.links_t.p_set.drop(columns=n.links_t.p_set.columns, inplace=True)
    return n


@pytest.fixture(scope="module")
def ac_dc_network_multiindexed(ac_dc_network):
    n = ac_dc_network
    n.snapshots = pd.MultiIndex.from_product([[2013], n.snapshots])
    gens_i = n.generators.index
    n.generators_t.p[gens_i] = np.random.rand(len(n.snapshots), len(gens_i))
    return n


@pytest.fixture(scope="module")
def storage_hvdc_network():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "opf-storage-hvdc",
        "opf-storage-data",
    )
    n = pypsa.Network(csv_folder)
    return n


@pytest.fixture(scope="module")
def pandapower_custom_network():
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
    net = pn.create_cigre_network_mv(with_der="all")
    return net
