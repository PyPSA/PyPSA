#!/usr/bin/env python3
"""
Created on Fri Sep 23 10:37:48 2022.

@author: fabian
"""

import os

import numpy as np
import pytest

import pypsa


@pytest.fixture
def consistent_network():
    n = pypsa.Network()
    n.add("Bus", "one")
    n.add("Bus", "two")
    n.add("Generator", "gen_one", bus="one", p_nom_max=10)
    n.add("Line", "line_one", bus0="one", bus1="two", x=0.01, r=0.01)
    n.add("Carrier", "AC")
    return n


@pytest.mark.skipif(os.name == "nt", reason="dtype confusing on Windows")
def test_consistency(consistent_network, caplog):
    consistent_network.consistency_check()
    assert not caplog.records


def test_missing_bus(consistent_network, caplog):
    consistent_network.add("Bus", "three")
    consistent_network.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"


def test_max_smaller_min(consistent_network, caplog):
    consistent_network.add(
        "Generator",
        "gen_two",
        bus="one",
        p_nom_max=10,
        p_nom_min=20,
        p_nom_extendable=True,
    )
    consistent_network.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"


def test_invalid_generator(consistent_network, caplog):
    consistent_network.add(
        "Generator",
        "gen_two",
        bus="one",
        p_max_pu=np.nan,
        p_min_pu=np.nan,
        p_nom_extendable=True,
    )
    consistent_network.add(
        "Generator",
        "gen_three",
        bus="one",
        p_max_pu=np.inf,
        p_min_pu=np.inf,
        p_nom_extendable=True,
    )
    consistent_network.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"


def test_bad_transformer(consistent_network, caplog):
    consistent_network.add("Transformer", "tranf_one", bus="one", s_nom=0)
    consistent_network.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"


def test_infeasible_capacity_limits(consistent_network, caplog):
    consistent_network.generators.loc[
        "gen_one", ["p_nom_extendable", "committable"]
    ] = (
        True,
        True,
    )
    consistent_network.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"


def test_nans_in_capacity_limits(consistent_network, caplog):
    consistent_network.generators.loc["gen_one", "p_nom_extendable"] = True
    consistent_network.generators.loc["gen_one", "p_nom_max"] = np.nan
    consistent_network.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"


def test_shapes_with_missing_idx(ac_dc_network_shapes, caplog):
    n = ac_dc_network_shapes
    n.add(
        "Shape",
        "missing_idx",
        geometry=n.shapes.geometry.iloc[0],
        component="Bus",
        idx="missing_idx",
    )
    n.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"
    assert caplog.records[-1].message.startswith("The following shapes")


def test_unknown_carriers(consistent_network, caplog):
    consistent_network.add("Generator", "wind", bus="hub", carrier="wind")
    consistent_network.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"
