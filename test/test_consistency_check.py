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
def consistent_n():
    n = pypsa.Network()
    n.add("Bus", "one")
    n.add("Bus", "two")
    n.add("Generator", "gen_one", bus="one", p_nom_max=10)
    n.add("Line", "line_one", bus0="one", bus1="two", x=0.01, r=0.01)
    n.add("Carrier", "AC")
    return n


@pytest.mark.skipif(os.name == "nt", reason="dtype confusing on Windows")
def test_consistency(consistent_n, caplog):
    consistent_n.consistency_check()
    assert not caplog.records


def test_missing_bus(consistent_n, caplog):
    consistent_n.add("Bus", "three")
    consistent_n.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"


def test_infeasible_capacity_limits(consistent_n, caplog):
    consistent_n.generators.loc["gen_one", ["p_nom_extendable", "committable"]] = (
        True,
        True,
    )
    consistent_n.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"


def test_nans_in_capacity_limits(consistent_n, caplog):
    consistent_n.generators.loc["gen_one", "p_nom_extendable"] = True
    consistent_n.generators.loc["gen_one", "p_nom_max"] = np.nan
    consistent_n.consistency_check()
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


def test_unknown_carriers(consistent_n, caplog):
    consistent_n.add("Generator", "wind", bus="hub", carrier="wind")
    consistent_n.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"


def test_inconsistent_e_sum_values(consistent_n, caplog):
    """
    Test that the consistency check raises a warning if the e_sum_min is greater than e_sum_max.
    """
    consistent_n.add(
        "Generator", "gen_two", bus="one", p_nom_max=10, e_sum_min=10, e_sum_max=5
    )
    consistent_n.consistency_check()
    assert caplog.records[-1].levelname == "WARNING"
