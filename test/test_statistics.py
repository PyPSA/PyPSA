# -*- coding: utf-8 -*-
import os

import numpy as np
import pytest

import pypsa
from pypsa.statistics import get_bus_and_carrier, get_country_and_carrier


@pytest.fixture
def ac_dc_network_r():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data",
        "results-lopf",
    )
    return pypsa.Network(csv_folder)


def test_default_unsolved(ac_dc_network):
    df = ac_dc_network.statistics()
    assert not df.empty


def test_default_solved(ac_dc_network_r):
    df = ac_dc_network_r.statistics()
    assert not df.empty

    df = ac_dc_network_r.statistics.capex()
    assert not df.empty

    df = ac_dc_network_r.statistics.opex()
    assert not df.empty

    df = ac_dc_network_r.statistics.energy_balance()
    assert not df.empty
    assert (
        round(
            df.groupby(level="bus_carrier").sum().sum()
            / df.where(lambda x: x > 0).groupby(level="bus_carrier").sum().sum(),
            3,
        )
        == 0
    )


def test_per_bus_carrier_unsolved(ac_dc_network):
    df = ac_dc_network.statistics(groupby=get_bus_and_carrier)
    assert not df.empty


def test_per_country_carrier_unsolved(ac_dc_network):
    n = ac_dc_network
    n.buses["country"] = ["UK", "UK", "UK", "UK", "DE", "DE", "DE", "NO", "NO"]
    df = n.statistics(groupby=get_country_and_carrier)
    assert not df.empty


def test_per_bus_carrier_solved(ac_dc_network_r):
    df = ac_dc_network_r.statistics(groupby=get_bus_and_carrier)
    assert not df.empty


def test_column_grouping_unsolved(ac_dc_network):
    df = ac_dc_network.statistics(groupby=["bus0", "carrier"], comps={"Link"})
    assert not df.empty


def test_column_grouping_solved(ac_dc_network_r):
    df = ac_dc_network_r.statistics(groupby=["bus0", "carrier"], comps={"Link"})
    assert not df.empty


def test_zero_profit_rule_branches(ac_dc_network_r):
    df = ac_dc_network_r.statistics(aggregate_time="sum")
    df = df.loc[["Line", "Link"]]
    assert np.allclose(df["Revenue"], df["Capital Expenditure"])
