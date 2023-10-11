# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.statistics import get_bus_and_carrier, get_country_and_carrier


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


def test_no_grouping(ac_dc_network_r):
    df = ac_dc_network_r.statistics(groupby=False)
    assert not df.empty


def test_bus_carrier_selection(ac_dc_network_r):
    df = ac_dc_network_r.statistics(groupby=False, bus_carrier="AC")
    assert not df.empty


def test_bus_carrier_selection_with_list(ac_dc_network_r):
    df = ac_dc_network_r.statistics(
        groupby=get_bus_and_carrier, bus_carrier=["AC", "DC"]
    )
    assert not df.empty


def test_transmission_carriers(ac_dc_network_r):
    n = ac_dc_network_r
    n.lines["carrier"] = "AC"
    df = pypsa.statistics.get_transmission_carriers(ac_dc_network_r)
    assert "AC" in df.unique(1)


def test_groupers(ac_dc_network_r):
    n = ac_dc_network_r
    c = "Generator"

    grouper = n.statistics.groupers.get_carrier(n, c)
    assert isinstance(grouper, pd.Series)

    grouper = n.statistics.groupers.get_bus_and_carrier(n, c)
    assert isinstance(grouper, list)
    assert all(isinstance(ds, pd.Series) for ds in grouper)

    grouper = n.statistics.groupers.get_name_bus_and_carrier(n, c)
    assert isinstance(grouper, list)
    assert all(isinstance(ds, pd.Series) for ds in grouper)

    grouper = n.statistics.groupers.get_country_and_carrier(n, c)
    assert isinstance(grouper, list)
    assert all(isinstance(ds, pd.Series) for ds in grouper)

    grouper = n.statistics.groupers.get_carrier_and_bus_carrier(n, c)
    assert isinstance(grouper, list)
    assert all(isinstance(ds, pd.Series) for ds in grouper)

    grouper = n.statistics.groupers.get_bus_and_carrier_and_bus_carrier(n, c)
    assert isinstance(grouper, list)
    assert all(isinstance(ds, pd.Series) for ds in grouper)
