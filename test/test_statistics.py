# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

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
    n = ac_dc_network_r
    revenue = n.statistics.revenue(aggregate_time="sum")
    capex = n.statistics.capex()
    comps = ["Line", "Link"]
    assert np.allclose(revenue[comps], capex[comps])


def test_net_and_gross_revenue(ac_dc_network_r):
    n = ac_dc_network_r
    target = n.statistics.revenue(aggregate_time="sum")
    revenue_out = n.statistics.revenue(aggregate_time="sum", kind="output")
    revenue_in = n.statistics.revenue(aggregate_time="sum", kind="input")
    revenue = revenue_in.add(revenue_out, fill_value=0)
    comps = ["Generator", "Line", "Link"]
    assert np.allclose(revenue[comps], target[comps])


def test_supply_withdrawal(ac_dc_network_r):
    n = ac_dc_network_r
    target = n.statistics.energy_balance()
    supply = n.statistics.energy_balance(kind="supply")
    withdrawal = n.statistics.energy_balance(kind="withdrawal")
    energy_balance = supply.sub(withdrawal, fill_value=0)
    assert np.allclose(energy_balance.reindex(target.index), target)


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


def test_storage_capacity(ac_dc_network_r):
    n = ac_dc_network_r
    df = n.statistics.installed_capacity(storage=True)
    assert df.empty

    df = n.statistics.optimal_capacity(storage=True)
    assert df.empty

    n.add("Store", "example", carrier="any", bus="Manchester", e_nom=10, e_nom_opt=5)
    df = n.statistics.installed_capacity(storage=True)
    assert not df.empty
    assert df.sum() == 10

    df = n.statistics.optimal_capacity(storage=True)
    assert not df.empty
    assert df.sum() == 5


def test_single_component(ac_dc_network_r):
    n = ac_dc_network_r
    df = n.statistics.installed_capacity(comps="Generator")
    assert not df.empty
    assert df.index.nlevels == 1


def test_multiindexed(ac_dc_network_multiindexed):
    n = ac_dc_network_multiindexed
    df = n.statistics()
    assert not df.empty
    assert df.columns.nlevels == 2
    assert df.columns.unique(1)[0] == 2013


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
