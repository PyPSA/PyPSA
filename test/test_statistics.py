import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.statistics import groupers


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


@pytest.mark.parametrize(
    "groupby",
    [
        "carrier",
        groupers.carrier,
        ["bus_carrier", "carrier"],
        [groupers.bus_carrier, groupers.carrier],
    ],
)
def test_grouping_by_keys_unsolved(ac_dc_network, groupby):
    df = ac_dc_network.statistics(groupby=groupby)
    assert not df.empty


@pytest.mark.parametrize(
    "groupby",
    [
        "carrier",
        groupers.carrier,
        ["bus_carrier", "carrier"],
        [groupers.bus_carrier, groupers.carrier],
        ["bus", "carrier"],
        ["country", "carrier"],
    ],
)
def test_grouping_by_keys_solved(ac_dc_network_r, groupby):
    df = ac_dc_network_r.statistics(groupby=groupby)
    assert not df.empty


def test_grouping_by_keys_with_specific_column_solved(ac_dc_network_r):
    df = ac_dc_network_r.statistics(groupby=["bus0", "carrier"], comps={"Link"})
    assert not df.empty


def test_grouping_by_new_registered_key(ac_dc_network_r):
    def new_grouper(n, c):
        return n.df(c).index.to_series()

    n = ac_dc_network_r
    pypsa.statistics.groupers.add_grouper("new_grouper", new_grouper)
    df = n.statistics.supply(groupby="new_grouper")
    assert not df.empty
    assert df.index.nlevels == 2

    df = n.statistics.supply(groupby=["new_grouper", "carrier"], comps="Link")
    assert not df.empty
    assert df.index.nlevels == 2


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


def test_no_time_aggregation(ac_dc_network_r):
    df = ac_dc_network_r.statistics.supply(aggregate_time=False)
    assert not df.empty
    assert isinstance(df, pd.DataFrame)


def test_bus_carrier_selection(ac_dc_network_r):
    df = ac_dc_network_r.statistics(groupby=False, bus_carrier="AC")
    assert not df.empty


def test_bus_carrier_selection_with_list(ac_dc_network_r):
    df = ac_dc_network_r.statistics(
        groupby=groupers["bus", "carrier"], bus_carrier=["AC", "DC"]
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


def test_aggregate_across_components(ac_dc_network_r):
    n = ac_dc_network_r
    df = n.statistics.installed_capacity(
        comps=["Generator", "Line"], aggregate_across_components=True
    )
    assert not df.empty
    assert "component" not in df.index.names

    df = n.statistics.supply(
        comps=["Generator", "Line"],
        aggregate_across_components=True,
        aggregate_time=False,
    )
    assert not df.empty
    assert "component" not in df.index.names


def test_multiindexed(ac_dc_network_multiindexed):
    n = ac_dc_network_multiindexed
    df = n.statistics()
    assert not df.empty
    assert df.columns.nlevels == 2
    assert df.columns.unique(1)[0] == 2013


def test_multiindexed_aggregate_across_components(ac_dc_network_multiindexed):
    n = ac_dc_network_multiindexed
    df = n.statistics.installed_capacity(
        comps=["Generator", "Line"], aggregate_across_components=True
    )
    assert not df.empty
    assert "component" not in df.index.names


def test_inactive_exclusion_in_static(ac_dc_network_r):
    n = ac_dc_network_r
    df = n.statistics()
    assert "Line" in df.index.unique(0)

    df = n.statistics(aggregate_time=False)
    assert "Line" in df.index.unique(0)

    n.lines["active"] = False
    df = n.statistics()
    assert "Line" not in df.index.unique(0)

    df = n.statistics(aggregate_time=False)
    assert "Line" not in df.index.unique(0)

    n.lines["active"] = True


def test_transmission_carriers(ac_dc_network_r):
    n = ac_dc_network_r
    n.lines["carrier"] = "AC"
    df = pypsa.statistics.get_transmission_carriers(ac_dc_network_r)
    assert "AC" in df.unique(1)


def test_groupers(ac_dc_network_r):
    n = ac_dc_network_r
    c = "Generator"

    grouper = groupers.carrier(n, c)
    assert isinstance(grouper, pd.Series)

    grouper = groupers.bus_carrier(n, c)
    assert isinstance(grouper, pd.Series)

    grouper = groupers["bus", "carrier"](n, c)
    assert isinstance(grouper, list)
    assert all(isinstance(ds, pd.Series) for ds in grouper)

    grouper = groupers["bus", "carrier"](n, c)
    assert isinstance(grouper, list)
    assert all(isinstance(ds, pd.Series) for ds in grouper)

    grouper = groupers["country", "carrier"](n, c)
    assert isinstance(grouper, list)
    assert all(isinstance(ds, pd.Series) for ds in grouper)

    grouper = groupers["carrier", "bus_carrier"](n, c)
    assert isinstance(grouper, list)
    assert all(isinstance(ds, pd.Series) for ds in grouper)

    grouper = groupers["bus", "carrier", "bus_carrier"](n, c)
    assert isinstance(grouper, list)
    assert all(isinstance(ds, pd.Series) for ds in grouper)


def test_parameters(ac_dc_network_r):
    n = ac_dc_network_r
    target = n.statistics.capex(nice_names=False).round(2)
    n.statistics.set_parameters(nice_names=False, round=2)
    df = n.statistics.capex()
    assert np.allclose(df, target)
    with pytest.raises(ValueError):
        # Test setting not existing parameters
        n.statistics.set_parameters(groupby=False)
        # Test setting wrong dtype of parameter
        n.statistics.set_parameters(round="one")
    # Test parameter representation
    isinstance(n.statistics.parameters, str)
