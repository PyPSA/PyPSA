import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.statistics import groupers
from pypsa.statistics.expressions import StatisticsAccessor


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_all_methods(ac_dc_network_r, stat_func):
    df = getattr(ac_dc_network_r.statistics, stat_func)
    assert not df().empty


def test_default_solved(ac_dc_network_r):
    df = ac_dc_network_r.statistics()
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


def test_drop_zero(ac_dc_network):
    n = ac_dc_network
    df = n.statistics.optimal_capacity(drop_zero=True)
    assert df.empty

    df = n.statistics.optimal_capacity()
    assert df.empty

    df = n.statistics.optimal_capacity(drop_zero=False)
    assert not df.empty
    assert np.any(df == 0)


def test_zero_profit_rule_branches(ac_dc_network_r):
    n = ac_dc_network_r
    revenue = n.statistics.revenue(aggregate_time="sum")
    capex = n.statistics.capex()
    comps = ["Line", "Link"]
    assert np.allclose(revenue[comps], capex[comps])


def test_net_and_gross_revenue(ac_dc_network_r):
    n = ac_dc_network_r
    target = n.statistics.revenue(aggregate_time="sum")
    revenue_out = n.statistics.revenue(aggregate_time="sum", direction="output")
    revenue_in = n.statistics.revenue(aggregate_time="sum", direction="input")
    revenue = revenue_in.add(revenue_out, fill_value=0)
    comps = ["Generator", "Line", "Link"]
    assert np.allclose(revenue[comps], target[comps])


def test_supply_withdrawal(ac_dc_network_r):
    n = ac_dc_network_r
    target = n.statistics.energy_balance()
    supply = n.statistics.energy_balance(direction="supply")
    withdrawal = n.statistics.energy_balance(direction="withdrawal")
    energy_balance = supply.sub(withdrawal, fill_value=0)
    assert np.allclose(energy_balance.reindex(target.index), target)


def test_opex():
    n = pypsa.Network()
    n.set_snapshots([0, 1, 2])
    n.snapshot_weightings.loc[:, :] = 2
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[0, 0, 5])
    n.add(
        "Generator",
        "gen",
        bus="bus",
        carrier="gen",
        p_nom=10,
        p_max_pu=[0, 1, 0],
        marginal_cost=2,
        marginal_cost_quadratic=0.2,
    )
    n.add(
        "Store",
        "sto",
        bus="bus",
        carrier="sto",
        e_nom=10,
        e_initial=0,
        marginal_cost_storage=0.5,
    )
    n.add("Bus", "bus2")
    n.add(
        "StorageUnit",
        "su",
        bus="bus2",
        carrier="su",
        marginal_cost=5,
        p_nom=1,
        max_hours=2,
        inflow=1,
        spill_cost=20,
    )

    n.optimize()

    opex = n.statistics.opex()

    assert opex.loc["Store", "sto"] == 2 * 0.5 * 10
    assert opex.loc["Generator", "gen"] == 2 * 2 * 5 + 2 * 0.2 * 5**2
    assert opex.loc["StorageUnit", "su"] == 2 * 20 * 2

    n.generators.marginal_cost_quadratic = 0
    n.generators.committable = True
    n.generators.start_up_cost = 4
    n.generators.shut_down_cost = 5
    n.generators.stand_by_cost = 9

    n.optimize()

    opex = n.statistics.opex()

    assert opex.loc["Store", "sto"] == 2 * 0.5 * 10
    assert opex.loc["Generator", "gen"] == 2 * 2 * 5 + 1 * 4 + 2 * 5 + 2 * 1 * 9

    opex = n.statistics.opex(cost_types="marginal_cost")

    assert opex.loc["Generator", "gen"] == 2 * 2 * 5
    with pytest.raises(KeyError):
        opex.loc["StorageUnit", "su"]
    with pytest.raises(KeyError):
        opex.loc["Store", "sto"]

    opex = n.statistics.opex(cost_types="marginal_cost_storage")

    assert opex.loc["Store", "sto"] == 2 * 0.5 * 10
    with pytest.raises(KeyError):
        opex.loc["Generator", "gen"]


def test_no_grouping(ac_dc_network_r):
    df = ac_dc_network_r.statistics(groupby=False)
    assert not df.empty


def test_no_time_aggregation(ac_dc_network_r):
    df = ac_dc_network_r.statistics.supply(aggregate_time=False)
    assert not df.empty
    assert isinstance(df, pd.DataFrame)


def test_carrier_selection(ac_dc_network_r):
    n = ac_dc_network_r
    df = n.statistics(carrier="AC")
    assert not df.empty
    assert "Line" in df.index.unique(0)
    assert list(df.index.unique(1)) == ["AC"]

    df = n.statistics(carrier=["AC"])
    assert "Line" in df.index.unique(0)
    assert list(df.index.unique(1)) == ["AC"]


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


def test_multiindexed(ac_dc_network_mi):
    n = ac_dc_network_mi
    df = n.statistics()
    assert not df.empty
    assert df.columns.nlevels == 2
    assert df.columns.unique(1)[0] == 2013


def test_multiindexed_aggregate_across_components(ac_dc_network_mi):
    n = ac_dc_network_mi
    df = n.statistics.installed_capacity(
        comps=["Generator", "Line"], aggregate_across_components=True
    )
    assert not df.empty
    assert "component" not in df.index.names


def test_inactive_exclusion_in_static(ac_dc_network_r):
    n = ac_dc_network_r
    df = n.statistics()
    assert "Line" in df.index.unique(0)

    n.lines["active"] = False
    df = n.statistics()
    assert "Line" not in df.index.unique(0)

    n.lines["active"] = True


def test_transmission_carriers(ac_dc_network_r):
    n = ac_dc_network_r
    n.lines["carrier"] = "AC"
    df = pypsa.statistics.get_transmission_carriers(ac_dc_network_r)
    assert "AC" in df.unique(1)


def test_system_cost(ac_dc_network_r):
    n = ac_dc_network_r
    capex = n.statistics.capex().sum()
    opex = n.statistics.opex().sum()
    system_cost = n.statistics.system_cost().sum()
    assert system_cost == capex + opex
