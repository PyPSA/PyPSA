# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.statistics import groupers
from pypsa.statistics.expressions import StatisticsAccessor, get_operation


def test_stats_alias(ac_dc_network):
    """Test that n.stats works as an alias for n.statistics."""
    n = ac_dc_network
    assert n.stats is n.statistics
    df_stats = n.stats()
    df_statistics = n.statistics()
    pd.testing.assert_frame_equal(df_stats, df_statistics)
    stats_installed_capacity = n.stats.installed_capacity()
    statistics_installed_capacity = n.statistics.installed_capacity()
    pd.testing.assert_series_equal(
        stats_installed_capacity, statistics_installed_capacity
    )


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
    df = ac_dc_network_r.statistics(groupby=["bus0", "carrier"], components={"Link"})
    assert not df.empty


def test_grouping_by_new_registered_key(ac_dc_network_r):
    def new_grouper(n, c):
        return n.c[c].static.index.to_series()

    n = ac_dc_network_r
    pypsa.statistics.groupers.add_grouper("new_grouper", new_grouper)
    df = n.statistics.supply(groupby="new_grouper")
    assert not df.empty
    assert df.index.nlevels == 2

    df = n.statistics.supply(groupby=["new_grouper", "carrier"], components="Link")
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
    revenue = n.statistics.revenue(groupby_time="sum")
    capex = n.statistics.capex()
    comps = ["Line", "Link"]
    assert np.allclose(revenue[comps], capex[comps])


def test_net_and_gross_revenue(ac_dc_network_r):
    n = ac_dc_network_r
    target = n.statistics.revenue(groupby_time="sum")
    revenue_out = n.statistics.revenue(groupby_time="sum", direction="output")
    revenue_in = n.statistics.revenue(groupby_time="sum", direction="input")
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

    n.c.generators.static.marginal_cost_quadratic = 0
    n.c.generators.static.committable = True
    n.c.generators.static.start_up_cost = 4
    n.c.generators.static.shut_down_cost = 5
    n.c.generators.static.stand_by_cost = 9

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
    df = ac_dc_network_r.statistics.supply(groupby_time=False)
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


@pytest.mark.parametrize(
    "stat_func",
    ["capex", "optimal_capacity", "installed_capacity"],
)
def test_bus_carrier_searches_all_ports(ac_dc_network_r, stat_func):
    n = ac_dc_network_r
    result = getattr(n.statistics, stat_func)(bus_carrier="DC")
    components = result.index.get_level_values("component")
    assert "Link" in components


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
    df = n.statistics.installed_capacity(components="Generator")
    assert not df.empty
    assert df.index.nlevels == 1


def test_aggregate_across_components(ac_dc_network_r):
    import warnings

    n = ac_dc_network_r
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        df = n.statistics.installed_capacity(
            components=["Generator", "Line"], aggregate_across_components=True
        )
        assert not df.empty
        assert "component" not in df.index.names

        df = n.statistics.supply(
            components=["Generator", "Line"],
            aggregate_across_components=True,
            groupby_time=False,
        )
        assert not df.empty
    assert "component" not in df.index.names


def test_multiindexed(ac_dc_periods):
    n = ac_dc_periods
    df = n.statistics()
    assert not df.empty
    assert df.columns.nlevels == 2
    assert df.columns.unique(1)[0] == 2013


def test_multiindexed_aggregate_across_components(ac_dc_periods):
    import warnings

    n = ac_dc_periods
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        df = n.statistics.installed_capacity(
            components=["Generator", "Line"], aggregate_across_components=True
        )
        assert not df.empty
        assert "component" not in df.index.names


def test_inactive_exclusion_in_static(ac_dc_network_r):
    n = ac_dc_network_r
    df = n.statistics()
    assert "Line" in df.index.unique(0)

    n.c.lines.static["active"] = False
    df = n.statistics()
    assert "Line" not in df.index.unique(0)

    n.c.lines.static["active"] = True


def test_transmission_carriers(ac_dc_network_r):
    n = ac_dc_network_r
    n.c.lines.static["carrier"] = "AC"
    df = pypsa.statistics.get_transmission_carriers(ac_dc_network_r)
    assert "AC" in df.unique(1)


def test_system_cost(ac_dc_network_r):
    n = ac_dc_network_r
    capex = n.statistics.capex().sum()
    opex = n.statistics.opex().sum()
    system_cost = n.statistics.system_cost().sum()
    assert system_cost == capex + opex


def test_prices(ac_dc_network_r):
    n = ac_dc_network_r

    # Test basic prices (load-weighted by default)
    prices = n.statistics.prices()
    assert isinstance(prices, pd.Series)
    assert len(prices) == len(n.buses)

    time_weighted = n.statistics.prices(weighting="time")
    load_weighted = n.statistics.prices(weighting="load")
    assert not time_weighted.equals(load_weighted)

    # Test bus carrier filtering
    ac_prices = n.statistics.prices(bus_carrier="AC")
    assert len(ac_prices) == sum(n.c.buses.static.carrier == "AC")

    # Test groupby bus_carrier
    grouped = n.statistics.prices(groupby="bus_carrier")
    assert set(grouped.index) == set(n.c.buses.static.carrier.unique())


@pytest.fixture
def network_with_nice_name():
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Carrier", "rural heat", nice_name="residential rural heat")
    n.add("Bus", "heat bus", carrier="rural heat", unit="MW")
    n.add(
        "Load",
        "heat load",
        bus="heat bus",
        carrier="rural heat",
        p_set=[1.0],
    )
    n.c.loads.dynamic.p = n.c.loads.dynamic.p_set.copy()
    return n


def test_energy_balance_bus_carrier_filter():
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Carrier", "rural heat")
    n.add("Bus", "heat bus", carrier="rural heat", unit="MW")
    n.add(
        "Load",
        "heat load",
        bus="heat bus",
        carrier="rural heat",
        p_set=[1.0],
    )
    n.c.loads.dynamic.p = n.c.loads.dynamic.p_set.copy()

    result = n.statistics.energy_balance(bus_carrier="rural heat")
    assert not result.empty
    assert "bus_carrier" in result.index.names
    assert "rural heat" in result.index.get_level_values("bus_carrier")


def test_energy_balance_bus_carrier_nice_name_filter(network_with_nice_name):
    n = network_with_nice_name

    displayed = n.statistics.energy_balance(nice_names=True)
    assert "residential rural heat" in displayed.index.get_level_values("bus_carrier")

    result = n.statistics.energy_balance(
        bus_carrier="residential rural heat", nice_names=True
    )
    assert not result.empty
    assert "residential rural heat" in result.index.get_level_values("bus_carrier")


def test_energy_balance_carrier_nice_name_filter(network_with_nice_name):
    n = network_with_nice_name

    result = n.statistics.energy_balance(
        carrier="residential rural heat", nice_names=True
    )
    assert not result.empty

    result = n.statistics.energy_balance(
        carrier="residential rural heat", nice_names=False
    )
    assert result.empty


@pytest.fixture(scope="module")
def multi_invest_network():
    n = pypsa.Network(snapshots=range(10))
    n.investment_periods = [2020, 2030, 2040]
    n.investment_period_weightings.loc[2020, "years"] = 10
    n.investment_period_weightings.loc[2020, "objective"] = 10
    n.investment_period_weightings.loc[2030, "years"] = 15
    n.investment_period_weightings.loc[2030, "objective"] = 15
    n.investment_period_weightings.loc[2040, "years"] = 5
    n.investment_period_weightings.loc[2040, "objective"] = 5
    n.add("Carrier", "wind")
    n.add("Carrier", "gas")
    n.add("Bus", "elec")
    n.add(
        "Generator",
        "wind-2020",
        bus="elec",
        carrier="wind",
        capital_cost=100,
        marginal_cost=0,
        p_nom_extendable=True,
        build_year=2020,
        lifetime=30,
    )
    n.add(
        "Generator",
        "gas-2020",
        bus="elec",
        carrier="gas",
        capital_cost=50,
        marginal_cost=30,
        p_nom_extendable=True,
        p_nom_min=10,
        build_year=2020,
        lifetime=20,
    )
    n.add(
        "Generator",
        "gas-2040",
        bus="elec",
        carrier="gas",
        capital_cost=40,
        marginal_cost=25,
        p_nom_extendable=True,
        build_year=2040,
        lifetime=20,
    )
    load = pd.DataFrame({"load": range(100, 100 + len(n.snapshots))}, index=n.snapshots)
    n.add("Load", "load", bus="elec", p_set=load["load"])
    n.optimize(solver_name="highs", multi_investment_periods=True)
    return n


class TestMultiInvest:
    @pytest.fixture(scope="class")
    def capex(self, multi_invest_network):
        return multi_invest_network.statistics.capex()

    @pytest.fixture(scope="class")
    def capex_ungrouped(self, multi_invest_network):
        return multi_invest_network.statistics.capex(groupby=False)

    @pytest.fixture(scope="class")
    def opex(self, multi_invest_network):
        return multi_invest_network.statistics.opex()

    @pytest.fixture(scope="class")
    def opex_ungrouped(self, multi_invest_network):
        return multi_invest_network.statistics.opex(groupby=False)

    @pytest.fixture(scope="class")
    def system_cost(self, multi_invest_network):
        return multi_invest_network.statistics.system_cost()

    def test_capex_structure(self, capex):
        assert isinstance(capex, pd.DataFrame)
        assert list(capex.columns) == [2020, 2030, 2040]
        assert (capex.fillna(0) >= 0).all().all()

    def test_capex_weighted(self, multi_invest_network, capex):
        weights = multi_invest_network.investment_period_weightings["objective"]
        annualized_2020 = capex[2020].dropna() / weights[2020]
        annualized_2030 = capex[2030].dropna() / weights[2030]
        pd.testing.assert_series_equal(
            annualized_2020, annualized_2030, rtol=1e-3, check_names=False
        )

    def test_capex_active_assets_filtering(self, capex_ungrouped):
        assert pd.isna(capex_ungrouped.loc[("Generator", "gas-2040"), 2020])
        assert capex_ungrouped.loc[("Generator", "gas-2040"), 2040] > 0.0

    def test_opex_structure(self, opex):
        assert isinstance(opex, pd.DataFrame)
        assert list(opex.columns) == [2020, 2030, 2040]
        assert not opex.empty
        assert (opex.fillna(0) >= 0).all().all()

    def test_opex_matches_marginal_cost(self, multi_invest_network, opex_ungrouped):
        n = multi_invest_network
        pw = n.investment_period_weightings["objective"]
        sw = n.snapshot_weightings.objective
        for period in n.investment_periods:
            active = n.c.generators.get_active_assets(period)
            mc = n.c.generators.static.loc[active, "marginal_cost"]
            mc = mc[mc > 0]
            p = n.c.generators.dynamic.p.loc[period, mc.index]
            expected = (p * mc).multiply(sw.loc[period], axis=0).sum() * pw[period]
            actual = (
                opex_ungrouped.loc["Generator"]
                .reindex(expected.index)[period]
                .fillna(0)
            )
            pd.testing.assert_series_equal(
                actual, expected, rtol=1e-3, atol=1e-6, check_names=False
            )

    def test_system_cost_equals_capex_plus_opex(self, capex, opex, system_cost):
        expected = capex.add(opex, fill_value=0).reindex(system_cost.index)
        pd.testing.assert_frame_equal(expected, system_cost, atol=1e-3)

    def test_system_cost_consistent_with_objective(
        self, multi_invest_network, system_cost
    ):
        assert np.nansum(system_cost.values) == pytest.approx(
            multi_invest_network.objective, rel=1e-3
        )


class TestGetOperation:
    @pytest.mark.parametrize(
        ("component", "expected_attr"),
        [("Link", "p0"), ("Line", "p0"), ("Generator", "p"), ("Store", "e")],
    )
    def test_get_operation(self, ac_dc_network_r, component, expected_attr):
        n = ac_dc_network_r
        pd.testing.assert_frame_equal(
            get_operation(n, component), n.components[component].dynamic[expected_attr]
        )

    def test_get_operation_multi_port(self, multiport_process_network):
        pd.testing.assert_frame_equal(
            get_operation(multiport_process_network, "Process"),
            multiport_process_network.c.processes.dynamic["p"],
        )


class TestMarketValue:
    mv_kwargs = {"nice_names": False, "round": None, "drop_zero": False}
    rtol = 1e-6

    def test_multiport(self, multiport_process_network):
        n = multiport_process_network
        mv = n.statistics.market_value(**self.mv_kwargs)

        operation = n.c.processes.dynamic["p"]["electrolyser"]
        prices = n.c.buses.dynamic["marginal_price"]
        rev_per_t = -(
            n.c.processes.dynamic["p0"]["electrolyser"] * prices["elec"]
            + n.c.processes.dynamic["p1"]["electrolyser"] * prices["h2"]
            + n.c.processes.dynamic["p2"]["electrolyser"] * prices["heat"]
        )
        expected = rev_per_t.mean() / operation.mean()
        np.testing.assert_allclose(
            mv.loc[("Process", "electrolyser")], expected, rtol=self.rtol
        )

    @pytest.mark.parametrize(
        ("bus_carrier", "port", "bus"),
        [
            ("AC", "p0", "elec"),
            ("H2", "p1", "h2"),
            ("heat", "p2", "heat"),
        ],
    )
    def test_with_bus_carrier(self, multiport_process_network, bus_carrier, port, bus):
        n = multiport_process_network
        mv = n.statistics.market_value(bus_carrier=bus_carrier, **self.mv_kwargs)

        reference_operation = n.c.processes.dynamic["p"]["electrolyser"]
        operation = n.c.processes.dynamic[port]["electrolyser"]
        prices = n.c.buses.dynamic["marginal_price"][bus]
        expected = -(operation * prices).mean() / reference_operation.mean()
        np.testing.assert_allclose(
            mv.loc[("Process", "electrolyser")], expected, rtol=self.rtol
        )

    def test_bus_carrier_additivity(self, multiport_process_network):
        n = multiport_process_network
        mv_kw = {**self.mv_kwargs, "groupby": False}
        mv_total = n.statistics.market_value(**mv_kw)
        mv_ac = n.statistics.market_value(bus_carrier="AC", **mv_kw)
        mv_h2 = n.statistics.market_value(bus_carrier="H2", **mv_kw)
        mv_heat = n.statistics.market_value(bus_carrier="heat", **mv_kw)

        expected = (
            mv_ac.loc[("Process", "electrolyser")]
            + mv_h2.loc[("Process", "electrolyser")]
            + mv_heat.loc[("Process", "electrolyser")]
        )
        np.testing.assert_allclose(
            mv_total.loc[("Process", "electrolyser")], expected, rtol=self.rtol
        )

    def test_generator(self, multiport_process_network):
        n = multiport_process_network
        mv = n.statistics.market_value(components="Generator", **self.mv_kwargs)
        operation = n.c.generators.dynamic["p"]["gen"]
        prices = n.c.buses.dynamic["marginal_price"]["elec"]
        expected = (operation * prices).mean() / operation.mean()
        np.testing.assert_allclose(mv.loc["AC"], expected, rtol=self.rtol)

    def test_withdrawing_load_sign(self):
        n = pypsa.Network()
        n.set_snapshots([0, 1])
        n.add("Carrier", "AC")
        n.add("Carrier", "load")
        n.add("Bus", "b", carrier="AC")
        n.add("Load", "l", bus="b", carrier="load", p_set=[10.0, 20.0])
        n.c.loads.dynamic["p"] = n.c.loads.dynamic["p_set"].copy()

        marginal_price = pd.DataFrame({"b": [50.0, 100.0]}, index=n.snapshots)
        marginal_price.columns.name = "name"
        n.c.buses.dynamic["marginal_price"] = marginal_price

        mv = n.statistics.market_value(components="Load", **self.mv_kwargs)

        p_load = np.array([10.0, 20.0])
        price_load = np.array([50.0, 100.0])
        expected = -(p_load * price_load).mean() / p_load.mean()
        np.testing.assert_allclose(mv.loc["load"], expected, rtol=self.rtol)
        assert mv.loc["load"] < 0

    def test_grouped_branches_is_finite(self, ac_dc_network_r):
        n = ac_dc_network_r
        mv = n.statistics.market_value(**self.mv_kwargs)

        assert np.isfinite(mv.loc[("Link", "DC")])
        assert np.isfinite(mv.loc[("Line", "AC")])
