# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT


import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.statistics import groupers
from pypsa.statistics.expressions import (
    StatisticsAccessor,
    get_operation,
    port_efficiency,
)


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


class TestPortEfficiency:
    """Tests for port_efficiency, including the segment parameter."""

    @pytest.fixture(scope="class")
    def base_network(self):
        idx = pd.Index([0, 1, 2])
        n = pypsa.Network()
        n.set_snapshots(idx)
        n.add("Bus", ["bus0", "bus1", "bus2", "bus3"], carrier="AC")
        n.add("Generator", "gen", bus="bus0", p_nom=100)
        return n

    @pytest.fixture(scope="class", params=["Link", "Process"])
    def component(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def eff_param(self, component):
        if component == "Link":
            return "efficiency"
        if component == "Process":
            return "rate"
        raise ValueError(f"Unexpected component type: {component}")

    @pytest.fixture(scope="class")
    def eff1(self, eff_param, component):
        return f"{eff_param}1" if component == "Process" else eff_param

    @pytest.fixture(scope="class")
    def static_link_eff(self, base_network, eff_param, eff1, component):
        """Minimal network with a multi-port Link and piecewise efficiency."""
        n = base_network.copy()
        n.add(
            component,
            f"{component.lower()}_static",
            bus0="bus0",
            bus1="bus1",
            bus2="bus2",
            **{eff1: 0.8, f"{eff_param}2": 0.5},
        )
        return n

    @pytest.fixture(scope="class")
    def dynamic_link_eff(self, base_network, eff_param, eff1, component):
        """Minimal network with a multi-port Link and piecewise efficiency."""
        n = base_network.copy()
        n.add(
            component,
            f"{component.lower()}_dyn",
            bus0="bus0",
            bus1="bus1",
            bus2="bus2",
            **{
                eff1: pd.Series([0.7, 0.8, 0.9], index=n.snapshots),
                f"{eff_param}2": pd.Series([0.1, 0.2, 0.3], index=n.snapshots),
            },
        )
        return n

    @pytest.fixture(scope="class")
    def segment_link_eff(self, base_network, eff1, component):
        """Network with a Link whose efficiency is defined piecewise per segment."""
        n = base_network.copy()
        n.add(
            component,
            f"{component.lower()}_seg",
            bus0="bus0",
            bus1="bus1",
            p_nom=100,
            **{eff1: {0.0: 0.4, 0.5: 0.5, 1.0: 0.6}},
        )
        return n

    @pytest.fixture(scope="class")
    def mixed_link_eff(self, base_network, eff_param, eff1, component):
        """Network with a Link whose efficiency is defined piecewise per segment."""
        n = base_network.copy()
        n.add(
            component,
            f"{component.lower()}_mix",
            bus0="bus0",
            bus1="bus1",
            bus2="bus2",
            bus3="bus3",
            p_nom=100,
            **{
                eff1: 0.5,
                f"{eff_param}2": {0.0: 0.4, 0.5: 0.5, 1.0: 0.6},
                f"{eff_param}3": pd.Series([0.1, 0.2, 0.3], index=n.snapshots),
            },
        )
        return n

    @pytest.fixture(scope="class")
    def default_dynamic_eff(self, mixed_link_eff, component):
        """Return default dynamic efficiency that is generated when a Link port has no dynamic efficiency defined."""
        dyn_eff = pd.DataFrame(
            1.0,
            index=mixed_link_eff.snapshots,
            columns=pd.Index([f"{component.lower()}_mix"], name="name"),
        )
        return dyn_eff

    # --- one-port components ---

    def test_one_port_returns_ones(self, base_network):
        result = port_efficiency(base_network, "Generator", port=0)
        assert (result == 1).all()

    def test_one_port_segment_flag_ignored(self, base_network):
        """segment=True on a one-port component still returns ones (not an error)."""
        result = port_efficiency(base_network, "Generator", port=0, segment=True)
        assert (result == 1).all()

    # --- passive branches ---

    def test_passive_branch_port0_returns_minus_ones(self, ac_dc_network):
        result = port_efficiency(ac_dc_network, "Line", port=0)
        assert (result == -1).all()

    def test_passive_branch_port1_returns_ones(self, ac_dc_network):
        result = port_efficiency(ac_dc_network, "Line", port=1)
        assert (result == 1).all()

    # --- Link static efficiency ---

    def test_static_link_port0_returns_minus_ones(self, static_link_eff, component):
        result = port_efficiency(static_link_eff, component, port=0)
        assert (result == -1).all()

    def test_static_link_port1_returns_efficiency(self, static_link_eff, component):
        result = port_efficiency(static_link_eff, component, port=1)
        assert result[f"{component.lower()}_static"] == pytest.approx(0.8)

    def test_static_link_port2_returns_efficiency2(self, static_link_eff, component):
        result = port_efficiency(static_link_eff, component, port=2)
        assert result[f"{component.lower()}_static"] == pytest.approx(0.5)

    # --- Link dynamic efficiency ---

    def test_dynamic_link_port1_returns_dataframe(self, dynamic_link_eff, component):
        result = port_efficiency(dynamic_link_eff, component, port=1, dynamic=True)
        assert isinstance(result, pd.DataFrame)

    def test_dynamic_link_port1_values(self, dynamic_link_eff, component, eff1):
        result = port_efficiency(dynamic_link_eff, component, port=1, dynamic=True)
        expected = dynamic_link_eff.components[component].dynamic[eff1]
        pd.testing.assert_frame_equal(result, expected)

    def test_dynamic_link_port2_returns_dataframe(
        self, dynamic_link_eff, component, eff_param
    ):
        result = port_efficiency(dynamic_link_eff, component, port=2, dynamic=True)
        expected = dynamic_link_eff.components[component].dynamic[f"{eff_param}2"]
        pd.testing.assert_frame_equal(result, expected)

    def test_dynamic_link_port2_values(self, dynamic_link_eff, component, eff_param):
        result = port_efficiency(dynamic_link_eff, component, port=2, dynamic=True)
        expected = dynamic_link_eff.components[component].dynamic[f"{eff_param}2"]
        pd.testing.assert_frame_equal(result, expected)

    # --- Link segment efficiency ---

    @pytest.mark.parametrize("port", [1, 2])
    def test_segment_and_dynamic_raises(self, segment_link_eff, component, port):
        with pytest.raises(ValueError, match="segment and dynamic"):
            port_efficiency(
                segment_link_eff, component, port=port, segment=True, dynamic=True
            )

    def test_segment_link_returns_dataframe(self, segment_link_eff, component):
        """When a Link has piecewise efficiency, segment=True returns a DataFrame."""
        result = port_efficiency(segment_link_eff, component, port=1, segment=True)
        assert isinstance(result, pd.DataFrame)

    def test_segment_link_values_within_breakpoints(
        self, segment_link_eff, component, eff1
    ):
        """Segment efficiency values should lie within the piecewise bounds."""
        result = port_efficiency(segment_link_eff, component, port=1, segment=True)
        expected = segment_link_eff.components[component].segments[eff1]
        pd.testing.assert_frame_equal(result, expected)

    def test_segment_link_no_segment_returns_series(
        self, segment_link_eff, component, eff1
    ):
        """Without segment=True, port_efficiency falls back to the static value."""
        result = port_efficiency(segment_link_eff, component, port=1, segment=False)
        expected = pd.Series(
            1.0,
            index=segment_link_eff.components[component].static.index,
            name=eff1,
        )
        pd.testing.assert_series_equal(result, expected)

    # --- mixed efficiency (parametrized over all combinations) ---

    def test_mixed_link_port0_always_minus_one(self, mixed_link_eff, component):
        result = port_efficiency(mixed_link_eff, component, port=0)
        assert (result == -1).all()

    @pytest.mark.parametrize(
        ("segment", "dynamic"), [(True, False), (False, True), (False, False)]
    )
    def test_mixed_link_port1_static(
        self, mixed_link_eff, default_dynamic_eff, segment, dynamic, component
    ):
        result = port_efficiency(
            mixed_link_eff, component, port=1, segment=segment, dynamic=dynamic
        )
        if dynamic:
            pd.testing.assert_frame_equal(result, default_dynamic_eff * 0.5)
        else:
            assert result.item() == 0.5

    def test_mixed_link_port2_segmented_request_static(self, mixed_link_eff, component):
        result = port_efficiency(
            mixed_link_eff, component, port=2, dynamic=False, segment=False
        )
        assert result.item() == 1

    def test_mixed_link_port2_segmented_request_dynamic(
        self, mixed_link_eff, default_dynamic_eff, component
    ):
        result = port_efficiency(
            mixed_link_eff, component, port=2, dynamic=True, segment=False
        )
        pd.testing.assert_frame_equal(result, default_dynamic_eff)

    def test_mixed_link_port2_segmented_request_segmented(
        self, mixed_link_eff, component, eff_param
    ):
        result = port_efficiency(
            mixed_link_eff, component, port=2, dynamic=False, segment=True
        )
        expected = mixed_link_eff.components[component].segments[f"{eff_param}2"]
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("segment", [True, False])
    def test_mixed_link_port3_dynamic_request_segmented_static(
        self, mixed_link_eff, segment, component
    ):
        result = port_efficiency(
            mixed_link_eff, component, port=3, dynamic=False, segment=segment
        )
        assert result.item() == 1

    def test_mixed_link_port3_dynamic_dynamic(
        self, mixed_link_eff, component, eff_param
    ):
        result = port_efficiency(
            mixed_link_eff, component, port=3, dynamic=True, segment=False
        )
        expected = mixed_link_eff.components[component].dynamic[f"{eff_param}3"]
        pd.testing.assert_frame_equal(result, expected)
