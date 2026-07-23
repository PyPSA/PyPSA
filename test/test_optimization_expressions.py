# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest
from linopy import LinearExpression

import pypsa
from pypsa.statistics import groupers

TOLERANCE = 1e-2


GROUPER_PARAMETERS = [
    groupers.carrier,
    groupers["carrier"],
    [groupers.bus, groupers.carrier],
    ["name", "bus", "carrier"],
    ["carrier", "bus_carrier"],
    ["bus", "carrier", "bus_carrier"],
    False,
]
KWARGS_PARAMETERS = [
    {"at_port": "all"},
    {"bus_carrier": "AC"},
    {"carrier": "AC"},
    {"nice_names": True},
    {"aggregate_across_components": True},
]
AGGREGRATE_TIME_PARAMETERS = ["sum", "mean", None]


@pytest.fixture
def prepared_network(ac_dc_network):
    n = ac_dc_network.copy()
    n.optimize.create_model()
    n.c.lines.static["carrier"] = n.c.lines.static.bus0.map(n.c.buses.static.carrier)
    n.c.generators.static.loc[n.c.generators.static.index[0], "p_nom_extendable"] = (
        False
    )
    return n


@pytest.fixture
def prepared_network_with_snapshot_subset(ac_dc_network):
    n = ac_dc_network.copy()
    n.optimize.create_model(snapshots=n.snapshots[:2])
    n.c.lines.static["carrier"] = n.c.lines.static.bus0.map(n.c.buses.static.carrier)
    n.c.generators.static.loc[n.c.generators.static.index[0], "p_nom_extendable"] = (
        False
    )
    return n


# Test one static function for each groupby option and other options
@pytest.mark.parametrize("groupby", GROUPER_PARAMETERS)
def test_expressions_capacity(prepared_network, groupby):
    n = prepared_network
    expr = n.optimize.expressions.capacity(groupby=groupby)
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


@pytest.mark.parametrize("aggregate_across_components", [True, False])
def test_expression_capacity_all_filtered(
    prepared_network, aggregate_across_components
):
    n = prepared_network
    expr = n.optimize.expressions.capacity(
        bus_carrier="non-existent",
        aggregate_across_components=aggregate_across_components,
    )
    assert isinstance(expr, LinearExpression)
    assert expr.size == 0


@pytest.mark.parametrize(
    "kwargs", KWARGS_PARAMETERS + [{"include_non_extendable": True}]
)
def test_expressions_capacity_other_options(prepared_network, kwargs):
    n = prepared_network
    expr = n.optimize.expressions.capacity(**kwargs)
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


def test_expressions_capex(prepared_network):
    n = prepared_network
    expr = n.optimize.expressions.capex()
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


# Test one dynamic function for each groupby option and other options
@pytest.mark.parametrize("groupby_time", AGGREGRATE_TIME_PARAMETERS)
@pytest.mark.parametrize("groupby", GROUPER_PARAMETERS)
def test_expressions_energy_balance(prepared_network, groupby, groupby_time):
    n = prepared_network
    expr = n.optimize.expressions.energy_balance(
        groupby=groupby, groupby_time=groupby_time
    )
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


@pytest.mark.parametrize("groupby_time", AGGREGRATE_TIME_PARAMETERS)
@pytest.mark.parametrize("groupby", GROUPER_PARAMETERS)
def test_expressions_energy_balance_with_snapshot_subset(
    prepared_network_with_snapshot_subset, groupby, groupby_time
):
    n = prepared_network_with_snapshot_subset
    expr = n.optimize.expressions.energy_balance(
        groupby=groupby, groupby_time=groupby_time
    )
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


def test_other_dynamic_expressions_with_snapshot_subset(
    prepared_network_with_snapshot_subset,
):
    n = prepared_network_with_snapshot_subset
    for expr_str in [
        "opex",
        "curtailment",
        "operation",
        "supply",
        "withdrawal",
        "transmission",
    ]:
        expr = getattr(n.optimize.expressions, expr_str)()
        assert isinstance(expr, LinearExpression)
        assert expr.size > 0


@pytest.mark.parametrize("kwargs", KWARGS_PARAMETERS)
def test_expressions_energy_balance_other_options(prepared_network, kwargs):
    n = prepared_network
    expr = n.optimize.expressions.energy_balance(**kwargs)
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


def test_expressions_supply(prepared_network):
    n = prepared_network
    expr = n.optimize.expressions.supply()
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


def test_expressions_withdrawal(prepared_network):
    n = prepared_network
    expr = n.optimize.expressions.withdrawal()
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


def test_expressions_storage_unit_supply_withdrawal():
    n = pypsa.Network(snapshots=range(6))
    n.add("Bus", "b")
    n.add("Generator", "g", bus="b", p_nom=100, marginal_cost=[5, 5, 30, 30, 5, 30])
    n.add("Load", "l", bus="b", p_set=50)
    n.add("StorageUnit", "su", bus="b", p_nom=50, max_hours=4)
    n.optimize()

    ex = n.optimize.expressions
    kwargs = {"components": ["StorageUnit"], "groupby": False, "groupby_time": False}
    supply = ex.supply(**kwargs).solution.to_numpy().ravel()
    withdrawal = ex.withdrawal(**kwargs).solution.to_numpy().ravel()
    dynamic = n.c["StorageUnit"].dynamic

    assert withdrawal.any()
    assert np.allclose(supply, dynamic["p_dispatch"]["su"])
    assert np.allclose(withdrawal, dynamic["p_store"]["su"])


def test_expressions_transmission(prepared_network):
    n = prepared_network
    expr = n.optimize.expressions.transmission()
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


def test_expressions_opex(prepared_network):
    n = prepared_network
    expr = n.optimize.expressions.opex()
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


def test_expressions_curtailment(prepared_network):
    n = prepared_network
    expr = n.optimize.expressions.curtailment()
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


def test_expressions_operation(prepared_network):
    n = prepared_network
    expr = n.optimize.expressions.operation()
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


@pytest.fixture
def non_extendable_network():
    """Network with only non-extendable generators."""

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024", periods=4, freq="h"))

    n.add("Carrier", ["AC", "solar", "gas"])
    n.add("Bus", "bus0", carrier="AC")
    n.add(
        "Generator",
        "solar0",
        bus="bus0",
        p_nom=100,
        p_nom_extendable=False,
        marginal_cost=0,
        capital_cost=1000,
        carrier="solar",
        p_max_pu=[1.0, 0.8, 0.6, 0.4],
    )
    n.add(
        "Generator",
        "gas0",
        bus="bus0",
        p_nom=100,
        p_nom_extendable=False,
        marginal_cost=50,
        capital_cost=500,
        carrier="gas",
    )
    n.add("Load", "load0", bus="bus0", p_set=[50, 50, 50, 50])

    n.optimize.create_model()
    return n


def test_curtailment_non_extendable_generators(non_extendable_network):
    """Test curtailment expression works with only non-extendable generators."""
    n = non_extendable_network
    curtailment_expr = n.optimize.expressions.curtailment(
        components=["Generator"], groupby_time=False
    )
    assert isinstance(curtailment_expr, LinearExpression)
    assert curtailment_expr.size > 0, (
        "Curtailment expression should not be empty for non-extendable generators"
    )


def test_capacity_non_extendable_generators(non_extendable_network):
    """Test capacity expression works with only non-extendable generators."""
    n = non_extendable_network
    capacity_expr = n.optimize.expressions.capacity(components=["Generator"])
    assert isinstance(capacity_expr, LinearExpression)
    # For constant-only expressions (no variables), size=0 but const has values
    assert len(capacity_expr.dims) > 0, (
        "Capacity expression should have dimensions for non-extendable generators"
    )
    # Verify the constants are present (p_nom values: 100 for each generator)
    assert (capacity_expr.const > 0).any(), (
        "Capacity expression should have non-zero constant values"
    )


def test_capex_non_extendable_generators(non_extendable_network):
    """Test capex expression works with only non-extendable generators."""
    n = non_extendable_network
    capex_expr = n.optimize.expressions.capex(components=["Generator"])
    assert isinstance(capex_expr, LinearExpression)
    # For constant-only expressions (no variables), size=0 but const has values
    assert len(capex_expr.dims) > 0, (
        "Capex expression should have dimensions for non-extendable generators"
    )
    # Verify the constants are present (p_nom * capital_cost)
    assert (capex_expr.const > 0).any(), (
        "Capex expression should have non-zero constant values"
    )


def test_concrete_at_port(prepared_network):
    n = prepared_network
    n.c.links.static["efficiency"] = 0.9
    n.c.links.static["efficiency2"] = 0.9
    expr = n.optimize.expressions.capacity("Link", at_port=["bus1", "bus2"])
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0
    assert (expr.coeffs == 0.9).all().item()

    expr = n.optimize.expressions.capacity("Link", at_port="all")
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


class TestExpressionsWithPiecewise:
    @staticmethod
    @pytest.fixture(scope="class")
    def piecewise_network_built(piecewise_network):
        piecewise_network.optimize.create_model(include_objective_constant=False)
        return piecewise_network

    def test_expressions_capex(self, piecewise_network_built):
        n = piecewise_network_built
        expr = n.optimize.expressions.capex().unstack("group")
        assert isinstance(expr, LinearExpression)
        assert str(expr.sel(component="StorageUnit", carrier="-")).endswith(
            "+10 StorageUnit-p_nom[storage0] + 1 StorageUnit-capital_cost_piecewise[storage1]"
        )
        assert "piecewise" not in str(expr.sel(component=["Link", "Generator"]))

    def test_expressions_opex(self, piecewise_network_built):
        n = piecewise_network_built
        expr = n.optimize.expressions.opex().unstack("group")
        assert isinstance(expr, LinearExpression)
        assert str(expr.sel(component="Generator", carrier="-")).endswith(
            "+0.6 Generator-p[0, gen1] + 0.6 Generator-p[1, gen1] + 1 Generator-marginal_cost_piecewise[0, gen0] + 1 Generator-marginal_cost_piecewise[1, gen0]"
        )
        assert "piecewise" not in str(expr.sel(component=["Link", "StorageUnit"]))

    def test_supply_minus_withdrawal_equals_energy_balance(self):
        """supply - withdrawal must reconstruct the net energy balance at the optimum."""
        n = pypsa.Network()
        n.set_snapshots(range(2))
        n.add("Bus", ["bus0", "bus1"])
        n.add("Generator", "gen0", bus="bus0", p_nom=300, marginal_cost=5)
        n.add(
            "Link",
            "link",
            bus0="bus0",
            bus1="bus1",
            p_nom=100,
            efficiency={0.0: 0.4, 0.5: 0.5, 1.0: 0.6},
        )
        n.add("Load", "load", bus="bus1", p_set=[20, 30])
        n.optimize.create_model(include_objective_constant=False)
        exprs = {
            k: getattr(n.optimize.expressions, k)().sum()
            for k in ("supply", "withdrawal", "energy_balance")
        }
        n.optimize.solve_model()
        s, w, eb = (
            exprs[k].solution.item() for k in ("supply", "withdrawal", "energy_balance")
        )
        assert s - w == pytest.approx(eb)

    def test_capex_schema_without_data_is_linear(self):
        """A piecewise schema without breakpoint data costs linearly (no KeyError)."""
        n = pypsa.Network()
        n.add("Bus", "bus")
        n.add(
            "Link",
            "link",
            bus0="bus",
            bus1="bus",
            p_nom_extendable=True,
            capital_cost=100,
        )
        n.optimize.create_model(include_objective_constant=False)
        assert not n.c.links.has_piecewise("capital_cost")
        expr = n.optimize.expressions.capex().unstack("group")
        assert "piecewise" not in str(expr)
        assert "Link-p_nom" in str(expr)

    def test_expressions_energy_balance(self, piecewise_network_built):
        n = piecewise_network_built
        expr = n.optimize.expressions.energy_balance().unstack("group")
        assert str(expr.sel(component="Link", carrier="AC")).endswith(
            "-1 Link-p[0, link] - 1 Link-p[1, link] + 1 Link-p1_piecewise[0, link] + 1 Link-p1_piecewise[1, link]"
        )
        assert "piecewise" not in str(
            expr.sel(component=["Generator", "StorageUnit", "Load"])
        )

    def test_expressions_supply_withdrawal(self, piecewise_network_built):
        """A positive piecewise port counts as supply, never as withdrawal."""
        n = piecewise_network_built
        supply = n.optimize.expressions.supply().unstack("group")
        assert "Link-p1_piecewise" in str(supply.sel(component="Link", carrier="AC"))
        withdrawal = n.optimize.expressions.withdrawal().unstack("group")
        assert "piecewise" not in str(withdrawal.sel(component="Link", carrier="AC"))

    def test_supply_withdrawal_with_negative_piecewise_port(self):
        """A negative piecewise port counts as withdrawal, reported positive."""
        n = pypsa.Network()
        n.set_snapshots(range(2))
        n.add("Bus", ["bus0", "bus1"])
        n.add("Generator", "gen0", bus="bus0", p_nom=150, marginal_cost=5)
        n.add(
            "Process",
            "proc",
            bus0="bus0",
            bus1="bus1",
            p_nom=100,
            rate0={0.1: -1 / 0.3, 0.5: -1 / 0.4, 1.0: -1 / 0.6},
            rate1=1,
        )
        n.add("Load", "load", bus="bus1", p_set=[20, 30])
        n.optimize.create_model(include_objective_constant=False)
        withdrawal = n.optimize.expressions.withdrawal().unstack("group")
        supply = n.optimize.expressions.supply().unstack("group")
        assert "-1 Process-p0_piecewise" in str(withdrawal.sel(component="Process"))
        assert "p0_piecewise" not in str(supply.sel(component="Process"))
