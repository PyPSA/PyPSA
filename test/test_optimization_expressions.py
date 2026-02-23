# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest
from linopy import LinearExpression

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
    import pandas as pd

    import pypsa

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
