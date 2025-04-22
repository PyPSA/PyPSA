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
    {"at_port": True},
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
    n.lines["carrier"] = n.lines.bus0.map(n.buses.carrier)
    n.generators.loc[n.generators.index[0], "p_nom_extendable"] = False
    return n


# Test one static function for each groupby option and other options
@pytest.mark.parametrize("groupby", GROUPER_PARAMETERS)
def test_expressions_capacity(prepared_network, groupby):
    n = prepared_network
    expr = n.optimize.expressions.capacity(groupby=groupby)
    assert isinstance(expr, LinearExpression)
    assert expr.size > 0


@pytest.mark.parametrize("aggregate_across_components", (True, False))
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
@pytest.mark.parametrize("aggregate_time", AGGREGRATE_TIME_PARAMETERS)
@pytest.mark.parametrize("groupby", GROUPER_PARAMETERS)
def test_expressions_energy_balance(prepared_network, groupby, aggregate_time):
    n = prepared_network
    expr = n.optimize.expressions.energy_balance(
        groupby=groupby, aggregate_time=aggregate_time
    )
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
