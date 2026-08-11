# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""At the optimum, optimization expressions must evaluate to the statistics values."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.statistics import groupers

TOLERANCE = 1e-2

BUS_CARRIER_GROUPBY = {"groupby": ["carrier", "bus_carrier"]}

# (expression method, statistics method, shared kwargs, expression sign)
Pair = tuple[str, str, dict[str, Any], int]

STATIC_PAIRS: list[Pair] = [
    ("capex", "capex", {}, 1),
    ("capacity", "optimal_capacity", {}, 1),
]
DYNAMIC_PAIRS: list[Pair] = [
    ("opex", "opex", {}, 1),
    ("curtailment", "curtailment", {}, 1),
    # supply/withdrawal clip coefficients, which only reproduces the statistics'
    # clip of the realised flow for components that cannot reverse direction
    ("supply", "supply", {"components": ["Generator"], **BUS_CARRIER_GROUPBY}, 1),
    ("withdrawal", "withdrawal", {"components": ["Load"], **BUS_CARRIER_GROUPBY}, 1),
    ("energy_balance", "energy_balance", BUS_CARRIER_GROUPBY, 1),
    # the expression weights the branch flow by the bus0 port efficiency, so it
    # reports the bus-side contribution where statistics reports the branch flow
    ("transmission", "transmission", {}, -1),
]
ALL_PAIRS = STATIC_PAIRS + DYNAMIC_PAIRS
REPRESENTATIVE_PAIRS = [
    pair for pair in ALL_PAIRS if pair[0] in ("capacity", "opex", "energy_balance")
]
# statistics covers all snapshots, so metrics with a non-zero contribution outside
# the optimized window cannot agree with an expression built on the window alone
SNAPSHOT_EXTENT_SENSITIVE = {"curtailment"}

GROUPER_PARAMETERS = [
    groupers.carrier,
    [groupers.bus, groupers.carrier],
    ["name", "bus", "carrier"],
    ["carrier", "bus_carrier"],
    False,
]
FILTER_PARAMETERS = [
    {"bus_carrier": "AC"},
    {"carrier": "AC"},
    {"nice_names": True},
]


def _ac_dc_network() -> pypsa.Network:
    n = pypsa.examples.ac_dc_meshed()
    n.c.lines.static["carrier"] = n.c.lines.static.bus0.map(n.c.buses.static.carrier)
    return n


@pytest.fixture(scope="module")
def solved_network():
    n = _ac_dc_network()
    n.optimize(include_objective_constant=False)
    return n


@pytest.fixture(scope="module")
def solved_multi_period():
    n = _ac_dc_network()
    n.snapshots = pd.MultiIndex.from_product([[2020, 2030], n.snapshots])
    n.investment_periods = [2020, 2030]
    n.optimize(multi_investment_periods=True, include_objective_constant=False)
    return n


@pytest.fixture(scope="module")
def solved_snapshot_subset():
    n = _ac_dc_network()
    n.optimize(snapshots=n.snapshots[:3], include_objective_constant=False)
    return n


def as_series(obj) -> pd.Series:
    """Expression solution or statistics result as a Series with canonical levels.

    Statistics spread periods and snapshots over columns while expressions carry
    them as dimensions, so both are flattened and their levels ordered alike.
    """
    if isinstance(obj, pd.DataFrame):
        series = obj.stack()
    elif isinstance(obj, pd.Series):
        series = obj
    elif not obj.solution.dims:
        # a filter matching nothing collapses the expression to a scalar
        series = pd.Series(dtype=float)
    else:
        series = obj.solution.to_series()
    if isinstance(series.index, pd.MultiIndex):
        series = series.reorder_levels(sorted(series.index.names))
    return series.sort_index()


def assert_matches(expr, stat, sign: int = 1) -> None:
    """Assert an expression evaluates to the statistics values on the common index."""
    left = sign * as_series(expr)
    right = as_series(stat)
    if left.empty and right.empty:
        return
    index = left.index.union(right.index)
    left = left.reindex(index).fillna(0.0)
    right = right.reindex(index).fillna(0.0)
    assert np.allclose(left, right, atol=TOLERANCE), (
        f"expression and statistics differ:\n{pd.concat([left, right], axis=1)}"
    )


def compare(n: pypsa.Network, pair: tuple, **kwargs) -> None:
    """Compare an expression/statistics method pair on the same arguments."""
    expr_name, stat_name, shared, sign = pair
    call = {**shared, **kwargs}
    expr = getattr(n.optimize.expressions, expr_name)(**call)
    stat = getattr(n.statistics, stat_name)(**call)
    assert_matches(expr, stat, sign)


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=lambda p: p[0])
def test_expression_solution_matches_statistics(solved_network, pair):
    compare(solved_network, pair)


@pytest.mark.parametrize("pair", DYNAMIC_PAIRS, ids=lambda p: p[0])
@pytest.mark.parametrize("groupby_time", ["sum", "mean", False])
def test_time_aggregation_matches_statistics(solved_network, pair, groupby_time):
    compare(solved_network, pair, groupby_time=groupby_time)


@pytest.mark.parametrize("pair", REPRESENTATIVE_PAIRS, ids=lambda p: p[0])
@pytest.mark.parametrize("groupby", GROUPER_PARAMETERS, ids=str)
def test_grouping_matches_statistics(solved_network, pair, groupby):
    expr_name, stat_name, _, sign = pair
    expr = getattr(solved_network.optimize.expressions, expr_name)(groupby=groupby)
    stat = getattr(solved_network.statistics, stat_name)(groupby=groupby)
    if groupby is False:
        # ungrouped expressions are indexed by component name alone
        stat = stat.droplevel("component")
    assert_matches(expr, stat, sign)


@pytest.mark.parametrize("kwargs", FILTER_PARAMETERS, ids=str)
@pytest.mark.parametrize("pair", REPRESENTATIVE_PAIRS, ids=lambda p: p[0])
def test_filters_match_statistics(solved_network, pair, kwargs):
    compare(solved_network, pair, **kwargs)


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=lambda p: p[0])
def test_multi_period_matches_statistics(solved_multi_period, pair):
    compare(solved_multi_period, pair)


@pytest.mark.parametrize("pair", DYNAMIC_PAIRS, ids=lambda p: p[0])
@pytest.mark.parametrize("groupby_time", ["sum", "mean"])
def test_multi_period_time_aggregation_matches_statistics(
    solved_multi_period, pair, groupby_time
):
    compare(solved_multi_period, pair, groupby_time=groupby_time)


@pytest.mark.parametrize(
    "pair",
    [pair for pair in ALL_PAIRS if pair[0] not in SNAPSHOT_EXTENT_SENSITIVE],
    ids=lambda p: p[0],
)
def test_snapshot_subset_matches_statistics(solved_snapshot_subset, pair):
    compare(solved_snapshot_subset, pair)


def test_operation_matches_dispatch(solved_network):
    """``operation`` has no statistics counterpart; check it against the dispatch."""
    n = solved_network
    expr = n.optimize.expressions.operation(
        components=["Generator"], groupby=False, groupby_time="sum"
    )
    weights = n.snapshot_weightings.generators
    dispatch = n.c.generators.dynamic.p.mul(weights, axis=0).sum()
    solution = as_series(expr)
    assert np.allclose(solution, dispatch.reindex(solution.index), atol=TOLERANCE)
