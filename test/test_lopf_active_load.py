# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for active (price-responsive) loads."""

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.optimization.common import (
    compute_constant_operation_assets,
    get_operational_assets,
)

DEMAND = [30.0, 40.0, 50.0, 60.0]


def _base_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Carrier", ["AC", "gas", "demand"])
    n.add("Bus", "bus", carrier="AC")
    n.add("Generator", "cheap", bus="bus", p_nom=100, marginal_cost=10, carrier="gas")
    n.add("Generator", "dear", bus="bus", p_nom=100, marginal_cost=50, carrier="gas")
    n.add("Load", "fixed", bus="bus", p_set=DEMAND, carrier="demand")
    return n


def _add_demand(n: pypsa.Network, as_generator: bool, **kwargs) -> None:
    """Add the same price-responsive demand as Load or as Generator(sign=-1)."""
    if as_generator:
        for attr in ("marginal_cost", "marginal_cost_piecewise"):
            if attr in kwargs:
                kwargs[attr] = -kwargs[attr]
        n.add("Generator", "flex", bus="bus", sign=-1, carrier="demand", **kwargs)
    else:
        n.add("Load", "flex", bus="bus", carrier="demand", **kwargs)


def _optimize(n: pypsa.Network, as_generator: bool) -> str:
    """Optimize, expecting the sign FutureWarning for negative generators."""
    ctx = (
        pytest.warns(FutureWarning, match="sign < 0") if as_generator else nullcontext()
    )
    with ctx:
        status, _ = n.optimize(solver_name="highs")
    return status


def _flex_dispatch(n: pypsa.Network, as_generator: bool) -> pd.Series:
    return (n.generators_t.p if as_generator else n.loads_t.p)["flex"]


def test_passive_loads_enter_as_constants():
    n = _base_network()
    n.optimize.create_model()
    assert "Load-p" not in n.model.variables
    loads = n.c.loads
    assert compute_constant_operation_assets(n, loads, n.snapshots).equals(
        pd.Index(["fixed"], name="name")
    )
    assert get_operational_assets(n, loads).empty

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"
    assert n.loads_t.p["fixed"].tolist() == DEMAND
    assert n.objective == pytest.approx(10 * sum(DEMAND))


def test_loads_without_capacity_stay_constants():
    n = _base_network()
    n.optimize.create_model()
    n_vars, n_cons = n.model.nvars, n.model.ncons

    # NaN `p_set` without capacity: zero consumption, no variable (e.g. loads
    # stored with the former default `p_set=0`, which is not written to files)
    n.add("Load", "empty", bus="bus")
    nan = float("nan")
    n.add("Load", "partial", bus="bus", p_set=[5, nan, nan, 5])
    n.optimize.create_model()
    assert "Load-p" not in n.model.variables
    assert (n.model.nvars, n.model.ncons) == (n_vars, n_cons)

    n.optimize.solve_model(solver_name="highs")
    assert n.loads_t.p["empty"].tolist() == [0.0] * 4
    assert n.loads_t.p["partial"].tolist() == [5.0, 0.0, 0.0, 5.0]
    np.testing.assert_allclose(n.buses_t.p["bus"], 0, atol=1e-6)


@pytest.mark.parametrize("as_generator", [False, True])
def test_linear_willingness_to_pay(as_generator):
    n = _base_network()
    _add_demand(n, as_generator, p_nom=100, marginal_cost=30)
    assert _optimize(n, as_generator) == "ok"
    # Demand is served from the cheap generator only, up to its capacity
    expected = [100 - d for d in DEMAND]
    np.testing.assert_allclose(_flex_dispatch(n, as_generator), expected)
    assert n.objective == pytest.approx(10 * 400 - 30 * sum(expected))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_nom": 100, "marginal_cost": 30},
        {"p_nom": 100, "marginal_cost": 60, "marginal_cost_quadratic": 0.5},
        {"p_nom": 80, "marginal_cost": 45, "p_min_pu": 0.2, "ramp_limit_up": 0.1},
        {"p_nom": 80, "marginal_cost": 45, "e_sum_max": 150},
        {
            "p_nom_extendable": True,
            "capital_cost": 50,
            "marginal_cost": 45,
            "p_min_pu": 0.5,
        },
    ],
    ids=["linear", "quadratic", "ramp", "e_sum", "extendable"],
)
def test_equivalence_with_negative_generator(kwargs):
    results = []
    for as_generator in (False, True):
        n = _base_network()
        _add_demand(n, as_generator, **kwargs.copy())
        assert _optimize(n, as_generator) == "ok"
        results.append((n.objective, _flex_dispatch(n, as_generator).to_numpy()))

    (obj_load, p_load), (obj_gen, p_gen) = results
    assert obj_load == pytest.approx(obj_gen, rel=1e-6)
    np.testing.assert_allclose(p_load, p_gen, rtol=1e-5, atol=1e-5)


def test_committable_load_equivalence():
    kwargs = {
        "p_nom": 80,
        "marginal_cost": 45,
        "committable": True,
        "p_min_pu": 0.5,
        "start_up_cost": 100,
        "min_up_time": 2,
    }
    results = []
    for as_generator in (False, True):
        n = _base_network()
        _add_demand(n, as_generator, **kwargs.copy())
        assert _optimize(n, as_generator) == "ok"
        results.append(n.objective)
        if not as_generator:
            assert "Load-status" in n.model.variables
    assert results[0] == pytest.approx(results[1], rel=1e-6)


def test_partially_given_p_set_fixes_snapshots():
    n = _base_network()
    nan = float("nan")
    n.add(
        "Load", "part", bus="bus", p_nom=100, marginal_cost=30, p_set=[10, nan, nan, 20]
    )
    n.optimize(solver_name="highs")
    p = n.loads_t.p["part"]
    assert p.iloc[0] == pytest.approx(10)
    assert p.iloc[3] == pytest.approx(20)
    assert p.iloc[1] == pytest.approx(60)
    assert "Load-p_set" in n.model.constraints


def test_complete_p_set_with_variable_feature_is_fixed_variable():
    n = _base_network()
    n.add("Load", "ext", bus="bus", p_set=10, p_nom_extendable=True, capital_cost=1)
    n.optimize.create_model()
    assert list(n.model["Load-p"].indexes["name"]) == ["ext"]
    n.optimize.solve_model(solver_name="highs")
    assert n.loads_t.p["ext"].tolist() == [10.0] * 4
    assert n.loads.p_nom_opt["ext"] == pytest.approx(10)


def test_mixed_constant_and_active_loads():
    n = _base_network()
    _add_demand(n, False, p_nom=100, marginal_cost=30)
    n.optimize.create_model()
    assert list(n.model["Load-p"].indexes["name"]) == ["flex"]
    n.optimize.solve_model(solver_name="highs")
    assert n.loads_t.p["fixed"].tolist() == DEMAND
    # bus balance closes
    np.testing.assert_allclose(n.buses_t.p["bus"], 0, atol=1e-6)


def test_statistics_consistent_with_objective():
    n = _base_network()
    _add_demand(
        n,
        False,
        p_nom_extendable=True,
        capital_cost=50,
        marginal_cost=60,
        marginal_cost_quadratic=0.5,
    )
    n.optimize(solver_name="highs")
    total = n.statistics.capex().sum() + n.statistics.opex().sum()
    assert total == pytest.approx(n.objective + n.objective_constant, rel=1e-6)
    # utility of consumption shows as negative operational expenditure
    assert n.statistics.opex(components="Load").sum() < 0
    assert n.statistics.withdrawal(components="Load").sum() == pytest.approx(
        n.loads_t.p.sum().sum()
    )


def test_model_expressions_include_constants_and_variables():
    n = _base_network()
    _add_demand(n, False, p_nom=100, marginal_cost=30)
    n.optimize.create_model()
    n.optimize.solve_model(solver_name="highs")
    withdrawal = n.optimize.expressions.withdrawal(components="Load").solution
    assert float(withdrawal.sum()) == pytest.approx(n.loads_t.p.sum().sum())
    opex = n.optimize.expressions.opex(components="Load").solution
    assert float(opex.sum()) == pytest.approx(
        n.statistics.opex(components="Load").sum()
    )


def test_future_warning_for_negative_generator_cost():
    n = _base_network()
    n.add("Generator", "flex", bus="bus", sign=-1, p_nom=100, marginal_cost=-30)
    with pytest.warns(FutureWarning, match="sign < 0"):
        n.optimize.create_model()


def test_power_flow_with_nan_load_set_point():
    n = _base_network()
    n.add("Load", "flex", bus="bus", p_nom=10)
    n.lpf()
    assert not n.buses_t.p.isna().any().any()


def test_rolling_horizon_with_active_load():
    n = _base_network()
    _add_demand(n, False, p_nom=100, marginal_cost=30)
    n.optimize.optimize_with_rolling_horizon(horizon=2, overlap=0, solver_name="highs")
    np.testing.assert_allclose(n.loads_t.p["flex"], [100 - d for d in DEMAND])
    assert n.loads_t.p["fixed"].tolist() == DEMAND


def test_stochastic_active_load():
    n = _base_network()
    _add_demand(n, False, p_nom=100, marginal_cost=30)
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.loads.static.loc[("high", "flex"), "marginal_cost"] = 60
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"
    p = n.loads_t.p
    np.testing.assert_allclose(p[("low", "flex")], [100 - d for d in DEMAND])
    np.testing.assert_allclose(p[("high", "flex")], [100.0] * 4)
