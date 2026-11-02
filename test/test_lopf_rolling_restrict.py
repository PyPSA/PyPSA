# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest
from test_lopf_rolling_horizon import get_network as get_base_network

import pypsa

pytest.importorskip("linopy.restrict")

KWARGS = {"include_objective_constant": False, "log_to_console": False}


def get_network(scenarios: bool = False) -> pypsa.Network:
    n = get_base_network(committable=False)
    n.add(
        "StorageUnit",
        "battery",
        bus="bus",
        p_nom=100,
        max_hours=4,
        marginal_cost=1,
        state_of_charge_initial=100,
    )
    n.add("Store", "tank", bus="bus", e_nom=200, e_initial=50, marginal_cost=2)
    if scenarios:
        n.set_scenarios({"low": 0.3, "high": 0.7})
        n.c.loads.dynamic.p_set.loc[:, ("high", "load")] *= 1.2
    return n


@pytest.mark.parametrize(
    "scenarios", [False, True], ids=["deterministic", "stochastic"]
)
@pytest.mark.parametrize(("horizon", "overlap"), [(3, 0), (4, 1), (5, 2)])
def test_restrict_matches_rebuild(horizon, overlap, scenarios):
    rebuild, restrict = get_network(scenarios), get_network(scenarios)

    rebuild.optimize.optimize_with_rolling_horizon(
        horizon=horizon, overlap=overlap, **KWARGS
    )
    restrict.optimize.optimize_with_rolling_horizon(
        horizon=horizon, overlap=overlap, method="restrict", **KWARGS
    )

    for c, attr in [
        ("generators", "p"),
        ("storage_units", "state_of_charge"),
        ("stores", "e"),
        ("buses", "marginal_price"),
    ]:
        pd.testing.assert_frame_equal(
            restrict.c[c].dynamic[attr], rebuild.c[c].dynamic[attr], atol=1e-6
        )
    opex = rebuild.statistics.opex()
    if scenarios:
        opex = opex.groupby(level="scenario").sum() * rebuild.scenario_weightings.weight
    assert restrict.objective == pytest.approx(opex.sum())


def test_restrict_matches_rebuild_single_period_of_investment_periods():
    rebuild, restrict = get_network(), get_network()
    for n in (rebuild, restrict):
        n.c.generators.static[["ramp_limit_up", "ramp_limit_down"]] = float("nan")
        n.set_investment_periods([2030, 2040])

    rebuild.optimize.optimize_with_rolling_horizon(horizon=4, overlap=1, **KWARGS)
    restrict.optimize.optimize_with_rolling_horizon(
        horizon=4, overlap=1, method="restrict", **KWARGS
    )

    pd.testing.assert_frame_equal(
        restrict.c.stores.dynamic.e, rebuild.c.stores.dynamic.e, atol=1e-6
    )
    pd.testing.assert_frame_equal(
        restrict.c.generators.dynamic.p, rebuild.c.generators.dynamic.p, atol=1e-6
    )


def coupling_constraint(n: pypsa.Network, sns: pd.Index) -> None:
    p = n.model["Generator-p"]
    n.model.add_constraints(p.sum() <= 1e5, name="total_generation")


def add_risk_preference(n: pypsa.Network) -> None:
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.set_risk_preference(alpha=0.9, omega=0.5)


OUT_OF_SCOPE = {
    "cyclic-storage-unit": (
        lambda n: n.add("StorageUnit", "x", bus="bus", cyclic_state_of_charge=True),
        {},
        "cyclic storage units",
    ),
    "cyclic-store": (
        lambda n: n.add("Store", "x", bus="bus", e_cyclic=True),
        {},
        "cyclic stores",
    ),
    "global-constraint": (
        lambda n: n.add("GlobalConstraint", "co2", sense="<=", constant=1e9),
        {},
        "global constraints",
    ),
    "e-sum-max": (
        lambda n: n.add("Generator", "x", bus="bus", p_nom=10, e_sum_max=1e6),
        {},
        "volume limits of Generator",
    ),
    "multi-investment-periods": (
        lambda n: n.set_investment_periods([2030, 2040]),
        {"multi_investment_periods": True},
        "multi-period investment",
    ),
    "committable": (
        lambda n: n.add("Generator", "x", bus="bus", p_nom=10, committable=True),
        {},
        "committable Generator",
    ),
    "risk-preference": (add_risk_preference, {}, "risk preference"),
    "modular": (
        lambda n: n.add(
            "Generator", "x", bus="bus", p_nom_extendable=True, p_nom_mod=10
        ),
        {},
        "linear problems only",
    ),
    "coupling-constraint": (
        lambda n: None,
        {"extra_functionality": coupling_constraint},
        "couples it to later snapshots",
    ),
    "file-io": (lambda n: None, {"io_api": "lp"}, "io_api='lp'"),
    "keep-files": (lambda n: None, {"keep_files": True}, "keep_files"),
}


@pytest.mark.parametrize(
    ("modify", "kwargs", "match"), OUT_OF_SCOPE.values(), ids=OUT_OF_SCOPE.keys()
)
def test_restrict_out_of_scope(modify, kwargs, match):
    n = get_network()
    modify(n)

    with pytest.raises(NotImplementedError, match=match):
        n.optimize.optimize_with_rolling_horizon(
            horizon=3, method="restrict", **KWARGS, **kwargs
        )


def test_restrict_infeasible_window():
    n = get_network()
    n.c.loads.dynamic.p_set.loc[4, "load"] = 5000

    with pytest.raises(RuntimeError, match=r"window \[3:5\] failed .* infeasible"):
        n.optimize.optimize_with_rolling_horizon(horizon=3, method="restrict", **KWARGS)


def test_rolling_horizon_unknown_method():
    with pytest.raises(ValueError, match="method must be"):
        get_network().optimize.optimize_with_rolling_horizon(method="persistent")
