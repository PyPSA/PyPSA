# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from typing import Any

import numpy as np
import pandas as pd
import pytest

import pypsa

NAN = float("nan")


@pytest.fixture(
    params=["storage_hvdc_network", "scigrid_de_network", "model_energy_network"]
)
def network(request):
    return request.getfixturevalue(request.param)


def test_optimization_equivalence(network):
    n = network
    names = n.storage_units.index
    n_store = n.copy()
    n_store.storage_units_to_stores()

    n.optimize()
    n_store.optimize()

    assert n_store.objective == pytest.approx(n.objective, rel=1e-6)
    e_nom_opt = n.storage_units.p_nom_opt * n.storage_units.max_hours
    pd.testing.assert_series_equal(
        n_store.stores.e_nom_opt[names], e_nom_opt, check_names=False, atol=1e-4
    )


def test_dispatch_equivalence(storage_hvdc_network):
    n = storage_hvdc_network
    names = n.storage_units.index
    # perturb costs so that the optimal dispatch is unique
    rng = np.random.default_rng(0)
    for c in ("generators", "storage_units"):
        index = getattr(n, c).index
        getattr(n, c + "_t")["marginal_cost"] = pd.DataFrame(
            getattr(n, c).marginal_cost.values
            + rng.uniform(0, 0.1, (len(n.snapshots), len(index))),
            index=n.snapshots,
            columns=index,
        )
    n_store = n.copy()
    n_store.storage_units_to_stores()

    n.optimize()
    n_store.optimize()

    su, st = n.storage_units_t, n_store.stores_t
    for attr in ("p_dispatch", "p_store", "spill"):
        pd.testing.assert_frame_equal(st[attr][names], su[attr][names], atol=1e-3)
    pd.testing.assert_frame_equal(st.e[names], su.state_of_charge[names], atol=1e-3)


def small_network(snapshots: int = 6, **su_kwargs: Any) -> pypsa.Network:
    n = pypsa.Network(snapshots=range(snapshots))
    n.add("Bus", "bus")
    mc = [10, 50, 20, 60, 15, 40, 12, 55, 18][:snapshots]
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=mc)
    load = [20, 80, 30, 60, 25, 40, 35, 55, 45][:snapshots]
    n.add("Load", "load", bus="bus", p_set=load)
    n.add("Store", "plain", bus="bus", e_nom=20, standing_loss=0.01)
    n.add("StorageUnit", "su", bus="bus", **{"p_nom": 10, **su_kwargs})
    return n


EFF = {"efficiency_store": 0.9, "efficiency_dispatch": 0.95}
SERIES = [0.5, 1, 0.8, 1, 0.3, 1]


@pytest.mark.parametrize(
    "su_kwargs",
    [
        {},
        {"max_hours": 4, **EFF},
        {"max_hours": 4, **EFF, "cyclic_state_of_charge": True, "standing_loss": 0.01},
        {"max_hours": 4, "efficiency_store": 0.9},
        {"max_hours": 4, "efficiency_dispatch": 0.9},
        {"max_hours": 6, **EFF, "inflow": [2, 8, 12, 0, 1, 3], "spill_cost": 1},
        {"max_hours": 6, "inflow": 12, "cyclic_state_of_charge": True},
        {"max_hours": 4, **EFF, "p_nom_extendable": True, "capital_cost": 5},
        {"max_hours": 4, "p_nom_extendable": True, "p_nom_max": 8, "capital_cost": 5},
        {
            "max_hours": 4,
            **EFF,
            "p_set": [-5, 5, NAN, 5, -5, NAN],
            "state_of_charge_initial": 20,
        },
        {"max_hours": 4, "standing_loss": 0.02, "state_of_charge_initial": 20},
        {"max_hours": 4, "marginal_cost": 2},
        {"max_hours": 4, **EFF, "marginal_cost": [1, 3, 2, 4, 1.5, 2.5]},
        {"max_hours": 4, "marginal_cost_storage": 0.5},
        {"max_hours": 4, **EFF, "p_max_pu": 0.7, "p_min_pu": -0.4},
        {"max_hours": 4, "p_max_pu": SERIES, "p_min_pu": [-x for x in SERIES]},
        {"max_hours": 4, **EFF, "state_of_charge_set": [NAN, NAN, 15, NAN, 5, NAN]},
    ],
)
def test_small_network_equivalence(su_kwargs: dict[str, Any]) -> None:
    n = small_network(**su_kwargs)
    n_store = n.copy()
    n_store.storage_units_to_stores()

    n.optimize()
    n_store.optimize()

    assert n_store.objective == pytest.approx(n.objective, rel=1e-6)
    su, st = n.storage_units_t, n_store.stores_t
    for attr in ("p", "p_dispatch", "p_store", "spill"):
        np.testing.assert_allclose(
            st[attr].get("su", 0), su[attr].get("su", 0), atol=1e-5
        )
    np.testing.assert_allclose(st.e["su"], su.state_of_charge["su"], atol=1e-5)
    np.testing.assert_allclose(st.p["plain"], n.stores_t.p["plain"], atol=1e-5)

    for stat in ("opex", "installed_capacity", "optimal_capacity"):
        kwargs = {"storage": True} if "capacity" in stat else {}
        expected = getattr(n.statistics, stat)(**kwargs).sum()
        assert getattr(n_store.statistics, stat)(**kwargs).sum() == pytest.approx(
            expected, rel=1e-6
        )
    for direction in ("supply", "withdrawal"):
        expected = n.statistics.energy_balance(direction=direction).sum()
        balance = n_store.statistics.energy_balance(direction=direction).sum()
        assert balance == pytest.approx(expected, rel=1e-6)


def test_rolling_horizon_equivalence() -> None:
    n = small_network(snapshots=9, max_hours=4, state_of_charge_initial=20, **EFF)
    n_store = n.copy()
    n_store.storage_units_to_stores()

    n.optimize.optimize_with_rolling_horizon(horizon=3, overlap=0)
    n_store.optimize.optimize_with_rolling_horizon(horizon=3, overlap=0)

    np.testing.assert_allclose(n_store.stores_t.p["su"], n.storage_units_t.p["su"])
    np.testing.assert_allclose(
        n_store.stores_t.e["su"], n.storage_units_t.state_of_charge["su"], atol=1e-5
    )


def test_capacity_expression_storage() -> None:
    n = small_network(max_hours=4, p_nom_extendable=True, capital_cost=5)
    n.optimize()
    expected = n.statistics.optimal_capacity(storage=True, groupby=False)
    n.storage_units_to_stores()
    n.optimize()

    capacity = n.optimize.expressions.capacity(storage=True, groupby=False)
    assert capacity.solution.sum() == pytest.approx(expected.sum(), rel=1e-6)
    optimal = n.statistics.optimal_capacity(storage=True, groupby=False)
    assert optimal.droplevel("component").to_dict() == pytest.approx(
        expected.droplevel("component").to_dict(), rel=1e-6
    )


@pytest.mark.parametrize(
    ("attr", "value", "store_attr", "expected"),
    [
        ("p_nom", 10, "e_nom", 40),
        ("p_nom_min", 5, "e_nom_min", 20),
        ("p_nom_max", 20, "e_nom_max", 80),
        ("p_nom_set", 10, "e_nom_set", 40),
        ("p_nom_mod", 5, "e_nom_mod", 20),
        ("p_nom_extendable", True, "e_nom_extendable", True),
        ("capital_cost", 8, "capital_cost", 2),
        ("overnight_cost", 400, "overnight_cost", 100),
        ("fom_cost", 4, "fom_cost", 1),
        ("discount_rate", 0.05, "discount_rate", 0.05),
        ("marginal_cost", 3, "marginal_cost_dispatch", 3),
        ("marginal_cost", [1, 2], "marginal_cost_dispatch", [1, 2]),
        ("marginal_cost_storage", 0.5, "marginal_cost_storage", 0.5),
        ("spill_cost", 2, "spill_cost", 2),
        ("state_of_charge_initial", 7, "e_initial", 7),
        ("state_of_charge_initial_per_period", True, "e_initial_per_period", True),
        ("state_of_charge_set", [5, NAN], "e_set", [5, NAN]),
        ("cyclic_state_of_charge", True, "e_cyclic", True),
        ("cyclic_state_of_charge_per_period", True, "e_cyclic_per_period", True),
        ("max_hours", 4, "max_hours", 4),
        ("efficiency_store", 0.9, "efficiency_store", 0.9),
        ("efficiency_dispatch", [0.9, 0.8], "efficiency_dispatch", [0.9, 0.8]),
        ("standing_loss", 0.01, "standing_loss", 0.01),
        ("inflow", 3, "inflow", 3),
        ("inflow", [1, 2], "inflow", [1, 2]),
        ("p_min_pu", -0.5, "p_min_pu", -0.5),
        ("p_max_pu", [0.5, 1], "p_max_pu", [0.5, 1]),
        ("p_set", [1, -1], "p_set", [1, -1]),
        ("build_year", 2020, "build_year", 2020),
        ("lifetime", 20, "lifetime", 20),
        ("active", False, "active", False),
    ],
)
def test_attribute_conversion(
    attr: str, value: Any, store_attr: str, expected: Any
) -> None:
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add("StorageUnit", "su", bus="bus", **{"max_hours": 4, attr: value})
    n.storage_units_to_stores()

    assert n.storage_units.empty
    if isinstance(expected, list):
        np.testing.assert_equal(n.stores_t[store_attr]["su"].tolist(), expected)
    else:
        assert n.stores.at["su", store_attr] == expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"p_dispatch_set": [1, NAN]}, "p_dispatch_set"),
        ({"p_store_set": [1, NAN]}, "p_store_set"),
        ({"marginal_cost_quadratic": 0.1}, "marginal_cost_quadratic"),
        ({"marginal_cost_quadratic": [0, 0.1]}, "marginal_cost_quadratic"),
        ({"marginal_cost": {0.0: 0.0, 1.0: 10.0}}, "piecewise"),
        ({"capital_cost": {0.0: 0.0, 1.0: 5.0}}, "piecewise"),
    ],
)
def test_unsupported_attribute_raises(kwargs: dict[str, Any], match: str) -> None:
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add("StorageUnit", "su", bus="bus", max_hours=4, **kwargs)
    n.add("StorageUnit", "plain", bus="bus", max_hours=4)
    with pytest.raises(ValueError, match=match):
        n.storage_units_to_stores()
    assert n.storage_units_to_stores(["plain"]).tolist() == ["plain"]


def test_convert_subset():
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add("StorageUnit", "a", bus="bus", p_nom=10, max_hours=2, marginal_cost=1)
    n.add("StorageUnit", "b", bus="bus", p_nom=5, max_hours=4, marginal_cost=[3, 4])
    b = n.storage_units.loc["b"].copy()

    assert n.storage_units_to_stores(["a"]).tolist() == ["a"]

    pd.testing.assert_series_equal(n.storage_units.loc["b"], b)
    assert n.storage_units_t.marginal_cost["b"].tolist() == [3, 4]
    assert n.stores.loc["a", ["e_nom", "marginal_cost_dispatch"]].tolist() == [20, 1]
    assert "a" not in n.stores_t.marginal_cost_dispatch


@pytest.mark.parametrize("max_hours", [np.inf, NAN])
def test_non_finite_max_hours_raises(max_hours: float) -> None:
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add("StorageUnit", "su", bus="bus")
    n.storage_units.loc["su", "max_hours"] = max_hours
    with pytest.raises(ValueError, match="finite `max_hours`"):
        n.storage_units_to_stores()


def test_storage_unit_future_warning():
    n = pypsa.Network()
    n.add("Bus", "bus")
    with pytest.warns(FutureWarning, match="storage_units_to_stores"):
        n.add("StorageUnit", "su", bus="bus")


def test_name_clash_raises():
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add("StorageUnit", "x", bus="bus")
    n.add("Store", "x", bus="bus")
    with pytest.raises(ValueError, match="already exist"):
        n.storage_units_to_stores()
