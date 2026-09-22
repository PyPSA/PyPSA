# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.fixture
def target_gen_p():
    target_path = (
        Path(__file__).parent
        / "data"
        / "storage-hvdc"
        / "results-lopf"
        / "generators-p.csv"
    )
    return pd.read_csv(target_path, index_col=0, parse_dates=True)


def test_optimize(storage_hvdc_network, target_gen_p):
    n = storage_hvdc_network
    n.optimize()
    equal(n.c.generators.dynamic.p.reindex_like(target_gen_p), target_gen_p, decimal=2)


def test_storage_energy_marginal_cost():
    n = pypsa.Network()
    n.snapshots = range(3)
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        marginal_cost=1,
        bus="bus",
        p_nom=3,
        p_max_pu=[1, 0, 0],
    )
    n.add("Load", "load", bus="bus", p_set=1)
    n.add(
        "Store",
        "store",
        bus="bus",
        marginal_cost_storage=0.2,
        e_initial=1,
        e_nom=10,
    )
    n.optimize()
    assert n.objective == 2.6


def test_spill_cost():
    sets_of_snapshots = 2
    p_set = [100, 100, 100, 100, 100]

    for has_spill_cost in [False, True]:
        n = pypsa.Network(snapshots=range(len(p_set) * sets_of_snapshots))

        n.add("Bus", "bus")

        # Add high capacity generator to help
        n.add(
            "Generator", "help", bus="bus", p_nom=1000, control="PQ", marginal_cost=100
        )

        # Add hydro unit
        if has_spill_cost:
            n.add(
                "StorageUnit",
                "hydro",
                bus="bus",
                p_nom=100,
                max_hours=10,
                inflow=[200, 200, 200, 200, 200, 50, 50, 50, 50, 50],
                spill_cost=1,
            )
        else:
            n.add(
                "StorageUnit",
                "hydro",
                bus="bus",
                p_nom=100,
                max_hours=10,
                inflow=[200, 200, 200, 200, 200, 50, 50, 50, 50, 50],
            )

        # Add Load
        n.add("Load", "load", bus="bus", p_set=p_set * sets_of_snapshots)

        overlap = 2
        for i in range(sets_of_snapshots):
            if i == 1:
                n.c.storage_units.static.state_of_charge_initial = (
                    n.c.storage_units.dynamic.state_of_charge.loc[n.snapshots[4]]
                )
            n.optimize(
                n.snapshots[i * len(p_set) : (i + 1) * len(p_set) + overlap],
            )

        spill = n.c.storage_units.dynamic["spill"].loc[:, "hydro"]
        total_spill = spill.sum()

        if has_spill_cost:
            assert total_spill == 0
        else:
            assert total_spill == 400


def test_storage_unit_p_set():
    """Test that p_set constrains net power (p_dispatch - p_store) for StorageUnit."""
    n = pypsa.Network()
    n.set_snapshots(range(4))

    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=10)
    n.add("Load", "load", bus="bus", p_set=[20, 30, 25, 35])

    n.add(
        "StorageUnit",
        "storage",
        bus="bus",
        p_nom=50,
        max_hours=2,
        p_set=[-10, 0, 5, 0],  # negative=charge, positive=discharge
        state_of_charge_initial=10,
    )

    n.optimize()

    # Check that p = p_dispatch - p_store equals p_set
    equal(n.c.storage_units.dynamic.p["storage"].values, [-10, 0, 5, 0], decimal=5)


def test_store_p_set():
    """Test p_set for Store components."""
    n = pypsa.Network()
    n.set_snapshots(range(4))

    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=10)
    n.add("Load", "load", bus="bus", p_set=[20, 30, 25, 35])

    n.add(
        "Store",
        "store",
        bus="bus",
        e_nom=100,
        p_set=[-10, 0, 5, 0],  # negative=charge, positive=discharge
        e_initial=10,
    )

    n.optimize()

    equal(n.c.stores.dynamic.p["store"].values, [-10, 0, 5, 0], decimal=5)


@pytest.mark.parametrize(
    ("cost_attr", "objective"),
    [
        ("marginal_cost", -2500),
        ("marginal_cost_dispatch", -500),
        ("marginal_cost_store", 0),
    ],
)
def test_store_marginal_cost_attributes(cost_attr, objective):
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=-5)
    n.add("Store", "store", bus="bus", e_nom=100, **{cost_attr: 20})
    n.optimize()
    assert n.objective == pytest.approx(objective)
    assert n.statistics.opex().sum() == pytest.approx(objective)


@pytest.mark.parametrize("efficiency_dispatch", [1, 0.9])
def test_store_max_hours_limits_power(efficiency_dispatch):
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=100)
    n.add("Load", "load", bus="bus", p_set=50)
    n.add(
        "Store",
        "store",
        bus="bus",
        e_nom=100,
        e_initial=100,
        max_hours=4,
        efficiency_dispatch=efficiency_dispatch,
    )
    n.optimize(assign_all_duals=True)
    assert (n.stores_t.p["store"] == 25).all()
    assert (n.generators_t.p["gen"] == 25).all()
    # energy capacity duals stay in mu_upper, power bound duals get their own name
    assert (n.stores_t.mu_upper["store"] == 0).all()
    assert (n.stores_t.mu_p_upper["store"].abs() == 100).all()


def test_store_inflow_and_spill():
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=5)
    n.add("Store", "store", bus="bus", e_nom=0, inflow=10, spill_cost=1)
    n.optimize()
    assert (n.stores_t.p["store"] == 5).all()
    assert (n.stores_t.spill["store"] == 5).all()
