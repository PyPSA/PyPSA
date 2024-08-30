import os

import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.fixture
def target_gen_p():
    target_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "opf-storage-hvdc",
        "opf-storage-data",
        "results",
        "generators-p.csv",
    )
    return pd.read_csv(target_path, index_col=0, parse_dates=True)


@pytest.fixture
def network():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "opf-storage-hvdc",
        "opf-storage-data",
    )
    return pypsa.Network(csv_folder)


def test_optimize(network, target_gen_p):
    network.optimize()
    equal(network.generators_t.p.reindex_like(target_gen_p), target_gen_p, decimal=2)


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
        network = pypsa.Network(snapshots=range(len(p_set) * sets_of_snapshots))

        network.add("Bus", "bus")

        # Add high capacity generator to help
        network.add(
            "Generator", "help", bus="bus", p_nom=1000, control="PQ", marginal_cost=100
        )

        # Add hydro unit
        if has_spill_cost:
            network.add(
                "StorageUnit",
                "hydro",
                bus="bus",
                p_nom=100,
                max_hours=10,
                inflow=[200, 200, 200, 200, 200, 50, 50, 50, 50, 50],
                spill_cost=1,
            )
        else:
            network.add(
                "StorageUnit",
                "hydro",
                bus="bus",
                p_nom=100,
                max_hours=10,
                inflow=[200, 200, 200, 200, 200, 50, 50, 50, 50, 50],
            )

        # Add Load
        network.add("Load", "load", bus="bus", p_set=p_set * sets_of_snapshots)

        overlap = 2
        for i in range(sets_of_snapshots):
            if i == 1:
                network.storage_units.state_of_charge_initial = (
                    network.storage_units_t.state_of_charge.loc[network.snapshots[4]]
                )
            network.optimize(
                network.snapshots[i * len(p_set) : (i + 1) * len(p_set) + overlap],
            )

        spill = network.storage_units_t["spill"].loc[:, "hydro"]
        total_spill = spill.sum()

        if has_spill_cost:
            assert total_spill == 0
        else:
            assert total_spill == 400
