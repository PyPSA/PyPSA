# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

import pypsa


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


def test_attribute_conversion():
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add(
        "StorageUnit",
        "su",
        bus="bus",
        p_nom=10,
        p_nom_max=20,
        max_hours=4,
        capital_cost=8,
        marginal_cost=[1, 2],
        efficiency_store=0.9,
        inflow=3,
        cyclic_state_of_charge=True,
        state_of_charge_set=[5, float("nan")],
    )
    n.storage_units_to_stores()

    assert n.storage_units.empty
    store = n.stores.loc["su"]
    assert store.e_nom == 40
    assert store.e_nom_max == 80
    assert store.max_hours == 4
    assert store.capital_cost == 2
    assert store.efficiency_store == 0.9
    assert store.inflow == 3
    assert store.e_cyclic
    assert n.stores_t.marginal_cost_dispatch["su"].tolist() == [1, 2]
    assert n.stores_t.e_set["su"].iloc[0] == 5


def test_name_clash_raises():
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add("StorageUnit", "x", bus="bus")
    n.add("Store", "x", bus="bus")
    with pytest.raises(ValueError, match="already exist"):
        n.storage_units_to_stores()
