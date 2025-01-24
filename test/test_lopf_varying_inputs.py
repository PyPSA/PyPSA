from numpy.testing import assert_array_almost_equal as equal

import pypsa


def test_time_dependent_generator_efficiency():
    n = pypsa.Network()
    s = [1, 0.25, 0.2]
    limit = sum(1 / i for i in s)
    n.snapshots = range(len(s))
    n.add("Bus", "bus")
    n.add("Carrier", "carrier", co2_emissions=1)
    n.add(
        "Generator",
        "gen",
        carrier="carrier",
        marginal_cost=1,
        bus="bus",
        p_nom=1,
        efficiency=s,
    )
    n.add("Load", "load", bus="bus", p_set=1)
    n.add("GlobalConstraint", "limit", constant=limit)
    status, _ = n.optimize()
    assert status == "ok"


def test_time_dependent_standing_losses_storage_units():
    n = pypsa.Network()
    s = [0, 0.1, 0.2]
    n.snapshots = range(len(s))
    n.add("Bus", "bus")
    n.add(
        "StorageUnit",
        "su",
        bus="bus",
        marginal_cost=1,
        p_nom=1,
        max_hours=1,
        state_of_charge_initial=1,
        standing_loss=s,
    )
    status, _ = n.optimize()
    assert status == "ok"
    equal(n.storage_units_t.state_of_charge.su.values, [1.0, 0.9, 0.72])


def test_time_dependent_standing_losses_stores():
    n = pypsa.Network()
    s = [0, 0.1, 0.2]
    n.snapshots = range(len(s))
    n.add("Bus", "bus")
    n.add(
        "Store",
        "sto",
        bus="bus",
        marginal_cost=1,
        e_nom=1,
        e_initial=1,
        standing_loss=s,
    )
    status, _ = n.optimize()
    assert status == "ok"
    equal(n.stores_t.e.sto.values, [1.0, 0.9, 0.72])
