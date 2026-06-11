# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import pypsa
from pypsa.consistency import ConsistencyError


@pytest.fixture
def basic_network():
    n = pypsa.Network()
    n.set_snapshots(range(10))
    n.add("Bus", "bus")
    return n


def test_basic_maintenance_scheduling(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=3,
    )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=50)
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["gen"].values
    assert maint.sum() == 3
    diffs = np.diff(np.where(maint)[0])
    assert (diffs == 1).all()


def test_optimal_timing(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "cheap",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add("Generator", "expensive", bus="bus", p_nom=100, marginal_cost=100)
    load = [80] * 10
    load[0] = 10
    load[1] = 10
    n.add("Load", "load", bus="bus", p_set=load)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["cheap"].values
    maint_periods = np.where(maint)[0]
    assert set(maint_periods) == {0, 1}


def test_partial_maintenance(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=2,
        maintenance_pu=0.5,
    )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=100)
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["gen"].values
    p = n.c.generators.dynamic.p["gen"].values
    assert maint.sum() == 2
    for t in range(len(maint)):
        if maint[t] > 0.5:
            assert p[t] <= 50 + 1e-5


def test_multiple_events(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=2,
        maintenance_events=2,
    )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=50)
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["gen"].values
    starts = n.c.generators.dynamic.maintenance_start["gen"].values
    assert maint.sum() == 4
    assert starts.sum() == 2
    assert all(maint[t : t + 2].sum() == 2 for t in np.where(starts > 0.5)[0])


def test_committable_maintainable(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        committable=True,
        p_min_pu=0.3,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=50)
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["gen"].values
    p = n.c.generators.dynamic.p["gen"].values
    assert maint.sum() == 2
    for t in range(len(maint)):
        if maint[t] > 0.5:
            assert p[t] < 1e-5


def test_extendable_maintainable(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=200,
        capital_cost=1,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add("Generator", "backup", bus="bus", p_nom=200, marginal_cost=100)
    n.add("Load", "load", bus="bus", p_set=80)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["gen"].values
    p = n.c.generators.dynamic.p["gen"].values
    assert maint.sum() == 2
    for t in range(len(maint)):
        if maint[t] > 0.5:
            assert p[t] < 1e-5


def test_link_maintenance(basic_network):
    n = basic_network
    n.add("Bus", "bus1")
    n.add("Generator", "gen", bus="bus", p_nom=200, marginal_cost=10)
    n.add(
        "Link",
        "link",
        bus0="bus",
        bus1="bus1",
        p_nom=100,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add("Generator", "backup", bus="bus1", p_nom=100, marginal_cost=100)
    n.add("Load", "load", bus="bus1", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.links.dynamic.maintenance["link"].values
    assert maint.sum() == 2
    p = n.c.links.dynamic.p0["link"].values
    for t in range(len(maint)):
        if maint[t] > 0.5:
            assert p[t] < 1e-5


def test_process_maintenance(basic_network):
    n = basic_network
    n.add("Bus", "bus1")
    n.add("Generator", "gen", bus="bus", p_nom=200, marginal_cost=10)
    n.add(
        "Process",
        "process",
        bus0="bus",
        bus1="bus1",
        p_nom=100,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add("Generator", "backup", bus="bus1", p_nom=100, marginal_cost=100)
    n.add("Load", "load", bus="bus1", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.processes.dynamic.maintenance["process"].values
    assert maint.sum() == 2
    p = n.c.processes.dynamic.p["process"].values
    for t in range(len(maint)):
        if maint[t] > 0.5:
            assert p[t] < 1e-5


def test_multiple_generators_no_simultaneous_overlap(basic_network):
    n = basic_network
    for i in range(2):
        n.add(
            "Generator",
            f"gen{i}",
            bus="bus",
            p_nom=100,
            marginal_cost=10,
            maintainable=True,
            maintenance_duration=2,
        )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=100)
    n.add("Load", "load", bus="bus", p_set=150)

    status = n.optimize()
    assert status[0] == "ok"

    m0 = n.c.generators.dynamic.maintenance["gen0"].values
    m1 = n.c.generators.dynamic.maintenance["gen1"].values
    simultaneous = (m0 > 0.5) & (m1 > 0.5)
    assert not simultaneous.any()


def test_no_maintenance_without_flag(basic_network):
    n = basic_network
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=10)
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"

    assert (
        "maintenance" not in n.c.generators.dynamic
        or n.c.generators.dynamic.maintenance.empty
    )


def test_mixed_fixed_generators_with_maintenance(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "maint",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add("Generator", "plain", bus="bus", p_nom=100, marginal_cost=20)
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"
    assert n.c.generators.dynamic.maintenance["maint"].sum() == 2


def test_mixed_committable_generators_with_maintenance(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "maint",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        committable=True,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add(
        "Generator", "plain", bus="bus", p_nom=100, marginal_cost=20, committable=True
    )
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"
    assert n.c.generators.dynamic.maintenance["maint"].sum() == 2


def test_mixed_extendable_generators_with_maintenance(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "maint",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=100,
        capital_cost=1,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add(
        "Generator",
        "plain",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=100,
        capital_cost=1,
        marginal_cost=20,
    )
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"
    assert n.c.generators.dynamic.maintenance["maint"].sum() == pytest.approx(2)


def test_single_snapshot_duration(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=1,
    )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=50)
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["gen"].values
    assert maint.sum() == 1


def test_full_horizon_duration():
    n = pypsa.Network()
    n.set_snapshots(range(5))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=5,
    )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=50)
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["gen"].values
    assert maint.sum() == 5
    assert (maint > 0.5).all()


def test_extendable_committable_maintainable(basic_network):
    n = basic_network
    n.add(
        "Generator",
        "gen",
        bus="bus",
        committable=True,
        p_nom_extendable=True,
        p_nom_max=200,
        capital_cost=1,
        marginal_cost=10,
        p_min_pu=0.3,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add("Generator", "backup", bus="bus", p_nom=200, marginal_cost=100)
    n.add("Load", "load", bus="bus", p_set=80)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["gen"].values
    p = n.c.generators.dynamic.p["gen"].values
    assert maint.sum() == 2
    for t in range(len(maint)):
        if maint[t] > 0.5:
            assert p[t] < 1e-5


def test_invalid_zero_duration(basic_network):
    n = basic_network.copy()
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=0,
    )
    n.add("Load", "load", bus="bus", p_set=50)

    with pytest.raises(ConsistencyError, match="maintenance_duration <= 0"):
        n.consistency_check(strict=["maintenance"])


def test_invalid_duration_exceeds_horizon(basic_network):
    n = basic_network.copy()
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=20,
    )
    n.add("Load", "load", bus="bus", p_set=50)

    with pytest.raises(ConsistencyError, match="maintenance_duration > total weighted"):
        n.consistency_check(strict=["maintenance"])


def test_invalid_zero_events(basic_network):
    n = basic_network.copy()
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=2,
        maintenance_events=0,
    )
    n.add("Load", "load", bus="bus", p_set=50)

    with pytest.raises(ConsistencyError, match="maintenance_events <= 0"):
        n.consistency_check(strict=["maintenance"])


def test_invalid_total_duration_exceeds_horizon(basic_network):
    n = basic_network.copy()
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=4,
        maintenance_events=3,
    )
    n.add("Load", "load", bus="bus", p_set=50)

    with pytest.raises(
        ConsistencyError,
        match="maintenance_duration .* maintenance_events > total weighted",
    ):
        n.consistency_check(strict=["maintenance"])


def test_invalid_infinite_p_nom_max(basic_network):
    n = basic_network.copy()
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom_extendable=True,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add("Load", "load", bus="bus", p_set=50)

    with pytest.raises(ConsistencyError, match="infinite p_nom_max"):
        n.consistency_check(strict=["maintenance"])


@pytest.mark.parametrize(("duration", "expected_snapshots"), [(4, 2), (3, 2), (2, 1)])
def test_weighted_duration(duration, expected_snapshots):
    n = pypsa.Network()
    n.set_snapshots(range(10))
    n.snapshot_weightings.loc[:, :] = 2.0
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=duration,
    )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=50)
    n.add("Load", "load", bus="bus", p_set=50)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["gen"].values
    assert maint.sum() == expected_snapshots
    assert (np.diff(np.where(maint > 0.5)[0]) == 1).all()


def test_non_uniform_weightings():
    n = pypsa.Network()
    n.set_snapshots(range(10))
    n.snapshot_weightings = n.snapshot_weightings.mul([3.0] * 5 + [1.0] * 5, axis=0)
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        maintainable=True,
        maintenance_duration=3,
    )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=50)
    load = [80.0] * 5 + [10.0] * 5
    n.add("Load", "load", bus="bus", p_set=load)

    status = n.optimize()
    assert status[0] == "ok"

    maint = n.c.generators.dynamic.maintenance["gen"].values > 0.5
    weights = n.snapshot_weightings.generators.values
    assert weights[maint].sum() >= 3
    assert set(np.where(maint)[0]) <= set(range(5, 10))
    assert maint.sum() == 3
