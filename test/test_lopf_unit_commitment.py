# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


def test_unit_commitment():
    """
    This test is based on https://docs.pypsa.org/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    expected_status = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=float).T

    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array([[4000, 6000, 5000, 0], [0, 0, 0, 800]], dtype=float).T

    equal(n.c.generators.dynamic.p.values, expected_dispatch)


def test_minimum_up_time():
    """
    This test is based on https://docs.pypsa.org/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        up_time_before=0,
        min_up_time=3,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 800, 5000, 3000])

    n.optimize()

    expected_status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T

    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array(
        [[3900, 0, 4900, 3000], [100, 800, 100, 0]], dtype=float
    ).T

    equal(n.c.generators.dynamic.p.values, expected_dispatch)


def test_minimum_up_time_up_time_before():
    """
    This test is based on https://docs.pypsa.org/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        up_time_before=1,
        min_up_time=4,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 800, 5000, 3000])

    n.optimize()

    expected_status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T

    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array(
        [[3900, 0, 4900, 3000], [100, 800, 100, 0]], dtype=float
    ).T

    equal(n.c.generators.dynamic.p.values, expected_dispatch)


def test_minimum_down_time():
    """
    This test is based on https://docs.pypsa.org/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """
    n = pypsa.Network()

    n.set_snapshots(range(4))

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        min_down_time=2,
        down_time_before=1,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=4000,
    )

    n.add("Load", "load", bus="bus", p_set=[3000, 800, 3000, 8000])

    n.optimize()

    expected_status = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float).T

    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array([[0, 0, 3000, 8000], [3000, 800, 0, 0]], dtype=float).T

    equal(n.c.generators.dynamic.p.values, expected_dispatch)


def test_minimum_down_time_up_time_before():
    """
    This test is based on https://docs.pypsa.org/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """
    n = pypsa.Network()

    n.set_snapshots(range(4))

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        min_down_time=3,
        down_time_before=2,
        up_time_before=0,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=4000,
    )

    n.add("Load", "load", bus="bus", p_set=[3000, 800, 3000, 8000])

    n.optimize()

    expected_status = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float).T

    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array([[0, 0, 3000, 8000], [3000, 800, 0, 0]], dtype=float).T

    equal(n.c.generators.dynamic.p.values, expected_dispatch)


def test_start_up_costs():
    n = pypsa.Network()

    n.snapshots = range(4)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.1,
        up_time_before=0,
        marginal_cost=20,
        start_up_cost=3000,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.0,
        up_time_before=1,
        start_up_cost=1000,
        p_nom=10000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    assert n.objective == 359000


def test_shut_down_costs():
    n = pypsa.Network()

    n.snapshots = range(4)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        shut_down_cost=1000,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        shut_down_cost=1000,
        p_min_pu=0.1,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    assert n.objective == 358000


def test_unit_commitment_rolling_horizon():
    n = pypsa.Network()
    n.snapshots = range(7)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.1,
        up_time_before=1,
        marginal_cost=20,
        min_up_time=2,
        min_down_time=2,
        start_up_cost=10000,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.01,
        start_up_cost=100,
        up_time_before=0,
        p_nom=10000,
    )
    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 800, 5000, 3000, 950, 800])

    n.optimize(snapshots=[0, 1, 2])
    n.optimize(snapshots=[2, 3, 4])
    n.optimize(snapshots=[4, 5, 6])

    expected_status = np.array(
        [[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1]], dtype=float
    ).T
    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array(
        [[4000, 6000, 0, 0, 0, 0, 0], [0, 0, 800, 5000, 3000, 950, 800]]
    ).T

    equal(n.c.generators.dynamic.p.values, expected_dispatch)


def test_linearized_unit_commitment():
    n = pypsa.Network()
    n.snapshots = pd.date_range("2022-01-01", "2022-02-09", freq="d")

    load = np.zeros(len(n.snapshots))
    load[:5] = 5
    load[5:10] = 6
    load[10:15] = 8
    load[15:20] = 10
    load[20:30] = 7
    load[30:40] = 6
    load *= 100

    n.add("Bus", "bus")

    for seed, i in enumerate(range(40), start=1):
        rng = np.random.default_rng(seed)  # Create a random number generator
        p_min_pu = rng.integers(1, 5) / 10
        marginal_cost = rng.integers(1, 11) * 10
        min_up_time = rng.integers(0, 6)
        min_down_time = rng.integers(0, 6)
        p_nom = rng.integers(1, 10) * 5
        start_up_cost = rng.integers(1, 5) * 100

        # the constraint tightening proposed in Baldick et al. depends on start_up_cost
        # and shut_down_cost being equal therefore, we force them to be equal for first
        # 20 generators
        shut_down_cost = rng.integers(1, 5) * 100 if i >= 20 else start_up_cost

        n.add(
            "Generator",
            f"{i}",
            bus="bus",
            committable=True,
            up_time_before=0,
            p_min_pu=p_min_pu,
            marginal_cost=marginal_cost,
            min_up_time=min_up_time,
            min_down_time=min_down_time,
            p_nom=p_nom,
            start_up_cost=start_up_cost,
            shut_down_cost=shut_down_cost,
        )
    n.add("Load", "load", bus="bus", p_set=load)

    n.optimize(linearized_unit_commitment=True)

    MILP_objective = 1241100
    assert round(n.objective / MILP_objective, 2) == 1


def test_link_unit_commitment():
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])

    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        up_time_before=0,
        min_up_time=3,
        start_up_cost=3333,
        p_nom=12000,
    )

    n.add(
        "Generator",
        "wind",
        bus="electricity",
        p_nom=800,
    )

    n.add("Load", "load", bus="electricity", p_set=[4000, 6000, 800, 5000])

    n.optimize()

    expected_status = [1.0, 1.0, 1.0, 1.0]

    equal(n.c.links.dynamic.status["OCGT"].values, expected_status)

    expected_dispatch = [3200.0, 5200.0, 600.0, 4200.0]

    equal(-n.c.links.dynamic.p1["OCGT"].values, expected_dispatch)

    assert round(n.objective, 1) == 267333.0


def test_link_ramp_limits():
    """
    Test that ramp limits work for Links.
    """
    n = pypsa.Network()

    snapshots = range(6)
    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])

    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        p_min_pu=0.1,
        efficiency=0.5,
        p_nom=10000,
        ramp_limit_up=0.3,  # 30% of p_nom per timestep = 3000 MW
        ramp_limit_down=0.4,  # 40% of p_nom per timestep = 4000 MW
        marginal_cost=20,
    )

    n.add("Generator", "backup", bus="electricity", marginal_cost=100, p_nom=10000)

    # Varying load to induce ramping
    n.add("Load", "load", bus="electricity", p_set=[2000, 7000, 1500, 5500, 5000, 2500])

    n.optimize()

    # Check that ramp limits are respected
    # For Links, use p0 (the power at bus0) which is the optimization variable
    p_diff = n.c.links.dynamic.p0["OCGT"].diff()
    max_ramp_up = 0.3 * 10000  # 3000 MW
    max_ramp_down = 0.4 * 10000  # 4000 MW

    # Check ramp up (positive changes)
    ramp_ups = p_diff[p_diff > 0]
    if not ramp_ups.empty:
        assert ramp_ups.max() <= max_ramp_up + 1e-4, (
            f"Ramp up limit violated: {ramp_ups.max()} > {max_ramp_up}"
        )

    # Check ramp down (negative changes)
    ramp_downs = p_diff[p_diff < 0].abs()
    if not ramp_downs.empty:
        assert ramp_downs.max() <= max_ramp_down + 1e-4, (
            f"Ramp down limit violated: {ramp_downs.max()} > {max_ramp_down}"
        )


def test_link_ramp_limits_rolling_horizon():
    """
    Test that ramp limits work for Links in rolling horizon optimization.
    This specifically tests the historical data retrieval for p0 in Links
    when sns[0] != n.snapshots[0].
    """
    n = pypsa.Network()

    snapshots = range(12)
    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])

    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        p_min_pu=0.1,
        efficiency=0.5,
        p_nom=10000,
        ramp_limit_up=0.3,  # 30% of p_nom per timestep = 3000 MW
        ramp_limit_down=0.4,  # 40% of p_nom per timestep = 4000 MW
        marginal_cost=20,
    )

    n.add("Generator", "backup", bus="electricity", marginal_cost=100, p_nom=10000)

    # Varying load to induce ramping with jumps > ramp limits (3000 up, 4000 down)
    n.add(
        "Load",
        "load",
        bus="electricity",
        p_set=[2000, 6000, 1000, 5500, 9000, 3000, 7500, 2500, 7000, 1500, 6000, 2000],
    )

    n.optimize.optimize_with_rolling_horizon(horizon=4, overlap=1)

    # Check that ramp limits are respected across all snapshots
    # For Links, use p0 (the power at bus0)
    p_diff = n.c.links.dynamic.p0["OCGT"].diff()
    max_ramp_up = 0.3 * 10000  # 3000 MW
    max_ramp_down = 0.4 * 10000  # 4000 MW

    # Check ramp up (positive changes)
    ramp_ups = p_diff[p_diff > 0]
    if not ramp_ups.empty:
        assert ramp_ups.max() <= max_ramp_up + 1e-4, (
            f"Ramp up limit violated: {ramp_ups.max()} > {max_ramp_up}"
        )

    # Check ramp down (negative changes)
    ramp_downs = p_diff[p_diff < 0].abs()
    if not ramp_downs.empty:
        assert ramp_downs.max() <= max_ramp_down + 1e-4, (
            f"Ramp down limit violated: {ramp_downs.max()} > {max_ramp_down}"
        )


def test_dynamic_ramp_rates():
    """
    This test checks that dynamic ramp rates are correctly applied when
    considering a unit outage represented by p_max_pu.
    """
    n = pypsa.Network()

    snapshots = range(15)
    n.set_snapshots(snapshots)
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=100)

    # vary marginal price of gen1 to induce ramping
    gen1_marginal = pd.Series(100, index=n.snapshots)
    gen1_marginal[[4, 5, 6, 10, 11, 12]] = 200

    static_ramp_up = 0.8
    static_ramp_down = 1
    p_max_pu = pd.Series(1, index=n.snapshots).astype(float)
    p_max_pu.loc[n.snapshots[0:6]] = 0.5  # 50% capacity outage for 6 periods

    n.add(
        "Generator",
        "gen1",
        bus="bus",
        p_nom=100,
        p_max_pu=p_max_pu,
        ramp_limit_up=static_ramp_up * p_max_pu,
        ramp_limit_down=static_ramp_down * p_max_pu,
        marginal_cost=gen1_marginal,
    )

    n.add("Generator", "gen2", bus="bus", p_nom=100, marginal_cost=150)

    n.optimize()

    assert (n.c.generators.dynamic.p.diff().loc[0:6, "gen1"]).max() <= 0.5 * 80
    assert (n.c.generators.dynamic.p.diff().loc[0:6, "gen1"]).min() >= -0.5 * 100
    assert (n.c.generators.dynamic.p.diff().loc[6:, "gen1"]).max() <= 80
    assert (n.c.generators.dynamic.p.diff().loc[6:, "gen1"]).min() >= -100


@pytest.mark.parametrize("direction", ["up", "down"])
def test_generator_ramp_constraints_mask_nan(direction):
    """
    See https://github.com/PyPSA/PyPSA/issues/1493
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=2, freq="h"))

    n.add("Bus", "bus")
    # Add load with sudden change to trigger ramping up/down
    p_set = [0.0, 50.0] if direction == "up" else [50.0, 10.0]
    n.add(
        "Load",
        "load",
        bus="bus",
        p_set=pd.Series(p_set, index=n.snapshots),
    )

    ramp_kw = {f"ramp_limit_{direction}": 0.5}

    # Generator with ramp limits
    n.add(
        "Generator",
        "gen_limited",
        bus="bus",
        p_nom=10,
        marginal_cost=1,
        **ramp_kw,
    )

    # Generator without ramp limits
    n.add(
        "Generator",
        "gen_unlimited",
        bus="bus",
        p_nom=1000,
        marginal_cost=10,
    )

    n.optimize(solver_name="highs")

    # Check labels to see which generators have active ramp constraints applied
    # Generator with undefined ramp limits should have label -1 (no constraint)
    # Limited generators should not have label -1 except for the very first snapshot
    key = f"Generator-p-ramp_limit_{direction}"
    constraints = n.model.constraints[key]
    labels_gen_limited = constraints.data["labels"].sel(name="gen_limited").to_pandas()
    assert (labels_gen_limited.loc[n.snapshots[1:]] != -1).all(), (
        f"ramp_{direction} constraint should be active for 'gen_limited'."
    )

    labels_gen_unlimited = (
        constraints.data["labels"].sel(name="gen_unlimited").to_pandas()
    )
    assert (labels_gen_unlimited == -1).all(), (
        f"ramp_{direction} constraint should be masked for 'gen_unlimited'."
    )


@pytest.mark.parametrize("direction", ["up", "down"])
def test_link_ramp_constraints_mask_nan(direction):
    """
    See https://github.com/PyPSA/PyPSA/issues/1493
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=2, freq="h"))

    n.add("Bus", "bus0")
    n.add("Bus", "bus1")

    # Add load with sudden change to trigger ramping up/down
    p_set = [0.0, 50.0] if direction == "up" else [50.0, 10.0]
    n.add(
        "Load",
        "load",
        bus="bus1",
        p_set=pd.Series(p_set, index=n.snapshots),
    )

    ramp_kw = {
        f"ramp_limit_{direction}": 0.5,
    }

    # Link with ramp limits
    n.add(
        "Link",
        "link_limited",
        bus0="bus0",
        bus1="bus1",
        p_nom=10,
        efficiency=1.0,
        marginal_cost=1,
        **ramp_kw,
    )

    # Link without ramp limits
    n.add(
        "Link",
        "link_unlimited",
        bus0="bus0",
        bus1="bus1",
        p_nom=1000,
        efficiency=1.0,
        marginal_cost=10,
    )

    n.add(
        "Generator",
        "generator",
        bus="bus0",
        p_nom=1000,
        marginal_cost=0.0,
    )

    n.optimize(solver_name="highs")

    # Check labels to see which links have active ramp constraints applied
    # Link with undefined ramp limits should have label -1 (no constraint)
    key = f"Link-p-ramp_limit_{direction}"
    constraints = n.model.constraints[key]
    labels_link_limited = (
        constraints.data["labels"].sel(name="link_limited").to_pandas()
    )
    assert (labels_link_limited.loc[n.snapshots[1:]] != -1).all(), (
        f"ramp_{direction} constraint should be active for 'link_limited'."
    )

    labels_link_unlimited = (
        constraints.data["labels"].sel(name="link_unlimited").to_pandas()
    )
    assert (labels_link_unlimited == -1).all(), (
        f"ramp_{direction} constraint should be masked for 'link_unlimited'."
    )


def test_dynamic_start_up_rates_for_commitables():
    """
    This test checks that start up ramp rate constraints within unit commitment functionality runs through and is considered correctly.
    """
    n = pypsa.Network()

    snapshots = range(15)
    n.set_snapshots(snapshots)
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=100)

    # vary marginal price of gen1 to induce ramping
    gen1_marginal = pd.Series(100, index=n.snapshots)
    gen1_marginal[[4, 5, 6, 10, 11, 12]] = 200

    n.add(
        "Generator",
        "gen1",
        bus="bus",
        p_nom=100,
        committable=True,
        p_min_pu=0.3,
        p_max_pu=1,
        ramp_limit_up=1,
        ramp_limit_start_up=0.3,
        ramp_limit_shut_down=1,
        start_up_cost=10,
        shut_down_cost=10,
        marginal_cost=gen1_marginal,
    )

    n.add("Generator", "gen2", bus="bus", p_nom=100, marginal_cost=150)

    status, _ = n.optimize(snapshots=n.snapshots)

    assert status == "ok"

    # Check that ramp_limit_start_up constraint is respected
    gen1_status = n.c.generators.dynamic.status["gen1"]
    gen1_p = n.c.generators.dynamic.p["gen1"]

    # Find startup events (status changes from 0 to 1)
    startup_snapshots = gen1_status[
        (gen1_status == 1) & (gen1_status.shift(1) == 0)
    ].index

    for snapshot in startup_snapshots:
        expected_max_startup = 0.3 * 100  # ramp_limit_start_up * p_nom
        assert gen1_p[snapshot] <= expected_max_startup, (
            f"Startup ramp limit violated at snapshot {snapshot}: {gen1_p[snapshot]} > {expected_max_startup}"
        )


def test_ramp_limits_multi_investment_period():
    n = pypsa.Network()
    sns = pd.date_range("2025-01-01", periods=5, freq="h").append(
        pd.date_range("2026-01-01", periods=5, freq="h")
    )
    n.set_snapshots(pd.MultiIndex.from_arrays([sns.year, sns]))
    n.investment_periods = [2025, 2026]
    n.investment_period_weightings["years"] = 5

    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom_extendable=True,
        ramp_limit_up=0.5,
        marginal_cost=10,
    )
    n.add("Load", "load", bus="bus", p_set=100)

    status, _ = n.optimize()
    assert status == "ok"


def test_committable_start_up_only_ramp_limit():
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus")

    load_profile = [0, 50, 100, 100, 50, 0]
    n.add("Load", "load", bus="bus", p_set=load_profile)

    n.add(
        "Generator",
        "gen_commit",
        bus="bus",
        p_nom=100,
        committable=True,
        p_min_pu=0.1,
        ramp_limit_start_up=0.4,
        marginal_cost=10,
    )
    n.add("Generator", "gen_backup", bus="bus", p_nom=200, marginal_cost=100)

    status, _ = n.optimize()
    assert status == "ok"

    gen_status = n.c.generators.dynamic.status["gen_commit"]
    gen_p = n.c.generators.dynamic.p["gen_commit"]
    startup_snapshots = gen_status[(gen_status == 1) & (gen_status.shift(1) == 0)].index

    for snapshot in startup_snapshots:
        expected_max = 0.4 * 100
        assert gen_p[snapshot] <= expected_max + 1e-5


def test_committable_shut_down_only_ramp_limit():
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus")

    load_profile = [100, 100, 50, 20, 10, 0]
    n.add("Load", "load", bus="bus", p_set=load_profile)

    n.add(
        "Generator",
        "gen_commit",
        bus="bus",
        p_nom=100,
        committable=True,
        p_min_pu=0.1,
        up_time_before=1,
        ramp_limit_shut_down=0.3,
        marginal_cost=10,
    )
    n.add("Generator", "gen_backup", bus="bus", p_nom=200, marginal_cost=100)

    status, _ = n.optimize()
    assert status == "ok"

    gen_status = n.c.generators.dynamic.status["gen_commit"]
    gen_p = n.c.generators.dynamic.p["gen_commit"]
    shutdown_snapshots = gen_status[
        (gen_status == 0) & (gen_status.shift(1) == 1)
    ].index

    for snapshot in shutdown_snapshots:
        prev_snapshot = snapshot - 1
        if prev_snapshot >= 0:
            expected_max_prev = 0.3 * 100
            assert gen_p[prev_snapshot] <= expected_max_prev + 1e-5


def test_committable_ramp_limits_multi_investment_period():
    n = pypsa.Network()
    sns = pd.date_range("2025-01-01", periods=5, freq="h").append(
        pd.date_range("2026-01-01", periods=5, freq="h")
    )
    n.set_snapshots(pd.MultiIndex.from_arrays([sns.year, sns]))
    n.investment_periods = [2025, 2026]
    n.investment_period_weightings["years"] = 5

    n.add("Bus", "bus")

    load_profile = [50, 80, 100, 80, 50, 50, 80, 100, 80, 50]
    n.add("Load", "load", bus="bus", p_set=load_profile)

    n.add(
        "Generator",
        "gen_commit",
        bus="bus",
        p_nom=100,
        committable=True,
        p_min_pu=0.3,
        ramp_limit_up=0.5,
        ramp_limit_start_up=0.4,
        ramp_limit_down=0.5,
        ramp_limit_shut_down=0.3,
        marginal_cost=10,
    )
    n.add("Generator", "gen_backup", bus="bus", p_nom=200, marginal_cost=100)

    status, _ = n.optimize()
    assert status == "ok"

    gen_p = n.c.generators.dynamic.p["gen_commit"]
    gen_status = n.c.generators.dynamic.status["gen_commit"]

    for i in range(1, len(n.snapshots)):
        curr_sns = n.snapshots[i]
        prev_sns = n.snapshots[i - 1]
        curr_status = gen_status[curr_sns]
        prev_status = gen_status[prev_sns]
        curr_p = gen_p[curr_sns]
        prev_p = gen_p[prev_sns]

        if curr_status == 1 and prev_status == 1:
            assert curr_p - prev_p <= 0.5 * 100 + 1e-5
            assert prev_p - curr_p <= 0.5 * 100 + 1e-5


def test_ramp_limit_start_up_binary_uc():
    """
    Test that ramp_limit_start_up parameter works correctly in binary unit commitment.
    """
    n = pypsa.Network()
    n.set_snapshots(range(4))

    n.add("Bus", ["gas", "electricity"])
    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    # Committable link with ramp_limit_start_up set, but ramp_limit_up is NaN
    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        ramp_limit_start_up=0.4,  # 40% of p_nom on start-up
        p_nom=10000,
        up_time_before=0,  # Starts from OFF state
        start_up_cost=3333,
    )

    # Backstop generator to ensure feasibility
    n.add(
        "Generator",
        "expensive_backstop",
        bus="electricity",
        p_nom=5000,
        marginal_cost=1e5,
    )

    n.add("Load", "load", bus="electricity", p_set=[4000, 5000, 2000, 5000])

    status, condition = n.optimize()

    assert status == "ok", f"Optimization failed with status {status}"

    # Get the link output (convert p1 to positive power output)
    link_output = -n.c.links.dynamic.p1.loc[:, "OCGT"].values

    # Expected ramp limit on start-up: 0.4 * 10000 = 4000 MW
    max_start_up_output = 0.4 * 10000

    # First snapshot: unit starts from OFF, should respect ramp_limit_start_up
    assert link_output[0] <= max_start_up_output * 1.01, (
        f"First snapshot output {link_output[0]} exceeds start-up ramp limit {max_start_up_output}"
    )

    # Verify unit is committed (status = 1)
    assert n.c.links.dynamic.status.loc[0, "OCGT"] == 1, (
        "Link should be committed at first snapshot"
    )

    # Verify start_up variable catches the start-up event
    assert n.c.links.dynamic.start_up.loc[0, "OCGT"] == 1, (
        "Link should show start-up at first snapshot"
    )


def test_infeasible_start_up_limit():
    """Test that introduction of infeasible start-up limit results in infeasible solution"""
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])

    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        min_up_time=3,
        start_up_cost=3333,
        p_nom=12000,
        up_time_before=0,
    )

    n.add(
        "Generator",
        "wind",
        bus="electricity",
        p_nom=800,
    )

    n.add("Load", "load", bus="electricity", p_set=[4000, 6000, 800, 5000])

    status, condition = n.optimize()
    assert status == "ok"

    n.c.links.static.loc["OCGT", "ramp_limit_start_up"] = 0.01

    status, condition = n.optimize()
    assert status == "warning"
    assert condition == "infeasible"


def test_ramp_limit_shut_down_binary_uc():
    """
    Test that ramp_limit_shut_down parameter works correctly in binary unit commitment (excluding first period as special case).
    """
    n = pypsa.Network()
    snapshots = range(4)
    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])
    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        ramp_limit_start_up=0.4,
        start_up_cost=3333,
        p_nom=10000,
        status=1,
        ramp_limit_shut_down=0.1,  # 10% of p_nom on shut-down
    )

    n.add(
        "Generator",
        "expensive_backstop",
        bus="electricity",
        p_nom=5000,
        marginal_cost=1e5,
    )

    n.add(
        "Generator",
        "slack_dump",
        bus="electricity",
        p_nom=20000,
        marginal_cost=1e6,
        sign=-1,
    )

    n.add("Load", "load", bus="electricity", p_set=[4000, 5000, 2000, 0])

    status, condition = n.optimize()

    assert status == "ok", f"Optimization failed with status {status}"

    # Get generator outputs
    gas_output = n.c.generators.dynamic.p.loc[:, "gas"].values
    backstop_output = n.c.generators.dynamic.p.loc[:, "expensive_backstop"].values

    # Expected pattern based on ramp_limit_shut_down constraint
    # Snapshot 0: 9000.0 MW (gas), 0.0 MW (backstop), 500.0 MW (slack)
    # Snapshot 1: 10000.0 MW (gas), 0.0 MW (backstop), 0.0 MW (slack)
    # Snapshot 2: 1000.0 MW (gas), 1500.0 MW (backstop), 0.0 MW (slack)
    # Snapshot 3: 0.0 MW (gas), 0.0 MW (backstop), 0.0 MW (slack)

    assert abs(gas_output[1] - 10000.0) < 1e-3, (
        f"Snapshot 1: Expected gas output 10000.0, got {gas_output[1]}"
    )
    assert abs(backstop_output[1] - 0.0) < 1e-3, (
        f"Snapshot 1: Expected backstop output 0.0, got {backstop_output[1]}"
    )

    assert abs(gas_output[2] - 1000.0) < 1e-3, (
        f"Snapshot 2: Expected gas output 1000.0, got {gas_output[2]}"
    )
    assert abs(backstop_output[2] - 1500.0) < 1e-3, (
        f"Snapshot 2: Expected backstop output 1500.0, got {backstop_output[2]}"
    )

    assert abs(gas_output[3] - 0.0) < 1e-3, (
        f"Snapshot 3: Expected gas output 0.0, got {gas_output[3]}"
    )
    assert abs(backstop_output[3] - 0.0) < 1e-3, (
        f"Snapshot 3: Expected backstop output 0.0, got {backstop_output[3]}"
    )


def test_ramp_limit_shut_down_first_snapshot_with_slack():
    """
    Test that ramp_limit_down and ramp_limit_shut_down constrain dispatch
    at snapshot 0 when unit starts ON at full capacity with zero demand.
    """
    n = pypsa.Network()
    n.set_snapshots(range(4))

    n.add("Bus", ["gas", "electricity"])
    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        ramp_limit_start_up=0.4,
        start_up_cost=3333,
        p_nom=10000,
        p_init=10000,
        ramp_limit_down=0.1,
        ramp_limit_shut_down=0.1,
    )

    n.add(
        "Generator",
        "expensive_backstop",
        bus="electricity",
        p_nom=5000,
        marginal_cost=1e5,
    )
    n.add(
        "Generator",
        "slack_dump",
        bus="electricity",
        p_nom=20000,
        marginal_cost=1e6,
        sign=-1,
    )
    n.add("Load", "load", bus="electricity", p_set=[0, 5000, 2000, 0])

    status, condition = n.optimize()

    assert status == "ok", f"Optimization failed with status {status}"

    gas_output = n.c.generators.dynamic.p.loc[:, "gas"].values
    slack_output = n.c.generators.dynamic.p.loc[:, "slack_dump"].values

    assert abs(gas_output[0] - 9000.0) < 1e-3, (
        f"Snapshot 0: Expected gas output 9000.0, got {gas_output[0]}"
    )
    assert abs(slack_output[0] - 4500.0) < 1e-3, (
        f"Snapshot 0: Expected slack output 4500.0, got {slack_output[0]}"
    )
