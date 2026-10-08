# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

import pypsa


def assert_ramp_limits_respected(n, tol=0.0):
    ramping = n.c.generators.dynamic.p.diff().fillna(0)
    static = n.c.generators.static
    assert (ramping <= static.eval("ramp_limit_up * p_nom_opt") + tol).all().all()
    assert (ramping >= -static.eval("ramp_limit_down * p_nom_opt") - tol).all().all()


def get_network(committable):
    n = pypsa.Network(snapshots=range(12))

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        ramp_limit_up=0.1,
        ramp_limit_down=0.3,
        marginal_cost=20,
        capital_cost=200,
        p_nom=1000,
        committable=committable,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        ramp_limit_up=0.5,
        ramp_limit_down=0.5,
        marginal_cost=40,
        capital_cost=200,
        p_nom=1000,
        committable=committable,
    )

    n.add("Load", "load", bus="bus", p_set=[400, 600, 500, 800] * 3)

    return n


@pytest.mark.parametrize("committable", [True, False])
def test_rolling_horizon(committable):
    n = get_network(committable)
    # now rolling horizon
    for sns in np.array_split(n.snapshots, 4):
        status, condition = n.optimize(snapshots=sns)
        assert status == "ok"

    assert_ramp_limits_respected(n)


@pytest.mark.parametrize("committable", [True, False])
def test_rolling_horizon_integrated(committable):
    n = get_network(committable)
    n.add(
        "StorageUnit",
        "storage",
        bus="bus",
        p_nom=100,
        p_nom_extendable=False,
        marginal_cost=10,
    )

    n.optimize.optimize_with_rolling_horizon(horizon=3)
    assert_ramp_limits_respected(n)


def test_rolling_horizon_integrated_overlap():
    n = get_network(committable=True)
    n.add(
        "StorageUnit",
        "storage",
        bus="bus",
        p_nom=100,
        p_nom_extendable=False,
        marginal_cost=10,
    )

    with pytest.raises(ValueError):
        n.optimize.optimize_with_rolling_horizon(horizon=1, overlap=2)

    n.optimize.optimize_with_rolling_horizon(horizon=3, overlap=1)
    assert_ramp_limits_respected(n)


def test_rolling_horizon_committable_ramp_limits():
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
        p_nom=10000,
        ramp_limit_up=0.5,
        ramp_limit_start_up=0.1,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=False,
        marginal_cost=70,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[1500, 5000, 5000, 800])

    n.optimize.optimize_with_rolling_horizon(
        linearized_unit_commitment=True,
        horizon=2,
    )

    # Check dispatch for the first two snapshots against expected values
    assert n.generators_t.p.loc[0, "coal"] == 1500.0
    assert (
        n.generators_t.p.loc[0, "gas"] == 0.0 or n.generators_t.p.loc[0, "gas"] == -0.0
    )

    assert n.generators_t.p.loc[1, "coal"] == 4500.0
    assert n.generators_t.p.loc[1, "gas"] == 500.0


def test_rolling_horizon_committable_overlap_matches_full_run():
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
        p_nom=10000,
        ramp_limit_up=0.5,
        ramp_limit_down=0.5,
        ramp_limit_start_up=0.1,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        p_min_pu=0.0,
        marginal_cost=70,
        p_nom=2000,
        ramp_limit_up=0.8,
        ramp_limit_down=0.8,
        ramp_limit_start_up=0.2,
    )

    n.add("Load", "load", bus="bus", p_set=[1500, 5000, 5000, 800])

    # Full-horizon reference solution
    status_full, cond_full = n.optimize(
        snapshots=n.snapshots,
        linearized_unit_commitment=True,
    )
    assert status_full == "ok", (
        f"Full-horizon optimization failed with status {status_full}, "
        f"condition {cond_full}"
    )
    p_full = n.generators_t.p.copy()

    # Rebuild the same network for rolling horizon
    n.model.solver_model = None
    n_rh = n.copy()

    n_rh.optimize.optimize_with_rolling_horizon(
        linearized_unit_commitment=True,
        horizon=2,
        overlap=1,
    )

    p_rh = n_rh.generators_t.p

    # Dispatch trajectory should match full-horizon run for all snapshots
    assert p_rh.equals(p_full)

    assert_ramp_limits_respected(n_rh)


@pytest.mark.parametrize(
    ("gas_kwargs", "gas_ramp", "load", "check_ramp"),
    [
        pytest.param({"p_nom": 1000}, 0.5, [1000] * 6, True, id="fixed"),
        pytest.param(
            {"p_nom_extendable": True, "capital_cost": 100},
            0.3,
            [400, 450, 800, 820, 900, 880],
            False,
            id="extendable",
        ),
    ],
)
def test_rolling_horizon_noncommittable_ramp_at_seam(
    gas_kwargs, gas_ramp, load, check_ramp
):
    """Regression test for issue #1644.

    Non-committable generators carry no commitment status at a rolling-horizon
    seam. Reading ``status=0`` for them corrupted their seam ramp terms (the
    start-up/shut-down terms for fixed capacity, the capacity-relative term for
    extendable capacity), pinning dispatch at the seam and turning the window
    spuriously infeasible.
    """
    n = pypsa.Network(snapshots=range(6))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_nom=500,
        p_min_pu=0.3,
        marginal_cost=20,
        ramp_limit_up=0.1,
        ramp_limit_down=0.1,
        ramp_limit_start_up=0.4,
        ramp_limit_shut_down=1.0,
    )
    n.add(
        "Generator",
        "gas",
        bus="bus",
        marginal_cost=40,
        ramp_limit_up=gas_ramp,
        ramp_limit_down=gas_ramp,
        **gas_kwargs,
    )
    n.add("Load", "load", bus="bus", p_set=load)

    n.optimize.optimize_with_rolling_horizon(linearized_unit_commitment=True, horizon=2)

    supply = n.c.generators.dynamic.p.sum(axis=1)
    demand = n.c.loads.dynamic.p_set.sum(axis=1)
    assert np.allclose(supply, demand)

    if check_ramp:
        assert_ramp_limits_respected(n, tol=1e-5)


def test_rolling_horizon_linearized_uc_with_ramp_limits():
    """
    Test rolling horizon with linearized UC and ramp limits on committables.

    Regression test for bug in issue #1454 where coordinate indexing in ramp limit constraints caused KeyError when using rolling horizon optimization with linearized unit commitment and ramp limits defined for committable generators.
    """
    n = pypsa.examples.scigrid_de()

    # Only first 4 snapshots for fast testing
    n.set_snapshots(n.snapshots[:4])

    # Set up a subset of committable generators with ramp limits
    disp = ["Gas", "Hard Coal", "Brown Coal", "Nuclear"]
    committable_mask = n.c.generators.static.carrier.isin(disp)
    n.c.generators.static.loc[committable_mask, "committable"] = True
    n.c.generators.static.loc[committable_mask, "ramp_limit_up"] = 0.5
    n.c.generators.static.loc[committable_mask, "ramp_limit_down"] = 0.5
    n.c.generators.static.loc[committable_mask, "ramp_limit_start_up"] = 0.5

    # This should complete without KeyError
    n.optimize.optimize_with_rolling_horizon(linearized_unit_commitment=True, horizon=2)

    # Lazy check for optimization going through
    assert n.objective > 0

    # Check ramping limits are respected for committable generators
    committable_gens = n.c.generators.static.index[committable_mask]
    ramping = n.c.generators.dynamic.p[committable_gens].diff().fillna(0)
    static = n.c.generators.static.loc[committable_gens]
    ramp_limits = static.eval("ramp_limit_up * p_nom_opt")
    assert (ramping.values <= ramp_limits.values[None, :] + 1e-5).all()


UNIT = {"Generator": {"bus": "bus"}, "Link": {"bus0": "fuel", "bus1": "bus"}}
UNIT["Process"] = UNIT["Link"]


@pytest.mark.parametrize(
    ("min_time", "horizon", "overlap"), [(4, 24, 8), (6, 4, 1)], ids=["long", "short"]
)
@pytest.mark.parametrize("linearized", [False, True], ids=["milp", "linearized"])
@pytest.mark.parametrize(
    ("component", "periods"),
    [("Generator", None), ("Link", None), ("Process", None), ("Generator", [1, 2])],
    ids=["Generator", "Link", "Process", "multi-period"],
)
def test_rolling_horizon_unit_commitment_seams(
    linearized, min_time, horizon, overlap, component, periods
):
    """Regression test for issue #1905.

    Start-ups, shut-downs and minimum up/down times stay consistent with the
    commitment status across window seams, also for fractional statuses.
    """
    n = pypsa.Network(snapshots=range(48))
    if periods:
        n.set_snapshots(pd.MultiIndex.from_product([periods, range(24)]))
        n.set_investment_periods(periods)
    n.add("Bus", ["bus", "fuel"])
    n.add("Generator", "supply", bus="fuel", p_nom=1000)
    n.add(
        component,
        "gen",
        **UNIT[component],
        marginal_cost=50,
        p_nom=100,
        p_min_pu=0.3,
        committable=True,
        min_up_time=min_time,
        min_down_time=min_time,
        start_up_cost=10000,
        shut_down_cost=5000,
        up_time_before=0,
        down_time_before=10,
    )
    availability = [0.8 if i % 24 < 12 else 0.3 for i in range(48)]
    n.add("Generator", "renewable", bus="bus", p_nom=80, p_max_pu=availability)
    load = [55.0] * 16 + [70.0] * 8 + [45.0, 55.0] * 4 + [35.0] * 16
    n.add("Load", "load", bus="bus", p_set=load)

    n.optimize.optimize_with_rolling_horizon(
        linearized_unit_commitment=linearized, horizon=horizon, overlap=overlap
    )

    dynamic = n.c[component].dynamic
    status = dynamic.status["gen"]
    start_up = dynamic.start_up["gen"]
    shut_down = dynamic.shut_down["gen"]
    switch = status.diff().fillna(status.iloc[0])
    assert np.allclose(start_up, switch.clip(lower=0), atol=1e-6)
    assert np.allclose(shut_down, (-switch).clip(lower=0), atol=1e-6)
    assert (start_up.rolling(min_time, min_periods=1).sum() <= status + 1e-6).all()
    assert (shut_down.rolling(min_time, min_periods=1).sum() <= 1 - status + 1e-6).all()
    assert shut_down.sum() > 0
