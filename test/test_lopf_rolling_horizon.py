# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import pypsa


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

    ramping = n.c.generators.dynamic.p.diff().fillna(0)
    assert (
        (ramping <= n.c.generators.static.eval("ramp_limit_up * p_nom_opt")).all().all()
    )
    assert (
        (ramping >= -n.c.generators.static.eval("ramp_limit_down * p_nom_opt"))
        .all()
        .all()
    )


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
    ramping = n.c.generators.dynamic.p.diff().fillna(0)
    assert (
        (ramping <= n.c.generators.static.eval("ramp_limit_up * p_nom_opt")).all().all()
    )
    assert (
        (ramping >= -n.c.generators.static.eval("ramp_limit_down * p_nom_opt"))
        .all()
        .all()
    )


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
    ramping = n.c.generators.dynamic.p.diff().fillna(0)
    assert (
        (ramping <= n.c.generators.static.eval("ramp_limit_up * p_nom_opt")).all().all()
    )
    assert (
        (ramping >= -n.c.generators.static.eval("ramp_limit_down * p_nom_opt"))
        .all()
        .all()
    )


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

    # Check ramping limits are respected
    ramping = n_rh.c.generators.dynamic.p.diff().fillna(0)
    static = n_rh.c.generators.static
    assert (ramping <= static.eval("ramp_limit_up * p_nom_opt")).all().all()
    assert (ramping >= -static.eval("ramp_limit_down * p_nom_opt")).all().all()


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
