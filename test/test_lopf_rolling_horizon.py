#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:08:25 2022.

@author: fabian
"""
import numpy as np
import pytest
from conftest import SUPPORTED_APIS, optimize

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


@pytest.mark.parametrize("api", SUPPORTED_APIS)
@pytest.mark.parametrize("committable", [True, False])
def test_rolling_horizon(api, committable):
    n = get_network(committable)
    # now rolling horizon
    for sns in np.array_split(n.snapshots, 4):
        status, condition = optimize(n, api, sns)
        assert status == "ok"

    ramping = n.generators_t.p.diff().fillna(0)
    assert (ramping <= n.generators.eval("ramp_limit_up * p_nom_opt")).all().all()
    assert (ramping >= -n.generators.eval("ramp_limit_down * p_nom_opt")).all().all()


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

    n.optimize.optimize_with_rolling_horizon(horizon=3, solver_name="glpk")
    ramping = n.generators_t.p.diff().fillna(0)
    assert (ramping <= n.generators.eval("ramp_limit_up * p_nom_opt")).all().all()
    assert (ramping >= -n.generators.eval("ramp_limit_down * p_nom_opt")).all().all()


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
        n.optimize.optimize_with_rolling_horizon(
            horizon=1, overlap=2, solver_name="glpk"
        )

    n.optimize.optimize_with_rolling_horizon(horizon=3, overlap=1, solver_name="glpk")
    ramping = n.generators_t.p.diff().fillna(0)
    assert (ramping <= n.generators.eval("ramp_limit_up * p_nom_opt")).all().all()
    assert (ramping >= -n.generators.eval("ramp_limit_down * p_nom_opt")).all().all()
