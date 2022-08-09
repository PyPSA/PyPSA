#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:08:25 2022.

@author: fabian
"""

import numpy as np
import pytest

import pypsa

solver_name = "glpk"


@pytest.mark.parametrize("pyomo", [True, False])
@pytest.mark.parametrize("committable", [True, False])
def test_rolling_horizon(pyomo, committable):
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

    # now rolling horizon
    for sns in np.array_split(n.snapshots, 4):
        status, condition = n.lopf(sns, solver_name=solver_name, pyomo=pyomo)
        assert status == "ok"

    ramping = n.generators_t.p.diff().fillna(0)
    assert (ramping <= n.generators.eval("ramp_limit_up * p_nom_opt")).all().all()
    assert (ramping >= -n.generators.eval("ramp_limit_down * p_nom_opt")).all().all()
