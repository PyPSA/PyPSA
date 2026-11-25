# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest
from linopy import available_solvers

import pypsa


@pytest.mark.parametrize("active_quadratic_cost", [0.0, 0.1])
@pytest.mark.parametrize("inactive_quadratic_cost", [0.0, 1.0])
def test_quadratic_costs_ignore_inactive_generators(
    active_quadratic_cost, inactive_quadratic_cost
):
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=10)
    n.add(
        "Generator",
        "active",
        bus="bus",
        p_nom=20,
        marginal_cost=1,
        marginal_cost_quadratic=active_quadratic_cost,
    )
    n.add(
        "Generator",
        "inactive",
        bus="bus",
        p_nom=20,
        marginal_cost=2,
        marginal_cost_quadratic=inactive_quadratic_cost,
        active=False,
    )

    status, condition = n.optimize(solver_name="highs")

    assert (status, condition) == ("ok", "optimal")
    assert n.objective == pytest.approx(2 * (10 + active_quadratic_cost * 100))
    assert "inactive" not in n.model["Generator-p"].indexes["name"]


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
def test_optimize_quadratic(ac_dc_network):
    n = ac_dc_network

    status, _ = n.optimize(solver_name="gurobi")

    assert status == "ok"

    gas_i = n.c.generators.static.index[n.c.generators.static.carrier == "gas"]

    objective_linear = n.objective

    # quadratic costs
    n.c.generators.static.loc[gas_i, "marginal_cost_quadratic"] = 2
    status, _ = n.optimize(solver_name="gurobi")

    assert status == "ok"
    assert n.objective > objective_linear
