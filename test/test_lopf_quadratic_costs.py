# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest
from linopy import available_solvers


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
