# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import pypsa


@pytest.fixture
def n():
    n = pypsa.Network()
    n.set_snapshots([0])

    n.add("Bus", "electricity")
    n.add("Bus", "hydrogen")

    n.add("Generator", "gen", bus="electricity", p_nom=100, marginal_cost=10)
    n.add("Load", "load", bus="electricity", p_set=20)

    n.add(
        "Process",
        "electrolyser",
        bus0="electricity",
        bus1="hydrogen",
        rate0=-1,
        rate1=0.8,
        p_nom=50,
    )
    n.add("Load", "h2_demand", bus="hydrogen", p_set=8)

    return n


def test_process_optimization_sign_convention(n):
    n.optimize()

    p = n.c.processes.dynamic["p"].loc[0, "electrolyser"]
    p0 = n.c.processes.dynamic["p0"].loc[0, "electrolyser"]
    p1 = n.c.processes.dynamic["p1"].loc[0, "electrolyser"]

    assert p > 0, f"Internal power p should be positive, got {p}"
    assert p0 > 0, f"p0 should be positive (withdrawing from bus0), got {p0}"
    assert p1 < 0, f"p1 should be negative (injecting into bus1), got {p1}"

    np.testing.assert_allclose(p0, p, rtol=1e-5)
    np.testing.assert_allclose(p1, -p * 0.8, rtol=1e-5)


def test_process_pf_sign_convention(n):
    n.c.processes.static.loc["electrolyser", "p_set"] = 10.0

    n.lpf()

    p0 = n.c.processes.dynamic["p0"].loc[0, "electrolyser"]
    p1 = n.c.processes.dynamic["p1"].loc[0, "electrolyser"]

    assert p0 > 0, f"p0 should be positive (withdrawing from bus0), got {p0}"
    assert p1 < 0, f"p1 should be negative (injecting into bus1), got {p1}"


def test_process_optimize_then_pf_consistency(n):
    n.optimize()

    opt_p = n.c.processes.dynamic["p"].loc[0, "electrolyser"]
    opt_p0 = n.c.processes.dynamic["p0"].loc[0, "electrolyser"]
    opt_p1 = n.c.processes.dynamic["p1"].loc[0, "electrolyser"]

    n.c.processes.static.loc["electrolyser", "p_set"] = opt_p
    n.lpf()

    pf_p0 = n.c.processes.dynamic["p0"].loc[0, "electrolyser"]
    pf_p1 = n.c.processes.dynamic["p1"].loc[0, "electrolyser"]

    np.testing.assert_allclose(
        pf_p0,
        opt_p0,
        rtol=1e-5,
        err_msg=f"p0 mismatch: optimization={opt_p0}, power_flow={pf_p0}",
    )
    np.testing.assert_allclose(
        pf_p1,
        opt_p1,
        rtol=1e-5,
        err_msg=f"p1 mismatch: optimization={opt_p1}, power_flow={pf_p1}",
    )
