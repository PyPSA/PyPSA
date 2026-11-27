# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest
from numpy.testing import assert_allclose


def test_optimize_with_power_flow(scipy_network):
    n = scipy_network

    res = n.optimize.optimize_and_run_non_linear_powerflow(snapshots=n.snapshots[:4])

    assert res["status"] == "ok"
    assert res["converged"].all().all()


def test_fix_optimal_dispatch_reproduces_flows(ac_dc_network):
    """A linear power flow on the fixed optimal dispatch reproduces the optimal flows."""
    n = ac_dc_network
    n.optimize()
    lines = n.c.lines.dynamic.p0.copy()
    links = n.c.links.dynamic.p0.copy()

    n.optimize.fix_optimal_dispatch()
    n.lpf()

    assert_allclose(n.c.lines.dynamic.p0, lines, atol=1e-6)
    assert_allclose(n.c.links.dynamic.p0, links, atol=1e-6)


def test_non_linear_power_flow_refuses_dc_sub_network(ac_dc_network):
    n = ac_dc_network

    with pytest.raises(NotImplementedError, match="only AC sub-networks"):
        n.pf(n.snapshots[:1])
