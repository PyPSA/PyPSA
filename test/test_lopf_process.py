# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import pypsa
from pypsa.descriptors import nominal_attrs


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


@pytest.fixture
def n_solved(n):
    n.optimize()
    return n


@pytest.fixture
def n_multiport():
    n = pypsa.Network()
    n.set_snapshots([0])

    n.add("Bus", "electricity")
    n.add("Bus", "hydrogen")
    n.add("Bus", "heat")

    n.add("Generator", "gen", bus="electricity", p_nom=100, marginal_cost=10)
    n.add("Load", "h2_demand", bus="hydrogen", p_set=7)
    n.add("Load", "heat_demand", bus="heat", p_set=2)

    n.add(
        "Process",
        "electrolyser",
        bus0="electricity",
        bus1="hydrogen",
        bus2="heat",
        rate0=-1,
        rate1=0.7,
        rate2=0.2,
        p_nom=50,
    )

    return n


@pytest.fixture
def n_multiport_solved(n_multiport):
    n_multiport.optimize()
    return n_multiport


def test_process_component_basics(n):
    assert "Process" in n.all_components
    assert "Process" in n.branch_components
    assert "Process" in n.controllable_branch_components
    assert "Process" not in n.passive_branch_components

    assert "electrolyser" in n.c.processes.static.index
    assert "0" in n.c.processes.ports
    assert "1" in n.c.processes.ports

    assert "Process" in nominal_attrs
    assert nominal_attrs["Process"] == "p_nom"

    assert ("Process", "electrolyser") in n.branches().index


def test_process_optimization_sign_convention(n_solved):
    p = n_solved.c.processes.dynamic["p"].loc[0, "electrolyser"]
    p0 = n_solved.c.processes.dynamic["p0"].loc[0, "electrolyser"]
    p1 = n_solved.c.processes.dynamic["p1"].loc[0, "electrolyser"]

    assert p > 0
    assert p0 > 0
    assert p1 < 0

    np.testing.assert_allclose(p0, p, rtol=1e-5)
    np.testing.assert_allclose(p1, -p * 0.8, rtol=1e-5)


def test_process_pf_sign_convention(n):
    n.c.processes.static.loc["electrolyser", "p_set"] = 10.0
    n.lpf()

    p0 = n.c.processes.dynamic["p0"].loc[0, "electrolyser"]
    p1 = n.c.processes.dynamic["p1"].loc[0, "electrolyser"]

    assert p0 > 0
    assert p1 < 0


def test_process_optimize_then_pf_consistency(n_solved):
    opt_p = n_solved.c.processes.dynamic["p"].loc[0, "electrolyser"]
    opt_p0 = n_solved.c.processes.dynamic["p0"].loc[0, "electrolyser"]
    opt_p1 = n_solved.c.processes.dynamic["p1"].loc[0, "electrolyser"]

    n_solved.c.processes.static.loc["electrolyser", "p_set"] = opt_p
    n_solved.lpf()

    np.testing.assert_allclose(
        n_solved.c.processes.dynamic["p0"].loc[0, "electrolyser"], opt_p0, rtol=1e-5
    )
    np.testing.assert_allclose(
        n_solved.c.processes.dynamic["p1"].loc[0, "electrolyser"], opt_p1, rtol=1e-5
    )


def test_process_statistics(n_solved):
    eb = n_solved.statistics.energy_balance()
    assert not eb.xs("Process", level="component").empty
    np.testing.assert_allclose(
        eb.groupby(level="bus_carrier").sum().sum(), 0, atol=1e-5
    )

    assert not n_solved.statistics.installed_capacity().empty
    assert not n_solved.statistics.supply().empty


def test_process_get_bounds_pu(n):
    min_pu, max_pu = n.c.processes.get_bounds_pu("p")
    assert float(min_pu.min()) == 0.0
    assert float(max_pu.max()) == 1.0

    with pytest.raises(ValueError, match="operational attributes"):
        n.c.processes.get_bounds_pu("invalid")


def test_process_multi_port_optimization(n_multiport_solved):
    n = n_multiport_solved

    p = n.c.processes.dynamic["p"].loc[0, "electrolyser"]
    np.testing.assert_allclose(
        n.c.processes.dynamic["p0"].loc[0, "electrolyser"], p, rtol=1e-5
    )
    np.testing.assert_allclose(
        n.c.processes.dynamic["p1"].loc[0, "electrolyser"], -p * 0.7, rtol=1e-5
    )
    np.testing.assert_allclose(
        n.c.processes.dynamic["p2"].loc[0, "electrolyser"], -p * 0.2, rtol=1e-5
    )

    assert n.c.processes.additional_ports == ["2"]


def test_process_multi_port_pf(n_multiport):
    n_multiport.c.processes.static.loc["electrolyser", "p_set"] = 10
    n_multiport.lpf()

    np.testing.assert_allclose(
        n_multiport.c.processes.dynamic["p0"].loc[0, "electrolyser"], 10, rtol=1e-5
    )
    np.testing.assert_allclose(
        n_multiport.c.processes.dynamic["p1"].loc[0, "electrolyser"], -7, rtol=1e-5
    )
    np.testing.assert_allclose(
        n_multiport.c.processes.dynamic["p2"].loc[0, "electrolyser"], -2, rtol=1e-5
    )


def test_process_ramp_limits():
    n = pypsa.Network()
    n.set_snapshots(range(3))

    n.add("Bus", "electricity")
    n.add("Bus", "hydrogen")
    n.add("Generator", "gen", bus="electricity", p_nom=100, marginal_cost=10)
    n.add("Load", "h2_demand", bus="hydrogen", p_set=[5, 10, 5])

    n.add(
        "Process",
        "electrolyser",
        bus0="electricity",
        bus1="hydrogen",
        rate0=-1,
        rate1=0.8,
        p_nom=50,
        ramp_limit_up=0.5,
        ramp_limit_down=0.5,
    )

    status, _ = n.optimize()
    assert status == "ok"

    ramp = n.c.processes.dynamic["p"]["electrolyser"].diff().dropna()
    assert (ramp <= 50 * 0.5 + 1e-5).all()
    assert (ramp >= -50 * 0.5 - 1e-5).all()
