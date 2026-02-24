# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd
import pytest

import pypsa
import pypsa.consistency
from pypsa.components._types.shared_layer.multiports import Multiport


class LinkSpec:
    type = "Link"

    @staticmethod
    def add(n, name, *, efficiency=0.9, **kw):
        n.add("Link", name, efficiency=efficiency, **kw)

    @staticmethod
    def dispatch(n, col, port):
        return n.c.links.dynamic[f"p{port}"][col]

    @staticmethod
    def delay_key(port):
        return "delay" if port == 1 else f"delay{port}"

    @staticmethod
    def cyclic_key(port):
        return "cyclic_delay" if port == 1 else f"cyclic_delay{port}"

    @staticmethod
    def eff_key(port):
        return f"efficiency{port}"

    @staticmethod
    def nom_opt(n, name):
        return n.c.links.static.at[name, "p_nom_opt"]


class ProcessSpec:
    type = "Process"

    @staticmethod
    def add(n, name, *, efficiency=0.9, **kw):
        n.add("Process", name, rate0=-1, rate1=efficiency, **kw)

    @staticmethod
    def dispatch(n, col, port):
        key = "p" if port == 0 else f"p{port}"
        return n.c.processes.dynamic[key][col]

    @staticmethod
    def delay_key(port):
        return f"delay{port}"

    @staticmethod
    def cyclic_key(port):
        return f"cyclic_delay{port}"

    @staticmethod
    def eff_key(port):
        return f"rate{port}"

    @staticmethod
    def nom_opt(n, name):
        return n.c.processes.static.at[name, "p_nom_opt"]


@pytest.fixture(params=[LinkSpec, ProcessSpec], ids=["Link", "Process"])
def spec(request):
    return request.param


@pytest.fixture
def base_network():
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "gen0", bus="bus0", p_nom=200, marginal_cost=10)
    n.add("Generator", "gen1", bus="bus1", p_nom=200, marginal_cost=50)
    return n


# ---------------------------------------------------------------------------
# Parametrized tests (Link + Process)
# ---------------------------------------------------------------------------


def test_delay_cyclic(base_network, spec):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    spec.add(
        n,
        "comp",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        **{spec.delay_key(1): 2, spec.cyclic_key(1): True},
    )
    n.optimize()
    p0 = spec.dispatch(n, "comp", 0).values
    p1 = spec.dispatch(n, "comp", 1).values
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 2) % 6], atol=1e-5)


def test_delay_non_cyclic(base_network, spec):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    spec.add(
        n,
        "comp",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        **{spec.delay_key(1): 2, spec.cyclic_key(1): False},
    )
    n.optimize()
    p0 = spec.dispatch(n, "comp", 0).values
    p1 = spec.dispatch(n, "comp", 1).values
    np.testing.assert_allclose(p1[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(p1[1], 0.0, atol=1e-5)
    for t in range(2, 6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[t - 2], atol=1e-5)


def test_multiport_different_delays(base_network, spec):
    n = base_network
    n.add("Bus", "bus2")
    n.add("Generator", "gen2", bus="bus2", p_nom=200, marginal_cost=50)
    n.add("Load", "load1", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add("Load", "load2", bus="bus2", p_set=[5, 10, 15, 20, 25, 30])
    spec.add(
        n,
        "comp",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        efficiency=0.9,
        p_nom=100,
        **{spec.eff_key(2): 0.8},
        **{
            spec.delay_key(1): 1,
            spec.cyclic_key(1): True,
            spec.delay_key(2): 3,
            spec.cyclic_key(2): True,
        },
    )
    n.optimize()
    p0 = spec.dispatch(n, "comp", 0).values
    p1 = spec.dispatch(n, "comp", 1).values
    p2 = spec.dispatch(n, "comp", 2).values
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 1) % 6], atol=1e-5)
        np.testing.assert_allclose(p2[t], -0.8 * p0[(t - 3) % 6], atol=1e-5)


def test_delay_mixed_delayed_and_non_delayed(base_network, spec):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    spec.add(
        n,
        "delayed",
        bus0="bus0",
        bus1="bus1",
        efficiency=0.9,
        p_nom=100,
        **{spec.delay_key(1): 2, spec.cyclic_key(1): True},
    )
    spec.add(n, "instant", bus0="bus0", bus1="bus1", efficiency=0.8, p_nom=100)
    n.optimize()
    p0_d = spec.dispatch(n, "delayed", 0).values
    p1_d = spec.dispatch(n, "delayed", 1).values
    p0_i = spec.dispatch(n, "instant", 0).values
    p1_i = spec.dispatch(n, "instant", 1).values
    for t in range(6):
        np.testing.assert_allclose(p1_d[t], -0.9 * p0_d[(t - 2) % 6], atol=1e-5)
    np.testing.assert_allclose(p1_i, -0.8 * p0_i, atol=1e-5)


def test_delay_extendable(base_network, spec):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    spec.add(
        n,
        "comp",
        bus0="bus0",
        bus1="bus1",
        p_nom_extendable=True,
        p_nom_max=200,
        capital_cost=1,
        **{spec.delay_key(1): 2, spec.cyclic_key(1): True},
    )
    n.optimize()
    assert spec.nom_opt(n, "comp") > 0
    p0 = spec.dispatch(n, "comp", 0).values
    p1 = spec.dispatch(n, "comp", 1).values
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 2) % 6], atol=1e-5)


def test_delay_non_uniform_weightings(spec):
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.snapshot_weightings.loc[:, "generators"] = [1, 2, 1, 2, 1, 2]
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "gen0", bus="bus0", p_nom=200, marginal_cost=10)
    n.add("Generator", "gen1", bus="bus1", p_nom=200, marginal_cost=50)
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    spec.add(
        n,
        "comp",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        **{spec.delay_key(1): 3, spec.cyclic_key(1): True},
    )
    n.optimize()
    p0 = spec.dispatch(n, "comp", 0).values
    p1 = spec.dispatch(n, "comp", 1).values
    expected_source = [4, 5, 0, 1, 2, 3]
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[expected_source[t]], atol=1e-5)


def test_delay_multi_invest_cyclic(spec):
    n = pypsa.Network()
    periods = [2020, 2030]
    snapshots = pd.MultiIndex.from_product([periods, range(6)])
    n.set_snapshots(snapshots)
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "gen0", bus="bus0", p_nom=200, marginal_cost=10)
    n.add("Generator", "gen1", bus="bus1", p_nom=200, marginal_cost=50)
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60] * 2)
    spec.add(
        n,
        "comp",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        **{spec.delay_key(1): 2, spec.cyclic_key(1): True},
    )
    n.optimize()
    for period in periods:
        p0 = spec.dispatch(n, "comp", 0).loc[period].values
        p1 = spec.dispatch(n, "comp", 1).loc[period].values
        for t in range(6):
            np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 2) % 6], atol=1e-5)


def test_delay_with_scenarios(spec):
    n = pypsa.Network()
    n.set_snapshots(range(5))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "gen0", bus="bus0", p_nom=500, marginal_cost=10)
    n.add("Generator", "gen1", bus="bus1", p_nom=500, marginal_cost=100)
    n.add("Load", "load1", bus="bus1", p_set=[0, 20, 40, 0, 0])
    spec.add(
        n,
        "delayed",
        bus0="bus0",
        bus1="bus1",
        p_nom=300,
        **{spec.delay_key(1): 1, spec.cyclic_key(1): True},
    )
    n.set_scenarios({"low": 0.5, "high": 0.5})
    status, _ = n.optimize()
    assert status == "ok"

    delay_weightings = n.snapshot_weightings.generators.loc[n.snapshots]
    src, valid = Multiport.get_delay_source_indexer(
        n.snapshots, delay_weightings, 1, True
    )
    for scenario in n.scenarios:
        p0 = spec.dispatch(n, (scenario, "delayed"), 0).to_numpy()
        p1 = spec.dispatch(n, (scenario, "delayed"), 1).to_numpy()
        expected_p1 = -0.9 * p0[src]
        expected_p1[~valid] = 0.0
        np.testing.assert_allclose(p1, expected_p1, atol=1e-5)


def test_consistency_negative_delay(spec):
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    spec.add(n, "comp", bus0="bus0", bus1="bus1", p_nom=100, **{spec.delay_key(1): -1})
    with pytest.raises(pypsa.consistency.ConsistencyError, match="Negative delay"):
        n.consistency_check()


def test_consistency_delay_exceeds_horizon(spec):
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    spec.add(n, "comp", bus0="bus0", bus1="bus1", p_nom=100, **{spec.delay_key(1): 6})
    with pytest.raises(pypsa.consistency.ConsistencyError, match="equal or exceed"):
        n.consistency_check()


# ---------------------------------------------------------------------------
# Link-only tests
# ---------------------------------------------------------------------------


def test_link_delay_zero_unchanged(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=0,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    np.testing.assert_allclose(p1, -0.9 * p0)


def test_link_delay_equal_to_horizon(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=6,
        cyclic_delay=True,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    np.testing.assert_allclose(p1, -0.9 * p0)


def test_link_delay_uses_generator_snapshot_weightings_non_cyclic(base_network):
    n = base_network
    n.snapshot_weightings.loc[:, "generators"] = [1, 2, 1, 2, 1, 2]
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=3,
        cyclic_delay=False,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    np.testing.assert_allclose(p1[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(p1[1], 0.0, atol=1e-10)
    expected_source = [0, 1, 2, 3]
    for t in range(2, 6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[expected_source[t - 2]])


def test_link_delay_with_scenarios_non_delayed_regression():
    n = pypsa.examples.ac_dc_meshed()
    n.set_scenarios({"low": 0.3, "high": 0.7})
    status, _ = n.optimize()
    assert status == "ok"


def test_link_delay_with_scenarios_multiport():
    n = pypsa.Network()
    n.set_snapshots(range(5))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Generator", "gen0", bus="bus0", p_nom=500, marginal_cost=10)
    n.add("Generator", "gen1", bus="bus1", p_nom=500, marginal_cost=100)
    n.add("Generator", "gen2", bus="bus2", p_nom=500, marginal_cost=100)
    n.add("Load", "load1", bus="bus1", p_set=[0, 20, 40, 0, 0])
    n.add("Load", "load2", bus="bus2", p_set=[10, 0, 0, 10, 0])
    n.add(
        "Link",
        "delayed",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        p_nom=300,
        efficiency=0.9,
        efficiency2=0.8,
        delay=1,
        cyclic_delay=True,
        delay2=2,
        cyclic_delay2=False,
    )
    n.add(
        "Link",
        "instant",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        p_nom=300,
        efficiency=0.95,
        efficiency2=0.85,
        delay=0,
        delay2=0,
    )
    n.set_scenarios({"low": 0.5, "high": 0.5})
    status, _ = n.optimize()
    assert status == "ok"

    delay_weightings = n.snapshot_weightings.generators.loc[n.snapshots]
    src1, valid1 = Multiport.get_delay_source_indexer(
        n.snapshots, delay_weightings, 1, True
    )
    src2, valid2 = Multiport.get_delay_source_indexer(
        n.snapshots, delay_weightings, 2, False
    )
    for scenario in n.scenarios:
        p0 = n.c.links.dynamic.p0[(scenario, "delayed")].to_numpy()
        p1 = n.c.links.dynamic.p1[(scenario, "delayed")].to_numpy()
        p2 = n.c.links.dynamic.p2[(scenario, "delayed")].to_numpy()
        expected_p1 = -0.9 * p0[src1]
        expected_p1[~valid1] = 0.0
        np.testing.assert_allclose(p1, expected_p1)
        expected_p2 = -0.8 * p0[src2]
        expected_p2[~valid2] = 0.0
        np.testing.assert_allclose(p2, expected_p2)


def test_link_delay_multi_invest_non_cyclic():
    n = pypsa.Network()
    periods = [2020, 2030]
    snapshots = pd.MultiIndex.from_product([periods, range(6)])
    n.set_snapshots(snapshots)
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "gen0", bus="bus0", p_nom=200, marginal_cost=10)
    n.add("Generator", "gen1", bus="bus1", p_nom=200, marginal_cost=50)
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60] * 2)
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=2,
        cyclic_delay=False,
    )
    n.optimize()
    for period in periods:
        p0 = n.c.links.dynamic.p0["link"].loc[period].values
        p1 = n.c.links.dynamic.p1["link"].loc[period].values
        np.testing.assert_allclose(p1[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(p1[1], 0.0, atol=1e-10)
        for t in range(2, 6):
            np.testing.assert_allclose(p1[t], -0.9 * p0[t - 2])
        np.testing.assert_allclose(p0[4], 0.0, atol=1e-10)
        np.testing.assert_allclose(p0[5], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Process-only tests
# ---------------------------------------------------------------------------


def test_process_delay_on_input_bus():
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "gen0", bus="bus0", p_nom=200, marginal_cost=10)
    n.add("Generator", "gen1", bus="bus1", p_nom=200, marginal_cost=50)
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Process",
        "electrolyser",
        bus0="bus0",
        bus1="bus1",
        rate0=-1,
        rate1=0.8,
        p_nom=100,
        delay0=2,
        cyclic_delay0=True,
    )
    n.optimize()
    p = n.c.processes.dynamic["p"]["electrolyser"].values
    p0 = n.c.processes.dynamic["p0"]["electrolyser"].values
    p1 = n.c.processes.dynamic["p1"]["electrolyser"].values
    for t in range(6):
        np.testing.assert_allclose(p0[t], -(-1) * p[(t - 2) % 6], atol=1e-5)
    np.testing.assert_allclose(p1, -0.8 * p, atol=1e-5)


def test_process_delay_cyclic_delay0_false():
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "gen0", bus="bus0", p_nom=200, marginal_cost=10)
    n.add("Generator", "gen1", bus="bus1", p_nom=200, marginal_cost=50)
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Process",
        "electrolyser",
        bus0="bus0",
        bus1="bus1",
        rate0=-1,
        rate1=0.8,
        p_nom=100,
        delay0=2,
        cyclic_delay0=False,
    )
    n.optimize()
    p = n.c.processes.dynamic["p"]["electrolyser"].values
    p0 = n.c.processes.dynamic["p0"]["electrolyser"].values
    np.testing.assert_allclose(p0[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(p0[1], 0.0, atol=1e-5)
    for t in range(2, 6):
        np.testing.assert_allclose(p0[t], p[t - 2], atol=1e-5)


# ---------------------------------------------------------------------------
# Unit tests (component-agnostic)
# ---------------------------------------------------------------------------


def test_delay_rounding_warning_fires(caplog):
    snapshots = pd.RangeIndex(8)
    weightings = pd.Series(4.0, index=snapshots)
    with caplog.at_level(logging.WARNING):
        Multiport.get_delay_source_indexer(
            snapshots, weightings, delay=3, is_cyclic=True
        )
    assert any("does not align" in r.message for r in caplog.records)


def test_delay_rounding_warning_silent(caplog):
    snapshots = pd.RangeIndex(6)
    weightings = pd.Series(1.0, index=snapshots)
    with caplog.at_level(logging.WARNING):
        Multiport.get_delay_source_indexer(
            snapshots, weightings, delay=2, is_cyclic=True
        )
    assert not any("does not align" in r.message for r in caplog.records)


def test_get_delay_source_indexer_multi_invest():
    snapshots = pd.MultiIndex.from_product(
        [[2020, 2030], range(4)], names=["period", "timestep"]
    )
    weightings = pd.Series(1.0, index=snapshots)
    src, valid = Multiport.get_delay_source_indexer(
        snapshots, weightings, delay=2, is_cyclic=True
    )
    expected_src = np.array([2, 3, 0, 1, 6, 7, 4, 5])
    np.testing.assert_array_equal(src, expected_src)
    np.testing.assert_array_equal(valid, np.ones(8, dtype=bool))


def test_get_delay_source_indexer_multi_invest_non_cyclic():
    snapshots = pd.MultiIndex.from_product(
        [[2020, 2030], range(4)], names=["period", "timestep"]
    )
    weightings = pd.Series(1.0, index=snapshots)
    src, valid = Multiport.get_delay_source_indexer(
        snapshots, weightings, delay=2, is_cyclic=False
    )
    expected_valid = np.array([False, False, True, True, False, False, True, True])
    np.testing.assert_array_equal(valid, expected_valid)


def test_consistency_delay_exceeds_period_horizon():
    n = pypsa.Network()
    snapshots = pd.MultiIndex.from_product(
        [[2020, 2030], range(4)], names=["period", "timestep"]
    )
    n.set_snapshots(snapshots)
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Link", "link", bus0="bus0", bus1="bus1", p_nom=100, delay=4)
    with pytest.raises(pypsa.consistency.ConsistencyError, match="equal or exceed"):
        n.consistency_check()


def test_delay_positions_raises_on_zero_total_weight():
    weights = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="positive total"):
        Multiport._delay_positions(weights, delay=1.0, is_cyclic=True)


def test_get_delay_source_indexer_empty_snapshots():
    snapshots = pd.RangeIndex(0)
    weightings = pd.Series(dtype=float)
    src, valid = Multiport.get_delay_source_indexer(
        snapshots, weightings, delay=2, is_cyclic=True
    )
    assert len(src) == 0
    assert len(valid) == 0
