# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd
import pytest

import pypsa
import pypsa.consistency
from pypsa.components._types.links import Links


@pytest.fixture
def base_network():
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "gen0", bus="bus0", p_nom=200, marginal_cost=10)
    n.add("Generator", "gen1", bus="bus1", p_nom=200, marginal_cost=50)
    return n


def test_link_delay_cyclic(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=2,
        cyclic_delay=True,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 2) % 6])


def test_link_delay_non_cyclic(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
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
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    np.testing.assert_allclose(p1[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(p1[1], 0.0, atol=1e-10)
    for t in range(2, 6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[t - 2])
    np.testing.assert_allclose(p0[4], 0.0, atol=1e-10)
    np.testing.assert_allclose(p0[5], 0.0, atol=1e-10)


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


def test_link_multiport_different_delays(base_network):
    n = base_network
    n.add("Bus", "bus2")
    n.add("Generator", "gen2", bus="bus2", p_nom=200, marginal_cost=50)
    n.add("Load", "load1", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add("Load", "load2", bus="bus2", p_set=[5, 10, 15, 20, 25, 30])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        p_nom=100,
        efficiency=0.9,
        efficiency2=0.8,
        delay=1,
        delay2=3,
        cyclic_delay=True,
        cyclic_delay2=True,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    p2 = n.c.links.dynamic.p2["link"].values
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 1) % 6])
        np.testing.assert_allclose(p2[t], -0.8 * p0[(t - 3) % 6])


def test_link_delay_mixed_delayed_and_non_delayed(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "delayed",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=2,
        cyclic_delay=True,
    )
    n.add(
        "Link",
        "instant",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.8,
        delay=0,
    )
    n.optimize()
    p0_d = n.c.links.dynamic.p0["delayed"].values
    p1_d = n.c.links.dynamic.p1["delayed"].values
    p0_i = n.c.links.dynamic.p0["instant"].values
    p1_i = n.c.links.dynamic.p1["instant"].values
    for t in range(6):
        np.testing.assert_allclose(p1_d[t], -0.9 * p0_d[(t - 2) % 6])
    np.testing.assert_allclose(p1_i, -0.8 * p0_i)


def test_link_delay_extendable(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom_extendable=True,
        p_nom_max=200,
        capital_cost=1,
        efficiency=0.9,
        delay=2,
        cyclic_delay=True,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    assert n.c.links.static.at["link", "p_nom_opt"] > 0
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 2) % 6])


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


def test_link_delay_uses_generator_snapshot_weightings_cyclic(base_network):
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
        cyclic_delay=True,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    expected_source = [4, 5, 0, 1, 2, 3]
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[expected_source[t]])


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


def test_link_delay_with_scenarios():
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

    from pypsa.components._types.links import Links

    delay_weightings = n.snapshot_weightings.generators.loc[n.snapshots]
    src1, valid1 = Links.get_delay_source_indexer(
        n.snapshots, delay_weightings, 1, True
    )
    src2, valid2 = Links.get_delay_source_indexer(
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


def test_delay_rounding_warning_fires(caplog):
    """Warn when delay doesn't align with snapshot boundaries (sub/super-snapshot)."""
    snapshots = pd.RangeIndex(8)
    weightings = pd.Series(4.0, index=snapshots)
    with caplog.at_level(logging.WARNING):
        Links.get_delay_source_indexer(snapshots, weightings, delay=3, is_cyclic=True)
    assert any("does not align" in r.message for r in caplog.records)


def test_delay_rounding_warning_silent(caplog):
    """No warning when delay aligns exactly with snapshot boundaries."""
    snapshots = pd.RangeIndex(6)
    weightings = pd.Series(1.0, index=snapshots)
    with caplog.at_level(logging.WARNING):
        Links.get_delay_source_indexer(snapshots, weightings, delay=2, is_cyclic=True)
    assert not any("does not align" in r.message for r in caplog.records)


def test_get_delay_source_indexer_multi_invest():
    """Delay wraps within each investment period, not across periods."""
    snapshots = pd.MultiIndex.from_product(
        [[2020, 2030], range(4)], names=["period", "timestep"]
    )
    weightings = pd.Series(1.0, index=snapshots)
    src, valid = Links.get_delay_source_indexer(
        snapshots, weightings, delay=2, is_cyclic=True
    )
    # Period 2020 (positions 0-3): wraps within [0,3] → [2,3,0,1]
    # Period 2030 (positions 4-7): wraps within [4,7] → [6,7,4,5]
    expected_src = np.array([2, 3, 0, 1, 6, 7, 4, 5])
    np.testing.assert_array_equal(src, expected_src)
    np.testing.assert_array_equal(valid, np.ones(8, dtype=bool))


def test_get_delay_source_indexer_multi_invest_non_cyclic():
    """Non-cyclic delay per period marks early snapshots invalid per period."""
    snapshots = pd.MultiIndex.from_product(
        [[2020, 2030], range(4)], names=["period", "timestep"]
    )
    weightings = pd.Series(1.0, index=snapshots)
    src, valid = Links.get_delay_source_indexer(
        snapshots, weightings, delay=2, is_cyclic=False
    )
    # Each period independently: first 2 snapshots invalid
    expected_valid = np.array([False, False, True, True, False, False, True, True])
    np.testing.assert_array_equal(valid, expected_valid)


def test_link_delay_multi_invest_cyclic():
    """End-to-end: delayed output wraps within each investment period."""
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
        cyclic_delay=True,
    )
    n.optimize()
    for period in periods:
        p0 = n.c.links.dynamic.p0["link"].loc[period].values
        p1 = n.c.links.dynamic.p1["link"].loc[period].values
        for t in range(6):
            np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 2) % 6])


def test_link_delay_multi_invest_non_cyclic():
    """End-to-end: non-cyclic delay applied independently per investment period."""
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


def test_consistency_delay_exceeds_period_horizon():
    """In multi-invest, delay is checked against per-period horizon."""
    n = pypsa.Network()
    snapshots = pd.MultiIndex.from_product(
        [[2020, 2030], range(4)], names=["period", "timestep"]
    )
    n.set_snapshots(snapshots)
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    # delay=4 equals per-period horizon (4 snapshots * weight 1.0 = 4.0)
    n.add("Link", "link", bus0="bus0", bus1="bus1", p_nom=100, delay=4)
    with pytest.raises(pypsa.consistency.ConsistencyError, match="equal or exceed"):
        n.consistency_check()


def test_consistency_negative_delay(base_network):
    n = base_network
    n.add("Link", "link", bus0="bus0", bus1="bus1", p_nom=100, delay=-1)
    with pytest.raises(pypsa.consistency.ConsistencyError, match="Negative delay"):
        n.consistency_check()


def test_consistency_delay_exceeds_horizon(base_network):
    n = base_network
    n.add("Link", "link", bus0="bus0", bus1="bus1", p_nom=100, delay=6)
    with pytest.raises(pypsa.consistency.ConsistencyError, match="equal or exceed"):
        n.consistency_check()


def test_delay_positions_raises_on_zero_total_weight():
    weights = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="positive total"):
        Links._delay_positions(weights, delay=1.0, is_cyclic=True)


def test_get_delay_source_indexer_empty_snapshots():
    snapshots = pd.RangeIndex(0)
    weightings = pd.Series(dtype=float)
    src, valid = Links.get_delay_source_indexer(
        snapshots, weightings, delay=2, is_cyclic=True
    )
    assert len(src) == 0
    assert len(valid) == 0
