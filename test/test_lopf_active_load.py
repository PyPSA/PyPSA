# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the active (dispatchable) Load component (issue #1736).

A load is active where its ``p_set`` is NaN (dispatched as a variable within
``[p_min_pu, p_max_pu] * p_nom``) and passive where ``p_set`` is set (entering
the nodal balance as a constant).
"""

import numpy as np
import pytest

import pypsa


@pytest.fixture
def base_network():
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "b")
    n.add("Generator", "cheap", bus="b", p_nom=100, marginal_cost=10)
    n.add("Generator", "expensive", bus="b", p_nom=100, marginal_cost=50)
    return n


def test_passive_load_classification(base_network):
    n = base_network
    n.add("Load", "base", bus="b", p_set=[30, 120, 60])
    # active_assets keeps its universal meaning (in the optimisation); passive
    # loads are excluded from dispatchable and reported by passive.
    assert list(n.c.loads.active_assets) == ["base"]
    assert list(n.c.loads.passive) == ["base"]
    assert list(n.c.loads.dispatchable) == []


def test_dispatchable_includes_inactive(base_network):
    # Like fixed/extendables, dispatchable/passive include inactive loads;
    # call sites intersect with active_assets.
    n = base_network
    n.add("Load", "flex", bus="b", p_nom=50, active=False)  # dispatchable, inactive
    n.add(
        "Load", "base", bus="b", p_set=[30, 120, 60], active=False
    )  # passive, inactive
    assert list(n.c.loads.active_assets) == []
    assert set(n.c.loads.dispatchable) == {"flex"}
    assert set(n.c.loads.passive) == {"base"}


def test_active_load_classification(base_network):
    n = base_network
    n.add("Load", "flex", bus="b", p_nom=50, marginal_cost=-40)  # p_set NaN
    assert list(n.c.loads.active_assets) == ["flex"]
    assert list(n.c.loads.dispatchable) == ["flex"]
    assert list(n.c.loads.passive) == []


def test_active_load_equivalent_to_negative_generator(base_network):
    """An active load must give the same optimum as a negative generator."""
    nl = base_network
    nl.add("Load", "base", bus="b", p_set=[30, 120, 60])
    nl.add("Load", "flex", bus="b", p_nom=50, p_min_pu=0, p_max_pu=1, marginal_cost=-40)
    nl.optimize(solver_name="highs")

    ng = pypsa.Network()
    ng.set_snapshots(range(3))
    ng.add("Bus", "b")
    ng.add("Generator", "cheap", bus="b", p_nom=100, marginal_cost=10)
    ng.add("Generator", "expensive", bus="b", p_nom=100, marginal_cost=50)
    ng.add("Load", "base", bus="b", p_set=[30, 120, 60])
    ng.add(
        "Generator",
        "flex",
        bus="b",
        p_nom=50,
        p_min_pu=-1,
        p_max_pu=0,
        marginal_cost=40,
    )
    ng.optimize(solver_name="highs")

    assert np.isclose(nl.objective, ng.objective)
    # served demand (positive) equals negated generator dispatch
    np.testing.assert_allclose(
        nl.loads_t.p["flex"], -ng.generators_t.p["flex"], atol=1e-6
    )


def test_passive_load_regression(base_network):
    """Passive loads still enter the balance as a constant (unchanged behaviour)."""
    n = base_network
    n.add("Load", "base", bus="b", p_set=[30, 120, 60])
    n.optimize(solver_name="highs")
    np.testing.assert_allclose(n.loads_t.p["base"], [30, 120, 60])


def test_mixed_p_set_pins_set_snapshots(base_network):
    """NaN snapshots dispatch freely; set snapshots are pinned to p_set."""
    n = base_network
    n.remove("Generator", "expensive")
    n.add(
        "Load", "flex", bus="b", p_nom=50, marginal_cost=-40, p_set=[np.nan, 20, np.nan]
    )
    n.optimize(solver_name="highs")
    # WTP (40) > cost (10) so free snapshots serve full p_nom; snapshot 1 pinned
    np.testing.assert_allclose(n.loads_t.p["flex"], [50, 20, 50])

    # the withdrawal expression must not double-count the pinned snapshot: it
    # equals the variable, not variable + p_set constant.
    w = n.statistics.withdrawal(components="Load", groupby_time=False)
    np.testing.assert_allclose(w.sum().to_numpy(), [50, 20, 50], atol=1e-6)


def test_committable_active_load(base_network):
    n = base_network
    n.remove("Generator", "expensive")
    n.add(
        "Load",
        "c",
        bus="b",
        p_nom=50,
        p_min_pu=0.5,
        marginal_cost=-40,
        committable=True,
    )
    n.optimize(solver_name="highs")
    assert "c" in n.loads_t.status.columns
    # serving is valuable, so the load stays on at full power
    np.testing.assert_allclose(n.loads_t.p["c"], 50)


def test_expressions_with_dispatchable_load(base_network):
    """Statistics expressions cover both passive and dispatchable loads."""
    n = base_network
    n.add("Load", "base", bus="b", p_set=[30, 120, 60])  # passive
    n.add("Load", "flex", bus="b", p_nom=50, marginal_cost=-40)  # dispatchable
    n.optimize(solver_name="highs")

    # withdrawal must equal served consumption of both loads
    w = n.statistics.withdrawal(components="Load", groupby_time=False)
    served = n.loads_t.p[["base", "flex"]].sum().sum()
    np.testing.assert_allclose(w.sum().sum(), served, atol=1e-6)


def test_zero_p_nom_active_load_is_noop(base_network):
    n = base_network
    n.add("Load", "l", bus="b")  # p_set NaN, p_nom default 0
    n.optimize(solver_name="highs")
    np.testing.assert_allclose(n.loads_t.p["l"], 0)
