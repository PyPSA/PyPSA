# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Test scenarios not implemented decorators."""

import pandas as pd
import pytest

import pypsa


def test_clustering():
    """Test that clustering methods raise ValueError when used with stochastic networks."""
    # Create a regular network
    n = pypsa.Network()
    n.add("Bus", "bus1", x=0, y=0)
    n.add("Bus", "bus2", x=1, y=1)

    # Add scenarios to make it a stochastic network
    n.scenarios = ["scenario1", "scenario2"]
    assert n.has_scenarios

    # Test that all clustering methods raise ValueError
    busmap = {"bus1": "cluster1", "bus2": "cluster1"}
    bus_weights = pd.Series([1, 1], index=["bus1", "bus2"])

    with pytest.raises(ValueError, match="not yet implemented for stochastic networks"):
        n.cluster.spatial.busmap_by_hac(2)

    with pytest.raises(ValueError, match="not yet implemented for stochastic networks"):
        n.cluster.spatial.busmap_by_kmeans(bus_weights, 2)

    with pytest.raises(ValueError, match="not yet implemented for stochastic networks"):
        n.cluster.spatial.busmap_by_greedy_modularity(2)

    with pytest.raises(ValueError, match="not yet implemented for stochastic networks"):
        n.cluster.spatial.cluster_by_hac(2)

    with pytest.raises(ValueError, match="not yet implemented for stochastic networks"):
        n.cluster.spatial.cluster_by_kmeans(bus_weights, 2)

    with pytest.raises(ValueError, match="not yet implemented for stochastic networks"):
        n.cluster.spatial.cluster_by_greedy_modularity(2)

    with pytest.raises(ValueError, match="not yet implemented for stochastic networks"):
        n.cluster.spatial.cluster_by_busmap(busmap)

    with pytest.raises(ValueError, match="not yet implemented for stochastic networks"):
        n.cluster.spatial.get_clustering_from_busmap(busmap)


@pytest.mark.parametrize("method", ["lpf", "pf", "lpf_contingency"])
def test_power_flow(method):
    """Test that power flow methods raise ValueError for stochastic networks."""
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", ["a", "b"], v_nom=380)
    n.add("Line", "l", bus0="a", bus1="b", x=0.1, r=0.01, s_nom=100)
    n.add("Generator", "g", bus="a", p_set=50, control="Slack")
    n.add("Load", "d", bus="b", p_set=50)
    n.set_scenarios(["s1", "s2"])

    with pytest.raises(ValueError, match="not yet implemented for stochastic networks"):
        getattr(n, method)()
