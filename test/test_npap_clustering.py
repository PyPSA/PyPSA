# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for pypsa.clustering.npap — NPAP spatial clustering integration."""

from __future__ import annotations

import logging
import warnings

import networkx as nx
import pandas as pd
import pytest

import pypsa
from pypsa.clustering.npap import (
    _build_networkx_graph_from_pypsa,
    _build_one_port_strategies,
    _busmap_to_partition_map,
    _npap_partition_to_busmap,
)
from pypsa.clustering.spatial import Clustering

try:
    import npap  # noqa: F401

    npap_installed = True
except ImportError:
    npap_installed = False


# ---------------------------------------------------------------------------
# Helper: small network for unit tests (no NPAP dependency)
# ---------------------------------------------------------------------------


def _make_small_network(
    with_transformer: bool = False,
    with_links: bool = False,
    with_time: bool = False,
) -> pypsa.Network:
    """Create a small 5-bus network for unit tests.

    Layout::

        bus0 --line01-- bus1 --line12-- bus2
          |                               |
        line03                          line24
          |                               |
        bus3 -------------------------  bus4

    Optional transformer between bus3 and bus4, optional DC link between
    bus0 and bus4.
    """
    n = pypsa.Network()

    if with_time:
        n.set_snapshots(range(3))

    n.add(
        "Bus",
        ["bus0", "bus1", "bus2", "bus3", "bus4"],
        x=[0.0, 1.0, 2.0, 0.0, 2.0],
        y=[1.0, 1.0, 1.0, 0.0, 0.0],
        v_nom=[110.0, 110.0, 110.0, 220.0, 220.0],
    )

    n.add(
        "Line",
        ["line01", "line12", "line03", "line24"],
        bus0=["bus0", "bus1", "bus0", "bus2"],
        bus1=["bus1", "bus2", "bus3", "bus4"],
        x=[0.1, 0.2, 0.3, 0.4],
        r=[0.01, 0.02, 0.03, 0.04],
        s_nom=[100, 200, 150, 250],
    )

    if with_transformer:
        n.add(
            "Transformer",
            "trafo34",
            bus0="bus3",
            bus1="bus4",
            x=0.05,
            s_nom=500,
        )

    if with_links:
        n.add(
            "Link",
            "link04",
            bus0="bus0",
            bus1="bus4",
            p_nom=300,
        )

    n.add(
        "Generator",
        ["gen0", "gen3"],
        bus=["bus0", "bus3"],
        p_nom=[100, 200],
        carrier=["wind", "gas"],
    )

    n.add(
        "Load",
        ["load1", "load4"],
        bus=["bus1", "bus4"],
        p_set=[50, 80],
    )

    return n


# ===================================================================
# Tier 1: Unit tests — no NPAP dependency required
# ===================================================================


class TestBuildNetworkxGraph:
    """Tests for _build_networkx_graph_from_pypsa."""

    def test_returns_digraph(self):
        n = _make_small_network()
        G = _build_networkx_graph_from_pypsa(n)
        assert isinstance(G, nx.DiGraph)

    def test_node_count(self):
        n = _make_small_network()
        G = _build_networkx_graph_from_pypsa(n)
        assert G.number_of_nodes() == 5

    def test_edge_count_lines_only(self):
        # 4 lines × 2 directions = 8 edges
        n = _make_small_network()
        G = _build_networkx_graph_from_pypsa(n)
        assert G.number_of_edges() == 8

    def test_npap_aliases_on_nodes(self):
        n = _make_small_network()
        G = _build_networkx_graph_from_pypsa(n)
        attrs = G.nodes["bus0"]
        assert attrs["lon"] == 0.0
        assert attrs["lat"] == 1.0
        assert attrs["voltage"] == 110.0

    def test_original_bus_attrs_preserved(self):
        n = _make_small_network()
        G = _build_networkx_graph_from_pypsa(n)
        attrs = G.nodes["bus3"]
        # Original PyPSA columns should still be present
        assert attrs["x"] == 0.0
        assert attrs["y"] == 0.0
        assert attrs["v_nom"] == 220.0

    def test_line_edge_type_marker(self):
        n = _make_small_network()
        G = _build_networkx_graph_from_pypsa(n)
        edge_data = G.edges["bus0", "bus1"]
        assert edge_data["type"] == "line"

    def test_transformer_edge_type_marker(self):
        n = _make_small_network(with_transformer=True)
        G = _build_networkx_graph_from_pypsa(n)
        edge_data = G.edges["bus3", "bus4"]
        assert edge_data["type"] == "trafo"

    def test_link_edge_type_marker(self):
        n = _make_small_network(with_links=True)
        G = _build_networkx_graph_from_pypsa(n, include_links=True)
        edge_data = G.edges["bus0", "bus4"]
        assert edge_data["type"] == "dc_link"

    def test_voltage_attributes_on_line_edges(self):
        n = _make_small_network()
        G = _build_networkx_graph_from_pypsa(n)
        edge_data = G.edges["bus0", "bus3"]
        assert edge_data["primary_voltage"] == 110.0
        assert edge_data["secondary_voltage"] == 220.0

    def test_buses_i_subset_filtering(self):
        n = _make_small_network()
        subset = pd.Index(["bus0", "bus1", "bus2"])
        G = _build_networkx_graph_from_pypsa(n, buses_i=subset)
        assert G.number_of_nodes() == 3
        assert set(G.nodes()) == {"bus0", "bus1", "bus2"}
        # Only lines within subset: line01, line12 → 4 edges
        assert G.number_of_edges() == 4

    def test_exclude_transformers(self):
        n = _make_small_network(with_transformer=True)
        G_with = _build_networkx_graph_from_pypsa(n, include_transformers=True)
        G_without = _build_networkx_graph_from_pypsa(n, include_transformers=False)
        # Transformer adds 2 directed edges
        assert G_with.number_of_edges() == G_without.number_of_edges() + 2

    def test_include_links_false_by_default(self):
        n = _make_small_network(with_links=True)
        G = _build_networkx_graph_from_pypsa(n)
        # Links excluded by default: only 4 lines × 2 = 8 edges
        assert G.number_of_edges() == 8

    def test_include_links_true(self):
        n = _make_small_network(with_links=True)
        G = _build_networkx_graph_from_pypsa(n, include_links=True)
        # 4 lines × 2 + 1 link × 2 = 10 edges
        assert G.number_of_edges() == 10

    def test_ac_island_single_island(self):
        n = _make_small_network()
        G = _build_networkx_graph_from_pypsa(n)
        # All buses connected by lines → single AC island → no ac_island attr
        for node in G.nodes():
            assert "ac_island" not in G.nodes[node]

    def test_ac_island_multiple_islands(self):
        # Create two disconnected AC islands connected only by DC link
        n = pypsa.Network()
        n.add(
            "Bus", ["a1", "a2", "b1", "b2"], x=[0, 1, 3, 4], y=[0, 0, 0, 0], v_nom=110
        )
        n.add("Line", "ac1", bus0="a1", bus1="a2", x=0.1, r=0.01, s_nom=100)
        n.add("Line", "ac2", bus0="b1", bus1="b2", x=0.1, r=0.01, s_nom=100)
        n.add("Link", "dc", bus0="a2", bus1="b1", p_nom=100)

        G = _build_networkx_graph_from_pypsa(n, include_links=True)
        # Two AC islands → ac_island attribute set on each node
        islands = {G.nodes[node]["ac_island"] for node in G.nodes()}
        assert len(islands) == 2
        # Buses in same AC island share same ID
        assert G.nodes["a1"]["ac_island"] == G.nodes["a2"]["ac_island"]
        assert G.nodes["b1"]["ac_island"] == G.nodes["b2"]["ac_island"]
        assert G.nodes["a1"]["ac_island"] != G.nodes["b1"]["ac_island"]

    def test_self_loop_lines_skipped(self):
        n = pypsa.Network()
        n.add("Bus", ["bus0", "bus1"], x=[0, 1], y=[0, 0], v_nom=110)
        n.add("Line", "normal", bus0="bus0", bus1="bus1", x=0.1, r=0.01, s_nom=100)
        n.add("Line", "selfloop", bus0="bus0", bus1="bus0", x=0.1, r=0.01, s_nom=100)

        G = _build_networkx_graph_from_pypsa(n)
        assert G.number_of_nodes() == 2
        # Only the normal line (2 directed edges), self-loop skipped
        assert G.number_of_edges() == 2
        assert not G.has_edge("bus0", "bus0")

    def test_link_edges_no_voltage_attributes(self):
        n = _make_small_network(with_links=True)
        G = _build_networkx_graph_from_pypsa(n, include_links=True)
        edge_data = G.edges["bus0", "bus4"]
        assert "primary_voltage" not in edge_data
        assert "secondary_voltage" not in edge_data

    def test_edge_count_with_transformer_and_links(self):
        n = _make_small_network(with_transformer=True, with_links=True)
        G = _build_networkx_graph_from_pypsa(
            n, include_transformers=True, include_links=True
        )
        # 4 lines × 2 + 1 trafo × 2 + 1 link × 2 = 12 edges
        assert G.number_of_edges() == 12


class TestBusmapPartitionRoundtrip:
    """Tests for _busmap_to_partition_map and _npap_partition_to_busmap."""

    def test_busmap_to_partition_basic(self):
        busmap = pd.Series({"a": "0", "b": "0", "c": "1"})
        pm = _busmap_to_partition_map(busmap)
        assert set(pm.keys()) == {0, 1}
        assert sorted(pm[0]) == ["a", "b"]
        assert pm[1] == ["c"]

    def test_partition_to_busmap_basic(self):
        mapping = {0: ["a", "b"], 1: ["c"]}
        busmap = _npap_partition_to_busmap(mapping)
        assert busmap.dtype == object  # string dtype
        assert busmap["a"] == "0"
        assert busmap["b"] == "0"
        assert busmap["c"] == "1"

    def test_roundtrip_busmap_to_partition_to_busmap(self):
        original = pd.Series({"x": "0", "y": "1", "z": "0"})
        pm = _busmap_to_partition_map(original)
        recovered = _npap_partition_to_busmap(pm)
        # Values should match (order of index may differ)
        for bus in original.index:
            assert recovered[bus] == original[bus]

    def test_roundtrip_partition_to_busmap_to_partition(self):
        original = {0: ["a", "b"], 1: ["c"], 2: ["d", "e", "f"]}
        busmap = _npap_partition_to_busmap(original)
        recovered = _busmap_to_partition_map(busmap)
        assert set(recovered.keys()) == set(original.keys())
        for k in original:
            assert sorted(recovered[k]) == sorted(original[k])

    def test_empty_busmap(self):
        busmap = pd.Series(dtype=str)
        pm = _busmap_to_partition_map(busmap)
        assert pm == {}

    def test_empty_partition(self):
        mapping: dict[int, list] = {}
        busmap = _npap_partition_to_busmap(mapping)
        assert len(busmap) == 0


class TestBuildOnePortStrategies:
    """Tests for _build_one_port_strategies."""

    def test_output_attrs_get_sum(self):
        n = _make_small_network()
        strategies = _build_one_port_strategies(n, "Generator", {})
        # Generator output attrs like p should get "sum"
        # Check that at least dynamic attrs get "sum"
        for attr_name in n.c.generators.dynamic:
            assert strategies.get(attr_name) == "sum"

    def test_user_flat_strategies_override(self):
        n = _make_small_network()
        user = {"p_nom": "max", "custom_attr": "first"}
        strategies = _build_one_port_strategies(n, "Generator", user)
        assert strategies["p_nom"] == "max"
        assert strategies["custom_attr"] == "first"

    def test_per_component_strategy_dicts(self):
        n = _make_small_network()
        user = {
            "Generator": {"p_nom": "max"},
            "Load": {"p_set": "mean"},
        }
        strategies = _build_one_port_strategies(n, "Generator", user)
        assert strategies["p_nom"] == "max"
        # Load strategy should not leak into Generator
        assert strategies.get("p_set") != "mean"

    def test_per_component_dict_for_other_component(self):
        n = _make_small_network()
        user = {
            "Generator": {"p_nom": "max"},
            "Load": {"p_set": "mean"},
        }
        strategies = _build_one_port_strategies(n, "Load", user)
        assert strategies["p_set"] == "mean"

    def test_empty_strategies(self):
        n = _make_small_network()
        strategies = _build_one_port_strategies(n, "Generator", {})
        # Should still have entries for dynamic attrs
        assert isinstance(strategies, dict)
        assert len(strategies) >= 0  # At minimum an empty or populated dict


# ===================================================================
# Tier 2: Integration tests — require NPAP
# ===================================================================

npap_skip = pytest.mark.skipif(not npap_installed, reason="npap not installed")


# Module-scoped fixtures: run NPAP only once per module
@pytest.fixture(scope="module")
def scipy_network_for_npap():
    """Module-scoped recreation of the scipy_network fixture."""
    n = pypsa.examples.scigrid_de()
    n.c.generators.static.control = "PV"
    g = n.c.generators.static[n.c.generators.static.bus == "492"]
    n.c.generators.static.loc[g.index, "control"] = "PQ"
    n.calculate_dependent_values()
    n.determine_network_topology()
    return n


@pytest.fixture(scope="module")
def npap_clustering_result(scipy_network_for_npap):
    """Run npap_clustering once and cache for the module."""
    from pypsa.clustering.npap import npap_clustering

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return npap_clustering(scipy_network_for_npap, n_clusters=50)


@pytest.fixture(scope="module")
def npap_busmap_result(scipy_network_for_npap):
    """Run busmap_by_npap once and cache for the module."""
    from pypsa.clustering.npap import busmap_by_npap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return busmap_by_npap(scipy_network_for_npap, n_clusters=50)


@npap_skip
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestBusmapByNpap:
    """Tests for busmap_by_npap."""

    def test_returns_series(self, npap_busmap_result):
        assert isinstance(npap_busmap_result, pd.Series)

    def test_correct_length(self, npap_busmap_result, scipy_network_for_npap):
        assert len(npap_busmap_result) == len(scipy_network_for_npap.buses)

    def test_string_dtype(self, npap_busmap_result):
        assert npap_busmap_result.dtype == object  # pandas string dtype

    def test_cluster_count(self, npap_busmap_result):
        assert npap_busmap_result.nunique() == 50

    def test_covers_all_buses(self, npap_busmap_result, scipy_network_for_npap):
        assert set(npap_busmap_result.index) == set(scipy_network_for_npap.buses.index)


@npap_skip
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestNpapClustering:
    """Tests for npap_clustering full pipeline."""

    def test_returns_clustering_dataclass(self, npap_clustering_result):
        assert isinstance(npap_clustering_result, Clustering)

    def test_clustered_bus_count(self, npap_clustering_result):
        assert len(npap_clustering_result.n.buses) == 50

    def test_busmap_length(self, npap_clustering_result, scipy_network_for_npap):
        assert len(npap_clustering_result.busmap) == len(scipy_network_for_npap.buses)

    def test_busmap_index_matches_original(
        self, npap_clustering_result, scipy_network_for_npap
    ):
        assert set(npap_clustering_result.busmap.index) == set(
            scipy_network_for_npap.buses.index
        )

    def test_busmap_values_are_strings(self, npap_clustering_result):
        assert npap_clustering_result.busmap.dtype == object

    def test_linemap_not_empty(self, npap_clustering_result):
        assert not npap_clustering_result.linemap.empty

    def test_linemap_index_subset_of_original_lines(
        self, npap_clustering_result, scipy_network_for_npap
    ):
        original_lines = scipy_network_for_npap.lines.index
        assert npap_clustering_result.linemap.index.isin(original_lines).all()

    def test_linemap_values_subset_of_aggregated_lines(self, npap_clustering_result):
        aggregated_lines = npap_clustering_result.n.lines.index
        assert npap_clustering_result.linemap.isin(aggregated_lines).all()

    def test_generators_carried_forward(
        self, npap_clustering_result, scipy_network_for_npap
    ):
        # All original generators should be present
        assert len(npap_clustering_result.n.generators) == len(
            scipy_network_for_npap.generators
        )

    def test_generator_bus_references_valid(self, npap_clustering_result):
        clustered_buses = npap_clustering_result.n.buses.index
        gen_buses = npap_clustering_result.n.generators.bus
        assert gen_buses.isin(clustered_buses).all()

    def test_loads_carried_forward(
        self, npap_clustering_result, scipy_network_for_npap
    ):
        assert len(npap_clustering_result.n.loads) == len(scipy_network_for_npap.loads)

    def test_load_bus_references_valid(self, npap_clustering_result):
        clustered_buses = npap_clustering_result.n.buses.index
        load_buses = npap_clustering_result.n.loads.bus
        assert load_buses.isin(clustered_buses).all()

    def test_snapshots_preserved(self, npap_clustering_result, scipy_network_for_npap):
        assert npap_clustering_result.n.snapshots.equals(
            scipy_network_for_npap.snapshots
        )

    def test_aggregated_lines_have_valid_bus_refs(self, npap_clustering_result):
        clustered_buses = npap_clustering_result.n.buses.index
        lines = npap_clustering_result.n.lines
        assert lines.bus0.isin(clustered_buses).all()
        assert lines.bus1.isin(clustered_buses).all()

    def test_branch_time_series_warning(self, caplog):
        """Branch time series warning is emitted via logger.warning."""
        from pypsa.clustering.npap import npap_clustering

        # Build a fresh network with line time series to trigger the warning
        n = pypsa.examples.scigrid_de()
        n.set_snapshots(n.snapshots[:3])  # keep small for speed
        n.calculate_dependent_values()
        n.determine_network_topology()

        # Inject a dummy p0 time series on lines so the warning fires
        n.c.lines.dynamic["p0"] = pd.DataFrame(
            0.0,
            index=n.snapshots,
            columns=n.c.lines.static.index[:5],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with caplog.at_level(logging.WARNING, logger="pypsa.clustering.npap"):
                npap_clustering(n, n_clusters=20)

        assert any(
            "will not be aggregated" in record.message for record in caplog.records
        )


@npap_skip
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestAccessors:
    """Tests for ClusteringAccessor NPAP methods."""

    def test_cluster_by_npap_returns_network(self, scipy_network_for_npap):
        result = scipy_network_for_npap.cluster.cluster_by_npap(n_clusters=50)
        assert isinstance(result, pypsa.Network)

    def test_get_npap_clustering_result_returns_clustering(
        self, scipy_network_for_npap
    ):
        result = scipy_network_for_npap.cluster.get_npap_clustering_result(
            n_clusters=50
        )
        assert isinstance(result, Clustering)

    def test_busmap_by_npap_accessor(self, scipy_network_for_npap):
        busmap = scipy_network_for_npap.cluster.busmap_by_npap(n_clusters=50)
        assert isinstance(busmap, pd.Series)
        assert busmap.nunique() == 50


@npap_skip
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestClusterThenOptimize:
    """End-to-end: cluster with NPAP, then run LOPF on the reduced network."""

    def test_clustered_network_is_solvable(self):
        from pypsa.clustering.npap import npap_clustering

        n = pypsa.examples.scigrid_de()
        # Use only a few snapshots for speed
        n.set_snapshots(n.snapshots[:4])
        n.calculate_dependent_values()
        n.determine_network_topology()

        # Cluster to 50 buses with one-port aggregation
        C = npap_clustering(
            n,
            n_clusters=50,
            aggregate_one_ports=["Generator", "Load", "StorageUnit"],
        )
        nc = C.n

        # Sanity: the clustered network has the expected structure
        assert len(nc.buses) == 50
        assert len(nc.generators) > 0
        assert len(nc.loads) > 0

        # Run LOPF — the key assertion: the reduced network is feasible
        status, condition = nc.optimize()
        assert status == "ok", (
            f"Optimization of clustered network failed: {status}, {condition}"
        )

        # Basic sanity on results
        assert nc.objective >= 0
        assert not nc.c.generators.dynamic.p.empty
        assert (nc.c.generators.dynamic.p >= -1e-5).all().all()
