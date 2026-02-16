# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Integration of NPAP partitioning and aggregation algorithms into PyPSA."""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd

from pypsa.clustering.spatial import (
    Clustering,
    aggregateoneport,
)

if TYPE_CHECKING:
    from pypsa import Network

logger = logging.getLogger(__name__)


def _build_networkx_graph_from_pypsa(
    n: Network,
    buses_i: pd.Index | None = None,
    include_transformers: bool = True,
    include_links: bool = False,
) -> nx.DiGraph:
    """Convert a PyPSA Network into a NetworkX DiGraph with NPAP-compatible attributes.

    All static attributes from PyPSA DataFrames are carried over to the
    NetworkX nodes and edges.  NPAP-specific aliases (``lon``, ``lat``,
    ``voltage``, ``primary_voltage``, ``secondary_voltage``) are added
    alongside the original PyPSA column names.

    Parameters
    ----------
    n : Network
        The PyPSA network to convert.
    buses_i : pd.Index | None, optional
        Subset of buses to include. If None, all buses are included.
    include_transformers : bool, optional
        Whether to include transformers as edges (default True).
    include_links : bool, optional
        Whether to include links as edges (default False).

    Returns
    -------
    nx.DiGraph
        A NetworkX DiGraph with NPAP-compatible node and edge attributes.

    """
    G = nx.DiGraph()

    # Get bus data
    buses = n.c.buses.static
    if buses_i is not None:
        buses = buses.loc[buses_i]

    bus_index = buses.index

    # Columns to skip when copying bus attributes (not meaningful on nodes)
    _bus_skip = {"generator"}

    # Add nodes: carry ALL bus static columns + NPAP aliases
    for bus_name, bus_data in buses.iterrows():
        node_attrs: dict[str, Any] = {
            "lon": bus_data.get("x", 0.0),
            "lat": bus_data.get("y", 0.0),
            "voltage": bus_data.get("v_nom", 0.0),
        }
        for col in buses.columns:
            if col in _bus_skip:
                continue
            val = bus_data[col]
            if not isinstance(val, float) or not pd.isna(val):
                node_attrs[col] = val
        G.add_node(bus_name, **node_attrs)

    # Columns to exclude from edge attribute copying (used for connectivity)
    _edge_skip = {"bus0", "bus1"}
    has_v_nom = "v_nom" in buses.columns

    def _add_branch_edges(
        df: pd.DataFrame, edge_type: str, add_voltages: bool = True
    ) -> None:
        """Add edges from a branch DataFrame, copying all static columns."""
        for _name, row in df.iterrows():
            bus0 = row["bus0"]
            bus1 = row["bus1"]
            if bus0 == bus1:
                continue

            edge_attrs: dict[str, Any] = {}
            for col in df.columns:
                if col in _edge_skip:
                    continue
                val = row[col]
                if not isinstance(val, float) or not pd.isna(val):
                    edge_attrs[col] = val

            # Set edge class marker LAST so it is never overwritten by a
            # PyPSA column of the same name (e.g. Line.type = line spec).
            edge_attrs["type"] = edge_type

            if add_voltages and has_v_nom:
                edge_attrs["primary_voltage"] = buses.at[bus0, "v_nom"]
                edge_attrs["secondary_voltage"] = buses.at[bus1, "v_nom"]

            G.add_edge(bus0, bus1, **edge_attrs)
            G.add_edge(bus1, bus0, **edge_attrs)

    # Add line edges
    lines = n.c.lines.static
    lines = lines[lines.bus0.isin(bus_index) & lines.bus1.isin(bus_index)]
    _add_branch_edges(lines, "line")

    # Add transformer edges
    if include_transformers:
        trafos = n.c.transformers.static
        trafos = trafos[trafos.bus0.isin(bus_index) & trafos.bus1.isin(bus_index)]
        _add_branch_edges(trafos, "trafo")

    # Add link edges
    if include_links:
        links = n.c.links.static
        links = links[links.bus0.isin(bus_index) & links.bus1.isin(bus_index)]
        _add_branch_edges(links, "dc_link", add_voltages=False)

    # AC island detection: compute connected components on AC-only subgraph.
    # Only assign ac_island when there are multiple islands (i.e. DC links
    # separate AC zones), otherwise the attribute is meaningless and triggers
    # unnecessary warnings in NPAP for algorithms that don't support it.
    ac_graph = nx.Graph()
    ac_graph.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        if data.get("type") in ("line", "trafo"):
            ac_graph.add_edge(u, v)

    components = list(nx.connected_components(ac_graph))
    if len(components) > 1:
        for island_id, component in enumerate(components):
            for node in component:
                G.nodes[node]["ac_island"] = island_id

    return G


def _busmap_to_partition_map(busmap: pd.Series) -> dict[int, list[str]]:
    """Convert a PyPSA busmap to NPAP's partition mapping format.

    Parameters
    ----------
    busmap : pd.Series
        Series mapping bus names to cluster labels.

    Returns
    -------
    dict[int, list[str]]
        Dictionary mapping integer cluster IDs to lists of bus names.

    """
    partition_map: dict[int, list[str]] = {}
    for cluster_label in busmap.unique():
        cluster_id = int(cluster_label)
        partition_map[cluster_id] = busmap[busmap == cluster_label].index.tolist()
    return partition_map


def _npap_partition_to_busmap(
    mapping: dict[int, list[Any]],
) -> pd.Series:
    """Convert NPAP's partition mapping to a PyPSA busmap.

    Parameters
    ----------
    mapping : dict[int, list[Any]]
        NPAP partition mapping (cluster_id -> list of bus names).

    Returns
    -------
    pd.Series
        Series with bus names as index and cluster labels as string values.

    """
    bus_to_cluster = {}
    for cluster_id, bus_names in mapping.items():
        for bus_name in bus_names:
            bus_to_cluster[bus_name] = str(cluster_id)
    return pd.Series(bus_to_cluster, dtype=str)


def busmap_by_npap(
    n: Network,
    n_clusters: int,
    strategy: str = "geographical_kmeans",
    buses_i: pd.Index | None = None,
    include_transformers: bool = True,
    include_links: bool = False,
    voltage_levels: list[float] | None = None,
    aggregate_parallel_edges: bool = False,
    parallel_edge_strategies: dict[str, str] | None = None,
    **kwargs: Any,
) -> pd.Series:
    """Partition a PyPSA network using NPAP algorithms, returning a busmap.

    Parameters
    ----------
    n : Network
        The PyPSA network to partition.
    n_clusters : int
        Number of clusters to create.
    strategy : str, optional
        NPAP partitioning strategy name (default "geographical_kmeans").
    buses_i : pd.Index | None, optional
        Subset of buses to cluster. If None, all buses are used.
    include_transformers : bool, optional
        Include transformers as edges in the graph (default True).
    include_links : bool, optional
        Include links as edges in the graph (default False).
    voltage_levels : list[float] | None, optional
        Target voltage levels for voltage-aware strategies. If None and
        a voltage-aware strategy is used, levels are auto-detected from v_nom.
    aggregate_parallel_edges : bool, optional
        Whether to aggregate parallel edges before partitioning (default False).
    parallel_edge_strategies : dict[str, str] | None, optional
        Strategies for parallel edge aggregation (NPAP strategy names).
    **kwargs : Any
        Additional keyword arguments passed to the NPAP partition method.

    Returns
    -------
    pd.Series
        Series with bus names as index and cluster labels as string values.

    Raises
    ------
    ModuleNotFoundError
        If the npap package is not installed.

    """
    if find_spec("npap") is None:
        msg = "Optional dependency 'npap' not found. Install via 'pip install npap'"
        raise ModuleNotFoundError(msg)

    from npap import PartitionAggregatorManager  # noqa: PLC0415

    # Build NetworkX graph
    G = _build_networkx_graph_from_pypsa(
        n,
        buses_i=buses_i,
        include_transformers=include_transformers,
        include_links=include_links,
    )

    # Create manager and load graph (bidirectional=False since graph is already bidirectional)
    manager = PartitionAggregatorManager()
    manager.load_data("networkx_direct", graph=G, bidirectional=False)

    # Handle MultiDiGraph or explicit parallel edge aggregation
    if isinstance(manager.get_current_graph(), nx.MultiDiGraph):
        manager.aggregate_parallel_edges(
            edge_properties=parallel_edge_strategies,
        )

    # For voltage-aware strategies, group by voltage levels
    if strategy.startswith("va_"):
        if voltage_levels is None:
            # Auto-detect voltage levels from bus v_nom
            buses = n.c.buses.static
            if buses_i is not None:
                buses = buses.loc[buses_i]
            voltage_levels = sorted(buses["v_nom"].dropna().unique().tolist())
            if not voltage_levels:
                voltage_levels = [220, 380]
            logger.info(
                "Auto-detected voltage levels for voltage-aware partitioning: %s",
                voltage_levels,
            )
        manager.group_by_voltage_levels(voltage_levels)

    # Run partition through NPAP's public API
    result = manager.partition(strategy, n_clusters=n_clusters, **kwargs)

    # Convert NPAP partition result to PyPSA busmap
    return _npap_partition_to_busmap(result.mapping)


def aggregate_network_by_npap(
    n: Network,
    busmap: pd.Series,
    node_strategies: dict[str, str] | None = None,
    line_strategies: dict[str, str] | None = None,
    transformer_strategies: dict[str, str] | None = None,
    link_strategies: dict[str, str] | None = None,
    aggregation_mode: Any | None = None,
    aggregation_profile: Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Aggregate buses, lines, transformers, and links using NPAP strategies.

    Uses NPAP's :class:`~npap.managers.AggregationManager` to aggregate network
    components.  Per-edge-type strategies (line / transformer / link) are
    passed via ``edge_type_properties`` in the :class:`AggregationProfile`.

    Parameters
    ----------
    n : Network
        The PyPSA network to aggregate.
    busmap : pd.Series
        Series mapping bus names to cluster labels.
    node_strategies : dict[str, str] | None, optional
        NPAP strategy names for node properties. Defaults to average for
        lat/lon/voltage.
    line_strategies : dict[str, str] | None, optional
        NPAP strategy names for line edge properties. Defaults to
        equivalent_reactance for x/r, sum for s_nom, average for length.
    transformer_strategies : dict[str, str] | None, optional
        NPAP strategy names for transformer edge properties.
    link_strategies : dict[str, str] | None, optional
        NPAP strategy names for link edge properties.
    aggregation_mode : AggregationMode | None, optional
        Pre-defined NPAP aggregation mode.  Overridden by
        *aggregation_profile* when both are given.
    aggregation_profile : AggregationProfile | None, optional
        Fully custom NPAP aggregation profile.  When supplied, per-type
        strategy dicts (*line_strategies*, etc.) are ignored.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys "buses", "lines", "transformers", "links",
        each mapping to a DataFrame of aggregated component data.

    Raises
    ------
    ModuleNotFoundError
        If the npap package is not installed.

    """
    if find_spec("npap") is None:
        msg = "Optional dependency 'npap' not found. Install via 'pip install npap'"
        raise ModuleNotFoundError(msg)

    from npap.interfaces import AggregationProfile  # noqa: PLC0415
    from npap.managers import AggregationManager  # noqa: PLC0415

    # Default strategies
    if node_strategies is None:
        node_strategies = {
            "lon": "average",
            "lat": "average",
            "voltage": "average",
            # String/categorical properties must use "first"
            "carrier": "first",
            "control": "first",
            "type": "first",
            "unit": "first",
            "location": "first",
            "sub_network": "first",
            "country": "first",
            "symbol": "first",
            "tags": "first",
            "substation_lv": "first",
            "substation_off": "first",
            "under_construction": "first",
        }
    if line_strategies is None:
        line_strategies = {
            "x": "equivalent_reactance",
            "r": "equivalent_reactance",
            "s_nom": "sum",
            "length": "average",
        }
    if transformer_strategies is None:
        transformer_strategies = {
            "x": "equivalent_reactance",
            "r": "equivalent_reactance",
            "s_nom": "sum",
        }
    if link_strategies is None:
        link_strategies = {
            "p_nom": "sum",
            "length": "average",
        }

    # Build graph and partition map
    G = _build_networkx_graph_from_pypsa(
        n, include_transformers=True, include_links=True
    )
    partition_map = _busmap_to_partition_map(busmap)

    # Build the AggregationProfile
    if aggregation_mode is not None and aggregation_profile is not None:
        msg = "Cannot specify both aggregation_mode and aggregation_profile"
        raise ValueError(msg)
    elif aggregation_profile is not None:
        profile = aggregation_profile
    elif aggregation_mode is not None:
        profile = AggregationManager.get_mode_profile(aggregation_mode)
        # Merge user node strategies into the mode profile
        profile.node_properties.update(node_strategies)
        # Merge per-type edge strategies
        profile.edge_type_properties.setdefault("line", {}).update(line_strategies)
        profile.edge_type_properties.setdefault("trafo", {}).update(
            transformer_strategies
        )
        profile.edge_type_properties.setdefault("dc_link", {}).update(link_strategies)
    else:
        profile = AggregationProfile(
            node_properties=node_strategies,
            edge_type_properties={
                "line": line_strategies,
                "trafo": transformer_strategies,
                "dc_link": link_strategies,
            },
            default_node_strategy="average",
            default_edge_strategy="sum",
            warn_on_defaults=False,
        )

    # Run aggregation through NPAP's public API
    agg_manager = AggregationManager()
    aggregated = agg_manager.aggregate(G, partition_map, profile)

    # Extract buses from aggregated node attributes
    buses_data: dict[str, dict[str, Any]] = {}
    for node, attrs in aggregated.nodes(data=True):
        buses_data[str(node)] = dict(attrs)

    buses_df = pd.DataFrame.from_dict(buses_data, orient="index")
    buses_df.index.name = "Bus"

    # Reconcile NPAP aliases with original PyPSA columns
    _alias_to_pypsa = {"lon": "x", "lat": "y", "voltage": "v_nom"}
    for alias, pypsa_name in _alias_to_pypsa.items():
        if alias not in buses_df.columns:
            continue
        if pypsa_name in buses_df.columns:
            buses_df = buses_df.drop(columns=[alias])
        else:
            buses_df = buses_df.rename(columns={alias: pypsa_name})

    # Drop internal attributes that are not PyPSA bus attributes
    for col in ["ac_island"]:
        if col in buses_df.columns:
            buses_df = buses_df.drop(columns=[col])

    # Extract edges by type from the aggregated graph.
    # Typed aggregation returns a MultiDiGraph with one edge per type per
    # cluster pair; untyped returns a DiGraph.
    _internal_cols = {"type", "primary_voltage", "secondary_voltage"}

    def _edges_to_df(edge_type: str) -> pd.DataFrame:
        """Collect aggregated edges of *edge_type* into a DataFrame."""
        rows: dict[str, dict[str, Any]] = {}
        for u, v, data in aggregated.edges(data=True):
            if data.get("type") != edge_type:
                continue
            c1, c2 = (min(u, v), max(u, v))
            row_name = f"{c1}-{c2}"
            if row_name in rows:
                continue  # already collected the canonical direction
            edge_attrs: dict[str, Any] = {
                "bus0": str(c1),
                "bus1": str(c2),
                **{k: val for k, val in data.items() if k not in _internal_cols},
            }
            rows[row_name] = edge_attrs
        return pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame()

    lines_df = _edges_to_df("line")
    trafos_df = _edges_to_df("trafo")
    links_df = _edges_to_df("dc_link")

    # Replace inf values produced by EquivalentReactanceStrategy with NaN
    for df in (lines_df, trafos_df, links_df):
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_mask = np.isinf(df[numeric_cols])
            if inf_mask.any().any():
                n_inf = int(inf_mask.sum().sum())
                logger.warning(
                    "Replaced %d infinite value(s) in aggregated edges with NaN",
                    n_inf,
                )
                df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return {
        "buses": buses_df,
        "lines": lines_df,
        "transformers": trafos_df,
        "links": links_df,
    }


def _build_one_port_strategies(
    n: Network,
    component: str,
    one_port_strategies: dict,
) -> dict:
    """Build effective strategies for one-port aggregation.

    Handles both flat strategy dicts (applied to all components) and
    per-component dicts.  Injects ``"sum"`` for solver output attributes
    that are present in the component data so that they aggregate cleanly
    instead of falling back to ``consense``.

    Parameters
    ----------
    n : Network
        The PyPSA network (used to inspect component metadata).
    component : str
        Component name (e.g. ``"Generator"``).
    one_port_strategies : dict
        User-supplied strategies — either a flat dict (applied to every
        component) or a dict of dicts keyed by component name.

    Returns
    -------
    dict
        Merged strategy dict ready for ``aggregateoneport(custom_strategies=…)``.

    """
    effective: dict[str, Any] = {}

    # Inject "sum" for solver output attributes present in the component data
    attrs = n.components[component]["defaults"]
    output_columns = attrs.index[attrs.static & attrs.status.str.startswith("Output")]
    for col in output_columns:
        if col in n.c[component].static.columns:
            effective[col] = "sum"

    # Also cover dynamic-only output attributes (e.g. p, q time series)
    for attr_name in n.c[component].dynamic:
        if attr_name not in effective:
            effective[attr_name] = "sum"

    # Layer user strategies on top
    if one_port_strategies:
        # Check if this is a per-component dict (value for this component is a dict)
        per_component = one_port_strategies.get(component)
        if isinstance(per_component, dict):
            effective.update(per_component)
        elif not any(isinstance(v, dict) for v in one_port_strategies.values()):
            # Flat dict — apply to all components
            effective.update(one_port_strategies)

    return effective


def npap_clustering(
    n: Network,
    n_clusters: int,
    strategy: str = "geographical_kmeans",
    buses_i: pd.Index | None = None,
    include_transformers: bool = True,
    include_links: bool = True,
    voltage_levels: list[float] | None = None,
    node_strategies: dict[str, str] | None = None,
    line_strategies: dict[str, str] | None = None,
    transformer_strategies: dict[str, str] | None = None,
    link_strategies: dict[str, str] | None = None,
    aggregate_one_ports: dict | None = None,
    one_port_strategies: dict | None = None,
    aggregation_mode: Any | None = None,
    aggregation_profile: Any | None = None,
    with_time: bool = True,
    **kwargs: Any,
) -> Clustering:
    """Partition and aggregate a PyPSA network using NPAP.

    This combines NPAP partitioning and aggregation for buses/lines/transformers/links
    with PyPSA's ``aggregateoneport()`` for generators, loads, storage_units, and stores.

    Parameters
    ----------
    n : Network
        The PyPSA network to cluster.
    n_clusters : int
        Number of clusters to create.
    strategy : str, optional
        NPAP partitioning strategy name (default "geographical_kmeans").
    buses_i : pd.Index | None, optional
        Subset of buses to cluster. If None, all buses are used.
    include_transformers : bool, optional
        Include transformers as edges (default True).
    include_links : bool, optional
        Include links as edges (default True).
    voltage_levels : list[float] | None, optional
        Target voltage levels for voltage-aware strategies.
    node_strategies : dict[str, str] | None, optional
        NPAP strategy names for node properties. Defaults to average for
        lat/lon/voltage.
    line_strategies : dict[str, str] | None, optional
        NPAP strategy names for line edge properties. Defaults to
        equivalent_reactance for x/r, sum for s_nom, average for length.
    transformer_strategies : dict[str, str] | None, optional
        NPAP strategy names for transformer edge properties.
    link_strategies : dict[str, str] | None, optional
        NPAP strategy names for link edge properties.
    aggregate_one_ports : dict | None, optional
        List or dict of one-port components to aggregate
        (e.g. ``["Generator", "Load"]``).
        If None, defaults to empty (no one-port aggregation).
    one_port_strategies : dict | None, optional
        Custom strategies for one-port aggregation.  May be a flat dict
        (applied to all components) or a dict of dicts keyed by component
        name.  Solver output attributes automatically get ``"sum"``.
    aggregation_mode : AggregationMode | None, optional
        Pre-defined NPAP aggregation mode passed to
        :func:`aggregate_network_by_npap`.
    aggregation_profile : AggregationProfile | None, optional
        Fully custom NPAP aggregation profile passed to
        :func:`aggregate_network_by_npap`.
    with_time : bool, optional
        Whether to include time-dependent data (default True).
    **kwargs : Any
        Additional keyword arguments passed to busmap_by_npap.

    Returns
    -------
    Clustering
        Named tuple with attributes n (clustered Network), busmap, and linemap.

    """
    if aggregate_one_ports is None:
        aggregate_one_ports = {}
    if one_port_strategies is None:
        one_port_strategies = {}

    # Step 1: Get busmap from NPAP partitioning
    busmap = busmap_by_npap(
        n,
        n_clusters=n_clusters,
        strategy=strategy,
        buses_i=buses_i,
        include_transformers=include_transformers,
        include_links=include_links,
        voltage_levels=voltage_levels,
        **kwargs,
    )

    # Step 2: Get aggregated components from NPAP
    npap_result = aggregate_network_by_npap(
        n,
        busmap,
        node_strategies=node_strategies,
        line_strategies=line_strategies,
        transformer_strategies=transformer_strategies,
        link_strategies=link_strategies,
        aggregation_mode=aggregation_mode,
        aggregation_profile=aggregation_profile,
    )

    # Step 3: Create new network
    clustered = n.__class__()

    # Add NPAP-aggregated buses
    buses_df = npap_result["buses"]
    if not buses_df.empty:
        clustered.add("Bus", buses_df.index, **buses_df)

    # Add NPAP-aggregated lines
    lines_df = npap_result["lines"]
    linemap = pd.Series(dtype=str)
    if not lines_df.empty:
        clustered.add("Line", lines_df.index, **lines_df)

        # Build linemap: original line names -> aggregated line names.
        # Only cross-cluster lines are included (same as spatial.py).
        orig_lines = n.c.lines.static
        mapped_bus0 = orig_lines.bus0.map(busmap)
        mapped_bus1 = orig_lines.bus1.map(busmap)
        valid = mapped_bus0.notna() & mapped_bus1.notna()
        cross_cluster = valid & (mapped_bus0 != mapped_bus1)
        cross_bus0 = mapped_bus0[cross_cluster].astype(int)
        cross_bus1 = mapped_bus1[cross_cluster].astype(int)

        # Canonical name using integer min/max to match _edges_to_df
        lo = np.minimum(cross_bus0, cross_bus1)
        hi = np.maximum(cross_bus0, cross_bus1)
        linemap = lo.astype(str) + "-" + hi.astype(str)
        linemap.name = None

        # Keep only lines whose aggregated counterpart exists
        linemap = linemap[linemap.isin(lines_df.index)]

    # Add NPAP-aggregated transformers
    trafos_df = npap_result["transformers"]
    if not trafos_df.empty and include_transformers:
        clustered.add("Transformer", trafos_df.index, **trafos_df)

    # Add NPAP-aggregated links
    links_df = npap_result["links"]
    if not links_df.empty and include_links:
        clustered.add("Link", links_df.index, **links_df)

    # Warn if branch time series exist but won't be aggregated
    _ts_components = ["Line"]
    if include_transformers:
        _ts_components.append("Transformer")
    if include_links:
        _ts_components.append("Link")
    for _comp in _ts_components:
        for _attr, _df in n.c[_comp].dynamic.items():
            if not _df.empty:
                logger.warning(
                    "Branch time series '%s' for %s will not be aggregated "
                    "by NPAP clustering. Use busmap_by_npap() with "
                    "get_clustering_from_busmap() for full time series support.",
                    _attr,
                    _comp,
                )
                break  # one warning per component suffices

    # Replace zero resistance values with a small epsilon. This can happen
    # when voltage-aware electrical strategies aggregate parallel branches.
    _r_epsilon = 1e-6
    for component in ("lines", "transformers"):
        static = getattr(clustered, component)
        if "r" in static.columns:
            zero_r = static["r"] == 0
            if zero_r.any():
                n_fixed = int(zero_r.sum())
                logger.info(
                    "Replaced %d zero r value(s) in %s with epsilon=%g",
                    n_fixed,
                    component,
                    _r_epsilon,
                )
                static.loc[zero_r, "r"] = _r_epsilon

    # Carry forward global constraints
    clustered.c.global_constraints.static = n.c.global_constraints.static

    if with_time:
        clustered.set_snapshots(n.snapshots)
        clustered.snapshot_weightings = n.snapshot_weightings.copy()
        if not n.investment_periods.empty:
            clustered.set_investment_periods(n.investment_periods)
            clustered.investment_period_weightings = (
                n.investment_period_weightings.copy()
            )

    # Step 4: Aggregate one-port components using PyPSA's aggregateoneport
    one_port_components = n.one_port_components.copy()

    for one_port in aggregate_one_ports:
        one_port_components.remove(one_port)
        custom = _build_one_port_strategies(n, one_port, one_port_strategies)
        new_static, new_dynamic = aggregateoneport(
            n,
            busmap,
            component=one_port,
            with_time=with_time,
            custom_strategies=custom,
        )
        clustered.add(one_port, new_static.index, **new_static)
        for attr, df in new_dynamic.items():
            if not df.empty:
                clustered._import_series_from_df(df, one_port, attr)

    # Collect remaining one-port components (remap bus references only)
    for c in n.components:
        if c.name not in one_port_components:
            continue
        remaining = c.static.assign(bus=c.static.bus.map(busmap)).dropna(subset=["bus"])
        clustered.add(c.name, remaining.index, **remaining)

    if with_time:
        for c in n.components:
            if c.name not in one_port_components:
                continue
            for attr, df in c.dynamic.items():
                if not df.empty:
                    clustered._import_series_from_df(df, c.name, attr)

    # Handle links that were not aggregated by NPAP (when include_links=False)
    if not include_links:
        bus_mappings = {
            "bus0": n.c.links.static.bus0.map(busmap),
            "bus1": n.c.links.static.bus1.map(busmap),
        }
        for port in n.c.links.additional_ports:
            col = f"bus{port}"
            if col in n.c.links.static.columns:
                bus_mappings[col] = n.c.links.static[col].map(busmap)

        new_links = (
            n.c.links.static.assign(**bus_mappings)
            .dropna(subset=["bus0", "bus1"])
            .loc[lambda df: df.bus0 != df.bus1]
        )
        if not new_links.empty:
            clustered.add("Link", new_links.index, **new_links)
            if with_time:
                for attr, df in n.c.links.dynamic.items():
                    if not df.empty:
                        clustered._import_series_from_df(df, "Link", attr)

    # Add carriers
    clustered.add("Carrier", n.c.carriers.static.index, **n.c.carriers.static)

    clustered.determine_network_topology()

    return Clustering(clustered, busmap, linemap)
