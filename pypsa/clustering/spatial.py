# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Functions for computing network clusters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from packaging.version import Version, parse
from pandas import Series

from pypsa.common import _scenarios_not_implemented
from pypsa.geo import haversine_pts

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable

    from pypsa import Network

logger = logging.getLogger(__name__)


def _sum_keep_na(s: Series) -> float:
    """Sum keeping all nan groups as nan instead of collapsing them to 0."""
    return s.sum(min_count=1)


DEFAULT_ONE_PORT_STRATEGIES = {
    "p": "sum",
    "q": "sum",
    "p_set": _sum_keep_na,
    "q_set": _sum_keep_na,
    "p_nom": pd.Series.sum,  # resolve infinities, see https://github.com/pandas-dev/pandas/issues/54161
    "p_nom_max": pd.Series.sum,  # resolve infinities, see https://github.com/pandas-dev/pandas/issues/54161
    "p_nom_min": "sum",
    "e_nom": pd.Series.sum,  # resolve infinities, see https://github.com/pandas-dev/pandas/issues/54161
    "e_nom_max": pd.Series.sum,  # resolve infinities, see https://github.com/pandas-dev/pandas/issues/54161
    "e_nom_min": "sum",
    "weight": "sum",
    "ramp_limit_up": "mean",
    "ramp_limit_down": "mean",
    "ramp_limit_start_up": "mean",
    "ramp_limit_shut_down": "mean",
    "build_year": lambda x: 0,
    "lifetime": lambda x: np.inf,
    "control": lambda x: "",
    "p_max_pu": "capacity_weighted_average",
    "p_min_pu": "capacity_weighted_average",
    "capital_cost": "capacity_weighted_average",
    "marginal_cost": "capacity_weighted_average",
    "efficiency": "capacity_weighted_average",
    "max_hours": "capacity_weighted_average",
    "inflow": "sum",
}

DEFAULT_BUS_STRATEGIES = {
    "x": "mean",
    "y": "mean",
    "v_nom": "max",
    "v_mag_pu_max": "min",
    "v_mag_pu_min": "max",
    "generator": lambda x: "",
}

DEFAULT_LINE_STRATEGIES = {
    "r": "reciprocal_voltage_weighted_average",
    "x": "reciprocal_voltage_weighted_average",
    "g": "voltage_weighted_average",
    "b": "voltage_weighted_average",
    "terrain_factor": "mean",
    "s_min_pu": "capacity_weighted_average",
    "s_max_pu": "capacity_weighted_average",
    "s_nom": pd.Series.sum,  # resolve infinities, see https://github.com/pandas-dev/pandas/issues/54161
    "s_nom_min": "sum",
    "s_nom_max": pd.Series.sum,  # resolve infinities, see https://github.com/pandas-dev/pandas/issues/54161
    "s_nom_extendable": "any",
    "num_parallel": "sum",
    "capital_cost": "length_capacity_weighted_average",
    "v_ang_min": "max",
    "v_ang_max": "min",
    "lifetime": "capacity_weighted_average",
    "build_year": "capacity_weighted_average",
}


def normed_or_uniform(x: pd.Series) -> pd.Series:
    """Normalize a series by dividing it by its sum.

    When the sum is zero, a uniform distribution is returned instead.

    Parameters
    ----------
    x : pandas.Series
        The input series to normalize.

    Returns
    -------
    pandas.Series
        The normalized series, or a uniform distribution if the input sum is zero.

    Examples
    --------
    >>> x = pd.Series([1, 2, 3])
    >>> normed_or_uniform(x)
    0    0.166667
    1    0.333333
    2    0.500000
    dtype: float64

    """
    if x.sum(skipna=False) > 0:
        return x / x.sum()
    return pd.Series(1.0 / len(x), x.index)


def make_consense(component: str, attr: str) -> Callable:
    """Return a function to verify attribute values of a cluster in a component.

    The values should either be the same or all null.

    Parameters
    ----------
    component : str
        The name of the component.
    attr : str
        The name of the attribute to verify.

    Returns
    -------
    Callable
        A function that checks whether all values in the Series are the same or all null.

    Raises
    ------
    AssertionError
        If the attribute values in a cluster are not the same or all null.

    """

    def consense(x: Series) -> object:
        v = x.iat[0]
        if not (x == v).all() and not x.isnull().all():
            msg = (
                f"In {component} cluster {x.name}, the values of attribute "
                f"{attr} do not agree:\n{x}"
            )
            raise ValueError(msg)
        return v

    return consense


def align_strategies(strategies: dict, keys: Iterable, component: str) -> dict:
    """Aligns the given strategies with the given keys.

    Parameters
    ----------
    strategies : dict
        The strategies to align.
    keys : list
        The keys to align the strategies with.
    component : str
        The component to align the strategies with.

    Returns
    -------
    dict
        The aligned strategies.

    """
    strategies |= {
        k: make_consense(component, k) for k in set(keys).difference(strategies)
    }
    return {k: strategies[k] for k in keys}


def flatten_multiindex(m: pd.MultiIndex, join: str = " ") -> pd.Index:
    """Flatten a multiindex by joining the levels with the given string.

    Parameters
    ----------
    m : pd.MultiIndex
        The multiindex to flatten.
    join : str, optional
        The string to join the levels with (default is " ").

    Returns
    -------
    pd.Index
        The flattened index.

    Examples
    --------
    >>> m = pd.MultiIndex.from_tuples([("a", "b"), ("c", "d")])
    >>> flatten_multiindex(m)
    Index(['a b', 'c d'], dtype='str')

    """
    return m if m.nlevels <= 1 else m.to_flat_index().str.join(join).str.strip()


def aggregateoneport(
    n: Network,
    busmap: dict,
    component: str,
    carriers: Iterable | None = None,
    buses: Iterable | None = None,
    with_time: bool = True,
    custom_strategies: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate one port components in the network based on the given busmap.

    Parameters
    ----------
    n : Network
        The network containing the generators.
    busmap : dict
        A dictionary mapping old bus IDs to new bus IDs.
    component : str
        The component to aggregate.
    carriers : list, optional
        List of carriers to be considered (default is all carriers).
    buses : list, optional
        List of buses to be considered (default is all buses).
    with_time : bool, optional
        Whether to include time-dependent attributes (default is True).
    custom_strategies : dict, optional
        Custom aggregation strategies (default is empty dict).

    Returns
    -------
    static : DataFrame
        DataFrame of the aggregated generators.
    dynamic : dict
        Dictionary of the aggregated dynamic data.

    """
    if custom_strategies is None:
        custom_strategies = {}
    c = component
    static = n.c[c].static
    attrs = n.components[c]["defaults"]
    if "carrier" in static.columns:
        if carriers is None:
            carriers = static.carrier.unique()
        to_aggregate = static.carrier.isin(carriers)
    else:
        to_aggregate = pd.Series(True, static.index)

    if buses is not None:
        to_aggregate |= static.bus.isin(buses)

    static = static[to_aggregate]
    static = static.assign(bus=static.bus.map(busmap))

    output_columns = attrs.index[attrs.static & attrs.status.str.startswith("Output")]
    columns = [c for c in static.columns if c not in output_columns]

    strategies = {**DEFAULT_ONE_PORT_STRATEGIES, **custom_strategies}
    static_strategies = align_strategies(strategies, columns, c)

    grouper = (
        [static.bus, static.carrier] if "carrier" in static.columns else static.bus
    )
    capacity = static.columns.intersection({"p_nom", "e_nom"})
    if len(capacity):
        capacity_weights = (
            static[capacity[0]].groupby(grouper).transform(normed_or_uniform)
        )
    if "weight" in static.columns:
        weights = static.weight.groupby(grouper).transform(normed_or_uniform)

    for k, v in static_strategies.items():
        if v == "weighted_average":
            static[k] = static[k] * weights
            static_strategies[k] = "sum"
        elif v == "capacity_weighted_average":
            static[k] = static[k] * capacity_weights
            static_strategies[k] = "sum"
        elif v == "weighted_min":
            static["p_nom_max"] /= weights
            static_strategies[k] = "min"

    aggregated = static.groupby(grouper).agg(static_strategies)
    aggregated.index = flatten_multiindex(aggregated.index).rename(c)

    non_aggregated = n.c[c].static[~to_aggregate]
    non_aggregated = non_aggregated.assign(bus=non_aggregated.bus.map(busmap))

    static = pd.concat([aggregated, non_aggregated], sort=False)
    static.fillna(attrs.default, inplace=True)

    dynamic = {}
    if with_time:
        dynamic_strategies = align_strategies(strategies, n.c[c].dynamic, c)
        for attr, data in n.c[c].dynamic.items():
            if data.empty:
                dynamic[attr] = data
                continue
            strategy = dynamic_strategies[attr]
            data = n.get_switchable_as_dense(c, attr)
            aggregated = data.loc[:, to_aggregate]

            if strategy == "weighted_average":
                aggregated = aggregated * weights
                aggregated = aggregated.T.groupby(grouper).sum().T
            elif strategy == "capacity_weighted_average":
                aggregated = aggregated * capacity_weights
                aggregated = aggregated.T.groupby(grouper).sum().T
            elif strategy == "weighted_min":
                aggregated = aggregated / weights
                aggregated = aggregated.T.groupby(grouper).min().T
            else:
                aggregated = aggregated.T.groupby(grouper).agg(strategy).T
            aggregated.columns = flatten_multiindex(aggregated.columns).rename(c)

            non_aggregated = data.loc[:, ~to_aggregate]

            dynamic[attr] = pd.concat([aggregated, non_aggregated], axis=1, sort=False)

            # filter out static values
            if attr in static:
                is_static = (dynamic[attr] == static[attr]).all()
                dynamic[attr] = dynamic[attr].loc[:, ~is_static]

    return static, dynamic


def aggregatebuses(
    n: Network, busmap: dict, custom_strategies: dict | None = None
) -> pd.DataFrame:
    """Aggregate buses in the network based on the given busmap.

    Parameters
    ----------
    n : Network
        The network containing the buses.
    busmap : dict
        A dictionary mapping old bus IDs to new bus IDs.
    custom_strategies : dict, optional
        Custom aggregation strategies (default is empty dict).

    Returns
    -------
    static : DataFrame
        DataFrame of the aggregated buses.

    """
    if custom_strategies is None:
        custom_strategies = {}
    c = "Bus"
    attrs = n.components[c]["defaults"]

    output_columns = attrs.index[attrs.static & attrs.status.str.startswith("Output")]
    columns = [c for c in n.c.buses.static.columns if c not in output_columns]

    strategies = {**DEFAULT_BUS_STRATEGIES, **custom_strategies}
    strategies = align_strategies(strategies, columns, c)

    aggregated = n.c.buses.static.groupby(busmap).agg(strategies)
    aggregated.index = flatten_multiindex(aggregated.index).rename(c)

    return aggregated


def aggregatelines(
    n: Network,
    busmap: dict,
    line_length_factor: float = 1.0,
    with_time: bool = True,
    custom_strategies: dict | None = None,
    bus_strategies: dict | None = None,
    custom_line_groupers: Iterable = [],
) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Aggregate lines in the network based on the given busmap.

    Parameters
    ----------
    n : Network
        The network containing the lines.
    busmap : dict
        A dictionary mapping old bus IDs to new bus IDs.
    line_length_factor : float, optional
        A factor to multiply the length of each line by (default is 1.0).
    with_time : bool, optional
        Whether to aggregate dynamic data (default is True).
    custom_strategies : dict, optional
        Custom aggregation strategies (default is empty dict).
    bus_strategies : dict, optional
        Custom aggregation strategies for buses (default is empty dict).
    custom_line_groupers : list, optional
        Additional custom groupers for the lines. Specifies that different column values are not aggregated. (default is empty list).

    Returns
    -------
    static : DataFrame
        DataFrame of the aggregated lines.
    dynamic : dict
        Dictionary of DataFrames of the aggregated dynamic data (if with_time is True).

    """
    if custom_strategies is None:
        custom_strategies = {}
    if bus_strategies is None:
        bus_strategies = {}
    attrs = n.components["Line"]["defaults"]
    static = n.c["Line"].static
    idx = static.index[static.bus0.map(busmap) != static.bus1.map(busmap)]
    static = static.loc[idx]

    orig_length = static.length
    orig_v_nom = static.bus0.map(n.c.buses.static.v_nom)

    bus_strategies = {**DEFAULT_BUS_STRATEGIES, **bus_strategies}
    cols = ["x", "y", "v_nom"]
    buses = (
        n.c.buses.static[cols].groupby(busmap).agg({c: bus_strategies[c] for c in cols})
    )

    static = static.assign(bus0=static.bus0.map(busmap), bus1=static.bus1.map(busmap))
    reverse_order = static.bus0 > static.bus1
    reverse_values = static.loc[reverse_order, ["bus1", "bus0"]].values
    static.loc[reverse_order, ["bus0", "bus1"]] = reverse_values

    output_columns = attrs.index[attrs.static & attrs.status.str.startswith("Output")]
    columns = [c for c in static.columns if c not in output_columns]

    strategies = {**DEFAULT_LINE_STRATEGIES, **custom_strategies}
    static_strategies = align_strategies(strategies, columns, "Line")

    grouper = (
        static.groupby(["bus0", "bus1", *custom_line_groupers]).ngroup().astype(str)
    )

    coords = buses[["x", "y"]]
    length = (
        haversine_pts(coords.loc[static.bus0], coords.loc[static.bus1])
        * line_length_factor
    )
    static = static.assign(length=length)

    length_factor = (static.length / orig_length).where(orig_length > 0, static.length)
    v_nom = pd.concat(
        [static.bus0.map(buses.v_nom), static.bus1.map(buses.v_nom)], axis=1
    ).max(axis=1)
    voltage_factor = (orig_v_nom / v_nom) ** 2
    capacity_weights = static.groupby(grouper).s_nom.transform(normed_or_uniform)

    for col, strategy in static_strategies.items():
        if strategy == "capacity_weighted_average":
            static[col] = static[col] * capacity_weights
            static_strategies[col] = "sum"
        elif strategy == "reciprocal_voltage_weighted_average":
            static[col] = voltage_factor / (length_factor * static[col])
            static_strategies[col] = lambda x: 1.0 / x.sum()
        elif strategy == "voltage_weighted_average":
            static[col] = voltage_factor * length_factor * static[col]
            static_strategies[col] = "sum"
        elif strategy == "length_capacity_weighted_average":
            static[col] = static[col] * length_factor * capacity_weights
            static_strategies[col] = "sum"

    static = static.groupby(grouper).agg(static_strategies)

    dynamic = {}
    if with_time:
        dynamic_strategies = align_strategies(strategies, n.c["Line"].dynamic, "Line")

        for attr, data in n.c.lines.dynamic.items():
            if data.empty:
                dynamic[attr] = data
                continue

            strategy = dynamic_strategies[attr]
            data = n.get_switchable_as_dense("Line", attr, inds=idx)

            if strategy == "capacity_weighted_average":
                data = data * capacity_weights
                data = data.T.groupby(grouper).sum().T
            else:
                data = data.T.groupby(grouper).agg(strategy).T

            dynamic[attr] = data

            # filter out static values
            if attr in static:
                is_static = (dynamic[attr] == static[attr]).all()
                dynamic[attr] = dynamic[attr].loc[:, ~is_static]

    return static, dynamic, grouper


@dataclass
class Clustering:
    """Clustering result."""

    n: Any
    busmap: pd.Series
    linemap: pd.Series


def _add_aggregated_one_port_components(
    n: Network,
    clustered: Network,
    busmap: dict | pd.Series,
    one_port_components: set[str],
    aggregate_one_ports: Iterable[str],
    with_time: bool,
    get_custom_strategies: Callable[[str], dict] | None = None,
) -> None:
    """Aggregate selected one-port components into a clustered network."""
    for one_port in aggregate_one_ports:
        one_port_components.remove(one_port)
        custom_strategies = (
            get_custom_strategies(one_port) if get_custom_strategies is not None else {}
        )
        new_static, new_dynamic = aggregateoneport(
            n,
            busmap,
            component=one_port,
            with_time=with_time,
            custom_strategies=custom_strategies,
        )
        clustered.add(one_port, new_static.index, **new_static)
        for attr, df in new_dynamic.items():
            if not df.empty:
                clustered._import_series_from_df(df, one_port, attr)


def _add_remaining_one_port_components(
    n: Network,
    clustered: Network,
    busmap: dict | pd.Series,
    one_port_components: set[str],
    with_time: bool,
) -> None:
    """Remap unaggregated one-port components into a clustered network."""
    for c in n.components:
        if c.name not in one_port_components:
            continue
        remaining_one_port_data = c.static.assign(bus=c.static.bus.map(busmap)).dropna(
            subset=["bus"]
        )
        clustered.add(c.name, remaining_one_port_data.index, **remaining_one_port_data)

    if with_time:
        for c in n.components:
            if c.name not in one_port_components:
                continue
            for attr, df in c.dynamic.items():
                if not df.empty:
                    clustered._import_series_from_df(df, c.name, attr)


def _build_networkx_graph_from_pypsa(
    n: Network,
    buses_i: pd.Index | None = None,
    include_transformers: bool = True,
    include_links: bool = False,
) -> nx.MultiGraph:
    """Convert a PyPSA Network into a NetworkX MultiGraph with NPAP-compatible attributes.

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
    nx.MultiGraph
        A NetworkX MultiGraph with NPAP-compatible node and edge attributes.

    """
    buses = n.c.buses.static
    if buses_i is not None:
        buses = buses.loc[buses_i]

    branch_components = ["Line"]
    if include_transformers:
        branch_components.append("Transformer")
    if include_links:
        branch_components.append("Link")

    G = n.graph(branch_components=branch_components)
    if buses_i is not None:
        G = G.subgraph(buses.index).copy()

    G.remove_edges_from(list(nx.selfloop_edges(G, keys=True)))

    # Columns to skip when copying bus attributes (not meaningful on nodes)
    _bus_skip = {"generator"}

    # Add nodes: carry ALL bus static columns + NPAP aliases
    for bus_name, bus_data in buses.iterrows():
        node_attrs: dict[str, Any] = {}
        for col in buses.columns:
            if col in _bus_skip:
                continue
            val = bus_data[col]
            if not isinstance(val, float) or not pd.isna(val):
                node_attrs[col] = val
        node_attrs.update(
            {
                "lon": bus_data.get("x", 0.0),
                "lat": bus_data.get("y", 0.0),
                "voltage": bus_data.get("v_nom", 0.0),
            }
        )
        G.nodes[bus_name].update(node_attrs)

    # Columns to exclude from edge attribute copying (used for connectivity)
    _edge_skip = {"bus0", "bus1"}
    has_v_nom = "v_nom" in buses.columns

    component_edge_types = {
        "Line": "line",
        "Transformer": "trafo",
        "Link": "dc_link",
    }

    for bus0, bus1, edge_key in G.edges(keys=True):
        component, branch_name = edge_key
        row = n.c[component].static.loc[branch_name]

        edge_attrs: dict[str, Any] = {}
        for col in row.index:
            if col in _edge_skip:
                continue
            val = row[col]
            if not isinstance(val, float) or not pd.isna(val):
                edge_attrs[col] = val

        # Set edge class marker LAST so it is never overwritten by a
        # PyPSA column of the same name (e.g. Line.type = line spec).
        edge_attrs["type"] = component_edge_types[component]

        if component != "Link" and has_v_nom:
            edge_attrs["primary_voltage"] = buses.at[row["bus0"], "v_nom"]
            edge_attrs["secondary_voltage"] = buses.at[row["bus1"], "v_nom"]

        G.edges[bus0, bus1, edge_key].update(edge_attrs)

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


def _as_npap_input_graph(G: nx.MultiGraph) -> nx.Graph | nx.MultiGraph:
    """Keep a MultiGraph only when it contains parallel branches."""
    if G.number_of_edges() == nx.Graph(G).number_of_edges():
        return nx.Graph(G)
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


def _aggregate_network_by_npap(
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
    from npap.managers import (  # noqa: PLC0415
        AggregationManager,
        PartitionAggregatorManager,
    )

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
            "active": "first",
        }
    else:
        line_strategies = {"active": "first", **line_strategies}
    if transformer_strategies is None:
        transformer_strategies = {
            "x": "equivalent_reactance",
            "r": "equivalent_reactance",
            "s_nom": "sum",
            "active": "first",
        }
    else:
        transformer_strategies = {"active": "first", **transformer_strategies}
    if link_strategies is None:
        link_strategies = {
            "p_nom": "sum",
            "length": "average",
            "active": "first",
        }
    else:
        link_strategies = {"active": "first", **link_strategies}

    # Build graph and partition map
    G = _build_networkx_graph_from_pypsa(
        n, include_transformers=True, include_links=True
    )
    loader = PartitionAggregatorManager()
    G = loader.load_data("networkx_direct", graph=_as_npap_input_graph(G))
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
    output_attrs = attrs.index[attrs.status.str.startswith("Output")]
    for attr_name in output_attrs:
        if (
            attr_name in n.c[component].static.columns
            or attr_name in n.c[component].dynamic
        ):
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


class SpatialClusteringMixin:
    """Mixin for spatial clustering methods.

    Class inherits to [`pypsa.clustering.SpatialClusteringAccessor`][]. All methods
    available via `n.cluster.spatial`.
    """

    _n: Network

    @_scenarios_not_implemented
    def busmap_by_kmeans(
        self,
        bus_weightings: pd.Series,
        n_clusters: int,
        buses_i: pd.Index | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Create a bus map from the clustering of buses in space with a weighting.

        Parameters
        ----------
        bus_weightings : pandas.Series
            Series of integer weights for buses, indexed by bus names.
        n_clusters : int
            Final number of clusters desired.
        buses_i : None|pandas.Index
            If not None (default), subset of buses to cluster.
        kwargs
            Any remaining arguments to be passed to KMeans (e.g. n_init, n_jobs).

        Returns
        -------
        busmap : pandas.Series
            Mapping of n.buses to k-means clusters (indexed by
            non-negative integers).

        """
        n = self._n

        if find_spec("sklearn") is None:
            msg = (
                "Optional dependency 'sklearn' not found."
                "Install via 'conda install -c conda-forge scikit-learn' "
                "or 'pip install scikit-learn'"
            )
            raise ModuleNotFoundError(msg)

        from sklearn.cluster import KMeans  # noqa: PLC0415

        if buses_i is None:
            buses_i = n.c.buses.static.index

        # since one cannot weight points directly in the scikit-learn
        # implementation of k-means, just add additional points at
        # same position
        points = n.c.buses.static.loc[buses_i, ["x", "y"]].values.repeat(
            bus_weightings.reindex(buses_i).astype(int), axis=0
        )

        kwargs.setdefault("n_init", "auto")
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, **kwargs)

        kmeans.fit(points)

        return pd.Series(
            data=kmeans.predict(n.c.buses.static.loc[buses_i, ["x", "y"]].values),
            index=buses_i,
        ).astype(str)

    @_scenarios_not_implemented
    def busmap_by_hac(
        self,
        n_clusters: int,
        buses_i: pd.Index | None = None,
        branch_components: Collection[str] | None = None,
        feature: pd.DataFrame | None = None,
        affinity: str | Callable = "euclidean",
        linkage: str = "ward",
        **kwargs: Any,
    ) -> pd.Series:
        """Create a busmap according to Hierarchical Agglomerative Clustering.

        Parameters
        ----------
        n_clusters : int
            Final number of clusters desired.
        buses_i: None | pandas.Index, default=None
            Subset of buses to cluster. If None, all buses are considered.
        branch_components: List, default=None
            Subset of all branch_components in the network. If None, all branch_components are considered.
        feature: None | pandas.DataFrame, default=None
            Feature to be considered for the clustering.
            The DataFrame must be indexed with buses_i.
            If None, all buses have the same similarity.
        affinity: str or Callable, default='euclidean'
            Metric used to compute the linkage.
            Can be "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed".
            If linkage is "ward", only "euclidean" is accepted.
            If "precomputed", a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
        linkage: 'ward', 'complete', 'average' or 'single', default='ward'
            Which linkage criterion to use.
            The linkage criterion determines which distance to use between sets of observation.
            The algorithm will merge the pairs of cluster that minimize this criterion.
            - 'ward' minimizes the variance of the clusters being merged.
            - 'average' uses the average of the distances of each observation of the two sets.
            - 'complete' or 'maximum' linkage uses the maximum distances between all observations of the two sets.
            - 'single' uses the minimum of the distances between all observations of the two sets.
        kwargs:
            Any remaining arguments to be passed to Hierarchical Clustering (e.g. memory, connectivity).

        Returns
        -------
        busmap : pandas.Series
            Mapping of n.buses to clusters (indexed by
            non-negative integers).

        """
        n = self._n

        if find_spec("sklearn") is None:
            msg = (
                "Optional dependency 'sklearn' not found."
                "Install via 'conda install -c conda-forge scikit-learn' "
                "or 'pip install scikit-learn'"
            )
            raise ModuleNotFoundError(msg)

        from sklearn.cluster import AgglomerativeClustering as HAC  # noqa: PLC0415

        if buses_i is None:
            buses_i = n.c.buses.static.index

        if branch_components is None:
            branch_components = n.branch_components

        if feature is None:
            logger.warning(
                "No feature is specified for Hierarchical Clustering. "
                "Falling back to default, where all buses have equal similarity. "
                "You can specify a feature as pandas.DataFrame indexed with buses_i."
            )

            feature = pd.DataFrame(index=buses_i, columns=[""], data=0)

        buses_x = n.c.buses.static.index.get_indexer(buses_i)

        adjacency_df = n.adjacency_matrix(
            branch_components=branch_components, return_dataframe=True
        )
        A = sp.csr_matrix(adjacency_df.values).tocsc()[buses_x][:, buses_x]

        labels = HAC(
            n_clusters=n_clusters,
            connectivity=A,
            metric=affinity,
            linkage=linkage,
            **kwargs,
        ).fit_predict(feature)

        return pd.Series(labels, index=buses_i, dtype=str)

    @_scenarios_not_implemented
    def busmap_by_greedy_modularity(  # noqa: D417
        self,
        n_clusters: int,
        buses_i: pd.Index | None = None,
    ) -> pd.Series:
        """Create a busmap according to Clauset-Newman-Moore greedy modularity maximization.

        See [CNM2004_1]_ for more details.

        Parameters
        ----------
        n_clusters : int
            Final number of clusters desired.
        buses_i: None | pandas.Index, default=None
            Subset of buses to cluster. If None, all buses are considered.

        Returns
        -------
        busmap : pandas.Series
            Mapping of n.buses to clusters (indexed by
            non-negative integers).

        References
        ----------
        [CNM2004_1] Clauset, A., Newman, M. E., & Moore, C.
            "Finding community structure in very large networks."
            Physical Review E 70(6), 2004.

        """
        n = self._n

        if parse(nx.__version__) < Version("2.8"):
            msg = (
                "The fuction `busmap_by_greedy_modularity` requires `networkx>=2.8`, "
                f"but version `networkx={nx.__version__}` is installed."
            )
            raise NotImplementedError(msg)

        if buses_i is None:
            buses_i = n.c.buses.static.index

        n.calculate_dependent_values()

        lines = n.c.lines.static.query("bus0 in @buses_i and bus1 in @buses_i")
        lines = (
            lines[["bus0", "bus1"]]
            .assign(weight=lines.s_nom / abs(lines.r + 1j * lines.x))
            .set_index(["bus0", "bus1"])
        )

        G = nx.Graph()
        G.add_nodes_from(buses_i)
        G.add_edges_from((u, v, {"weight": w}) for (u, v), w in lines.itertuples())

        communities = nx.community.greedy_modularity_communities(
            G, best_n=n_clusters, cutoff=n_clusters, weight="weight"
        )
        busmap = pd.Series(buses_i, buses_i)
        for c in np.arange(len(communities)):
            busmap.loc[list(communities[c])] = str(c)
        busmap.index = busmap.index.astype(str)

        return busmap

    @_scenarios_not_implemented
    def cluster_by_kmeans(
        self,
        bus_weightings: pd.Series,
        n_clusters: int,
        line_length_factor: float = 1.0,
        **kwargs: Any,
    ) -> Network:
        """Cluster the network according to k-means clustering of the buses.

        Buses can be weighted by an integer in the series `bus_weightings`.

        Note that this clustering method completely ignores the branches of the network.

        Parameters
        ----------
        bus_weightings : pandas.Series
            Series of integer weights for buses, indexed by bus names.
        n_clusters : int
            Final number of clusters desired.
        line_length_factor : float
            Factor to multiply the spherical distance between new buses in order to get new
            line lengths.
        kwargs
            Any remaining arguments to be passed to KMeans (e.g. n_init, n_jobs)

        Returns
        -------
        Network
            The clustered network.

        """
        busmap = self.busmap_by_kmeans(
            bus_weightings=bus_weightings, n_clusters=n_clusters, **kwargs
        )
        return self.get_clustering_from_busmap(
            busmap, line_length_factor=line_length_factor
        ).n

    @_scenarios_not_implemented
    def cluster_by_hac(  # noqa: D417
        self,
        n_clusters: int,
        buses_i: pd.Index | None = None,
        branch_components: Collection[str] | None = None,
        feature: pd.DataFrame | None = None,
        affinity: str | Callable = "euclidean",
        linkage: str = "ward",
        line_length_factor: float = 1.0,
        **kwargs: Any,
    ) -> Network:
        """Cluster the network using Hierarchical Agglomerative Clustering.

        Parameters
        ----------
        n_clusters : int
            Final number of clusters desired.
        buses_i: None | pandas.Index, default=None
            Subset of buses to cluster. If None, all buses are considered.
        branch_components: List, default=["Line", "Link"]
            Subset of all branch_components in the network.
        feature: None | pandas.DataFrame, default=None
            Feature to be considered for the clustering.
            The DataFrame must be indexed with buses_i.
            If None, all buses have the same similarity.
        affinity: str or Callable, default='euclidean'
            Metric used to compute the linkage.
            Can be "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed".
            If linkage is "ward", only "euclidean" is accepted.
            If "precomputed", a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
        linkage: 'ward', 'complete', 'average' or 'single', default='ward'
            Which linkage criterion to use.
            The linkage criterion determines which distance to use between sets of observation.
            The algorithm will merge the pairs of cluster that minimize this criterion.
            - 'ward' minimizes the variance of the clusters being merged.
            - 'average' uses the average of the distances of each observation of the two sets.
            - 'complete' or 'maximum' linkage uses the maximum distances between all observations of the two sets.
            - 'single' uses the minimum of the distances between all observations of the two sets.
        line_length_factor: float, default=1.0
            Factor to multiply the spherical distance between two new buses in order to get new line lengths.
        kwargs:
            Any remaining arguments to be passed to Hierarchical Clustering (e.g. memory, connectivity).

        Returns
        -------
        Network
            The clustered network.

        """
        busmap = self.busmap_by_hac(
            n_clusters,
            buses_i,
            branch_components,
            feature,
            affinity,
            linkage,
            **kwargs,
        )
        return self.get_clustering_from_busmap(
            busmap, line_length_factor=line_length_factor
        ).n

    @_scenarios_not_implemented
    def cluster_by_greedy_modularity(  # noqa: D417
        self,
        n_clusters: int,
        buses_i: pd.Index | None = None,
        line_length_factor: float = 1.0,
    ) -> Network:
        """Cluster the network using Clauset-Newman-Moore greedy modularity maximization.

        See [CNM2004_2]_ for more details.

        Parameters
        ----------
        n_clusters : int
            Final number of clusters desired.
        buses_i: None | pandas.Index, default=None
            Subset of buses to cluster. If None, all buses are considered.
        line_length_factor: float, default=1.0
            Factor to multiply the spherical distance between two new buses to get new line lengths.

        Returns
        -------
        Network
            The clustered network.

        References
        ----------
        [CNM2004_2] Clauset, A., Newman, M. E., & Moore, C.
            "Finding community structure in very large networks."
            Physical Review E 70(6), 2004.

        """
        busmap = self.busmap_by_greedy_modularity(n_clusters, buses_i)
        return self.get_clustering_from_busmap(
            busmap, line_length_factor=line_length_factor
        ).n

    @_scenarios_not_implemented
    def busmap_by_npap(
        self,
        n_clusters: int,
        strategy: str = "geographical_kmeans",
        buses_i: pd.Index | None = None,
        include_transformers: bool = True,
        include_links: bool = False,
        voltage_levels: list[float] | None = None,
        parallel_edge_strategies: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Partition a PyPSA network using NPAP algorithms, returning a busmap.

        Parameters
        ----------
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
        n = self._n

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

        # Create manager and load graph
        manager = PartitionAggregatorManager()
        manager.load_data("networkx_direct", graph=_as_npap_input_graph(G))

        # By default, NPAP's partitioning algorithms do not handle parallel edges
        if isinstance(manager.get_current_graph(), nx.MultiDiGraph):
            edge_properties = {
                "type": "first",
                "primary_voltage": "first",
                "secondary_voltage": "first",
                "x": "equivalent_reactance",
                "r": "equivalent_reactance",
                "s_nom": "sum",
                "p_nom": "sum",
                "length": "average",
                "active": "first",
            }
            if parallel_edge_strategies is not None:
                edge_properties.update(parallel_edge_strategies)
            manager.aggregate_parallel_edges(
                edge_properties=edge_properties,
                default_strategy="first",
                warn_on_defaults=False,
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

    @_scenarios_not_implemented
    def get_npap_clustering_from_busmap(
        self,
        busmap: pd.Series,
        include_transformers: bool = True,
        include_links: bool = True,
        node_strategies: dict[str, str] | None = None,
        line_strategies: dict[str, str] | None = None,
        transformer_strategies: dict[str, str] | None = None,
        link_strategies: dict[str, str] | None = None,
        aggregate_one_ports: dict | None = None,
        one_port_strategies: dict | None = None,
        aggregation_mode: Any | None = None,
        aggregation_profile: Any | None = None,
        with_time: bool = True,
    ) -> Clustering:
        """Aggregate a PyPSA network from a busmap using NPAP.

        This combines NPAP aggregation for buses/lines/transformers/links
        with PyPSA's ``aggregateoneport()`` for generators, loads, storage_units, and stores.

        Parameters
        ----------
        busmap : pd.Series
            Series mapping bus names to cluster labels.
        include_transformers : bool, optional
            Include transformers as edges (default True).
        include_links : bool, optional
            Include links as edges (default True).
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
            Pre-defined NPAP aggregation mode passed to the NPAP aggregation helper.
        aggregation_profile : AggregationProfile | None, optional
            Fully custom NPAP aggregation profile passed to the NPAP aggregation helper.
        with_time : bool, optional
            Whether to include time-dependent data (default True).

        Returns
        -------
        Clustering
            Named tuple with attributes n (clustered Network), busmap, and linemap.

        """
        n = self._n

        if aggregate_one_ports is None:
            aggregate_one_ports = {}
        if one_port_strategies is None:
            one_port_strategies = {}

        # Get aggregated components from NPAP
        npap_result = _aggregate_network_by_npap(
            n,
            busmap,
            node_strategies=node_strategies,
            line_strategies=line_strategies,
            transformer_strategies=transformer_strategies,
            link_strategies=link_strategies,
            aggregation_mode=aggregation_mode,
            aggregation_profile=aggregation_profile,
        )

        # Create new network
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

        # Aggregate one-port components using PyPSA's aggregateoneport
        one_port_components = n.one_port_components.copy()

        _add_aggregated_one_port_components(
            n,
            clustered,
            busmap,
            one_port_components,
            aggregate_one_ports,
            with_time,
            lambda one_port: _build_one_port_strategies(
                n, one_port, one_port_strategies
            ),
        )

        # Collect remaining one-port components (remap bus references only)
        _add_remaining_one_port_components(
            n, clustered, busmap, one_port_components, with_time
        )

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

    @_scenarios_not_implemented
    def cluster_by_npap(
        self,
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
    ) -> Network:
        """Cluster a PyPSA network using NPAP.

        This combines NPAP partitioning and aggregation for buses/lines/transformers/links
        with PyPSA's ``aggregateoneport()`` for generators, loads, storage_units, and stores.

        Parameters
        ----------
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
            Pre-defined NPAP aggregation mode passed to the NPAP aggregation helper.
        aggregation_profile : AggregationProfile | None, optional
            Fully custom NPAP aggregation profile passed to the NPAP aggregation helper.
        with_time : bool, optional
            Whether to include time-dependent data (default True).
        **kwargs : Any
            Additional keyword arguments passed to busmap_by_npap.

        Returns
        -------
        Network
            The clustered network.

        """
        busmap = self.busmap_by_npap(
            n_clusters=n_clusters,
            strategy=strategy,
            buses_i=buses_i,
            include_transformers=include_transformers,
            include_links=include_links,
            voltage_levels=voltage_levels,
            **kwargs,
        )
        return self.get_npap_clustering_from_busmap(
            busmap,
            include_transformers=include_transformers,
            include_links=include_links,
            node_strategies=node_strategies,
            line_strategies=line_strategies,
            transformer_strategies=transformer_strategies,
            link_strategies=link_strategies,
            aggregate_one_ports=aggregate_one_ports,
            one_port_strategies=one_port_strategies,
            aggregation_mode=aggregation_mode,
            aggregation_profile=aggregation_profile,
            with_time=with_time,
        ).n

    @_scenarios_not_implemented
    def cluster_by_busmap(
        self,
        busmap: dict,
        with_time: bool = True,
        line_length_factor: float = 1.0,
        aggregate_generators_weighted: bool = False,
        aggregate_one_ports: dict | None = None,
        aggregate_generators_carriers: Iterable | None = None,
        scale_link_capital_costs: bool = True,
        bus_strategies: dict | None = None,
        one_port_strategies: dict | None = None,
        generator_strategies: dict | None = None,
        line_strategies: dict | None = None,
        aggregate_generators_buses: Iterable | None = None,
        custom_line_groupers: list | None = None,
    ) -> Network:
        """Cluster the network spatially by busmap.

        This function calls [`get_clustering_from_busmap`][pypsa.clustering.SpatialClusteringAccessor.get_clustering_from_busmap] internally.
        For more information, see the documentation of that function.

        Returns
        -------
        n : pypsa.Network

        """
        return self.get_clustering_from_busmap(
            busmap,
            with_time=with_time,
            line_length_factor=line_length_factor,
            aggregate_generators_weighted=aggregate_generators_weighted,
            aggregate_one_ports=aggregate_one_ports,
            aggregate_generators_carriers=aggregate_generators_carriers,
            scale_link_capital_costs=scale_link_capital_costs,
            bus_strategies=bus_strategies,
            one_port_strategies=one_port_strategies,
            generator_strategies=generator_strategies,
            line_strategies=line_strategies,
            aggregate_generators_buses=aggregate_generators_buses,
            custom_line_groupers=custom_line_groupers,
        ).n

    @_scenarios_not_implemented
    def get_clustering_from_busmap(
        self,
        busmap: dict,
        with_time: bool = True,
        line_length_factor: float = 1.0,
        aggregate_generators_weighted: bool = False,
        aggregate_one_ports: dict | None = None,
        aggregate_generators_carriers: Iterable | None = None,
        scale_link_capital_costs: bool = True,
        bus_strategies: dict | None = None,
        one_port_strategies: dict | None = None,
        generator_strategies: dict | None = None,
        line_strategies: dict | None = None,
        aggregate_generators_buses: Iterable | None = None,
        custom_line_groupers: list | None = None,
    ) -> Clustering:
        """Get a clustering result from a busmap."""
        n = self._n

        if bus_strategies is None:
            bus_strategies = {}
        if one_port_strategies is None:
            one_port_strategies = {}
        if generator_strategies is None:
            generator_strategies = {}
        if line_strategies is None:
            line_strategies = {}
        if aggregate_one_ports is None:
            aggregate_one_ports = {}
        if custom_line_groupers is None:
            custom_line_groupers = []

        buses = aggregatebuses(n, busmap, custom_strategies=bus_strategies)
        lines, lines_t, linemap = aggregatelines(
            n,
            busmap,
            line_length_factor,
            with_time=with_time,
            custom_strategies=line_strategies,
            bus_strategies=bus_strategies,
            custom_line_groupers=custom_line_groupers,
        )

        clustered = n.__class__()

        clustered.add("Bus", buses.index, **buses)
        clustered.add("Line", lines.index, **lines)

        # Carry forward global constraints to clustered n.
        clustered.c.global_constraints.static = n.c.global_constraints.static

        if with_time:
            clustered.set_snapshots(n.snapshots)
            clustered.snapshot_weightings = n.snapshot_weightings.copy()
            if not n.investment_periods.empty:
                clustered.set_investment_periods(n.investment_periods)
                clustered.investment_period_weightings = (
                    n.investment_period_weightings.copy()
                )
            for attr, df in lines_t.items():
                if not df.empty:
                    clustered._import_series_from_df(df, "Line", attr)

        one_port_components = n.one_port_components.copy()

        if aggregate_generators_weighted:
            # TODO: Remove this in favour of the more general approach below.
            one_port_components.remove("Generator")
            generators, generators_dynamic = aggregateoneport(
                n,
                busmap,
                "Generator",
                carriers=aggregate_generators_carriers,
                buses=aggregate_generators_buses,
                with_time=with_time,
                custom_strategies=generator_strategies,
            )
            clustered.add("Generator", generators.index, **generators)
            if with_time:
                for attr, df in generators_dynamic.items():
                    if not df.empty:
                        clustered._import_series_from_df(df, "Generator", attr)

        _add_aggregated_one_port_components(
            n,
            clustered,
            busmap,
            one_port_components,
            aggregate_one_ports,
            with_time,
            lambda one_port: one_port_strategies.get(one_port, {}),
        )

        # Collect remaining one ports

        _add_remaining_one_port_components(
            n, clustered, busmap, one_port_components, with_time
        )

        bus_mappings = {
            "bus0": n.c.links.static.bus0.map(busmap),
            "bus1": n.c.links.static.bus1.map(busmap),
        }

        # Also add additional ports if they exist
        for port in n.c.links.additional_ports:
            col = f"bus{port}"
            if col in n.c.links.static.columns:
                bus_mappings[col] = n.c.links.static[col].map(busmap)

        new_links = (
            n.c.links.static.assign(**bus_mappings)
            .dropna(subset=["bus0", "bus1"])  # Only require bus0 and bus1 to be non-NaN
            .loc[lambda df: df.bus0 != df.bus1]
        )

        new_links["length"] = np.where(
            new_links.length.notnull() & (new_links.length > 0),
            line_length_factor
            * haversine_pts(
                buses.loc[new_links["bus0"], ["x", "y"]],
                buses.loc[new_links["bus1"], ["x", "y"]],
            ),
            0,
        )
        if scale_link_capital_costs:
            new_links["capital_cost"] *= (
                new_links.length / n.c.links.static.length
            ).fillna(1)

        clustered.add("Link", new_links.index, **new_links)

        if with_time:
            for attr, df in n.c.links.dynamic.items():
                if not df.empty:
                    clustered._import_series_from_df(df, "Link", attr)

        clustered.add("Carrier", n.c.carriers.static.index, **n.c.carriers.static)

        clustered.determine_network_topology()

        return Clustering(clustered, busmap, linemap)


################
# Reduce stubs/dead-ends, i.e. nodes with valency 1, iteratively to remove tree-like structures


def busmap_by_stubs(
    n: Network, matching_attrs: Iterable[str] | None = None
) -> pd.Series:
    """Create a busmap by reducing stubs and stubby trees.

    In other words sequentially reducing dead-ends.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    matching_attrs : None|[str]
        bus attributes clusters have to agree on

    Returns
    -------
    busmap : pandas.Series
        Mapping of n.c.buses.static to k-means clusters (indexed by
        non-negative integers).

    """
    busmap = pd.Series(n.c.buses.static.index, n.c.buses.static.index)

    G = n.graph()

    def attrs_match(u: str, v: str) -> bool:
        return (
            matching_attrs is None
            or (
                n.c.buses.static.loc[u, matching_attrs]
                == n.c.buses.static.loc[v, matching_attrs]
            ).all()
        )

    while True:
        stubs = []
        for u in G.nodes:
            neighbours = list(G.adj[u].keys())
            if len(neighbours) == 1:
                (v,) = neighbours
                if attrs_match(u, v):
                    busmap[busmap == u] = v
                    stubs.append(u)
        G.remove_nodes_from(stubs)
        if not stubs:
            break
    return busmap


# Backward-compatible module-level functions


def busmap_by_kmeans(
    n: Network,
    bus_weightings: pd.Series,
    n_clusters: int,
    buses_i: pd.Index | None = None,
    **kwargs: Any,
) -> pd.Series:
    """Create a bus map from the clustering of buses in space with a weighting.

    Parameters
    ----------
    n : pypsa.Network
        The buses must have coordinates x, y.
    bus_weightings : pandas.Series
        Series of integer weights for buses, indexed by bus names.
    n_clusters : int
        Final number of clusters desired.
    buses_i : None|pandas.Index
        If not None (default), subset of buses to cluster.
    kwargs
        Any remaining arguments to be passed to KMeans (e.g. n_init, n_jobs).

    Returns
    -------
    busmap : pandas.Series
        Mapping of n.buses to k-means clusters (indexed by
        non-negative integers).

    """
    obj = SpatialClusteringMixin()
    obj._n = n
    return obj.busmap_by_kmeans(
        bus_weightings=bus_weightings,
        n_clusters=n_clusters,
        buses_i=buses_i,
        **kwargs,
    )


def busmap_by_hac(
    n: Network,
    n_clusters: int,
    buses_i: pd.Index | None = None,
    branch_components: Collection[str] | None = None,
    feature: pd.DataFrame | None = None,
    affinity: str | Callable = "euclidean",
    linkage: str = "ward",
    **kwargs: Any,
) -> pd.Series:
    """Create a busmap according to Hierarchical Agglomerative Clustering.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    n_clusters : int
        Final number of clusters desired.
    buses_i: None | pandas.Index, default=None
        Subset of buses to cluster. If None, all buses are considered.
    branch_components: List, default=None
        Subset of all branch_components in the network.
    feature: None | pandas.DataFrame, default=None
        Feature to be considered for the clustering.
    affinity: str or Callable, default='euclidean'
        Metric used to compute the linkage.
    linkage: 'ward', 'complete', 'average' or 'single', default='ward'
        Which linkage criterion to use.
    kwargs:
        Any remaining arguments to be passed to Hierarchical Clustering.

    Returns
    -------
    busmap : pandas.Series
        Mapping of n.buses to clusters.

    """
    obj = SpatialClusteringMixin()
    obj._n = n
    return obj.busmap_by_hac(
        n_clusters,
        buses_i,
        branch_components,
        feature,
        affinity,
        linkage,
        **kwargs,
    )


def busmap_by_greedy_modularity(
    n: Network, n_clusters: int, buses_i: pd.Index | None = None
) -> pd.Series:
    """Create a busmap according to Clauset-Newman-Moore greedy modularity maximization.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    n_clusters : int
        Final number of clusters desired.
    buses_i: None | pandas.Index, default=None
        Subset of buses to cluster.

    Returns
    -------
    busmap : pandas.Series
        Mapping of n.buses to clusters.

    """
    obj = SpatialClusteringMixin()
    obj._n = n
    return obj.busmap_by_greedy_modularity(n_clusters, buses_i)


def kmeans_clustering(
    n: Network,
    bus_weightings: pd.Series,
    n_clusters: int,
    line_length_factor: float = 1.0,
    **kwargs: Any,
) -> Clustering:
    """Cluster the network according to k-means clustering of the buses.

    Buses can be weighted by an integer in the series `bus_weightings`.

    Note that this clustering method completely ignores the branches of the network.

    Parameters
    ----------
    n : pypsa.Network
        The buses must have coordinates x, y.
    bus_weightings : pandas.Series
        Series of integer weights for buses, indexed by bus names.
    n_clusters : int
        Final number of clusters desired.
    line_length_factor : float
        Factor to multiply the spherical distance between new buses in order to get new
        line lengths.
    kwargs
        Any remaining arguments to be passed to KMeans (e.g. n_init, n_jobs)

    Returns
    -------
    Clustering : named tuple
        A named tuple containing network, busmap and linemap

    """
    obj = SpatialClusteringMixin()
    obj._n = n
    busmap = obj.busmap_by_kmeans(
        bus_weightings=bus_weightings, n_clusters=n_clusters, **kwargs
    )
    return obj.get_clustering_from_busmap(busmap, line_length_factor=line_length_factor)


def hac_clustering(  # noqa: D417
    n: Network,
    n_clusters: int,
    buses_i: pd.Index | None = None,
    branch_components: Collection[str] | None = None,
    feature: pd.DataFrame | None = None,
    affinity: str | Callable = "euclidean",
    linkage: str = "ward",
    line_length_factor: float = 1.0,
    **kwargs: Any,
) -> Clustering:
    """Cluster the network using Hierarchical Agglomerative Clustering.

    Parameters
    ----------
    n_clusters : int
        Final number of clusters desired.
    buses_i: None | pandas.Index, default=None
        Subset of buses to cluster. If None, all buses are considered.
    branch_components: List, default=["Line", "Link"]
        Subset of all branch_components in the network.
    feature: None | pandas.DataFrame, default=None
        Feature to be considered for the clustering.
        The DataFrame must be indexed with buses_i.
        If None, all buses have the same similarity.
    affinity: str or Callable, default=’euclidean’
        Metric used to compute the linkage.
        Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
        If linkage is “ward”, only “euclidean” is accepted.
        If “precomputed”, a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
    linkage: ‘ward’, ‘complete’, ‘average’ or ‘single’, default=’ward’
        Which linkage criterion to use.
        The linkage criterion determines which distance to use between sets of observation.
        The algorithm will merge the pairs of cluster that minimize this criterion.
        - ‘ward’ minimizes the variance of the clusters being merged.
        - ‘average’ uses the average of the distances of each observation of the two sets.
        - ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.
        - ‘single’ uses the minimum of the distances between all observations of the two sets.
    line_length_factor: float, default=1.0
        Factor to multiply the spherical distance between two new buses in order to get new line lengths.
    kwargs:
        Any remaining arguments to be passed to Hierarchical Clustering (e.g. memory, connectivity).


    Returns
    -------
    Clustering : named tuple
        A named tuple containing network, busmap and linemap

    """
    obj = SpatialClusteringMixin()
    obj._n = n
    busmap = obj.busmap_by_hac(
        n_clusters,
        buses_i,
        branch_components,
        feature,
        affinity,
        linkage,
        **kwargs,
    )
    return obj.get_clustering_from_busmap(busmap, line_length_factor=line_length_factor)


def greedy_modularity_clustering(
    n: Network,
    n_clusters: int,
    buses_i: pd.Index | None = None,
    line_length_factor: float = 1.0,
) -> Clustering:
    """Create a busmap according to Clauset-Newman-Moore greedy modularity maximization.

    See [CNM2004_2]_ for more details.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    n_clusters : int
        Final number of clusters desired.
    buses_i: None | pandas.Index, default=None
        Subset of buses to cluster. If None, all buses are considered.
    line_length_factor: float, default=1.0
        Factor to multiply the spherical distance between two new buses to get new line lengths.

    Returns
    -------
    Clustering : named tuple
        A named tuple containing network, busmap and linemap.

    References
    ----------
    [CNM2004_2] Clauset, A., Newman, M. E., & Moore, C.
        "Finding community structure in very large networks."
        Physical Review E 70(6), 2004.

    """
    obj = SpatialClusteringMixin()
    obj._n = n
    busmap = obj.busmap_by_greedy_modularity(n_clusters, buses_i)
    return obj.get_clustering_from_busmap(busmap, line_length_factor=line_length_factor)


def get_clustering_from_busmap(
    n: Network,
    busmap: dict,
    with_time: bool = True,
    line_length_factor: float = 1.0,
    aggregate_generators_weighted: bool = False,
    aggregate_one_ports: dict | None = None,
    aggregate_generators_carriers: Iterable | None = None,
    scale_link_capital_costs: bool = True,
    bus_strategies: dict | None = None,
    one_port_strategies: dict | None = None,
    generator_strategies: dict | None = None,
    line_strategies: dict | None = None,
    aggregate_generators_buses: Iterable | None = None,
    custom_line_groupers: list | None = None,
) -> Clustering:
    """Get a clustering result from a busmap.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    busmap : dict
        A dictionary mapping old bus IDs to new bus IDs.
    with_time : bool, optional
        Whether to include time-dependent attributes (default is True).
    line_length_factor : float, optional
        Factor to multiply line lengths (default is 1.0).
    aggregate_generators_weighted : bool, optional
        Whether to aggregate generators weighted (default is False).
    aggregate_one_ports : dict, optional
        One-port components to aggregate.
    aggregate_generators_carriers : list, optional
        Carriers to aggregate generators by.
    scale_link_capital_costs : bool, optional
        Whether to scale link capital costs (default is True).
    bus_strategies : dict, optional
        Custom aggregation strategies for buses.
    one_port_strategies : dict, optional
        Custom aggregation strategies for one-port components.
    generator_strategies : dict, optional
        Custom aggregation strategies for generators.
    line_strategies : dict, optional
        Custom aggregation strategies for lines.
    aggregate_generators_buses : list, optional
        Buses to aggregate generators by.
    custom_line_groupers : list, optional
        Additional custom groupers for lines.

    Returns
    -------
    Clustering
        A named tuple containing network, busmap and linemap.

    """
    obj = SpatialClusteringMixin()
    obj._n = n
    return obj.get_clustering_from_busmap(
        busmap,
        with_time=with_time,
        line_length_factor=line_length_factor,
        aggregate_generators_weighted=aggregate_generators_weighted,
        aggregate_one_ports=aggregate_one_ports,
        aggregate_generators_carriers=aggregate_generators_carriers,
        scale_link_capital_costs=scale_link_capital_costs,
        bus_strategies=bus_strategies,
        one_port_strategies=one_port_strategies,
        generator_strategies=generator_strategies,
        line_strategies=line_strategies,
        aggregate_generators_buses=aggregate_generators_buses,
        custom_line_groupers=custom_line_groupers,
    )


def busmap_by_npap(
    n: Network,
    n_clusters: int,
    strategy: str = "geographical_kmeans",
    buses_i: pd.Index | None = None,
    include_transformers: bool = True,
    include_links: bool = False,
    voltage_levels: list[float] | None = None,
    parallel_edge_strategies: dict[str, str] | None = None,
    **kwargs: Any,
) -> pd.Series:
    """Partition a PyPSA network using NPAP algorithms, returning a busmap.

    Parameters
    ----------
    n : pypsa.Network
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
        Target voltage levels for voltage-aware strategies.
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
    obj = SpatialClusteringMixin()
    obj._n = n
    return obj.busmap_by_npap(
        n_clusters=n_clusters,
        strategy=strategy,
        buses_i=buses_i,
        include_transformers=include_transformers,
        include_links=include_links,
        voltage_levels=voltage_levels,
        parallel_edge_strategies=parallel_edge_strategies,
        **kwargs,
    )


def get_npap_clustering_from_busmap(
    n: Network,
    busmap: pd.Series,
    include_transformers: bool = True,
    include_links: bool = True,
    node_strategies: dict[str, str] | None = None,
    line_strategies: dict[str, str] | None = None,
    transformer_strategies: dict[str, str] | None = None,
    link_strategies: dict[str, str] | None = None,
    aggregate_one_ports: dict | None = None,
    one_port_strategies: dict | None = None,
    aggregation_mode: Any | None = None,
    aggregation_profile: Any | None = None,
    with_time: bool = True,
) -> Clustering:
    """Aggregate a PyPSA network from a busmap using NPAP.

    This combines NPAP aggregation for buses/lines/transformers/links
    with PyPSA's ``aggregateoneport()`` for generators, loads, storage_units, and stores.

    Parameters
    ----------
    n : Network
        The PyPSA network to cluster.
    busmap : pd.Series
        Series mapping bus names to cluster labels.
    include_transformers : bool, optional
        Include transformers as edges (default True).
    include_links : bool, optional
        Include links as edges (default True).
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
        Pre-defined NPAP aggregation mode passed to the NPAP aggregation helper.
    aggregation_profile : AggregationProfile | None, optional
        Fully custom NPAP aggregation profile passed to the NPAP aggregation helper.
    with_time : bool, optional
        Whether to include time-dependent data (default True).

    Returns
    -------
    Clustering
        Named tuple with attributes n (clustered Network), busmap, and linemap.

    """
    obj = SpatialClusteringMixin()
    obj._n = n
    return obj.get_npap_clustering_from_busmap(
        busmap,
        include_transformers=include_transformers,
        include_links=include_links,
        node_strategies=node_strategies,
        line_strategies=line_strategies,
        transformer_strategies=transformer_strategies,
        link_strategies=link_strategies,
        aggregate_one_ports=aggregate_one_ports,
        one_port_strategies=one_port_strategies,
        aggregation_mode=aggregation_mode,
        aggregation_profile=aggregation_profile,
        with_time=with_time,
    )


def cluster_by_npap(
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
) -> Network:
    """Cluster a PyPSA network using NPAP.

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
        Pre-defined NPAP aggregation mode passed to the NPAP aggregation helper.
    aggregation_profile : AggregationProfile | None, optional
        Fully custom NPAP aggregation profile passed to the NPAP aggregation helper.
    with_time : bool, optional
        Whether to include time-dependent data (default True).
    **kwargs : Any
        Additional keyword arguments passed to busmap_by_npap.

    Returns
    -------
    Network
        The clustered network.

    """
    obj = SpatialClusteringMixin()
    obj._n = n
    return obj.cluster_by_npap(
        n_clusters=n_clusters,
        strategy=strategy,
        buses_i=buses_i,
        include_transformers=include_transformers,
        include_links=include_links,
        voltage_levels=voltage_levels,
        node_strategies=node_strategies,
        line_strategies=line_strategies,
        transformer_strategies=transformer_strategies,
        link_strategies=link_strategies,
        aggregate_one_ports=aggregate_one_ports,
        one_port_strategies=one_port_strategies,
        aggregation_mode=aggregation_mode,
        aggregation_profile=aggregation_profile,
        with_time=with_time,
        **kwargs,
    )
