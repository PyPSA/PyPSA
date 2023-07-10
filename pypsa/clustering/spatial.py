# -*- coding: utf-8 -*-
"""
Functions for computing network clusters.
"""

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2023 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import logging
from collections import namedtuple
from functools import reduce
from importlib.util import find_spec

import networkx as nx
import numpy as np
import pandas as pd
from deprecation import deprecated
from packaging.version import Version, parse
from pandas import Series

logger = logging.getLogger(__name__)


from pypsa import io
from pypsa.components import Network
from pypsa.geo import haversine_pts

DEFAULT_ONE_PORT_STRATEGIES = dict(
    p="sum",
    q="sum",
    p_set="sum",
    q_set="sum",
    p_nom="sum",
    p_nom_max="sum",
    p_nom_min="sum",
    e_nom="sum",
    e_nom_max="sum",
    e_nom_min="sum",
    weight="sum",
    ramp_limit_up="mean",
    ramp_limit_down="mean",
    ramp_limit_start_up="mean",
    ramp_limit_shut_down="mean",
    build_year=lambda x: 0,
    lifetime=lambda x: np.inf,
    p_max_pu="capacity_weighted_average",
    p_min_pu="capacity_weighted_average",
    capital_cost="capacity_weighted_average",
    marginal_cost="capacity_weighted_average",
    efficiency="capacity_weighted_average",
    max_hours="capacity_weighted_average",
)

DEFAULT_BUS_STRATEGIES = dict(
    x="mean",
    y="mean",
    v_nom="max",
    v_mag_pu_max="min",
    v_mag_pu_min="max",
)


def normed_or_uniform(x):
    """
    Normalize a series by dividing it by its sum, unless the sum is zero, in
    which case return a uniform distribution.

    Parameters
    ----------
    x : pandas.Series
        The input series to normalize.

    Returns
    -------
    pandas.Series
        The normalized series, or a uniform distribution if the input sum is zero.
    """
    if x.sum(skipna=False) > 0:
        return x / x.sum()
    else:
        return pd.Series(1.0 / len(x), x.index)


def weighted_average(key: str, weights: str) -> float:
    """
    Define a function to calculate the weighted average of a given key in a
    pandas DataFrame group.

    Parameters
    ----------
    key : str
        The key of the column to calculate the weighted average on.
    weights : str
        The key of the column to use as weights for the weighted average.

    Returns
    -------
    float
        The weighted average of the given key in the pandas DataFrame group.
    """

    def weighted_average_func(group: pd.DataFrame) -> float:
        return (group[key] * normed_or_uniform(group[weights])).sum()

    return weighted_average_func


def make_consense(component: str, attr: str) -> callable:
    """
    Returns a function to verify attribute values of a cluster in a component.
    The values should either be the same or all null.

    Parameters
    ----------
    component : str
        The name of the component.
    attr : str
        The name of the attribute to verify.

    Returns
    -------
    callable
        A function that checks whether all values in the Series are the same or all null.

    Raises
    ------
    AssertionError
        If the attribute values in a cluster are not the same or all null.
    """

    def consense(x: Series) -> object:
        v = x.iat[0]
        assert (x == v).all() or x.isnull().all(), (
            f"In {component} cluster {x.name}, the values of attribute "
            f"{attr} do not agree:\n{x}"
        )
        return v

    return consense


def align_strategies(strategies, keys, component):
    """
    Aligns the given strategies with the given keys.

    Parameters
    ----------
    strategies : dict
        The strategies to align.
    keys : list
        The keys to align the strategies with.

    Returns
    -------
    dict
        The aligned strategies.
    """
    strategies |= {
        k: make_consense(component, k) for k in set(keys).difference(strategies)
    }
    return {k: v for k, v in strategies.items() if k in keys}


@deprecated(details="Use `make_consense` instead.")
def _make_consense(component: str, attr: str) -> callable:
    return make_consense(component, attr)


def flatten_multiindex(m, join=" "):
    if m.nlevels <= 1:
        return m
    return m.to_flat_index().str.join(join).str.strip()


def aggregateoneport(
    n,
    busmap,
    component,
    carriers=None,
    buses=None,
    with_time=True,
    custom_strategies=dict(),
):
    """
    Aggregate one port components in the network based on the given busmap.

    Parameters
    ----------
    network : Network
        The network containing the generators.
    busmap : dict
        A dictionary mapping old bus IDs to new bus IDs.
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
    df : DataFrame
        DataFrame of the aggregated generators.
    pnl : dict
        Dictionary of the aggregated pnl data.
    """
    c = component
    df = n.df(c)
    attrs = n.components[c]["attrs"]

    if "carrier" in df.columns:
        if carriers is None:
            carriers = df.carrier.unique()
        to_aggregate = df.carrier.isin(carriers)
    else:
        to_aggregate = pd.Series(True, df.index)

    if buses is not None:
        to_aggregate |= df.bus.isin(buses)

    df = df[to_aggregate]
    df = df.assign(bus=df.bus.map(busmap))

    columns = set(attrs.index[attrs.static & attrs.status.str.startswith("Input")])
    columns = columns & set(df.columns)

    strategies = {**DEFAULT_ONE_PORT_STRATEGIES, **custom_strategies}
    static_strategies = align_strategies(strategies, columns, c)

    grouper = [df.bus, df.carrier] if "carrier" in df.columns else df.bus
    capacity_weights = df.p_nom.groupby(grouper, axis=0).transform(normed_or_uniform)
    if "weight" in df.columns:
        weights = df.weight.groupby(grouper, axis=0).transform(normed_or_uniform)

    for k, v in static_strategies.items():
        if v == "weighted_average":
            df[k] = df[k] * weights
            static_strategies[k] = "sum"
        elif v == "capacity_weighted_average":
            df[k] = df[k] * capacity_weights
            static_strategies[k] = "sum"
        elif v == "weighted_min":
            df["p_nom_max"] /= weights
            static_strategies[k] = "min"

    aggregated = df.groupby(grouper, axis=0).agg(static_strategies)
    aggregated.index = flatten_multiindex(aggregated.index).rename(c)

    non_aggregated = n.df(c)[~to_aggregate]
    non_aggregated = non_aggregated.assign(bus=non_aggregated.bus.map(busmap))

    df = pd.concat([aggregated, non_aggregated], sort=False)

    pnl = dict()
    if with_time:
        dynamic_strategies = align_strategies(strategies, n.pnl(c), c)
        for attr, data in n.pnl(c).items():
            strategy = dynamic_strategies[attr]
            cols = data.columns
            aggregated = data.loc[:, to_aggregate[cols]]

            if strategy == "weighted_average":
                aggregated = aggregated * weights[cols]
                aggregated = aggregated.groupby(grouper, axis=1).sum()
            elif strategy == "capacity_weighted_average":
                aggregated = aggregated * capacity_weights[cols]
                aggregated = aggregated.groupby(grouper, axis=1).sum()
            elif strategy == "weighted_min":
                aggregated = aggregated / weights[cols]
                aggregated = aggregated.groupby(grouper, axis=1).min()
            else:
                aggregated = aggregated.groupby(grouper, axis=1).agg(strategy)
            aggregated.columns = flatten_multiindex(aggregated.columns).rename(c)

            non_aggregated = data.loc[:, ~to_aggregate[cols]]

            pnl[attr] = pd.concat([aggregated, non_aggregated], axis=1, sort=False)

    return df, pnl


@deprecated(details="Use `aggregateoneport` instead.")
def aggregategenerators(
    n,
    busmap,
    carriers=None,
    buses=None,
    with_time=True,
    custom_strategies=dict(),
):
    return aggregateoneport(
        n, busmap, "Generator", carriers, buses, with_time, custom_strategies
    )


def aggregatebuses(n, busmap, custom_strategies=dict()):
    """
    Aggregate buses in the network based on the given busmap.

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
    df : DataFrame
        DataFrame of the aggregated buses.
    """
    c = "Bus"
    attrs = n.components[c]["attrs"]

    columns = set(attrs.index[attrs.static & attrs.status.str.startswith("Input")])
    columns = columns & set(n.buses.columns)

    strategies = {**DEFAULT_BUS_STRATEGIES, **custom_strategies}
    strategies = align_strategies(strategies, columns, c)

    aggregated = n.buses.groupby(busmap).agg(strategies)
    aggregated.index = flatten_multiindex(aggregated.index).rename(c)

    return aggregated


def aggregatelines(network, buses, interlines, line_length_factor=1.0, with_time=True):
    # make sure all lines have same bus ordering
    positive_order = interlines.bus0_s < interlines.bus1_s
    interlines_p = interlines[positive_order]
    interlines_n = interlines[~positive_order].rename(
        columns={"bus0_s": "bus1_s", "bus1_s": "bus0_s"}
    )
    interlines_c = pd.concat((interlines_p, interlines_n), sort=False)

    attrs = network.components["Line"]["attrs"]
    columns = set(
        attrs.index[attrs.static & attrs.status.str.startswith("Input")]
    ).difference(("name", "bus0", "bus1"))

    consense = {
        attr: _make_consense("Bus", attr)
        for attr in (
            columns
            | {"sub_network"}
            - {
                "r",
                "x",
                "g",
                "b",
                "terrain_factor",
                "s_nom",
                "s_nom_min",
                "s_nom_max",
                "s_nom_extendable",
                "length",
                "v_ang_min",
                "v_ang_max",
            }
        )
    }

    def aggregatelinegroup(l):
        # l.name is a tuple of the groupby index (bus0_s, bus1_s)
        length_s = (
            haversine_pts(
                buses.loc[l.name[0], ["x", "y"]], buses.loc[l.name[1], ["x", "y"]]
            )
            * line_length_factor
        )
        v_nom_s = buses.loc[list(l.name), "v_nom"].max()

        voltage_factor = (np.asarray(network.buses.loc[l.bus0, "v_nom"]) / v_nom_s) ** 2
        non_zero_len = l.length != 0
        length_factor = (length_s / l.length[non_zero_len]).reindex(
            l.index, fill_value=1
        )

        data = dict(
            r=1.0 / (voltage_factor / (length_factor * l["r"])).sum(),
            x=1.0 / (voltage_factor / (length_factor * l["x"])).sum(),
            g=(voltage_factor * length_factor * l["g"]).sum(),
            b=(voltage_factor * length_factor * l["b"]).sum(),
            terrain_factor=l["terrain_factor"].mean(),
            s_max_pu=(l["s_max_pu"] * normed_or_uniform(l["s_nom"])).sum(),
            s_nom=l["s_nom"].sum(),
            s_nom_min=l["s_nom_min"].sum(),
            s_nom_max=l["s_nom_max"].sum(),
            s_nom_extendable=l["s_nom_extendable"].any(),
            num_parallel=l["num_parallel"].sum(),
            capital_cost=(
                length_factor * normed_or_uniform(l["s_nom"]) * l["capital_cost"]
            ).sum(),
            length=length_s,
            sub_network=consense["sub_network"](l["sub_network"]),
            v_ang_min=l["v_ang_min"].max(),
            v_ang_max=l["v_ang_max"].min(),
            lifetime=(l["lifetime"] * normed_or_uniform(l["s_nom"])).sum(),
            build_year=(l["build_year"] * normed_or_uniform(l["s_nom"])).sum(),
        )
        data.update((f, consense[f](l[f])) for f in columns.difference(data))
        return pd.Series(data, index=[f for f in l.columns if f in columns])

    lines = interlines_c.groupby(["bus0_s", "bus1_s"]).apply(aggregatelinegroup)
    lines["name"] = [str(i + 1) for i in range(len(lines))]

    linemap_p = interlines_p.join(lines["name"], on=["bus0_s", "bus1_s"])["name"]
    linemap_n = interlines_n.join(lines["name"], on=["bus0_s", "bus1_s"])["name"]
    linemap = pd.concat((linemap_p, linemap_n), sort=False)

    lines_t = dict()

    if with_time:
        for attr, df in network.lines_t.items():
            lines_agg_b = df.columns.to_series().map(linemap).dropna()
            df_agg = df.loc[:, lines_agg_b.index]
            if not df_agg.empty:
                if (attr == "s_max_pu") or (attr == "s_min_pu"):
                    weighting = network.lines.groupby(linemap).s_nom.apply(
                        normed_or_uniform
                    )
                    df_agg = df_agg.multiply(weighting.loc[df_agg.columns], axis=1)
                pnl_df = df_agg.groupby(linemap, axis=1).sum()
                pnl_df.columns = flatten_multiindex(pnl_df.columns).rename("name")
                lines_t[attr] = pnl_df

    return lines, linemap_p, linemap_n, linemap, lines_t


def get_buses_linemap_and_lines(
    n: Network,
    busmap: pd.DataFrame,
    line_length_factor: float = 1.0,
    bus_strategies={},
    with_time: bool = True,
):
    """
    Compute new buses and lines based on the given network and busmap.

    Parameters
    ----------
    n : pypsa.Network
        The network to compute the buses and lines for.
    busmap : pandas.DataFrame
        The mapping of buses to clusters.
    line_length_factor : float, optional
        The factor to multiply the line length by, by default 1.0
    bus_strategies : dict, optional
        The strategies to use for aggregating buses, by default {}
    with_time : bool, optional
        Whether to include time-dependent data, by default True

    Returns
    -------
    tuple
        A tuple containing the new buses, the line map, the positive line map, the negative line map, the new lines, and the time-dependent lines.
    """
    # Compute new buses
    buses = aggregatebuses(n, busmap, custom_strategies=bus_strategies)

    lines = n.lines.assign(
        bus0_s=n.lines.bus0.map(busmap), bus1_s=n.lines.bus1.map(busmap)
    )
    interlines = lines.loc[lines["bus0_s"] != lines["bus1_s"]]

    # Aggregate lines
    lines, linemap_p, linemap_n, linemap, lines_t = aggregatelines(
        n, buses, interlines, line_length_factor, with_time=with_time
    )

    # If there are no lines, return an empty DataFrame
    if lines.empty:
        lines = lines.drop(columns=["bus0_s", "bus1_s"]).reset_index(drop=True)
    else:
        lines = lines.reset_index()
        lines = lines.rename(columns={"bus0_s": "bus0", "bus1_s": "bus1"})
    lines = lines.set_index("name").rename_axis(index="Line")

    return (buses, linemap, linemap_p, linemap_n, lines, lines_t)


Clustering = namedtuple(
    "Clustering",
    ["network", "busmap", "linemap", "linemap_positive", "linemap_negative"],
)


def get_clustering_from_busmap(
    n,
    busmap,
    with_time=True,
    line_length_factor=1.0,
    aggregate_generators_weighted=False,
    aggregate_one_ports={},
    aggregate_generators_carriers=None,
    scale_link_capital_costs=True,
    bus_strategies=dict(),
    one_port_strategies=dict(),
    generator_strategies=dict(),
    aggregate_generators_buses=None,
):
    buses, linemap, linemap_p, linemap_n, lines, lines_t = get_buses_linemap_and_lines(
        n, busmap, line_length_factor, bus_strategies, with_time
    )

    clustered = Network()

    io.import_components_from_dataframe(clustered, buses, "Bus")
    io.import_components_from_dataframe(clustered, lines, "Line")

    # Carry forward global constraints to clustered n.
    clustered.global_constraints = n.global_constraints

    if with_time:
        clustered.set_snapshots(n.snapshots)
        clustered.snapshot_weightings = n.snapshot_weightings.copy()
        for attr, df in lines_t.items():
            if not df.empty:
                io.import_series_from_dataframe(clustered, df, "Line", attr)

    one_port_components = n.one_port_components.copy()

    if aggregate_generators_weighted:
        one_port_components.remove("Generator")
        generators, generators_pnl = aggregateoneport(
            n,
            busmap,
            "Generator",
            carriers=aggregate_generators_carriers,
            buses=aggregate_generators_buses,
            with_time=with_time,
            custom_strategies=generator_strategies,
        )
        io.import_components_from_dataframe(clustered, generators, "Generator")
        if with_time:
            for attr, df in generators_pnl.items():
                if not df.empty:
                    io.import_series_from_dataframe(clustered, df, "Generator", attr)

    for one_port in aggregate_one_ports:
        one_port_components.remove(one_port)
        new_df, new_pnl = aggregateoneport(
            n,
            busmap,
            component=one_port,
            with_time=with_time,
            custom_strategies=one_port_strategies.get(one_port, {}),
        )
        io.import_components_from_dataframe(clustered, new_df, one_port)
        for attr, df in new_pnl.items():
            io.import_series_from_dataframe(clustered, df, one_port, attr)

    ##
    # Collect remaining one ports

    for c in n.iterate_components(one_port_components):
        io.import_components_from_dataframe(
            clustered,
            c.df.assign(bus=c.df.bus.map(busmap)).dropna(subset=["bus"]),
            c.name,
        )

    if with_time:
        for c in n.iterate_components(one_port_components):
            for attr, df in c.pnl.items():
                if not df.empty:
                    io.import_series_from_dataframe(clustered, df, c.name, attr)

    new_links = (
        n.links.assign(bus0=n.links.bus0.map(busmap), bus1=n.links.bus1.map(busmap))
        .dropna(subset=["bus0", "bus1"])
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
        new_links["capital_cost"] *= (new_links.length / n.links.length).fillna(1)

    io.import_components_from_dataframe(clustered, new_links, "Link")

    if with_time:
        for attr, df in n.links_t.items():
            if not df.empty:
                io.import_series_from_dataframe(clustered, df, "Link", attr)

    io.import_components_from_dataframe(clustered, n.carriers, "Carrier")

    clustered.determine_network_topology()

    return Clustering(clustered, busmap, linemap, linemap_p, linemap_n)


################
# k-Means clustering based on bus properties


def busmap_by_kmeans(network, bus_weightings, n_clusters, buses_i=None, **kwargs):
    """
    Create a bus map from the clustering of buses in space with a weighting.

    Parameters
    ----------
    network : pypsa.Network
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
        Mapping of network.buses to k-means clusters (indexed by
        non-negative integers).
    """
    if find_spec("sklearn") is None:
        raise ModuleNotFoundError(
            "Optional dependency 'sklearn' not found."
            "Install via 'conda install -c conda-forge scikit-learn' "
            "or 'pip install scikit-learn'"
        )

    from sklearn.cluster import KMeans

    if buses_i is None:
        buses_i = network.buses.index

    # since one cannot weight points directly in the scikit-learn
    # implementation of k-means, just add additional points at
    # same position
    points = network.buses.loc[buses_i, ["x", "y"]].values.repeat(
        bus_weightings.reindex(buses_i).astype(int), axis=0
    )

    kwargs.setdefault("n_init", "auto")
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, **kwargs)

    kmeans.fit(points)

    busmap = pd.Series(
        data=kmeans.predict(network.buses.loc[buses_i, ["x", "y"]].values),
        index=buses_i,
    ).astype(str)

    return busmap


def kmeans_clustering(
    network, bus_weightings, n_clusters, line_length_factor=1.0, **kwargs
):
    """
    Cluster the network according to k-means clustering of the buses.

    Buses can be weighted by an integer in the series `bus_weightings`.

    Note that this clustering method completely ignores the branches of the network.

    Parameters
    ----------
    network : pypsa.Network
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

    busmap = busmap_by_kmeans(network, bus_weightings, n_clusters, **kwargs)

    return get_clustering_from_busmap(
        network, busmap, line_length_factor=line_length_factor
    )


################
# Hierarchical Clustering
def busmap_by_hac(
    network,
    n_clusters,
    buses_i=None,
    branch_components=None,
    feature=None,
    affinity="euclidean",
    linkage="ward",
    **kwargs,
):
    """
    Create a busmap according to Hierarchical Agglomerative Clustering.

    Parameters
    ----------
    network : pypsa.Network
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
    affinity: str or callable, default=’euclidean’
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
    kwargs:
        Any remaining arguments to be passed to Hierarchical Clustering (e.g. memory, connectivity).

    Returns
    -------
    busmap : pandas.Series
        Mapping of network.buses to clusters (indexed by
        non-negative integers).
    """

    if find_spec("sklearn") is None:
        raise ModuleNotFoundError(
            "Optional dependency 'sklearn' not found."
            "Install via 'conda install -c conda-forge scikit-learn' "
            "or 'pip install scikit-learn'"
        )

    from sklearn.cluster import AgglomerativeClustering as HAC

    if buses_i is None:
        buses_i = network.buses.index

    if branch_components is None:
        branch_components = network.branch_components

    if feature is None:
        logger.warning(
            "No feature is specified for Hierarchical Clustering. "
            "Falling back to default, where all buses have equal similarity. "
            "You can specify a feature as pandas.DataFrame indexed with buses_i."
        )

        feature = pd.DataFrame(index=buses_i, columns=[""], data=0)

    buses_x = network.buses.index.get_indexer(buses_i)

    A = network.adjacency_matrix(branch_components=branch_components).tocsc()[buses_x][
        :, buses_x
    ]

    # TODO: maybe change the deprecated argument 'affinity' to 'metric'
    labels = HAC(
        n_clusters=n_clusters,
        connectivity=A,
        metric=affinity,
        linkage=linkage,
        **kwargs,
    ).fit_predict(feature)

    busmap = pd.Series(labels, index=buses_i, dtype=str)

    return busmap


def hac_clustering(
    network,
    n_clusters,
    buses_i=None,
    branch_components=None,
    feature=None,
    affinity="euclidean",
    linkage="ward",
    line_length_factor=1.0,
    **kwargs,
):
    """
    Cluster the network using Hierarchical Agglomerative Clustering.

    Parameters
    ----------
    network : pypsa.Network
        The buses must have coordinates x, y.
    buses_i: None | pandas.Index, default=None
        Subset of buses to cluster. If None, all buses are considered.
    branch_components: List, default=["Line", "Link"]
        Subset of all branch_components in the network.
    feature: None | pandas.DataFrame, default=None
        Feature to be considered for the clustering.
        The DataFrame must be indexed with buses_i.
        If None, all buses have the same similarity.
    affinity: str or callable, default=’euclidean’
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

    busmap = busmap_by_hac(
        network,
        n_clusters,
        buses_i,
        branch_components,
        feature,
        affinity,
        linkage,
        **kwargs,
    )

    return get_clustering_from_busmap(
        network, busmap, line_length_factor=line_length_factor
    )


################
# Cluserting based on Modularity (on electrical parameters of the network)
def busmap_by_greedy_modularity(network, n_clusters, buses_i=None):
    """
    Create a busmap according to Clauset-Newman-Moore greedy modularity
    maximization [1].

    Parameters
    ----------
    network : pypsa.Network
    n_clusters : int
        Final number of clusters desired.
    buses_i: None | pandas.Index, default=None
        Subset of buses to cluster. If None, all buses are considered.

    Returns
    -------
    busmap : pandas.Series
        Mapping of network.buses to clusters (indexed by
        non-negative integers).

    References
    ----------
    .. [1] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    """
    if parse(nx.__version__) < Version("2.8"):
        raise NotImplementedError(
            "The fuction `busmap_by_greedy_modularity` requires `networkx>=2.8`, "
            f"but version `networkx={nx.__version__}` is installed."
        )

    if buses_i is None:
        buses_i = network.buses.index

    network.calculate_dependent_values()

    lines = network.lines.query("bus0 in @buses_i and bus1 in @buses_i")
    lines = (
        lines[["bus0", "bus1"]]
        .assign(weight=lines.s_nom / abs(lines.r + 1j * lines.x))
        .set_index(["bus0", "bus1"])
    )

    G = nx.Graph()
    G.add_nodes_from(buses_i)
    G.add_edges_from((u, v, dict(weight=w)) for (u, v), w in lines.itertuples())

    communities = nx.community.greedy_modularity_communities(
        G, best_n=n_clusters, cutoff=n_clusters, weight="weight"
    )
    busmap = pd.Series(buses_i, buses_i)
    for c in np.arange(len(communities)):
        busmap.loc[list(communities[c])] = str(c)
    busmap.index = busmap.index.astype(str)

    return busmap


def greedy_modularity_clustering(
    network, n_clusters, buses_i=None, line_length_factor=1.0
):
    """
    Create a busmap according to Clauset-Newman-Moore greedy modularity
    maximization [1].

    Parameters
    ----------
    network : pypsa.Network
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
    .. [1] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    """

    busmap = busmap_by_greedy_modularity(network, n_clusters, buses_i)

    return get_clustering_from_busmap(
        network, busmap, line_length_factor=line_length_factor
    )


################
# Reduce stubs/dead-ends, i.e. nodes with valency 1, iteratively to remove tree-like structures


def busmap_by_stubs(network, matching_attrs=None):
    """
    Create a busmap by reducing stubs and stubby trees (i.e. sequentially
    reducing dead-ends).

    Parameters
    ----------
    network : pypsa.Network

    matching_attrs : None|[str]
        bus attributes clusters have to agree on

    Returns
    -------
    busmap : pandas.Series
        Mapping of network.buses to k-means clusters (indexed by
        non-negative integers).
    """
    busmap = pd.Series(network.buses.index, network.buses.index)

    G = network.graph()

    def attrs_match(u, v):
        return (
            matching_attrs is None
            or (
                network.buses.loc[u, matching_attrs]
                == network.buses.loc[v, matching_attrs]
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
        if len(stubs) == 0:
            break
    return busmap
