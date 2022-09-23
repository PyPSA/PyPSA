# -*- coding: utf-8 -*-

"""
Functions for computing network clusters.
"""

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2022 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import logging
from collections import namedtuple
from functools import reduce
from importlib.util import find_spec

import networkx as nx
import numpy as np
import pandas as pd
from packaging.version import Version, parse

logger = logging.getLogger(__name__)


from pypsa import io
from pypsa.components import Network
from pypsa.geo import haversine_pts


def _normed(s):
    tot = s.sum()
    if tot == 0:
        return 1.0
    else:
        return s / tot


def _flatten_multiindex(m, join=" "):
    if m.nlevels <= 1:
        return m
    levels = map(m.get_level_values, range(m.nlevels))
    return reduce(lambda x, y: x + join + y, levels, next(levels)).str.rstrip()


def _make_consense(component, attr):
    def consense(x):
        v = x.iat[0]
        assert (
            x == v
        ).all() or x.isnull().all(), "In {} cluster {} the values of attribute {} do not agree:\n{}".format(
            component, x.name, attr, x
        )
        return v

    return consense


def aggregategenerators(
    network, busmap, with_time=True, carriers=None, custom_strategies=dict()
):
    if carriers is None:
        carriers = network.generators.carrier.unique()

    gens_agg_b = network.generators.carrier.isin(carriers)
    attrs = network.components["Generator"]["attrs"]
    generators = network.generators.loc[gens_agg_b].assign(
        bus=lambda df: df.bus.map(busmap)
    )
    columns = (
        set(attrs.index[attrs.static & attrs.status.str.startswith("Input")])
        | {"weight"}
    ) & set(generators.columns) - {"control"}
    grouper = [generators.bus, generators.carrier]

    def normed_or_uniform(x):
        return (
            x / x.sum() if x.sum(skipna=False) > 0 else pd.Series(1.0 / len(x), x.index)
        )

    weighting = generators.weight.groupby(grouper, axis=0).transform(normed_or_uniform)
    generators["capital_cost"] *= weighting

    strategies = {
        "p_nom_max": pd.Series.min,
        "weight": pd.Series.sum,
        "p_nom": pd.Series.sum,
        "capital_cost": pd.Series.sum,
        "efficiency": pd.Series.mean,
        "ramp_limit_up": pd.Series.mean,
        "ramp_limit_down": pd.Series.mean,
        "ramp_limit_start_up": pd.Series.mean,
        "ramp_limit_shut_down": pd.Series.mean,
        "build_year": lambda x: 0,
        "lifetime": lambda x: np.inf,
    }
    strategies.update(custom_strategies)
    if strategies["p_nom_max"] is pd.Series.min:
        generators["p_nom_max"] /= weighting
    strategies.update(
        (attr, _make_consense("Generator", attr))
        for attr in columns.difference(strategies)
    )
    new_df = generators.groupby(grouper, axis=0).agg(strategies)
    new_df.index = _flatten_multiindex(new_df.index).rename("name")

    new_df = pd.concat(
        [
            new_df,
            network.generators.loc[~gens_agg_b].assign(
                bus=lambda df: df.bus.map(busmap)
            ),
        ],
        axis=0,
        sort=False,
    )

    new_pnl = dict()
    if with_time:
        for attr, df in network.generators_t.items():
            pnl_gens_agg_b = df.columns.to_series().map(gens_agg_b)
            df_agg = df.loc[:, pnl_gens_agg_b]
            if not df_agg.empty:
                if attr == "p_max_pu":
                    df_agg = df_agg.multiply(weighting.loc[df_agg.columns], axis=1)
                pnl_df = df_agg.groupby(grouper, axis=1).sum()
                pnl_df.columns = _flatten_multiindex(pnl_df.columns).rename("name")
                new_pnl[attr] = pd.concat(
                    [df.loc[:, ~pnl_gens_agg_b], pnl_df], axis=1, sort=False
                )

    return new_df, new_pnl


def aggregateoneport(
    network, busmap, component, with_time=True, custom_strategies=dict()
):

    if network.df(component).empty:
        return network.df(component), network.pnl(component)

    attrs = network.components[component]["attrs"]
    old_df = getattr(network, network.components[component]["list_name"]).assign(
        bus=lambda df: df.bus.map(busmap)
    )
    columns = set(
        attrs.index[attrs.static & attrs.status.str.startswith("Input")]
    ) & set(old_df.columns)
    grouper = old_df.bus if "carrier" not in columns else [old_df.bus, old_df.carrier]

    def aggregate_max_hours(max_hours):
        if (max_hours == max_hours.iloc[0]).all():
            return max_hours.iloc[0]
        else:
            return (max_hours * _normed(old_df.p_nom.reindex(max_hours.index))).sum()

    default_strategies = dict(
        p=pd.Series.sum,
        q=pd.Series.sum,
        p_set=pd.Series.sum,
        q_set=pd.Series.sum,
        p_nom=pd.Series.sum,
        p_nom_max=pd.Series.sum,
        p_nom_min=pd.Series.sum,
        max_hours=aggregate_max_hours,
    )
    strategies = {
        attr: default_strategies.get(attr, _make_consense(component, attr))
        for attr in columns
    }
    strategies.update(custom_strategies)
    new_df = old_df.groupby(grouper).agg(strategies)
    new_df.index = _flatten_multiindex(new_df.index).rename("name")

    new_pnl = dict()

    def normed_or_uniform(x):
        return (
            x / x.sum() if x.sum(skipna=False) > 0 else pd.Series(1.0 / len(x), x.index)
        )

    if "e_nom" in new_df.columns:
        weighting = old_df.e_nom.groupby(grouper, axis=0).transform(normed_or_uniform)
    elif "p_nom" in new_df.columns:
        weighting = old_df.p_nom.groupby(grouper, axis=0).transform(normed_or_uniform)

    if with_time:
        old_pnl = network.pnl(component)
        for attr, df in old_pnl.items():
            if not df.empty:
                if attr in ["e_min_pu", "e_max_pu", "p_min_pu", "p_max_pu"]:
                    df = df.multiply(weighting.loc[df.columns], axis=1)
                pnl_df = df.groupby(grouper, axis=1).sum()
                pnl_df.columns = _flatten_multiindex(pnl_df.columns).rename("name")
                new_pnl[attr] = pnl_df
    return new_df, new_pnl


def aggregatebuses(network, busmap, custom_strategies=dict()):
    attrs = network.components["Bus"]["attrs"]
    columns = set(
        attrs.index[attrs.static & attrs.status.str.startswith("Input")]
    ) & set(network.buses.columns)

    strategies = dict(
        x=pd.Series.mean,
        y=pd.Series.mean,
        v_nom=pd.Series.max,
        v_mag_pu_max=pd.Series.min,
        v_mag_pu_min=pd.Series.max,
    )
    strategies.update(
        (attr, _make_consense("Bus", attr)) for attr in columns.difference(strategies)
    )
    strategies.update(custom_strategies)

    return (
        network.buses.groupby(busmap)
        .agg(strategies)
        .reindex(
            columns=[
                f
                for f in network.buses.columns
                if f in columns or f in custom_strategies
            ]
        )
    )


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
            s_max_pu=(l["s_max_pu"] * _normed(l["s_nom"])).sum(),
            s_nom=l["s_nom"].sum(),
            s_nom_min=l["s_nom_min"].sum(),
            s_nom_max=l["s_nom_max"].sum(),
            s_nom_extendable=l["s_nom_extendable"].any(),
            num_parallel=l["num_parallel"].sum(),
            capital_cost=(
                length_factor * _normed(l["s_nom"]) * l["capital_cost"]
            ).sum(),
            length=length_s,
            sub_network=consense["sub_network"](l["sub_network"]),
            v_ang_min=l["v_ang_min"].max(),
            v_ang_max=l["v_ang_max"].min(),
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
                    weighting = network.lines.groupby(linemap).s_nom.apply(_normed)
                    df_agg = df_agg.multiply(weighting.loc[df_agg.columns], axis=1)
                pnl_df = df_agg.groupby(linemap, axis=1).sum()
                pnl_df.columns = _flatten_multiindex(pnl_df.columns).rename("name")
                lines_t[attr] = pnl_df

    return lines, linemap_p, linemap_n, linemap, lines_t


def get_buses_linemap_and_lines(
    network, busmap, line_length_factor=1.0, bus_strategies=dict(), with_time=True
):
    # compute new buses
    buses = aggregatebuses(network, busmap, bus_strategies)

    lines = network.lines.assign(
        bus0_s=lambda df: df.bus0.map(busmap), bus1_s=lambda df: df.bus1.map(busmap)
    )

    # lines between different clusters
    interlines = lines.loc[lines["bus0_s"] != lines["bus1_s"]]
    lines, linemap_p, linemap_n, linemap, lines_t = aggregatelines(
        network, buses, interlines, line_length_factor, with_time
    )
    return (
        buses,
        linemap,
        linemap_p,
        linemap_n,
        lines.reset_index()
        .rename(columns={"bus0_s": "bus0", "bus1_s": "bus1"}, copy=False)
        .set_index("name"),
        lines_t,
    )


Clustering = namedtuple(
    "Clustering",
    ["network", "busmap", "linemap", "linemap_positive", "linemap_negative"],
)


def get_clustering_from_busmap(
    network,
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
):

    buses, linemap, linemap_p, linemap_n, lines, lines_t = get_buses_linemap_and_lines(
        network, busmap, line_length_factor, bus_strategies, with_time
    )

    network_c = Network()

    io.import_components_from_dataframe(network_c, buses, "Bus")
    io.import_components_from_dataframe(network_c, lines, "Line")

    # Carry forward global constraints to clustered network.
    network_c.global_constraints = network.global_constraints

    if with_time:
        network_c.set_snapshots(network.snapshots)
        network_c.snapshot_weightings = network.snapshot_weightings.copy()
        for attr, df in lines_t.items():
            if not df.empty:
                io.import_series_from_dataframe(network_c, df, "Line", attr)

    one_port_components = network.one_port_components.copy()

    if aggregate_generators_weighted:
        one_port_components.remove("Generator")
        generators, generators_pnl = aggregategenerators(
            network,
            busmap,
            with_time=with_time,
            carriers=aggregate_generators_carriers,
            custom_strategies=generator_strategies,
        )
        io.import_components_from_dataframe(network_c, generators, "Generator")
        if with_time:
            for attr, df in generators_pnl.items():
                if not df.empty:
                    io.import_series_from_dataframe(network_c, df, "Generator", attr)

    for one_port in aggregate_one_ports:
        one_port_components.remove(one_port)
        new_df, new_pnl = aggregateoneport(
            network,
            busmap,
            component=one_port,
            with_time=with_time,
            custom_strategies=one_port_strategies.get(one_port, {}),
        )
        io.import_components_from_dataframe(network_c, new_df, one_port)
        for attr, df in new_pnl.items():
            io.import_series_from_dataframe(network_c, df, one_port, attr)

    ##
    # Collect remaining one ports

    for c in network.iterate_components(one_port_components):
        io.import_components_from_dataframe(
            network_c,
            c.df.assign(bus=c.df.bus.map(busmap)).dropna(subset=["bus"]),
            c.name,
        )

    if with_time:
        for c in network.iterate_components(one_port_components):
            for attr, df in c.pnl.items():
                if not df.empty:
                    io.import_series_from_dataframe(network_c, df, c.name, attr)

    new_links = (
        network.links.assign(
            bus0=network.links.bus0.map(busmap), bus1=network.links.bus1.map(busmap)
        )
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
        new_links["capital_cost"] *= (new_links.length / network.links.length).fillna(1)

    io.import_components_from_dataframe(network_c, new_links, "Link")

    if with_time:
        for attr, df in network.links_t.items():
            if not df.empty:
                io.import_series_from_dataframe(network_c, df, "Link", attr)

    io.import_components_from_dataframe(network_c, network.carriers, "Carrier")

    network_c.determine_network_topology()

    return Clustering(network_c, busmap, linemap, linemap_p, linemap_n)


################
# k-Means clustering based on bus properties


def busmap_by_kmeans(network, bus_weightings, n_clusters, buses_i=None, **kwargs):
    """
    Create a bus map from the clustering of buses in space with a weighting.

    Parameters
    ----------
    network : pypsa.Network
        The buses must have coordinates x,y.
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
        The buses must have coordinates x,y.
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

    labels = HAC(
        n_clusters=n_clusters,
        connectivity=A,
        affinity=affinity,
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
        The buses must have coordinates x,y.
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
        network, n_clusters, buses_i, branch_components, feature, **kwargs
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
