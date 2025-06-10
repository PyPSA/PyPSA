"""Functions for computing network clusters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
from deprecation import deprecated
from packaging.version import Version, parse
from pandas import Series

from pypsa.geo import haversine_pts

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable

    from pypsa import Network

logger = logging.getLogger(__name__)


DEFAULT_ONE_PORT_STRATEGIES = {
    "p": "sum",
    "q": "sum",
    "p_set": "sum",
    "q_set": "sum",
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
    Index(['a b', 'c d'], dtype='object')

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
    static = n.static(c)
    attrs = n.components[c]["attrs"]
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
            static[capacity[0]].groupby(grouper, axis=0).transform(normed_or_uniform)
        )
    if "weight" in static.columns:
        weights = static.weight.groupby(grouper, axis=0).transform(normed_or_uniform)

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

    non_aggregated = n.static(c)[~to_aggregate]
    non_aggregated = non_aggregated.assign(bus=non_aggregated.bus.map(busmap))

    static = pd.concat([aggregated, non_aggregated], sort=False)
    static.fillna(attrs.default, inplace=True)

    dynamic = {}
    if with_time:
        dynamic_strategies = align_strategies(strategies, n.dynamic(c), c)
        for attr, data in n.dynamic(c).items():
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
    attrs = n.components[c]["attrs"]

    output_columns = attrs.index[attrs.static & attrs.status.str.startswith("Output")]
    columns = [c for c in n.buses.columns if c not in output_columns]

    strategies = {**DEFAULT_BUS_STRATEGIES, **custom_strategies}
    strategies = align_strategies(strategies, columns, c)

    aggregated = n.buses.groupby(busmap).agg(strategies)
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
    attrs = n.components["Line"]["attrs"]
    static = n.static("Line")
    idx = static.index[static.bus0.map(busmap) != static.bus1.map(busmap)]
    static = static.loc[idx]

    orig_length = static.length
    orig_v_nom = static.bus0.map(n.buses.v_nom)

    bus_strategies = {**DEFAULT_BUS_STRATEGIES, **bus_strategies}
    cols = ["x", "y", "v_nom"]
    buses = n.buses[cols].groupby(busmap).agg({c: bus_strategies[c] for c in cols})

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
    ).max(1)
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
        dynamic_strategies = align_strategies(strategies, n.dynamic("Line"), "Line")

        for attr, data in n.lines_t.items():
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

    @property
    @deprecated(
        deprecated_in="0.32",
        removed_in="1.0",
        details="Use `clustering.n` instead.",
    )
    def network(self) -> Network:
        """Get the network.

        !!! warning "Deprecated in 0.32"
            Use `clustering.n` instead.
        """
        return self.n


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
    """Get a clustering result from a busmap."""
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
    clustered.global_constraints = n.global_constraints

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

    for one_port in aggregate_one_ports:
        one_port_components.remove(one_port)
        new_static, new_dynamic = aggregateoneport(
            n,
            busmap,
            component=one_port,
            with_time=with_time,
            custom_strategies=one_port_strategies.get(one_port, {}),
        )
        clustered.add(one_port, new_static.index, **new_static)
        for attr, df in new_dynamic.items():
            clustered._import_series_from_df(df, one_port, attr)

    # Collect remaining one ports

    for c in n.iterate_components(one_port_components):
        remaining_one_port_data = c.static.assign(bus=c.static.bus.map(busmap)).dropna(
            subset=["bus"]
        )
        clustered.add(c.name, remaining_one_port_data.index, **remaining_one_port_data)

    if with_time:
        for c in n.iterate_components(one_port_components):
            for attr, df in c.dynamic.items():
                if not df.empty:
                    clustered._import_series_from_df(df, c.name, attr)

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

    clustered.add("Link", new_links.index, **new_links)

    if with_time:
        for attr, df in n.links_t.items():
            if not df.empty:
                clustered._import_series_from_df(df, "Link", attr)

    clustered.add("Carrier", n.carriers.index, **n.carriers)

    clustered.determine_network_topology()

    return Clustering(clustered, busmap, linemap)


################
# k-Means clustering based on bus properties


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
    if find_spec("sklearn") is None:
        msg = (
            "Optional dependency 'sklearn' not found."
            "Install via 'conda install -c conda-forge scikit-learn' "
            "or 'pip install scikit-learn'"
        )
        raise ModuleNotFoundError(msg)

    from sklearn.cluster import KMeans

    if buses_i is None:
        buses_i = n.buses.index

    # since one cannot weight points directly in the scikit-learn
    # implementation of k-means, just add additional points at
    # same position
    points = n.buses.loc[buses_i, ["x", "y"]].values.repeat(
        bus_weightings.reindex(buses_i).astype(int), axis=0
    )

    kwargs.setdefault("n_init", "auto")
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, **kwargs)

    kmeans.fit(points)

    return pd.Series(
        data=kmeans.predict(n.buses.loc[buses_i, ["x", "y"]].values),
        index=buses_i,
    ).astype(str)


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
    busmap = busmap_by_kmeans(n, bus_weightings, n_clusters, **kwargs)

    return get_clustering_from_busmap(n, busmap, line_length_factor=line_length_factor)


################
# Hierarchical Clustering
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
        Subset of all branch_components in the network. If None, all branch_components are considered.
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
    kwargs:
        Any remaining arguments to be passed to Hierarchical Clustering (e.g. memory, connectivity).

    Returns
    -------
    busmap : pandas.Series
        Mapping of n.buses to clusters (indexed by
        non-negative integers).

    """
    if find_spec("sklearn") is None:
        msg = (
            "Optional dependency 'sklearn' not found."
            "Install via 'conda install -c conda-forge scikit-learn' "
            "or 'pip install scikit-learn'"
        )
        raise ModuleNotFoundError(msg)

    from sklearn.cluster import AgglomerativeClustering as HAC

    if buses_i is None:
        buses_i = n.buses.index

    if branch_components is None:
        branch_components = n.branch_components

    if feature is None:
        logger.warning(
            "No feature is specified for Hierarchical Clustering. "
            "Falling back to default, where all buses have equal similarity. "
            "You can specify a feature as pandas.DataFrame indexed with buses_i."
        )

        feature = pd.DataFrame(index=buses_i, columns=[""], data=0)

    buses_x = n.buses.index.get_indexer(buses_i)

    A = n.adjacency_matrix(branch_components=branch_components).tocsc()[buses_x][
        :, buses_x
    ]

    labels = HAC(
        n_clusters=n_clusters,
        connectivity=A,
        metric=affinity,
        linkage=linkage,
        **kwargs,
    ).fit_predict(feature)

    return pd.Series(labels, index=buses_i, dtype=str)


def hac_clustering(
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
    n : pypsa.Network
        The buses must have coordinates x, y.
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
    busmap = busmap_by_hac(
        n,
        n_clusters,
        buses_i,
        branch_components,
        feature,
        affinity,
        linkage,
        **kwargs,
    )

    return get_clustering_from_busmap(n, busmap, line_length_factor=line_length_factor)


def busmap_by_greedy_modularity(
    n: Network, n_clusters: int, buses_i: pd.Index | None = None
) -> pd.Series:
    """Create a busmap according to Clauset-Newman-Moore greedy modularity maximization.

    See [CNM2004_1]_ for more details.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
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
    .. [CNM2004_1] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.

    """
    if parse(nx.__version__) < Version("2.8"):
        msg = (
            "The fuction `busmap_by_greedy_modularity` requires `networkx>=2.8`, "
            f"but version `networkx={nx.__version__}` is installed."
        )
        raise NotImplementedError(msg)

    if buses_i is None:
        buses_i = n.buses.index

    n.calculate_dependent_values()

    lines = n.lines.query("bus0 in @buses_i and bus1 in @buses_i")
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
    .. [CNM2004_2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.

    """
    busmap = busmap_by_greedy_modularity(n, n_clusters, buses_i)

    return get_clustering_from_busmap(n, busmap, line_length_factor=line_length_factor)


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
        Mapping of n.buses to k-means clusters (indexed by
        non-negative integers).

    """
    busmap = pd.Series(n.buses.index, n.buses.index)

    G = n.graph()

    def attrs_match(u: str, v: str) -> bool:
        return (
            matching_attrs is None
            or (n.buses.loc[u, matching_attrs] == n.buses.loc[v, matching_attrs]).all()
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
