## Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Functions for computing network clusters
"""

__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"

import numpy as np
import pandas as pd
import networkx as nx
from collections import OrderedDict, namedtuple
from six.moves import map, range, reduce
from six import itervalues, iteritems

import logging
logger = logging.getLogger(__name__)


from .components import Network
from .geo import haversine_pts

from . import io

def _normed(s):
    tot = s.sum()
    if tot == 0:
        return 1.
    else:
        return s/tot

def _flatten_multiindex(m, join=' '):
    if m.nlevels <= 1: return m
    levels = map(m.get_level_values, range(m.nlevels))
    return reduce(lambda x, y: x+join+y, levels, next(levels))

def _make_consense(component, attr):
    def consense(x):
        v = x.iat[0]
        assert ((x == v).all() or x.isnull().all()), (
            "In {} cluster {} the values of attribute {} do not agree:\n{}"
            .format(component, x.name, attr, x)
        )
        return v
    return consense

def aggregategenerators(network, busmap, with_time=True, carriers=None, custom_strategies=dict()):
    if carriers is None:
        carriers = network.generators.carrier.unique()

    gens_agg_b = network.generators.carrier.isin(carriers)
    attrs = network.components["Generator"]["attrs"]
    generators = (network.generators.loc[gens_agg_b]
                  .assign(bus=lambda df: df.bus.map(busmap)))
    columns = (set(attrs.index[attrs.static & attrs.status.str.startswith('Input')]) |
               {'weight'}) & set(generators.columns) - {'control'}
    grouper = [generators.bus, generators.carrier]

    def normed_or_uniform(x):
        return x/x.sum() if x.sum(skipna=False) > 0 else pd.Series(1./len(x), x.index)
    weighting = generators.weight.groupby(grouper, axis=0).transform(normed_or_uniform)
    generators['capital_cost'] *= weighting
    strategies = {'p_nom_max': np.min, 'weight': np.sum, 'p_nom': np.sum, 'capital_cost': np.sum}
    strategies.update(custom_strategies)
    if strategies['p_nom_max'] is np.min:
        generators['p_nom_max'] /= weighting

    strategies.update((attr, _make_consense('Generator', attr))
                      for attr in columns.difference(strategies))
    new_df = generators.groupby(grouper, axis=0).agg(strategies)
    new_df.index = _flatten_multiindex(new_df.index).rename("name")

    new_df = pd.concat([new_df,
                        network.generators.loc[~gens_agg_b]
                        .assign(bus=lambda df: df.bus.map(busmap))], axis=0, sort=False)

    new_pnl = dict()
    if with_time:
        for attr, df in iteritems(network.generators_t):
            pnl_gens_agg_b = df.columns.to_series().map(gens_agg_b)
            df_agg = df.loc[:, pnl_gens_agg_b]
            if not df_agg.empty:
                if attr == 'p_max_pu':
                    df_agg = df_agg.multiply(weighting.loc[df_agg.columns], axis=1)
                pnl_df = df_agg.groupby(grouper, axis=1).sum()
                pnl_df.columns = _flatten_multiindex(pnl_df.columns).rename("name")
                new_pnl[attr] = pd.concat([df.loc[:, ~pnl_gens_agg_b], pnl_df], axis=1, sort=False)

    return new_df, new_pnl

def aggregateoneport(network, busmap, component, with_time=True, custom_strategies=dict()):
    attrs = network.components[component]["attrs"]
    old_df = getattr(network, network.components[component]["list_name"]).assign(bus=lambda df: df.bus.map(busmap))
    columns = set(attrs.index[attrs.static & attrs.status.str.startswith('Input')]) & set(old_df.columns)
    grouper = old_df.bus if 'carrier' not in columns else [old_df.bus, old_df.carrier]

    def aggregate_max_hours(max_hours):
        if (max_hours == max_hours.iloc[0]).all():
            return max_hours.iloc[0]
        else:
            return (max_hours * _normed(old_df.p_nom.reindex(max_hours.index))).sum()

    default_strategies = dict(p=np.sum, q=np.sum, p_set=np.sum, q_set=np.sum,
                              p_nom=np.sum, p_nom_max=np.sum, p_nom_min=np.sum,
                              max_hours=aggregate_max_hours)
    strategies = {attr: default_strategies.get(attr, _make_consense(component, attr))
                  for attr in columns}
    strategies.update(custom_strategies)
    new_df = old_df.groupby(grouper).agg(strategies)
    new_df.index = _flatten_multiindex(new_df.index).rename("name")

    new_pnl = dict()
    if with_time:
        old_pnl = network.pnl(component)
        for attr, df in iteritems(old_pnl):
            if not df.empty:
                pnl_df = df.groupby(grouper, axis=1).sum()
                pnl_df.columns = _flatten_multiindex(pnl_df.columns).rename("name")
                new_pnl[attr] = pnl_df

    return new_df, new_pnl

def aggregatebuses(network, busmap, custom_strategies=dict()):
    attrs = network.components["Bus"]["attrs"]
    columns = set(attrs.index[attrs.static & attrs.status.str.startswith('Input')]) & set(network.buses.columns)

    strategies = dict(x=np.mean, y=np.mean,
                      v_nom=np.max,
                      v_mag_pu_max=np.min, v_mag_pu_min=np.max)
    strategies.update((attr, _make_consense("Bus", attr))
                      for attr in columns.difference(strategies))
    strategies.update(custom_strategies)

    return network.buses \
            .groupby(busmap).agg(strategies) \
            .reindex(columns=[f
                              for f in network.buses.columns
                              if f in columns or f in custom_strategies])

def aggregatelines(network, buses, interlines, line_length_factor=1.0):

    #make sure all lines have same bus ordering
    positive_order = interlines.bus0_s < interlines.bus1_s
    interlines_p = interlines[positive_order]
    interlines_n = interlines[~ positive_order].rename(columns={"bus0_s":"bus1_s", "bus1_s":"bus0_s"})
    interlines_c = pd.concat((interlines_p,interlines_n), sort=False)

    attrs = network.components["Line"]["attrs"]
    columns = set(attrs.index[attrs.static & attrs.status.str.startswith('Input')]).difference(('name', 'bus0', 'bus1'))

    consense = {
        attr: _make_consense('Bus', attr)
        for attr in (columns | {'sub_network'}
                     - {'r', 'x', 'g', 'b', 'terrain_factor', 's_nom',
                        's_nom_min', 's_nom_max', 's_nom_extendable',
                        'length', 'v_ang_min', 'v_ang_max'})
    }

    def aggregatelinegroup(l):

        # l.name is a tuple of the groupby index (bus0_s, bus1_s)
        length_s = haversine_pts(buses.loc[l.name[0], ['x', 'y']],
                                 buses.loc[l.name[1], ['x', 'y']]) * line_length_factor
        v_nom_s = buses.loc[list(l.name),'v_nom'].max()

        voltage_factor = (np.asarray(network.buses.loc[l.bus0,'v_nom'])/v_nom_s)**2
        length_factor = (length_s/l['length'])

        data = dict(
            r=1./(voltage_factor/(length_factor * l['r'])).sum(),
            x=1./(voltage_factor/(length_factor * l['x'])).sum(),
            g=(voltage_factor * length_factor * l['g']).sum(),
            b=(voltage_factor * length_factor * l['b']).sum(),
            terrain_factor=l['terrain_factor'].mean(),
            s_nom=l['s_nom'].sum(),
            s_nom_min=l['s_nom_min'].sum(),
            s_nom_max=l['s_nom_max'].sum(),
            s_nom_extendable=l['s_nom_extendable'].any(),
            num_parallel=l['num_parallel'].sum(),
            capital_cost=(length_factor * _normed(l['s_nom']) * l['capital_cost']).sum(),
            length=length_s,
            sub_network=consense['sub_network'](l['sub_network']),
            v_ang_min=l['v_ang_min'].max(),
            v_ang_max=l['v_ang_max'].min()
        )
        data.update((f, consense[f](l[f])) for f in columns.difference(data))
        return pd.Series(data, index=[f for f in l.columns if f in columns])

    lines = interlines_c.groupby(['bus0_s', 'bus1_s']).apply(aggregatelinegroup)
    lines['name'] = [str(i+1) for i in range(len(lines))]

    linemap_p = interlines_p.join(lines['name'], on=['bus0_s', 'bus1_s'])['name']
    linemap_n = interlines_n.join(lines['name'], on=['bus0_s', 'bus1_s'])['name']
    linemap = pd.concat((linemap_p,linemap_n), sort=False)

    return lines, linemap_p, linemap_n, linemap

def get_buses_linemap_and_lines(network, busmap, line_length_factor=1.0, bus_strategies=dict()):
    # compute new buses
    buses = aggregatebuses(network, busmap, bus_strategies)

    lines = network.lines.assign(bus0_s=lambda df: df.bus0.map(busmap),
                                 bus1_s=lambda df: df.bus1.map(busmap))

    # lines between different clusters
    interlines = lines.loc[lines['bus0_s'] != lines['bus1_s']]
    lines, linemap_p, linemap_n, linemap = aggregatelines(network, buses, interlines, line_length_factor)
    return (buses,
            linemap,
            linemap_p,
            linemap_n,
            lines.reset_index()
                 .rename(columns={'bus0_s': 'bus0', 'bus1_s': 'bus1'}, copy=False)
                 .set_index('name'))

Clustering = namedtuple('Clustering', ['network', 'busmap', 'linemap',
                                       'linemap_positive', 'linemap_negative'])

def get_clustering_from_busmap(network, busmap, with_time=True, line_length_factor=1.0,
                               aggregate_generators_weighted=False, aggregate_one_ports={},
                               aggregate_generators_carriers=None,
                               scale_link_capital_costs=True,
                               bus_strategies=dict(), one_port_strategies=dict(),
                               generator_strategies=dict()):

    buses, linemap, linemap_p, linemap_n, lines = get_buses_linemap_and_lines(network, busmap, line_length_factor, bus_strategies)

    network_c = Network()

    io.import_components_from_dataframe(network_c, buses, "Bus")
    io.import_components_from_dataframe(network_c, lines, "Line")

    if with_time:
        network_c.snapshot_weightings = network.snapshot_weightings.copy()
        network_c.set_snapshots(network.snapshots)

    one_port_components = network.one_port_components.copy()

    if aggregate_generators_weighted:
        one_port_components.remove("Generator")
        generators, generators_pnl = aggregategenerators(network, busmap, with_time=with_time,
                                                         carriers=aggregate_generators_carriers,
                                                         custom_strategies=generator_strategies)
        io.import_components_from_dataframe(network_c, generators, "Generator")
        if with_time:
            for attr, df in iteritems(generators_pnl):
                if not df.empty:
                    io.import_series_from_dataframe(network_c, df, "Generator", attr)

    for one_port in aggregate_one_ports:
        one_port_components.remove(one_port)
        new_df, new_pnl = aggregateoneport(network, busmap, component=one_port, with_time=with_time,
                                           custom_strategies=one_port_strategies.get(one_port, {}))
        io.import_components_from_dataframe(network_c, new_df, one_port)
        for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(network_c, df, one_port, attr)


    ##
    # Collect remaining one ports

    for c in network.iterate_components(one_port_components):
        io.import_components_from_dataframe(
            network_c,
            c.df.assign(bus=c.df.bus.map(busmap)).dropna(subset=['bus']),
            c.name
        )

    if with_time:
        for c in network.iterate_components(one_port_components):
            for attr, df in iteritems(c.pnl):
                if not df.empty:
                    io.import_series_from_dataframe(network_c, df, c.name, attr)

    new_links = (network.links.assign(bus0=network.links.bus0.map(busmap),
                                      bus1=network.links.bus1.map(busmap))
                        .dropna(subset=['bus0', 'bus1'])
                        .loc[lambda df: df.bus0 != df.bus1])

    new_links['length'] = np.where(
        new_links.length.notnull() & (new_links.length > 0),
        line_length_factor *
        haversine_pts(buses.loc[new_links['bus0'], ['x', 'y']],
                      buses.loc[new_links['bus1'], ['x', 'y']]),
        0
    )
    if scale_link_capital_costs:
        new_links['capital_cost'] *= (new_links.length/network.links.length).fillna(1)

    io.import_components_from_dataframe(network_c, new_links, "Link")

    if with_time:
        for attr, df in iteritems(network.links_t):
            if not df.empty:
                io.import_series_from_dataframe(network_c, df, "Link", attr)

    io.import_components_from_dataframe(network_c, network.carriers, "Carrier")

    network_c.determine_network_topology()

    return Clustering(network_c, busmap, linemap, linemap_p, linemap_n)


################
# Length

def busmap_by_linemask(network, mask):
    mask = network.lines.loc[:,['bus0', 'bus1']].assign(mask=mask).set_index(['bus0','bus1'])['mask']
    G = nx.OrderedGraph()
    G.add_nodes_from(network.buses.index)
    G.add_edges_from(mask.index[mask])
    return pd.Series(OrderedDict((n, str(i))
                                 for i, g in enumerate(nx.connected_components(G))
                                 for n in g),
                     name='name')

def busmap_by_length(network, length):
    return busmap_by_linemask(network, network.lines.length < length)

def length_clustering(network, length):
    busmap = busmap_by_length(network, length=length)
    return get_clustering_from_busmap(network, busmap)

################
# SpectralClustering

try:
    # available using pip as scikit-learn
    from sklearn.cluster import spectral_clustering as sk_spectral_clustering

    def busmap_by_spectral_clustering(network, n_clusters, **kwds):
        lines = network.lines.loc[:,['bus0', 'bus1']].assign(weight=network.lines.num_parallel).set_index(['bus0','bus1'])
        lines.weight+=0.1
        G = nx.Graph()
        G.add_nodes_from(network.buses.index)
        G.add_edges_from((u,v,dict(weight=w)) for (u,v),w in lines.itertuples())
        return pd.Series(list(map(str,sk_spectral_clustering(nx.adjacency_matrix(G), n_clusters, **kwds) + 1)),
                         index=network.buses.index)

    def spectral_clustering(network, n_clusters=8, **kwds):
        busmap = busmap_by_spectral_clustering(network, n_clusters=n_clusters, **kwds)
        return get_clustering_from_busmap(network, busmap)

except ImportError:
    pass

################
# Louvain

try:
    # available using pip as python-louvain
    import community

    def busmap_by_louvain(network):
        lines = network.lines.loc[:,['bus0', 'bus1']].assign(weight=network.lines.num_parallel).set_index(['bus0','bus1'])
        lines.weight+=0.1
        G = nx.Graph()
        G.add_nodes_from(network.buses.index)
        G.add_edges_from((u,v,dict(weight=w)) for (u,v),w in lines.itertuples())
        b=community.best_partition(G)
        list_cluster=[]
        for i in b:
            list_cluster.append(str(b[i]))
        return pd.Series(list_cluster,index=network.buses.index)

    def louvain_clustering(network, **kwds):
        busmap = busmap_by_louvain(network)
        return get_clustering_from_busmap(network, busmap)

except ImportError:
    pass


################
# k-Means clustering based on bus properties

try:
    # available using pip as scikit-learn
    from sklearn.cluster import KMeans

    def busmap_by_kmeans(network, bus_weightings, n_clusters, buses_i=None, ** kwargs):
        """
        Create a bus map from the clustering of buses in space with a
        weighting.

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
            Any remaining arguments to be passed to KMeans (e.g. n_init,
            n_jobs).

        Returns
        -------
        busmap : pandas.Series
            Mapping of network.buses to k-means clusters (indexed by
            non-negative integers).
        """

        if buses_i is None:
            buses_i = network.buses.index

        # since one cannot weight points directly in the scikit-learn
        # implementation of k-means, just add additional points at
        # same position
        points = (network.buses.loc[buses_i, ["x","y"]].values
                  .repeat(bus_weightings.reindex(buses_i).astype(int), axis=0))

        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, ** kwargs)

        kmeans.fit(points)

        busmap = pd.Series(data=kmeans.predict(network.buses.loc[buses_i, ["x","y"]]),
                           index=buses_i).astype(str)

        return busmap

    def kmeans_clustering(network, bus_weightings, n_clusters, line_length_factor=1.0, ** kwargs):
        """
        Cluster then network according to k-means clustering of the
        buses.

        Buses can be weighted by an integer in the series
        `bus_weightings`.

        Note that this clustering method completely ignores the
        branches of the network.

        Parameters
        ----------
        network : pypsa.Network
            The buses must have coordinates x,y.
        bus_weightings : pandas.Series
            Series of integer weights for buses, indexed by bus names.
        n_clusters : int
            Final number of clusters desired.
        line_length_factor : float
            Factor to multiply the crow-flies distance between new buses in order to get new
            line lengths.
        kwargs
            Any remaining arguments to be passed to KMeans (e.g. n_init, n_jobs)


        Returns
        -------
        Clustering : named tuple
            A named tuple containing network, busmap and linemap
        """

        busmap = busmap_by_kmeans(network, bus_weightings, n_clusters, ** kwargs)
        return get_clustering_from_busmap(network, busmap, line_length_factor=line_length_factor)

except ImportError:
    pass






################
# Rectangular grid clustering


def busmap_by_rectangular_grid(buses, divisions=10):
    busmap = pd.Series(0, index=buses.index)
    if isinstance(divisions, tuple):
        divisions_x, divisions_y = divisions
    else:
        divisions_x = divisions_y = divisions
    gb = buses.groupby([pd.cut(buses.x, divisions_x), pd.cut(buses.y, divisions_y)])
    for nk, oks in enumerate(itervalues(gb.groups)):
        busmap.loc[oks] = nk
    return busmap

def rectangular_grid_clustering(network, divisions):
    busmap = busmap_by_rectangular_grid(network.buses, divisions)
    return get_clustering_from_busmap(network, busmap)





################
# Reduce stubs/dead-ends, i.e. nodes with valency 1, iteratively to remove tree-like structures

def busmap_by_stubs(network, matching_attrs=None):
    """Create a busmap by reducing stubs and stubby trees
    (i.e. sequentially reducing dead-ends).

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
        return (matching_attrs is None or
                (network.buses.loc[u, matching_attrs] ==
                 network.buses.loc[v, matching_attrs]).all())

    while True:
        stubs = []
        for u in G.nodes:
            neighbours = list(G.adj[u].keys())
            if len(neighbours) == 1:
                v, = neighbours
                if attrs_match(u, v):
                    busmap[busmap == u] = v
                    stubs.append(u)
        G.remove_nodes_from(stubs)
        if len(stubs) == 0:
            break
    return busmap

def stubs_clustering(network,use_reduced_coordinates=True, line_length_factor=1.0):
    """Cluster network by reducing stubs and stubby trees
    (i.e. sequentially reducing dead-ends).

    Parameters
    ----------
    network : pypsa.Network
    use_reduced_coordinates : boolean
        If True, do not average clusters, but take from busmap.
    line_length_factor : float
        Factor to multiply the crow-flies distance between new buses in order to get new
        line lengths.

    Returns
    -------
    Clustering : named tuple
        A named tuple containing network, busmap and linemap
    """

    busmap = busmap_by_stubs(network)

    #reset coordinates to the new reduced guys, rather than taking an average
    if use_reduced_coordinates:
        # TODO : FIX THIS HACK THAT HAS UNEXPECTED SIDE-EFFECTS,
        # i.e. network is changed in place!!
        network.buses.loc[busmap.index,['x','y']] = network.buses.loc[busmap,['x','y']].values

    return get_clustering_from_busmap(network, busmap, line_length_factor=line_length_factor)
