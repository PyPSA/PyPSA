## Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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

from __future__ import absolute_import

__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"

import numpy as np
import pandas as pd
import networkx as nx
from collections import OrderedDict, namedtuple
from itertools import repeat
from six.moves import zip, range

from .descriptors import OrderedGraph
from . import components, io
from . import Network

def _consense(x):
    v = x.iat[0]
    assert (x == v).all()
    return v

def _haversine(coords):
    lon, lat = np.deg2rad(np.asarray(coords)).T
    a = np.sin((lat[1]-lat[0])/2.)**2 + np.cos(lat[0]) * np.cos(lat[1]) * np.sin((lon[0] - lon[1])/2.)**2
    return 6371.000 * 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )

def aggregatebuses(network, busmap):
    columns = set(network.component_simple_descriptors[components.Bus])
    strategies = dict(x=np.mean, y=np.mean,
                      v_mag_max=np.min, v_mag_min=np.max, v_nom=np.max)
    strategies.update(zip(columns.difference(strategies), repeat(_consense)))

    return network.buses \
            .groupby(busmap).agg(strategies) \
            .reindex_axis([f for f in network.buses.columns if f in columns], axis=1)

def aggregatelines(network, buses, interlines):
    columns = set(network.component_simple_descriptors[components.Line]).difference(('bus0', 'bus1'))

    def aggregatelinegroup(l):
        # l.name is a tuple of the groupby index (bus0_s, bus1_s)
        length_s = _haversine(buses.loc[list(l.name),['x', 'y']])
        v_nom_s = _consense(buses.loc[list(l.name),'v_nom'])

        voltage_factor = (np.asarray(network.buses.loc[l.bus0,'v_nom'])/v_nom_s)**2
        length_factor = length_s/l['length']

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
            capital_cost=l['capital_cost'].sum(),
            length=length_s,
            sub_network=_consense(l['sub_network']),
            v_ang_min=l['v_ang_min'].max(),
            v_ang_max=l['v_ang_max'].min()
        )
        data.update((f, _consense(l[f])) for f in columns.difference(data))
        return pd.Series(data, index=[f for f in l.columns if f in columns])

    lines = interlines.groupby(['bus0_s', 'bus1_s']).apply(aggregatelinegroup)
    lines['name'] = [str(i+1) for i in range(len(lines))]
    return lines

def get_buses_linemap_and_lines(network, busmap):
    # compute new buses
    buses = aggregatebuses(network, busmap)

    lines = network.lines

    lines['bus0_s'] = lines.bus0.map(busmap)
    lines['bus1_s'] = lines.bus1.map(busmap)

    # lines between different clusters
    interlines = lines.loc[lines['bus0_s'] != lines['bus1_s']]
    lines = aggregatelines(network, buses, interlines)

    linemap = interlines.join(lines['name'], on=['bus0_s', 'bus1_s'])['name']
    return (buses,
            linemap,
            lines.reset_index()
                 .rename(columns={'bus0_s': 'bus0', 'bus1_s': 'bus1'}, copy=False)
                 .set_index('name'))

# network building stuff

def _build_network_from_buses_lines(buses, lines):
    network = Network()

    io.import_components_from_dataframe(network, buses, "Bus")
    io.import_components_from_dataframe(network, lines, "Line")

    network.build_graph()
    network.determine_network_topology()

    return network

Clustering = namedtuple('Clustering', ['network', 'busmap', 'linemap'])

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
    buses, linemap, lines = get_buses_linemap_and_lines(network, busmap)
    return Clustering(_build_network_from_buses_lines(buses, lines), busmap, linemap)

################
# SpectralClustering

try:
    # available using pip as scikit-learn
    from sklearn.cluster import spectral_clustering as sk_spectral_clustering

    def busmap_by_spectral_clustering(network, n_clusters, **kwds):
        lines = network.lines.loc[:,['bus0', 'bus1']].assign(weight=1./network.lines.x).set_index(['bus0','bus1'])
        G = OrderedGraph()
        G.add_nodes_from(network.buses.index)
        G.add_edges_from((u,v,dict(weight=w)) for (u,v),w in lines.itertuples())
        return pd.Series(sk_spectral_clustering(nx.adjacency_matrix(G), n_clusters, **kwds) + 1,
                         index=network.buses.index)

    def spectral_clustering(network, n_clusters=8, **kwds):
        busmap = busmap_by_spectral_clustering(network, n_clusters=n_clusters, **kwds)
        buses, linemap, lines = get_buses_linemap_and_lines(network, busmap)
        return Clustering(_build_network_from_buses_lines(buses, lines), busmap, linemap)

except ImportError:
    pass

################
# Louvain

try:
    # available using pip as python-louvain
    import community

    def busmap_by_louvain(network, level=-1):
        lines = network.lines.loc[:,['bus0', 'bus1']].assign(weight=1./network.lines.x).set_index(['bus0','bus1'])
        G = nx.Graph()
        G.add_nodes_from(network.buses.index)
        G.add_edges_from((u,v,dict(weight=w)) for (u,v),w in lines.itertuples())
        dendrogram = community.generate_dendrogram(G)
        if level < 0:
            level += len(dendrogram)
        return pd.Series(community.partition_at_level(dendrogram, level=level),
                         index=network.buses.index)

    def louvain_clustering(network, level=-1, **kwds):
        busmap = busmap_by_louvain(network, level=level)
        buses, linemap, lines = get_buses_linemap_and_lines(network, busmap)
        return Clustering(_build_network_from_buses_lines(buses, lines), busmap, linemap)

except ImportError:
    pass
