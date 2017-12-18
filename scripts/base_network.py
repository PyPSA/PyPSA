# coding: utf-8

import yaml
import pandas as pd
import numpy as np
import scipy as sp, scipy.spatial
from scipy.sparse import csgraph
from operator import attrgetter
from six import iteritems
from itertools import count, chain

import shapely, shapely.prepared, shapely.wkt
from shapely.geometry import Point

from vresutils import shapes as vshapes

import logging
logger = logging.getLogger(__name__)

import pypsa

def _find_closest_bus(buses, pos):
    if (not hasattr(_find_closest_bus, 'kdtree')) or len(_find_closest_bus.kdtree.data) != len(buses.index):
        _find_closest_bus.kdtree = sp.spatial.cKDTree(buses.loc[:,["x", "y"]].values)
    return buses.index[_find_closest_bus.kdtree.query(pos)[1]]

def _load_buses_from_eg():
    buses = (pd.read_csv(snakemake.input.eg_buses, quotechar="'",
                         true_values='t', false_values='f',
                         dtype=dict(bus_id="str"))
            .set_index("bus_id")
            .drop(['under_construction', 'station_id'], axis=1)
            .rename(columns=dict(voltage='v_nom')))

    buses['carrier'] = buses.pop('dc').map({True: 'DC', False: 'AC'})

    # remove all buses outside of all countries including exclusive economic zones (offshore)
    europe_shape = vshapes.country_cover(snakemake.config['countries'])
    europe_shape_exterior = shapely.geometry.Polygon(shell=europe_shape.exterior) # no holes
    europe_shape_exterior_prepped = shapely.prepared.prep(europe_shape_exterior)
    buses_in_europe_b = buses[['x', 'y']].apply(lambda p: europe_shape_exterior_prepped.contains(Point(p)), axis=1)

    buses_with_v_nom_to_keep_b = buses.v_nom.isin(snakemake.config['electricity']['voltages']) | buses.v_nom.isnull()
    logger.info("Removing buses with voltages {}".format(pd.Index(buses.v_nom.unique()).dropna().difference(snakemake.config['electricity']['voltages'])))

    return pd.DataFrame(buses.loc[buses_in_europe_b & buses_with_v_nom_to_keep_b])

def _load_transformers_from_eg(buses):
    transformers = (pd.read_csv(snakemake.input.eg_transformers, quotechar="'",
                                true_values='t', false_values='f',
                                dtype=dict(transformer_id='str', bus0='str', bus1='str'))
                    .set_index('transformer_id'))

    transformers = _remove_dangling_branches(transformers, buses)

    return transformers

def _load_converters_from_eg(buses):
    converters = (pd.read_csv(snakemake.input.eg_converters, quotechar="'",
                              true_values='t', false_values='f',
                              dtype=dict(converter_id='str', bus0='str', bus1='str'))
                  .set_index('converter_id'))

    converters = _remove_dangling_branches(converters, buses)

    converters['carrier'] = 'B2B'

    return converters


def _load_links_from_eg(buses):
    links = (pd.read_csv(snakemake.input.eg_links, quotechar="'", true_values='t', false_values='f',
                         dtype=dict(link_id='str', bus0='str', bus1='str', under_construction="bool"))
             .set_index('link_id'))

    links['length'] /= 1e3

    links = _remove_dangling_branches(links, buses)

    # Add DC line parameters
    links['carrier'] = 'DC'

    return links

def _load_lines_from_eg(buses):
    lines = (pd.read_csv(snakemake.input.eg_lines, quotechar="'", true_values='t', false_values='f',
                         dtype=dict(line_id='str', bus0='str', bus1='str',
                                    underground="bool", under_construction="bool"))
             .set_index('line_id')
             .rename(columns=dict(voltage='v_nom', circuits='num_parallel')))

    lines['length'] /= 1e3

    lines = _remove_dangling_branches(lines, buses)

    return lines

def _split_aclines_with_several_voltages(buses, lines, transformers):
    ## Split AC lines with multiple voltages
    def namer(string, start=0): return (string.format(x) for x in count(start=start))
    busname = namer("M{:02}")
    trafoname = namer("M{:02}")
    linename = namer("M{:02}")

    def find_or_add_lower_v_nom_bus(bus, v_nom):
        candidates = transformers.loc[(transformers.bus1 == bus) &
                                      (transformers.bus0.map(buses.v_nom) == v_nom),
                                      'bus0']
        if len(candidates):
            return candidates.iloc[0]
        new_bus = next(busname)
        buses.loc[new_bus] = pd.Series({'v_nom': v_nom, 'symbol': 'joint', 'carrier': 'AC',
                                        'x': buses.at[bus, 'x'], 'y': buses.at[bus, 'y'],
                                        'under_construction': buses.at[bus, 'under_construction']})

        transformers.loc[next(trafoname)] = pd.Series({'bus0': new_bus, 'bus1': bus})
        return new_bus

    voltage_levels = lines.v_nom.unique()

    for line in lines.tags.str.extract(r'"text_"=>"\(?(\d+)\+(\d+).*?"', expand=True).dropna().itertuples():
        v_nom = int(line._2)
        if lines.at[line.Index, 'num_parallel'] > 1:
            lines.at[line.Index, 'num_parallel'] -= 1

        if v_nom in voltage_levels:
            bus0 = find_or_add_lower_v_nom_bus(lines.at[line.Index, 'bus0'], v_nom)
            bus1 = find_or_add_lower_v_nom_bus(lines.at[line.Index, 'bus1'], v_nom)
            lines.loc[next(linename)] = pd.Series(
                dict(chain(iteritems({'bus0': bus0, 'bus1': bus1, 'v_nom': v_nom, 'circuits': 1}),
                           iteritems({k: lines.at[line.Index, k]
                                      for k in ('underground', 'under_construction',
                                                'tags', 'geometry', 'length')})))
            )

    return buses, lines, transformers

def _apply_parameter_corrections(n):
    with open(snakemake.input.parameter_corrections) as f:
        corrections = yaml.load(f)

    for component, attrs in iteritems(corrections):
        df = n.df(component)
        for attr, repls in iteritems(attrs):
            for i, r in iteritems(repls):
                if i == 'oid':
                    df["oid"] = df.tags.str.extract('"oid"=>"(\d+)"', expand=False)
                    r = df.oid.map(repls["oid"]).dropna()
                elif i == 'index':
                    r = pd.Series(repls["index"])
                else:
                    raise NotImplementedError()
                df.loc[r.index, attr] = r

def _set_electrical_parameters_lines(lines):
    v_noms = snakemake.config['electricity']['voltages']
    linetypes = snakemake.config['lines']['types']

    for v_nom in v_noms:
        lines.loc[lines["v_nom"] == v_nom, 'type'] = linetypes[v_nom]

    lines['s_max_pu'] = snakemake.config['lines']['s_max_pu']

    return lines

def _set_electrical_parameters_links(links):
    links['p_max_pu'] = snakemake.config['links']['s_max_pu']
    links['p_min_pu'] = -1. * snakemake.config['links']['s_max_pu']

    links_p_nom = pd.read_csv(snakemake.input.links_p_nom)

    tree = sp.spatial.KDTree(np.vstack([
        links_p_nom[['x1', 'y1', 'x2', 'y2']],
        links_p_nom[['x2', 'y2', 'x1', 'y1']]
    ]))

    dist, ind = tree.query(
        np.asarray([np.asarray(shapely.wkt.loads(s))[[0, -1]].flatten()
                    for s in links.geometry]),
        distance_upper_bound=1.5
    )

    links_p_nom["j"] =(
        pd.DataFrame(dict(D=dist, i=links_p_nom.index[ind % len(links_p_nom)]), index=links.index)
        .groupby('i').D.idxmin()
    )

    p_nom = links_p_nom.dropna(subset=["j"]).set_index("j")["Power (MW)"]
    links.loc[p_nom.index, "p_nom"] = p_nom

    links.loc[links.under_construction.astype(bool), "p_nom"] = 0.

    return links

def _set_electrical_parameters_transformers(transformers):
    config = snakemake.config['transformers']

    ## Add transformer parameters
    transformers["x"] = config.get('x', 0.1)
    transformers["s_nom"] = config.get('s_nom', 2000)
    transformers['type'] = config.get('type', '')

    return transformers

def _remove_dangling_branches(branches, buses):
    return pd.DataFrame(branches.loc[branches.bus0.isin(buses.index) & branches.bus1.isin(buses.index)])

def _connect_close_buses(network, radius=1.):
    adj = network.graph(["Line", "Transformer", "Link"]).adj

    n_lines_added = 0
    n_transformers_added = 0
    ac_buses = network.buses[network.buses.carrier == 'AC']

    for i,u in enumerate(ac_buses.index):

        vs = ac_buses[["x","y"]].iloc[i+1:]
        distance_km = pypsa.geo.haversine(vs, ac_buses.loc[u,["x","y"]])

        for j,v in enumerate(vs.index):
            km = distance_km[j,0]

            if km < radius:
                if u in adj[v]:
                    continue
                #print(u,v,ac_buses.at[u,"v_nom"], ac_buses.at[v,"v_nom"],km)

                if ac_buses.at[u,"v_nom"] != ac_buses.at[v,"v_nom"]:
                    network.add("Transformer","extra_trafo_{}_{}".format(u,v),s_nom=2000,bus0=u,bus1=v,x=0.1)
                    n_transformers_added += 1
                else:
                    network.add("Line","extra_line_{}_{}".format(u,v),s_nom=4000,bus0=u,bus1=v,x=0.1)
                    n_lines_added += 1

    logger.info("Added {} lines and {} transformers to connect buses less than {} km apart."
                .format(n_lines_added, n_transformers_added, radius))

    return network

def _remove_connected_components_smaller_than(network, min_size):
    network.determine_network_topology()

    sub_network_sizes = network.buses.groupby('sub_network').size()
    subs_to_remove = sub_network_sizes.index[sub_network_sizes < min_size]

    logger.info("Removing {} small sub_networks (synchronous zones) with less than {} buses. In total {} buses."
                .format(len(subs_to_remove), min_size, network.buses.sub_network.isin(subs_to_remove).sum()))

    return network[~network.buses.sub_network.isin(subs_to_remove)]

def _remove_unconnected_components(network):
    _, labels = csgraph.connected_components(network.adjacency_matrix(), directed=False)
    component = pd.Series(labels, index=network.buses.index)

    component_sizes = component.value_counts()
    components_to_remove = component_sizes.iloc[1:]

    logger.info("Removing {} unconnected network components with less than {} buses. In total {} buses."
                .format(len(components_to_remove), components_to_remove.max(), components_to_remove.sum()))

    return network[component == component_sizes.index[0]]

def base_network():
    buses = _load_buses_from_eg()

    links = _load_links_from_eg(buses)
    converters = _load_converters_from_eg(buses)

    lines = _load_lines_from_eg(buses)
    transformers = _load_transformers_from_eg(buses)

    # buses, lines, transformers = _split_aclines_with_several_voltages(buses, lines, transformers)

    lines = _set_electrical_parameters_lines(lines)
    links = _set_electrical_parameters_links(links)
    transformers = _set_electrical_parameters_transformers(transformers)

    n = pypsa.Network()
    n.name = 'PyPSA-Eur'

    n.set_snapshots(pd.date_range(snakemake.config['historical_year'], periods=8760, freq='h'))

    n.import_components_from_dataframe(buses, "Bus")
    n.import_components_from_dataframe(lines, "Line")
    n.import_components_from_dataframe(transformers, "Transformer")
    n.import_components_from_dataframe(links, "Link")
    n.import_components_from_dataframe(converters, "Link")

    if 'T' in snakemake.wildcards.opts.split('-'):
        raise NotImplemented

    # n = _connect_close_buses(n, radius=1.)

    n = _remove_unconnected_components(n)

    _apply_parameter_corrections(n)

    # Workaround: import_components_from_dataframe does not preserve types of foreign columns
    n.lines['underground'] = n.lines['underground'].astype(bool)
    n.lines['under_construction'] = n.lines['under_construction'].astype(bool)
    n.links['underground'] = n.links['underground'].astype(bool)
    n.links['under_construction'] = n.links['under_construction'].astype(bool)

    return n

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        snakemake.input = Dict(
            eg_buses='../data/entsoegridkit/buses.csv',
            eg_lines='../data/entsoegridkit/lines.csv',
            eg_links='../data/entsoegridkit/links.csv',
            eg_converters='../data/entsoegridkit/converters.csv',
            eg_transformers='../data/entsoegridkit/transformers.csv',
            parameter_corrections='../data/parameter_corrections.yaml',
            links_p_nom='../data/links_p_nom.csv'
        )

        snakemake.wildcards = Dict(opts='LC')
        with open('../config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.output = ['../networks/base_LC.h5']

    logger.setLevel(level=snakemake.config['logging_level'])

    n = base_network()
    n.export_to_hdf5(snakemake.output[0])
