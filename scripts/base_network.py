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

    n = _remove_unconnected_components(n)

    _apply_parameter_corrections(n)

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
        snakemake.output = ['../networks/base_LC.nc']

    logger.setLevel(level=snakemake.config['logging_level'])

    n = base_network()
    n.export_to_netcdf(snakemake.output[0])
