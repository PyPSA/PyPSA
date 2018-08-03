# coding: utf-8

import yaml
import pandas as pd
import geopandas as gpd
import numpy as np
import scipy as sp, scipy.spatial
from scipy.sparse import csgraph
from six import iteritems
from six.moves import filter

from shapely.geometry import Point
import shapely, shapely.prepared, shapely.wkt

from vresutils.graph import BreadthFirstLevels

import logging
logger = logging.getLogger(__name__)

import pypsa

def _load_buses_from_eg():
    buses = (pd.read_csv(snakemake.input.eg_buses, quotechar="'",
                         true_values='t', false_values='f',
                         dtype=dict(bus_id="str", under_construction='bool'))
            .set_index("bus_id")
            .drop(['station_id'], axis=1)
            .rename(columns=dict(voltage='v_nom')))

    buses['carrier'] = buses.pop('dc').map({True: 'DC', False: 'AC'})
    buses['under_construction'] = buses['under_construction'].fillna(False).astype(bool)

    # remove all buses outside of all countries including exclusive economic zones (offshore)
    europe_shape = gpd.read_file(snakemake.input.europe_shape).loc[0, 'geometry']
    europe_shape_prepped = shapely.prepared.prep(europe_shape)
    buses_in_europe_b = buses[['x', 'y']].apply(lambda p: europe_shape_prepped.contains(Point(p)), axis=1)

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
        if attrs is None: continue

        for attr, repls in iteritems(attrs):
            for i, r in iteritems(repls):
                if i == 'oid':
                    df["oid"] = df.tags.str.extract('"oid"=>"(\d+)"', expand=False)
                    r = df.oid.map(repls["oid"]).dropna()
                elif i == 'index':
                    r = pd.Series(repls["index"])
                else:
                    raise NotImplementedError()
                df.loc[r.index, attr] = r.astype(df[attr].dtype)

def _set_electrical_parameters_lines(lines):
    v_noms = snakemake.config['electricity']['voltages']
    linetypes = snakemake.config['lines']['types']

    for v_nom in v_noms:
        lines.loc[lines["v_nom"] == v_nom, 'type'] = linetypes[v_nom]

    lines['s_max_pu'] = snakemake.config['lines']['s_max_pu']
    if not snakemake.config['lines']['with_under_construction']:
        lines.loc[lines.under_construction.astype(bool), 'num_parallel'] = 0.

    return lines

def _set_lines_s_nom_from_linetypes(n):
    n.lines['s_nom'] = (
        np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) *
        n.lines['v_nom'] * n.lines.num_parallel
    )

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

    if not snakemake.config['links']['with_under_construction']:
        links.loc[links.under_construction.astype(bool), "p_nom"] = 0.

    return links

def _set_electrical_parameters_converters(converters):
    converters['p_max_pu'] = snakemake.config['links']['s_max_pu']
    converters['p_min_pu'] = -1. * snakemake.config['links']['s_max_pu']

    converters['p_nom'] = 2000

    # Converters are combined with links
    converters['under_construction'] = False
    converters['underground'] = False

    return converters

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

def _set_countries_and_substations(n):

    buses = n.buses

    def buses_in_shape(shape):
        shape = shapely.prepared.prep(shape)
        return pd.Series(
            np.fromiter((shape.contains(Point(x, y))
                        for x, y in buses.loc[:,["x", "y"]].values),
                        dtype=bool, count=len(buses)),
            index=buses.index
        )

    countries = snakemake.config['countries']
    country_shapes = gpd.read_file(snakemake.input.country_shapes).set_index('id')['geometry']
    offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes).set_index('id')['geometry']
    substation_b = buses['symbol'].str.contains('substation', case=False)

    def prefer_voltage(x, which):
        index = x.index
        if len(index) == 1:
            return pd.Series(index, index)
        key = (x.index[0]
               if x['v_nom'].isnull().all()
               else getattr(x['v_nom'], 'idx' + which)())
        return pd.Series(key, index)

    gb = buses.loc[substation_b].groupby(['x', 'y'], as_index=False,
                                         group_keys=False, sort=False)
    bus_map_low = gb.apply(prefer_voltage, 'min')
    lv_b = (bus_map_low == bus_map_low.index).reindex(buses.index, fill_value=False)
    bus_map_high = gb.apply(prefer_voltage, 'max')
    hv_b = (bus_map_high == bus_map_high.index).reindex(buses.index, fill_value=False)

    onshore_b = pd.Series(False, buses.index)
    offshore_b = pd.Series(False, buses.index)

    for country in countries:
        onshore_shape = country_shapes[country]
        onshore_country_b = buses_in_shape(onshore_shape)
        onshore_b |= onshore_country_b

        buses.loc[onshore_country_b, 'country'] = country

        if country not in offshore_shapes: continue
        offshore_country_b = buses_in_shape(offshore_shapes[country])
        offshore_b |= offshore_country_b

        buses.loc[offshore_country_b, 'country'] = country

    buses['substation_lv'] = lv_b & onshore_b & (~ buses['under_construction'])
    buses['substation_off'] = (offshore_b | (hv_b & onshore_b)) & (~ buses['under_construction'])

    # Nearest country in numbers of hops defines country of homeless buses
    c_nan_b = buses.country.isnull()
    c = n.buses['country']
    graph = n.graph()
    n.buses.loc[c_nan_b, 'country'] = \
        [(next(filter(len, map(lambda x: c.loc[x].dropna(), BreadthFirstLevels(graph, [b]))))
          .value_counts().index[0])
         for b in buses.index[c_nan_b]]

    return buses

def base_network():
    buses = _load_buses_from_eg()

    links = _load_links_from_eg(buses)
    converters = _load_converters_from_eg(buses)

    lines = _load_lines_from_eg(buses)
    transformers = _load_transformers_from_eg(buses)

    lines = _set_electrical_parameters_lines(lines)
    transformers = _set_electrical_parameters_transformers(transformers)
    links = _set_electrical_parameters_links(links)
    converters = _set_electrical_parameters_converters(converters)

    n = pypsa.Network()
    n.name = 'PyPSA-Eur'

    n.set_snapshots(pd.date_range(freq='h', **snakemake.config['snapshots']))

    n.import_components_from_dataframe(buses, "Bus")
    n.import_components_from_dataframe(lines, "Line")
    n.import_components_from_dataframe(transformers, "Transformer")
    n.import_components_from_dataframe(links, "Link")
    n.import_components_from_dataframe(converters, "Link")

    n = _remove_unconnected_components(n)

    _set_lines_s_nom_from_linetypes(n)

    _apply_parameter_corrections(n)

    _set_countries_and_substations(n)

    return n

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict
        snakemake = MockSnakemake(
            path='..',
            wildcards={},
            input=Dict(
                eg_buses='data/entsoegridkit/buses.csv',
                eg_lines='data/entsoegridkit/lines.csv',
                eg_links='data/entsoegridkit/links.csv',
                eg_converters='data/entsoegridkit/converters.csv',
                eg_transformers='data/entsoegridkit/transformers.csv',
                parameter_corrections='data/parameter_corrections.yaml',
                links_p_nom='data/links_p_nom.csv'
            ),
            output = ['networks/base_LC.nc']
        )

    logging.basicConfig(level=snakemake.config['logging_level'])

    n = base_network()
    n.export_to_netcdf(snakemake.output[0])
