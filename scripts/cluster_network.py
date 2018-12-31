# coding: utf-8

import pandas as pd
idx = pd.IndexSlice

import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
import scipy as sp
from scipy.sparse.csgraph import connected_components
import xarray as xr
import geopandas as gpd
import shapely
import networkx as nx
from shutil import copyfile

from six import iteritems
from six.moves import reduce

import pyomo.environ as po

import pypsa
from pypsa.io import import_components_from_dataframe, import_series_from_dataframe
from pypsa.networkclustering import (busmap_by_stubs, busmap_by_kmeans,
                                     _make_consense, get_clustering_from_busmap,
                                     aggregategenerators, aggregateoneport)
def normed(x):
    return (x/x.sum()).fillna(0.)

def weighting_for_country(n, x):
    conv_carriers = {'OCGT', 'PHS', 'hydro'}
    gen = (n
           .generators.loc[n.generators.carrier.isin(conv_carriers)]
           .groupby('bus').p_nom.sum()
           .reindex(n.buses.index, fill_value=0.) +
           n
           .storage_units.loc[n.storage_units.carrier.isin(conv_carriers)]
           .groupby('bus').p_nom.sum()
           .reindex(n.buses.index, fill_value=0.))
    load = n.loads_t.p_set.mean().groupby(n.loads.bus).sum()

    b_i = x.index
    g = normed(gen.reindex(b_i, fill_value=0))
    l = normed(load.reindex(b_i, fill_value=0))

    w= g + l
    return (w * (100. / w.max())).clip(lower=1.).astype(int)


## Plot weighting for Germany

def plot_weighting(n, country, country_shape=None):
    n.plot(bus_sizes=(2*weighting_for_country(n.buses.loc[n.buses.country == country])).reindex(n.buses.index, fill_value=1))
    if country_shape is not None:
        plt.xlim(country_shape.bounds[0], country_shape.bounds[2])
        plt.ylim(country_shape.bounds[1], country_shape.bounds[3])


# # Determining the number of clusters per country

def distribute_clusters(n, n_clusters):
    load = n.loads_t.p_set.mean().groupby(n.loads.bus).sum()
    loadc = load.groupby([n.buses.country, n.buses.sub_network]).sum()
    n_cluster_per_country = n_clusters * normed(loadc)
    one_cluster_b = n_cluster_per_country < 0.5
    n_one_cluster, n_one_cluster_prev = one_cluster_b.sum(), 0

    while n_one_cluster > n_one_cluster_prev:
        n_clusters_rem = n_clusters - one_cluster_b.sum()
        assert n_clusters_rem > 0
        n_cluster_per_country[~one_cluster_b] = n_clusters_rem * normed(loadc[~one_cluster_b])
        one_cluster_b = n_cluster_per_country < 0.5
        n_one_cluster, n_one_cluster_prev = one_cluster_b.sum(), n_one_cluster

    n_cluster_per_country[one_cluster_b] = 1.1
    n_cluster_per_country[~one_cluster_b] = n_cluster_per_country[~one_cluster_b] + 0.5

    return n_cluster_per_country.astype(int)

def distribute_clusters_exactly(n, n_clusters):
    for d in [0, 1, -1, 2, -2]:
        n_cluster_per_country = distribute_clusters(n, n_clusters + d)
        if n_cluster_per_country.sum() == n_clusters:
            return n_cluster_per_country
    else:
        return distribute_clusters(n, n_clusters)

def distribute_clusters_optim(n, n_clusters, solver_name=None):
    if solver_name is None:
        solver_name = snakemake.config['solver']['solver']['name']

    L = (n.loads_t.p_set.mean()
         .groupby(n.loads.bus).sum()
         .groupby([n.buses.country, n.buses.sub_network]).sum()
         .pipe(normed))

    m = po.ConcreteModel()
    m.n = po.Var(list(L.index), bounds=(1, None), domain=po.Integers)
    m.tot = po.Constraint(expr=(po.summation(m.n) == n_clusters))
    m.objective = po.Objective(expr=po.sum((m.n[i] - L.loc[i]*n_clusters)**2
                                           for i in L.index),
                               sense=po.minimize)

    opt = po.SolverFactory(solver_name)
    if isinstance(opt, pypsa.opf.PersistentSolver):
        opt.set_instance(m)
    results = opt.solve(m)
    assert results['Solver'][0]['Status'].key == 'ok', "Solver returned non-optimally: {}".format(results)

    return pd.Series(m.n.get_values(), index=L.index).astype(int)

def busmap_for_n_clusters(n, n_clusters):
    n.determine_network_topology()

    if 'snakemake' in globals():
        solver_name = snakemake.config['solving']['solver']['name']
    else:
        solver_name = "gurobi"

    n_clusters = distribute_clusters_optim(n, n_clusters, solver_name=solver_name)

    def busmap_for_country(x):
        prefix = x.name[0] + x.name[1] + ' '
        if len(x) == 1:
            return pd.Series(prefix + '0', index=x.index)
        weight = weighting_for_country(n, x)
        return prefix + busmap_by_kmeans(n, weight, n_clusters[x.name], buses_i=x.index, n_init=1000, max_iter=30000, tol=1e-6)
    return n.buses.groupby(['country', 'sub_network'], group_keys=False).apply(busmap_for_country)

def plot_busmap_for_n_clusters(n, n_clusters=50):
    busmap = busmap_for_n_clusters(n, n_clusters)
    cs = busmap.unique()
    cr = sns.color_palette("hls", len(cs))
    n.plot(bus_colors=busmap.map(dict(zip(cs, cr))))
    del cs, cr

def clustering_for_n_clusters(n, n_clusters, aggregate_carriers=None,
                              line_length_factor=1.25, potential_mode='conservative'):

    if potential_mode == 'conservative':
        p_nom_max_strategy = np.min
    elif potential_mode == 'heuristic':
        p_nom_max_strategy = np.sum
    else:
        raise AttributeError("potential_mode should be one of 'conservative' or 'heuristic', "
                             "but is '{}'".format(potential_mode))

    clustering = get_clustering_from_busmap(
        n, busmap_for_n_clusters(n, n_clusters),
        bus_strategies=dict(country=_make_consense("Bus", "country")),
        aggregate_generators_weighted=True,
        aggregate_generators_carriers=aggregate_carriers,
        aggregate_one_ports=["Load", "StorageUnit"],
        line_length_factor=line_length_factor,
        generator_strategies={'p_nom_max': p_nom_max_strategy}
    )

    return clustering

def save_to_geojson(s, fn):
    if os.path.exists(fn):
        os.unlink(fn)
    df = s.reset_index()
    schema = {**gpd.io.file.infer_schema(df), 'geometry': 'Unknown'}
    df.to_file(fn, driver='GeoJSON', schema=schema)

def cluster_regions(busmaps, input=None, output=None):
    if input is None: input = snakemake.input
    if output is None: output = snakemake.output

    busmap = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])

    for which in ('regions_onshore', 'regions_offshore'):
        regions = gpd.read_file(getattr(input, which)).set_index('name')
        geom_c = regions.geometry.groupby(busmap).apply(shapely.ops.cascaded_union)
        regions_c = gpd.GeoDataFrame(dict(geometry=geom_c))
        regions_c.index.name = 'name'
        save_to_geojson(regions_c, getattr(output, which))

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict
        snakemake = MockSnakemake(
            wildcards=Dict(network='elec', simpl='', clusters='45'),
            input=Dict(
                network='networks/{network}_s{simpl}.nc',
                regions_onshore='resources/regions_onshore_{network}_s{simpl}.geojson',
                regions_offshore='resources/regions_offshore_{network}_s{simpl}.geojson'
            ),
            output=Dict(
                network='networks/{network}_s{simpl}_{clusters}.nc',
                regions_onshore='resources/regions_onshore_{network}_s{simpl}_{clusters}.geojson',
                regions_offshore='resources/regions_offshore_{network}_s{simpl}_{clusters}.geojson'
            )
        )

    logging.basicConfig(level=snakemake.config['logging_level'])

    n = pypsa.Network(snakemake.input.network)

    renewable_carriers = pd.Index([tech
                                   for tech in n.generators.carrier.unique()
                                   if tech.split('-', 2)[0] in snakemake.config['renewable']])

    if snakemake.wildcards.clusters.endswith('m'):
        n_clusters = int(snakemake.wildcards.clusters[:-1])
        aggregate_carriers = None # All
    else:
        n_clusters = int(snakemake.wildcards.clusters)
        aggregate_carriers = pd.Index(n.generators.carrier.unique()).difference(renewable_carriers)

    if n_clusters == len(n.buses):
        # Fast-path if no clustering is necessary
        busmap = n.buses.index.to_series()
        linemap = n.lines.index.to_series()
        clustering = pypsa.networkclustering.Clustering(n, busmap, linemap, linemap, pd.Series(dtype='O'))
    else:
        line_length_factor = snakemake.config['lines']['length_factor']

        def consense(x):
            v = x.iat[0]
            assert ((x == v).all() or x.isnull().all()), (
                "The `potential` configuration option must agree for all renewable carriers, for now!"
            )
            return v
        potential_mode = consense(pd.Series([snakemake.config['renewable'][tech]['potential']
                                             for tech in renewable_carriers]))
        clustering = clustering_for_n_clusters(n, n_clusters, aggregate_carriers,
                                               line_length_factor=line_length_factor,
                                               potential_mode=potential_mode)

    clustering.network.export_to_netcdf(snakemake.output.network)
    with pd.HDFStore(snakemake.output.clustermaps, mode='w') as store:
        with pd.HDFStore(snakemake.input.clustermaps, mode='r') as clustermaps:
            for attr in clustermaps.keys():
                store.put(attr, clustermaps[attr], format="table", index=False)
        for attr in ('busmap', 'linemap', 'linemap_positive', 'linemap_negative'):
            store.put(attr, getattr(clustering, attr), format="table", index=False)

    cluster_regions((clustering.busmap,))


