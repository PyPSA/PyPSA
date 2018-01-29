# coding: utf-8

import pandas as pd
idx = pd.IndexSlice

import os
import numpy as np
import scipy as sp
import xarray as xr
import geopandas as gpd
import shapely

from six.moves import reduce

import pypsa
from pypsa.networkclustering import (busmap_by_stubs, busmap_by_kmeans,
                                     _make_consense, get_clustering_from_busmap)
def normed(x):
    return (x/x.sum()).fillna(0.)

def simplify_network_to_380(n):
    ## All goes to v_nom == 380

    n.buses['v_nom'] = 380.

    linetype_380, = n.lines.loc[n.lines.v_nom == 380., 'type'].unique()
    lines_v_nom_b = n.lines.v_nom != 380.
    n.lines.loc[lines_v_nom_b, 'num_parallel'] *= (n.lines.loc[lines_v_nom_b, 'v_nom'] / 380.)**2
    n.lines.loc[lines_v_nom_b, 'v_nom'] = 380.
    n.lines.loc[lines_v_nom_b, 'type'] = linetype_380

    # Replace transformers by lines
    trafo_map = pd.Series(n.transformers.bus1.values, index=n.transformers.bus0.values)
    trafo_map = trafo_map[~trafo_map.index.duplicated(keep='first')]
    several_trafo_b = trafo_map.isin(trafo_map.index)
    trafo_map.loc[several_trafo_b] = trafo_map.loc[several_trafo_b].map(trafo_map)
    missing_buses_i = n.buses.index.difference(trafo_map.index)
    trafo_map = trafo_map.append(pd.Series(missing_buses_i, missing_buses_i))

    for c in n.one_port_components|n.branch_components:
        df = n.df(c)
        for col in df.columns:
            if col.startswith('bus'):
                df[col] = df[col].map(trafo_map)

    n.mremove("Transformer", n.transformers.index)
    n.mremove("Bus", n.buses.index.difference(trafo_map))

    return n, trafo_map

def remove_stubs(n):
    n.determine_network_topology()

    busmap = busmap_by_stubs(n, ['carrier', 'country'])

    n.buses.loc[busmap.index, ['x','y']] = n.buses.loc[busmap, ['x','y']].values

    clustering = get_clustering_from_busmap(
        n, busmap,
        bus_strategies=dict(country=_make_consense("Bus", "country")),
        line_length_factor=snakemake.config['lines']['length_factor'],
        aggregate_generators_weighted=True,
        aggregate_one_ports=["Load", "StorageUnit"]
    )

    return clustering.network, busmap

def weighting_for_country(x):
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
    return (w * (100. / w.max())).astype(int)

    return weighting_for_country


## Plot weighting for Germany

def plot_weighting(n, country):
    n.plot(bus_sizes=(2*weighting_for_country(n.buses.loc[n.buses.country == country])).reindex(n.buses.index, fill_value=1))
    p = vshapes.countries()['DE']
    plt.xlim(p.bounds[0], p.bounds[2])
    plt.ylim(p.bounds[1], p.bounds[3])


# # Determining the number of clusters per country

def distribute_clusters(n_clusters):
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

def distribute_clusters_exactly(n_clusters):
    for d in [0, 1, -1, 2, -2]:
        n_cluster_per_country = distribute_clusters(n_clusters + d)
        if n_cluster_per_country.sum() == n_clusters:
            return n_cluster_per_country
    else:
        return distribute_clusters(n_clusters)

def busmap_for_n_clusters(n_clusters):
    n_clusters = distribute_clusters_exactly(n_clusters)
    def busmap_for_country(x):
        prefix = x.name[0] + x.name[1] + ' '
        if len(x) == 1:
            return pd.Series(prefix + '0', index=x.index)
        weight = weighting_for_country(x)
        return prefix + busmap_by_kmeans(n, weight, n_clusters[x.name], buses_i=x.index)
    return n.buses.groupby(['country', 'sub_network'], group_keys=False).apply(busmap_for_country)

def plot_busmap_for_n_clusters(n_clusters=50):
    busmap = busmap_for_n_clusters(n_clusters)
    cs = busmap.unique()
    cr = sns.color_palette("hls", len(cs))
    n.plot(bus_colors=busmap.map(dict(zip(cs, cr))))
    del cs, cr

def clustering_for_n_clusters(n_clusters):
    clustering = get_clustering_from_busmap(
        n, busmap_for_n_clusters(n_clusters),
        bus_strategies=dict(country=_make_consense("Bus", "country")),
        aggregate_generators_weighted=True,
        aggregate_one_ports=["Load", "StorageUnit"]
    )

    # set n-1 security margin to 0.5 for 37 clusters and to 0.7 from 200 clusters
    # (there was already one of 0.7 in-place)
    s_max_pu = np.clip(0.5 + 0.2 * (n_clusters - 37) / (200 - 37), 0.5, 0.7)
    clustering.network.lines['s_max_pu'] = s_max_pu

    return clustering

def save_to_geojson(s, fn):
    if os.path.exists(fn):
        os.unlink(fn)
    s.reset_index().to_file(fn, driver='GeoJSON')

def cluster_regions(busmaps):
    busmap = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])

    for which in ('regions_onshore', 'regions_offshore'):
        regions = gpd.read_file(getattr(snakemake.input, which)).set_index('name')
        geom_c = regions.geometry.groupby(clustering.busmap).apply(shapely.ops.cascaded_union)
        regions_c = gpd.GeoDataFrame(dict(geometry=geom_c))
        save_to_geojson(regions_c, getattr(snakemake.output, which))

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('../config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.wildcards = Dict(clusters='37')
        snakemake.input = Dict(network='../networks/elec.nc',
                               regions_onshore='../resources/regions_onshore.geojson',
                               regions_offshore='../resources/regions_offshore.geojson')
        snakemake.output = Dict(network='../networks/elec_{clusters}.nc'.format(**snakemake.wildcards),
                                regions_onshore='../resources/regions_onshore_{clusters}.geojson'.format(**snakemake.wildcards),
                                regions_offshore='../resources/regions_offshore_{clusters}.geojson'.format(**snakemake.wildcards))

    n = pypsa.Network(snakemake.input.network)

    n, trafo_map = simplify_network_to_380(n)

    n, stub_map = remove_stubs(n)

    n_clusters = int(snakemake.wildcards.clusters)
    clustering = clustering_for_n_clusters(n_clusters)

    clustering.network.export_to_netcdf(snakemake.output.network)

    cluster_regions((trafo_map, stub_map, clustering.busmap))
