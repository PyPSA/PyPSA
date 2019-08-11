# coding: utf-8
"""
Creates networks clustered to ``{cluster}`` number of zones with aggregated buses, generators and transmission corridors.

Relevant Settings
-----------------

.. code:: yaml

    renewable: (keys)
        {technology}:
            potential:

    solving:
        solver:
            name:

    lines:
        length_factor:

Inputs
------

- ``resources/regions_onshore_{network}_s{simpl}.geojson``: confer :ref:`simplify`
- ``resources/regions_offshore_{network}_s{simpl}.geojson``: confer :ref:`simplify`
- ``resources/clustermaps_{network}_s{simpl}.h5``: confer :ref:`simplify`
- ``networks/{network}_s{simpl}.nc``: confer :ref:`simplify`

Outputs
-------

- ``resources/regions_onshore_{network}_s{simpl}_{clusters}.geojson``:
- ``resources/regions_offshore_{network}_s{simpl}_{clusters}.geojson``:
- ``resources/clustermaps_{network}_s{simpl}_{clusters}.h5``:
- ``networks/{network}_s{simpl}_{clusters}.nc``:

Description
-----------

The rule cluster_network instead clusters the network to a given number of buses.

    -Why is this cluster function used?
    -Why the user can set a number behind the elec_sXXX for simplification?

As you found out for yourself, elec_s100_50.nc for example is a network in which simplify_network clusters the network to 100 buses and in a second step cluster_network reduces it down to 50 buses.

Well, let me provide a use-case where this makes sense:

In preliminary tests, it turns out, that the principal effect of changing spatial resolution is actually only partially due to the transmission network. It is more important to differentiate between wind generators with higher capacity factors from those with lower capacity factors, ie to have a higher spatial resolution in the renewable generation than in the number of buses.

This two-step clustering can take advantage of that fact (and allows to study it)
by looking at networks like networks/elec_s100_50m.nc (note the additional m in the cluster wildcard). For this example simplify_network clusters to 100 buses and then cluster_network clusters to 50m buses, which means 50 buses for the network topology but only moving instead of aggregating the generators to the clustered buses. So in this network you still have up to 100 different wind generators, 2 at each bus on average.

In combination these two features allow you to study the spatial resolution of the transmission network separately from the spatial resolution of renewable generators. Beware: There is no clear evidence telling you what is a good representation of the full model. These options are under active study.

    Why we have a cluster function inside of the simplification method?

Why are you asking three times the same question?

    Is it possible to run the model without the simplification method / rule?
    I tryed to run the snakemake without the s for simplification.

No, the network clustering methods in PyPSA's networkclustering module don't work reliably with multiple voltage levels and transformers. If it is somehow necessary for you we could include switches to make Step 2 and 3 optional as well. But that's about it.

.. tip::
    The rule ``cluster_all_networks`` runs
    for all ``scenario`` s in the configuration file 
    the rule ``cluster_network``.

"""

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
    conv_carriers = {'OCGT','CCGT','PHS', 'hydro'}
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

def distribute_clusters(n, n_clusters, solver_name=None):
    if solver_name is None:
        solver_name = snakemake.config['solving']['solver']['name']

    L = (n.loads_t.p_set.mean()
         .groupby(n.loads.bus).sum()
         .groupby([n.buses.country, n.buses.sub_network]).sum()
         .pipe(normed))

    N = n.buses.groupby(['country', 'sub_network']).size()

    assert n_clusters >= len(N) and n_clusters <= N.sum(), \
        "Number of clusters must be {} <= n_clusters <= {} for this selection of countries.".format(len(N), N.sum())

    m = po.ConcreteModel()
    def n_bounds(model, *n_id):
        return (1, N[n_id])
    m.n = po.Var(list(L.index), bounds=n_bounds, domain=po.Integers)
    m.tot = po.Constraint(expr=(po.summation(m.n) == n_clusters))
    m.objective = po.Objective(expr=sum((m.n[i] - L.loc[i]*n_clusters)**2 for i in L.index),
                               sense=po.minimize)

    opt = po.SolverFactory(solver_name)
    if not opt.has_capability('quadratic_objective'):
        logger.warn(f'The configured solver `{solver_name}` does not support quadratic objectives. Falling back to `ipopt`.')
        opt = po.SolverFactory('ipopt')

    results = opt.solve(m)
    assert results['Solver'][0]['Status'].key == 'ok', "Solver returned non-optimally: {}".format(results)

    return pd.Series(m.n.get_values(), index=L.index).astype(int)

def busmap_for_n_clusters(n, n_clusters, solver_name, algorithm="kmeans", **algorithm_kwds):
    if algorithm == "kmeans":
        algorithm_kwds.setdefault('n_init', 1000)
        algorithm_kwds.setdefault('max_iter', 30000)
        algorithm_kwds.setdefault('tol', 1e-6)

    n.determine_network_topology()

    n_clusters = distribute_clusters(n, n_clusters, solver_name=solver_name)

    def reduce_network(n, buses):
        nr = pypsa.Network()
        nr.import_components_from_dataframe(buses, "Bus")
        nr.import_components_from_dataframe(n.lines.loc[n.lines.bus0.isin(buses.index) & n.lines.bus1.isin(buses.index)], "Line")
        return nr

    def busmap_for_country(x):
        prefix = x.name[0] + x.name[1] + ' '
        logger.debug("Determining busmap for country {}".format(prefix[:-1]))
        if len(x) == 1:
            return pd.Series(prefix + '0', index=x.index)
        weight = weighting_for_country(n, x)

        if algorithm == "kmeans":
            return prefix + busmap_by_kmeans(n, weight, n_clusters[x.name], buses_i=x.index, **algorithm_kwds)
        elif algorithm == "spectral":
            return prefix + busmap_by_spectral_clustering(reduce_network(n, x), n_clusters[x.name], **algorithm_kwds)
        elif algorithm == "louvain":
            return prefix + busmap_by_louvain(reduce_network(n, x), n_clusters[x.name], **algorithm_kwds)
        else:
            raise ArgumentError("`algorithm` must be one of 'kmeans', 'spectral' or 'louvain'")

    return (n.buses.groupby(['country', 'sub_network'], group_keys=False, squeeze=True)
            .apply(busmap_for_country).rename('busmap'))

def plot_busmap_for_n_clusters(n, n_clusters=50):
    busmap = busmap_for_n_clusters(n, n_clusters)
    cs = busmap.unique()
    cr = sns.color_palette("hls", len(cs))
    n.plot(bus_colors=busmap.map(dict(zip(cs, cr))))
    del cs, cr

def clustering_for_n_clusters(n, n_clusters, aggregate_carriers=None,
                              line_length_factor=1.25, potential_mode='simple',
                              solver_name="cbc", algorithm="kmeans"):

    if potential_mode == 'simple':
        p_nom_max_strategy = np.sum
    elif potential_mode == 'conservative':
        p_nom_max_strategy = np.min
    else:
        raise AttributeError("potential_mode should be one of 'simple' or 'conservative', "
                             "but is '{}'".format(potential_mode))

    clustering = get_clustering_from_busmap(
        n, busmap_for_n_clusters(n, n_clusters, solver_name, algorithm),
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
                regions_offshore='resources/regions_offshore_{network}_s{simpl}.geojson',
                clustermaps='resources/clustermaps_{network}_s{simpl}.h5'
            ),
            output=Dict(
                network='networks/{network}_s{simpl}_{clusters}.nc',
                regions_onshore='resources/regions_onshore_{network}_s{simpl}_{clusters}.geojson',
                regions_offshore='resources/regions_offshore_{network}_s{simpl}_{clusters}.geojson',
                clustermaps='resources/clustermaps_{network}_s{simpl}_{clusters}.h5'
            )
        )

    logging.basicConfig(level=snakemake.config['logging_level'])

    n = pypsa.Network(snakemake.input.network)

    renewable_carriers = pd.Index([tech
                                   for tech in n.generators.carrier.unique()
                                   if tech.split('-', 2)[0] in snakemake.config['renewable']])

    if snakemake.wildcards.clusters.endswith('m'):
        n_clusters = int(snakemake.wildcards.clusters[:-1])
        aggregate_carriers = pd.Index(n.generators.carrier.unique()).difference(renewable_carriers)
    else:
        n_clusters = int(snakemake.wildcards.clusters)
        aggregate_carriers = None # All

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
                                               potential_mode=potential_mode,
                                               solver_name=snakemake.config['solving']['solver']['name'])

    clustering.network.export_to_netcdf(snakemake.output.network)
    with pd.HDFStore(snakemake.output.clustermaps, mode='w') as store:
        with pd.HDFStore(snakemake.input.clustermaps, mode='r') as clustermaps:
            for attr in clustermaps.keys():
                store.put(attr, clustermaps[attr], format="table", index=False)
        for attr in ('busmap', 'linemap', 'linemap_positive', 'linemap_negative'):
            store.put(attr, getattr(clustering, attr), format="table", index=False)

    cluster_regions((clustering.busmap,))
