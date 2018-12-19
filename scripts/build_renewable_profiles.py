#!/usr/bin/env python

import os
import atlite
import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool

import glaes as gl
import geokit as gk
from osgeo import gdal
from scipy.sparse import csr_matrix, vstack

from pypsa.geo import haversine
from vresutils import landuse as vlanduse
from vresutils.array import spdiag

import progressbar as pgb
import logging
logger = logging.getLogger(__name__)

bounds = dx = dy = gebco = clc = natura = None
def init_globals(n_bounds, n_dx, n_dy):
    # global in each process of the multiprocessing.Pool
    global bounds, dx, dy, gebco, clc, natura

    bounds = n_bounds
    dx = n_dx
    dy = n_dy

    gebco = gk.raster.loadRaster(snakemake.input.gebco)
    gebco.SetProjection(gk.srs.loadSRS(4326).ExportToWkt())

    clc = gk.raster.loadRaster(snakemake.input.corine)
    clc.SetProjection(gk.srs.loadSRS(3035).ExportToWkt())

    natura = gk.raster.loadRaster(snakemake.input.natura)

def downsample_to_coarse_grid(bounds, dx, dy, mask, data):
    # The GDAL warp function with the 'average' resample algorithm needs a band of zero values of at least
    # the size of one coarse cell around the original raster or it produces erroneous results
    orig = mask.createRaster(data=data)
    padded_extent = mask.extent.castTo(bounds.srs).pad(max(dx, dy)).castTo(mask.srs)
    padded = padded_extent.fit((mask.pixelWidth, mask.pixelHeight)).warp(orig, mask.pixelWidth, mask.pixelHeight)
    orig = None # free original raster
    average = bounds.createRaster(dx, dy, dtype=gdal.GDT_Float32)
    assert gdal.Warp(average, padded, resampleAlg='average') == 1, "gdal warp failed: %s" % gdal.GetLastErrorMsg()
    return average

def calculate_potential(gid):
    feature = gk.vector.extractFeature(snakemake.input.regions, where=gid)
    ec = gl.ExclusionCalculator(feature.geom)

    corine = config.get("corine", {})
    if isinstance(corine, list):
        corine = {'grid_codes': corine}
    if "grid_codes" in corine:
        ec.excludeRasterType(clc, value=corine["grid_codes"], invert=True)
    if corine.get("distance", 0.) > 0.:
        ec.excludeRasterType(clc, value=corine["distance_grid_codes"], buffer=corine["distance"])

    if config.get("natura", False):
        ec.excludeRasterType(natura, value=1)
    if "max_depth" in config:
        ec.excludeRasterType(gebco, (None, -config["max_depth"]))

    # TODO compute a distance field as a raster beforehand
    if 'max_shore_distance' in config:
        ec.excludeVectorType(snakemake.input.country_shapes, buffer=config['max_shore_distance'], invert=True)
    if 'min_shore_distance' in config:
        ec.excludeVectorType(snakemake.input.country_shapes, buffer=config['min_shore_distance'])

    availability = downsample_to_coarse_grid(bounds, dx, dy, ec.region, np.where(ec.region.mask, ec._availability, 0))

    return csr_matrix(gk.raster.extractMatrix(availability).flatten() / 100.)


if __name__ == '__main__':
    pgb.streams.wrap_stderr()
    logging.basicConfig(level=snakemake.config['logging_level'])

    config = snakemake.config['renewable'][snakemake.wildcards.technology]

    time = pd.date_range(freq='m', **snakemake.config['snapshots'])
    params = dict(years=slice(*time.year[[0, -1]]), months=slice(*time.month[[0, -1]]))

    cutout = atlite.Cutout(config['cutout'],
                        cutout_dir=os.path.dirname(snakemake.input.cutout),
                        **params)

    minx, maxx, miny, maxy = cutout.extent
    dx = (maxx - minx) / (cutout.shape[1] - 1)
    dy = (maxy - miny) / (cutout.shape[0] - 1)
    bounds = gk.Extent.from_xXyY((minx - dx/2., maxx + dx/2.,
                                  miny - dy/2., maxy + dy/2.))

    # Use GLAES to compute available potentials and the transition matrix

    with Pool(initializer=init_globals, initargs=(bounds, dx, dy),
              maxtasksperchild=20, processes=snakemake.config['atlite'].get('nprocesses', 2)) as pool:
        regions = gk.vector.extractFeatures(snakemake.input.regions, onlyAttr=True)
        buses = pd.Index(regions['name'], name="bus")
        widgets = [
            pgb.widgets.Percentage(),
            ' ', pgb.widgets.SimpleProgress(format='(%s)' % pgb.widgets.SimpleProgress.DEFAULT_FORMAT),
            ' ', pgb.widgets.Bar(),
            ' ', pgb.widgets.Timer(),
            ' ', pgb.widgets.ETA()
        ]
        progressbar = pgb.ProgressBar(prefix='Compute GIS potentials: ', widgets=widgets, max_value=len(regions))
        matrix = vstack(list(progressbar(pool.imap(calculate_potential, regions.index))))

    potentials = config['capacity_per_sqkm'] * vlanduse._cutout_cell_areas(cutout)
    potmatrix = matrix * spdiag(potentials.ravel())
    potmatrix.data[potmatrix.data < 1.] = 0 # ignore weather cells where only less than 1 MW can be installed
    potmatrix.eliminate_zeros()

    with pgb.ProgressBar(prefix='Compute capacity factors: ', max_value=1) as progressbar:
        progressbar.update(0)
        resource = config['resource']
        func = getattr(cutout, resource.pop('method'))
        correction_factor = config.get('correction_factor', 1.)
        if correction_factor != 1.:
            logger.warning('correction_factor is set as {}'.format(correction_factor))
        capacity_factor = correction_factor * func(capacity_factor=True, **resource).stack(spatial=('y', 'x')).values
        layoutmatrix = potmatrix * spdiag(capacity_factor)
        progressbar.update(1)


    with pgb.ProgressBar(prefix='Compute profiles: ', max_value=1) as progressbar:
        progressbar.update(0)
        profile, capacities = func(matrix=layoutmatrix, index=buses, per_unit=True,
                                   return_capacity=True, **resource)
        progressbar.update(1)


    p_nom_max_meth = config.get('potential', 'conservative')

    if p_nom_max_meth == 'conservative':
        # p_nom_max has to be calculated for each bus and is the minimal ratio
        # (min over all weather grid cells of the bus region) between the available
        # potential (potmatrix) and the used normalised layout (layoutmatrix /
        # capacities), so we would like to calculate i.e. potmatrix / (layoutmatrix /
        # capacities). Since layoutmatrix = potmatrix * capacity_factor, this
        # corresponds to capacities/max(capacity factor in the voronoi cell)
        p_nom_max = xr.DataArray([1./np.max(capacity_factor[inds]) if len(inds) else 0.
                                  for inds in np.split(potmatrix.indices, potmatrix.indptr[1:-1])], [buses]) * capacities
    elif p_nom_max_meth == 'heuristic':
        p_nom_max = 0.8 * xr.DataArray(np.asarray(potmatrix.sum(axis=1)).squeeze(), [buses])
    else:
        raise AssertionError('Config key `potential` should be one of "conservative" (default) or "heuristic",'
                             ' not "{}"'.format(p_nom_max_meth))

    layout = xr.DataArray(np.asarray(potmatrix.sum(axis=0)).reshape(cutout.shape),
                          [cutout.meta.indexes[ax] for ax in ['y', 'x']])

    # Determine weighted average distance from substation
    cell_coords = cutout.grid_coordinates()

    average_distance = []
    for i in regions.index:
        row = layoutmatrix[i]
        distances = haversine(regions.loc[i, ['x', 'y']], cell_coords[row.indices])[0]
        average_distance.append((distances * (row.data / row.data.sum())).sum())

    average_distance = xr.DataArray(average_distance, [buses])

    ds = xr.merge([(correction_factor * profile).rename('profile'),
                   capacities.rename('weight'),
                   p_nom_max.rename('p_nom_max'),
                   layout.rename('potential'),
                   average_distance.rename('average_distance')])

    if snakemake.wildcards.technology.startswith("offwind"):
        import geopandas as gpd
        from shapely.geometry import LineString

        offshore_shape = gpd.read_file(snakemake.input.offshore_shapes).set_index('id').unary_union
        underwater_fraction = []
        for i in regions.index:
            row = layoutmatrix[i]
            centre_of_mass = (cell_coords[row.indices] * (row.data / row.data.sum())[:,np.newaxis]).sum(axis=0)
            line = LineString([centre_of_mass, regions.loc[i, ['x', 'y']]])
            underwater_fraction.append(line.intersection(offshore_shape).length / line.length)

        ds['underwater_fraction'] = xr.DataArray(underwater_fraction, [buses])

    (ds.sel(bus=(ds['profile'].mean('time') > config.get('min_p_max_pu', 0.)) & (ds['p_nom_max'] > 0.))
     .to_netcdf(snakemake.output.profile))
