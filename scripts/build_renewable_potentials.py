import atlite
import xarray as xr
import pandas as pd

from vresutils import landuse as vlanduse

config = snakemake.config['renewable'][snakemake.wildcards.technology]

cutout = atlite.Cutout(config['cutout'])

total_capacity = config['capacity_per_sqm'] * vlanduse._cutout_cell_areas(cutout)
potentials = xr.DataArray(total_capacity *
                          vlanduse.corine_for_cutout(cutout, fn=snakemake.input.corine,
                                                     natura_fn=snakemake.input.natura, **config['corine']),
                          [cutout.meta.indexes['y'], cutout.meta.indexes['x']])

if 'height_cutoff' in config:
    potentials.values[(cutout.meta['height'] < - config['height_cutoff']).transpose(*potentials.dims)] = 0.

potentials.to_netcdf(snakemake.output[0])
