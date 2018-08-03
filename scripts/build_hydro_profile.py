#!/usr/bin/env python

import os
import atlite
import pandas as pd
import geopandas as gpd
from vresutils import hydro as vhydro
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=snakemake.config['logging_level'])

cutout = atlite.Cutout(snakemake.config['renewable']['hydro']['cutout'],
                       cutout_dir=os.path.dirname(snakemake.input.cutout))

countries = snakemake.config['countries']
country_shapes = gpd.read_file(snakemake.input.country_shapes).set_index('id')['geometry'].reindex(countries)
country_shapes.index.name = 'countries'

eia_stats = vhydro.get_eia_annual_hydro_generation(snakemake.input.eia_hydro_generation).reindex(columns=countries)
inflow = cutout.runoff(shapes=country_shapes,
                       smooth=True,
                       lower_threshold_quantile=True,
                       normalize_using_yearly=eia_stats)

inflow.to_netcdf(snakemake.output[0])
