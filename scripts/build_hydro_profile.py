#!/usr/bin/env python
"""
Build hydroelectric inflow time-series for each country

See also
--------
build_renewable_profiles
"""

import os
import atlite
import pandas as pd
import geopandas as gpd
from vresutils import hydro as vhydro
import logging


if __name__ == "__main__":
    logger.basicConfig(level=snakemake.config['logging_level'])

    config = snakemake.config['renewable']['hydro']
    cutout = atlite.Cutout(config['cutout'],
                        cutout_dir=os.path.dirname(snakemake.input.cutout))

    countries = snakemake.config['countries']
    country_shapes = gpd.read_file(snakemake.input.country_shapes).set_index('name')['geometry'].reindex(countries)
    country_shapes.index.name = 'countries'

    eia_stats = vhydro.get_eia_annual_hydro_generation(snakemake.input.eia_hydro_generation).reindex(columns=countries)
    inflow = cutout.runoff(shapes=country_shapes,
                        smooth=True,
                        lower_threshold_quantile=True,
                        normalize_using_yearly=eia_stats)

    if 'clip_min_inflow' in config:
        inflow.values[inflow.values < config['clip_min_inflow']] = 0.

    inflow.to_netcdf(snakemake.output[0])
