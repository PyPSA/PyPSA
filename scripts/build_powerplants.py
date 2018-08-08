# coding: utf-8

import logging
import pandas as pd
from scipy.spatial import cKDTree as KDTree

import pypsa
import powerplantmatching as ppm

if 'snakemake' not in globals():
    from vresutils.snakemake import MockSnakemake, Dict

    snakemake = MockSnakemake(
        input=Dict(base_network='networks/base.nc'),
        output=['resources/powerplants.csv']
    )

logging.basicConfig(level=snakemake.config['logging_level'])

n = pypsa.Network(snakemake.input.base_network)

ppl = (ppm.collection.matched_data()
       [lambda df : ~df.Fueltype.isin(('Solar', 'Wind'))]
       .pipe(ppm.cleaning.clean_technology)
       .assign(Fueltype=lambda df: (
           df.Fueltype.where(df.Fueltype != 'Natural Gas',
                             df.Technology.replace('Steam Turbine', 'OCGT').fillna('OCGT'))))
       .pipe(ppm.utils.fill_geoposition, parse=True, only_saved_locs=True)
       .pipe(ppm.heuristics.fill_missing_duration))

# ppl.loc[(ppl.Fueltype == 'Other') & ppl.Technology.str.contains('CCGT'), 'Fueltype'] = 'CCGT'
# ppl.loc[(ppl.Fueltype == 'Other') & ppl.Technology.str.contains('Steam Turbine'), 'Fueltype'] = 'CCGT'

ppl = ppl.loc[ppl.lon.notnull() & ppl.lat.notnull()]

substation_lv_i = n.buses.index[n.buses['substation_lv']]
kdtree = KDTree(n.buses.loc[substation_lv_i, ['x','y']].values)
ppl = ppl.assign(bus=substation_lv_i[kdtree.query(ppl[['lon','lat']].values)[1]])

ppl.to_csv(snakemake.output[0])
