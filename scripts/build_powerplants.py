# coding: utf-8

import logging
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as KDTree
import pycountry as pyc

import pypsa
import powerplantmatching as ppm

def country_alpha_2(name):
    try:
        cntry = pyc.countries.get(name=name)
    except KeyError:
        cntry = None
    if cntry is None:
        cntry = pyc.countries.get(official_name=name)
    return cntry.alpha_2

def add_my_carriers(ppl):
    switch = snakemake.config['electricity']['my_carriers_switch']
    if switch == 'replace-all' or switch == 'replace-selection' or switch == 'add':
        countries_dict = snakemake.config['countries_dict'] # dictionary, eg. GB: United Kindgom
        for country in snakemake.config['electricity']['my_carriers_for_countries']:
            dirname = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
            add_ppls = pd.read_csv(dirname + "/resources/powerplants_" + country + ".csv", index_col=0)
            if switch == 'replace-all' or switch == 'replace-selection':
                to_drop = ppl[ppl.Country == countries_dict[country]]
                if switch == 'replace-selection':
                    to_drop = to_drop[to_drop.Fueltype.isin(add_ppls.groupby('Fueltype').mean().index)]
                ppl = ppl.drop(to_drop.index)
            ppl = ppl.append(add_ppls, sort='False')
    else:
        logger.warning('my_carriers_switch is invalid keyword, try one of [add, replace-all, replace-selection]. powerplants remain unchanged.')
    return ppl

def restrict_buildyear(ppl):
    year = snakemake.config['electricity']['restrict_buildyear']
    search_pattern = [str(int(year)+x) for x in range(1,2050-int(year))]
    logger.info('restricting build year of generators to ' + str(year) + '...')
    for pattern in search_pattern: #do it in forloop+contains instead of map as YearCommissioned might have the weirdest formats
        ppl = ppl[ppl['YearCommissioned'].str.contains(pattern) == False]
    return ppl

if __name__ == "__main__":
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
        .pipe(ppm.utils.fill_geoposition))
    ppl = add_my_carriers(ppl) # add carriers from own powerplant files
    ppl = restrict_buildyear(ppl)

    # ppl.loc[(ppl.Fueltype == 'Other') & ppl.Technology.str.contains('CCGT'), 'Fueltype'] = 'CCGT'
    # ppl.loc[(ppl.Fueltype == 'Other') & ppl.Technology.str.contains('Steam Turbine'), 'Fueltype'] = 'CCGT'

    ppl = ppl.loc[ppl.lon.notnull() & ppl.lat.notnull()]

    ppl_country = ppl.Country.map(country_alpha_2)
    countries = n.buses.country.unique()
    cntries_without_ppl = []

    for cntry in countries:
        substation_lv_i = n.buses.index[n.buses['substation_lv'] & (n.buses.country == cntry)]
        ppl_b = ppl_country == cntry
        if not ppl_b.any():
            cntries_without_ppl.append(cntry)
            continue

        kdtree = KDTree(n.buses.loc[substation_lv_i, ['x','y']].values)
        ppl.loc[ppl_b, 'bus'] = substation_lv_i[kdtree.query(ppl.loc[ppl_b, ['lon','lat']].values)[1]]

    if cntries_without_ppl:
        logging.warning("No powerplants known in: {}".format(", ".join(cntries_without_ppl)))

    bus_null_b = ppl["bus"].isnull()
    if bus_null_b.any():
        logging.warning("Couldn't find close bus for {} powerplants".format(bus_null_b.sum()))

    ppl.to_csv(snakemake.output[0])
