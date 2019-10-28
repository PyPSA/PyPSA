# coding: utf-8
"""
Retrieves conventional powerplant capacities and locations from `powerplantmatching <https://github.com/FRESNA/powerplantmatching>`_, assigns these to buses and creates a ``.csv`` file.

Relevant Settings
-----------------

.. code:: yaml

    enable:
        powerplantmatching:

.. seealso:: 
    Documentation of the configuration file ``config.yaml`` at
    :ref:`toplevel_cf`

Inputs
------

- ``networks/base.nc``: confer :ref:`base`.

Outputs
-------

- ``resource/powerplants.csv``: A list of conventional power plants (i.e. neither wind nor solar) with fields for name, fuel type, technology, country, capacity in MW, duration, commissioning year, retrofit year, latitude, longitude, and dam information as documented in the `powerplantmatching README <https://github.com/FRESNA/powerplantmatching/blob/master/README.md>`_; additionally it includes information on the closest substation/bus in ``networks/base.nc``.

    .. image:: ../img/powerplantmatching.png
        :scale: 30 %

    **Source:** `powerplantmatching on GitHub <https://github.com/FRESNA/powerplantmatching>`_

Description
-----------

"""

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

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict

        snakemake = MockSnakemake(
            input=Dict(base_network='networks/base.nc'),
            output=['resources/powerplants.csv']
        )

    logging.basicConfig(level=snakemake.config['logging_level'])

    n = pypsa.Network(snakemake.input.base_network)

    ppm.powerplants(from_url=True)

    ppl = (ppm.collection.matched_data()
        [lambda df : ~df.Fueltype.isin(('Solar', 'Wind'))]
        .pipe(ppm.cleaning.clean_technology)
        .assign(Fueltype=lambda df: (
            df.Fueltype.where(df.Fueltype != 'Natural Gas',
                                df.Technology.replace('Steam Turbine', 'OCGT').fillna('OCGT'))))
        .pipe(ppm.utils.fill_geoposition))

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
