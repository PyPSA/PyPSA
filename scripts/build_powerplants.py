# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Retrieves conventional powerplant capacities and locations from `powerplantmatching <https://github.com/FRESNA/powerplantmatching>`_, assigns these to buses and creates a ``.csv`` file. It is possible to amend the powerplant database with custom entries provided in ``data/custom_powerplants.csv``.

Relevant Settings
-----------------

.. code:: yaml

    electricity:
      powerplants_filter:
      custom_powerplants:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity`

Inputs
------

- ``networks/base.nc``: confer :ref:`base`.
- ``data/custom_powerplants.csv``: custom powerplants in the same format as `powerplantmatching <https://github.com/FRESNA/powerplantmatching>`_ provides

Outputs
-------

- ``resource/powerplants.csv``: A list of conventional power plants (i.e. neither wind nor solar) with fields for name, fuel type, technology, country, capacity in MW, duration, commissioning year, retrofit year, latitude, longitude, and dam information as documented in the `powerplantmatching README <https://github.com/FRESNA/powerplantmatching/blob/master/README.md>`_; additionally it includes information on the closest substation/bus in ``networks/base.nc``.

    .. image:: ../img/powerplantmatching.png
        :scale: 30 %

    **Source:** `powerplantmatching on GitHub <https://github.com/FRESNA/powerplantmatching>`_

Description
-----------

The configuration options ``electricity: powerplants_filter`` and ``electricity: custom_powerplants`` can be used to control whether data should be retrieved from the original powerplants database or from custom amendmends. These specify `pandas.query <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_ commands.

1. Adding all powerplants from custom:

    .. code:: yaml

        powerplants_filter: false
        custom_powerplants: true

2. Replacing powerplants in e.g. Germany by custom data:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: true

    or

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: Country in ['Germany']


3. Adding additional built year constraints:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany'] and YearCommissioned <= 2015
        custom_powerplants: YearCommissioned <= 2015

"""

import logging
from _helpers import configure_logging

import pypsa
import powerplantmatching as pm
import pandas as pd
import numpy as np

from scipy.spatial import cKDTree as KDTree

logger = logging.getLogger(__name__)


def add_custom_powerplants(ppl):
    custom_ppl_query = snakemake.config['electricity']['custom_powerplants']
    if not custom_ppl_query:
        return ppl
    add_ppls = pd.read_csv(snakemake.input.custom_powerplants, index_col=0,
                           dtype={'bus': 'str'})
    if isinstance(custom_ppl_query, str):
        add_ppls.query(custom_ppl_query, inplace=True)
    return ppl.append(add_ppls, sort=False, ignore_index=True, verify_integrity=True)


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_powerplants')
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)
    countries = n.buses.country.unique()

    ppl = (pm.powerplants(from_url=True)
           .powerplant.fill_missing_decommyears()
           .powerplant.convert_country_to_alpha2()
           .query('Fueltype not in ["Solar", "Wind"] and Country in @countries')
           .replace({'Technology': {'Steam Turbine': 'OCGT'}})
            .assign(Fueltype=lambda df: (
                    df.Fueltype
                      .where(df.Fueltype != 'Natural Gas',
                             df.Technology.replace('Steam Turbine',
                                                   'OCGT').fillna('OCGT')))))

    ppl_query = snakemake.config['electricity']['powerplants_filter']
    if isinstance(ppl_query, str):
        ppl.query(ppl_query, inplace=True)

    ppl = add_custom_powerplants(ppl) # add carriers from own powerplant files

    cntries_without_ppl = [c for c in countries if c not in ppl.Country.unique()]

    for c in countries:
        substation_i = n.buses.query('substation_lv and country == @c').index
        kdtree = KDTree(n.buses.loc[substation_i, ['x','y']].values)
        ppl_i = ppl.query('Country == @c').index

        tree_i = kdtree.query(ppl.loc[ppl_i, ['lon','lat']].values)[1]
        ppl.loc[ppl_i, 'bus'] = substation_i.append(pd.Index([np.nan]))[tree_i]

    if cntries_without_ppl:
        logging.warning(f"No powerplants known in: {', '.join(cntries_without_ppl)}")

    bus_null_b = ppl["bus"].isnull()
    if bus_null_b.any():
        logging.warning(f"Couldn't find close bus for {bus_null_b.sum()} powerplants")

    ppl.to_csv(snakemake.output[0])
