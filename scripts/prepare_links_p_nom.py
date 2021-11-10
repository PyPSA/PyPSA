#!/usr/bin/env python

# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Extracts capacities of HVDC links from `Wikipedia <https://en.wikipedia.org/wiki/List_of_HVDC_projects>`_.

Relevant Settings
-----------------

.. code:: yaml

    enable:
        prepare_links_p_nom:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`toplevel_cf`

Inputs
------

*None*

Outputs
-------

- ``data/links_p_nom.csv``: A plain download of https://en.wikipedia.org/wiki/List_of_HVDC_projects#Europe plus extracted coordinates.

Description
-----------

*None*

"""

import logging
from _helpers import configure_logging

import pandas as pd

logger = logging.getLogger(__name__)


def multiply(s):
    return s.str[0].astype(float) * s.str[1].astype(float)


def extract_coordinates(s):
    regex = (r"(\d{1,2})°(\d{1,2})′(\d{1,2})″(N|S) "
             r"(\d{1,2})°(\d{1,2})′(\d{1,2})″(E|W)")
    e = s.str.extract(regex, expand=True)
    lat = (e[0].astype(float) + (e[1].astype(float) + e[2].astype(float)/60.)/60.)*e[3].map({'N': +1., 'S': -1.})
    lon = (e[4].astype(float) + (e[5].astype(float) + e[6].astype(float)/60.)/60.)*e[7].map({'E': +1., 'W': -1.})
    return lon, lat


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake #rule must be enabled in config
        snakemake = mock_snakemake('prepare_links_p_nom', simpl='', network='elec')
    configure_logging(snakemake)

    links_p_nom = pd.read_html('https://en.wikipedia.org/wiki/List_of_HVDC_projects', header=0, match="SwePol")[0]

    mw = "Power (MW)"
    m_b = links_p_nom[mw].str.contains('x').fillna(False)

    links_p_nom.loc[m_b, mw] = links_p_nom.loc[m_b, mw].str.split('x').pipe(multiply)
    links_p_nom[mw] = links_p_nom[mw].str.extract("[-/]?([\d.]+)", expand=False).astype(float)

    links_p_nom['x1'], links_p_nom['y1'] = extract_coordinates(links_p_nom['Converterstation 1'])
    links_p_nom['x2'], links_p_nom['y2'] = extract_coordinates(links_p_nom['Converterstation 2'])

    links_p_nom.dropna(subset=['x1', 'y1', 'x2', 'y2']).to_csv(snakemake.output[0], index=False)
