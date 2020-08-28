# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# coding: utf-8
"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such as

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** per tonne emissions of carbon-dioxide (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        emission_prices:
        USD2013_to_EUR2013:
        discountrate:
        marginal_cost:
        capital_cost:

    electricity:
        co2limit:
        max_hours:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`costs_cf`, :ref:`electricity_cf`

Inputs
------

- ``data/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``networks/{network}_s{simpl}_{clusters}.nc``: confer :ref:`cluster`

Outputs
-------

- ``networks/{network}_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Complete PyPSA network that will be handed to the ``solve_network`` rule.

Description
-----------

.. tip::
    The rule :mod:`prepare_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`prepare_network`.

"""

import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging

from add_electricity import load_costs, update_transmission_costs
from six import iteritems

import numpy as np
import re
import pypsa
import pandas as pd

idx = pd.IndexSlice

def add_co2limit(n, Nyears=1., factor=None):

    if factor is not None:
        annual_emissions = factor*snakemake.config['electricity']['co2base']
    else:
        annual_emissions = snakemake.config['electricity']['co2limit']

    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=annual_emissions * Nyears)


def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']
    if exclude_co2: emission_prices.pop('co2')
    ep = (pd.Series(emission_prices).rename(lambda x: x+'_emissions') *
          n.carriers.filter(like='_emissions')).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators['marginal_cost'] += gen_ep
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units['marginal_cost'] += su_ep


def set_line_s_max_pu(n):
    # set n-1 security margin to 0.5 for 37 clusters and to 0.7 from 200 clusters
    n_clusters = len(n.buses)
    s_max_pu = np.clip(0.5 + 0.2 * (n_clusters - 37) / (200 - 37), 0.5, 0.7)
    n.lines['s_max_pu'] = s_max_pu


def set_transmission_limit(n, ll_type, factor, Nyears=1):
    links_dc_b = n.links.carrier == 'DC' if not n.links.empty else pd.Series()

    _lines_s_nom = (np.sqrt(3) * n.lines.type.map(n.line_types.i_nom) *
                   n.lines.num_parallel *  n.lines.bus0.map(n.buses.v_nom))
    lines_s_nom = n.lines.s_nom.where(n.lines.type == '', _lines_s_nom)


    col = 'capital_cost' if ll_type == 'c' else 'length'
    ref = (lines_s_nom @ n.lines[col] +
           n.links[links_dc_b].p_nom @ n.links[links_dc_b][col])

    costs = load_costs(Nyears, snakemake.input.tech_costs,
                       snakemake.config['costs'],
                       snakemake.config['electricity'])
    update_transmission_costs(n, costs, simple_hvdc_costs=False)

    if factor == 'opt' or float(factor) > 1.0:
        n.lines['s_nom_min'] = lines_s_nom
        n.lines['s_nom_extendable'] = True

        n.links.loc[links_dc_b, 'p_nom_min'] = n.links.loc[links_dc_b, 'p_nom']
        n.links.loc[links_dc_b, 'p_nom_extendable'] = True

    if factor != 'opt':
        con_type = 'expansion_cost' if ll_type == 'c' else 'volume_expansion'
        rhs = float(factor) * ref
        n.add('GlobalConstraint', f'l{ll_type}_limit',
              type=f'transmission_{con_type}_limit',
              sense='<=', constant=rhs, carrier_attribute='AC, DC')
    return n



def average_every_nhours(n, offset):
    logger.info('Resampling the network to {}'.format(offset))
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name+"_t")
        for k, df in iteritems(c.pnl):
            if not df.empty:
                pnl[k] = df.resample(offset).mean()

    return m


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('prepare_network', network='elec', simpl='',
                                  clusters='40', ll='v0.3', opts='Co2L-24H')
    configure_logging(snakemake)

    opts = snakemake.wildcards.opts.split('-')

    n = pypsa.Network(snakemake.input[0])
    Nyears = n.snapshot_weightings.sum()/8760.

    set_line_s_max_pu(n)

    for o in opts:
        m = re.match(r'^\d+h$', o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break
    else:
        logger.info("No resampling")

    for o in opts:
        if "Co2L" in o:
            m = re.findall("[0-9]*\.?[0-9]+$", o)
            if len(m) > 0:
                add_co2limit(n, Nyears, float(m[0]))
            else:
                add_co2limit(n, Nyears)

    for o in opts:
        oo = o.split("+")
        if oo[0].startswith(tuple(n.carriers.index)):
            carrier = oo[0]
            cost_factor = float(oo[1])
            if carrier == "AC":  # lines do not have carrier
                n.lines.capital_cost *= cost_factor
            else:
                comps = {"Generator", "Link", "StorageUnit"}
                for c in n.iterate_components(comps):
                    sel = c.df.carrier.str.contains(carrier)
                    c.df.loc[sel,"capital_cost"] *= cost_factor

    if 'Ep' in opts:
        add_emission_prices(n)

    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    set_transmission_limit(n, ll_type, factor, Nyears)

    n.export_to_netcdf(snakemake.output[0])
