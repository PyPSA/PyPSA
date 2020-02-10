# coding: utf-8
"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such as

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** of carbon-dioxide emissions (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying a limit on the **cost** of transmission expansion,
- specifying a limit on the **volume** of transmission expansion, and
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
    n.generators['marginal_cost'] += n.generators.carrier.map(ep)
    n.storage_units['marginal_cost'] += n.storage_units.carrier.map(ep)

def set_line_s_max_pu(n):
    # set n-1 security margin to 0.5 for 37 clusters and to 0.7 from 200 clusters
    n_clusters = len(n.buses)
    s_max_pu = np.clip(0.5 + 0.2 * (n_clusters - 37) / (200 - 37), 0.5, 0.7)
    n.lines['s_max_pu'] = s_max_pu

def set_line_cost_limit(n, lc, Nyears=1.):
    links_dc_b = n.links.carrier == 'DC' if not n.links.empty else pd.Series()

    lines_s_nom = n.lines.s_nom.where(
        n.lines.type == '',
        np.sqrt(3) * n.lines.num_parallel *
        n.lines.type.map(n.line_types.i_nom) *
        n.lines.bus0.map(n.buses.v_nom)
    )

    n.lines['capital_cost_lc'] = n.lines['capital_cost']
    n.links['capital_cost_lc'] = n.links['capital_cost']
    total_line_cost = ((lines_s_nom * n.lines['capital_cost_lc']).sum() +
                       n.links.loc[links_dc_b].eval('p_nom * capital_cost_lc').sum())

    if lc == 'opt':
        costs = load_costs(Nyears, snakemake.input.tech_costs,
                           snakemake.config['costs'], snakemake.config['electricity'])
        update_transmission_costs(n, costs, simple_hvdc_costs=False)
    else:
        # Either line_volume cap or cost
        n.lines['capital_cost'] = 0.
        n.links.loc[links_dc_b, 'capital_cost'] = 0.

    if lc == 'opt' or float(lc) > 1.0:
        n.lines['s_nom_min'] = lines_s_nom
        n.lines['s_nom_extendable'] = True

        n.links.loc[links_dc_b, 'p_nom_min'] = n.links.loc[links_dc_b, 'p_nom']
        n.links.loc[links_dc_b, 'p_nom_extendable'] = True

        if lc != 'opt':
            line_cost = float(lc) * total_line_cost
            n.add('GlobalConstraint', 'lc_limit',
                  type='transmission_expansion_cost_limit',
                  sense='<=', constant=line_cost, carrier_attribute='AC, DC')
    return n

def set_line_volume_limit(n, lv, Nyears=1.):
    links_dc_b = n.links.carrier == 'DC' if not n.links.empty else pd.Series()

    lines_s_nom = n.lines.s_nom.where(
        n.lines.type == '',
        np.sqrt(3) * n.lines.num_parallel *
        n.lines.type.map(n.line_types.i_nom) *
        n.lines.bus0.map(n.buses.v_nom)
    )

    total_line_volume = ((lines_s_nom * n.lines['length']).sum() +
                         n.links.loc[links_dc_b].eval('p_nom * length').sum())

    if lv == 'opt':
        costs = load_costs(Nyears, snakemake.input.tech_costs,
                           snakemake.config['costs'], snakemake.config['electricity'])
        update_transmission_costs(n, costs, simple_hvdc_costs=True)
    else:
        # Either line_volume cap or cost
        n.lines['capital_cost'] = 0.
        n.links.loc[links_dc_b, 'capital_cost'] = 0.

    if lv == 'opt' or float(lv) > 1.0:
        n.lines['s_nom_min'] = lines_s_nom
        n.lines['s_nom_extendable'] = True

        n.links.loc[links_dc_b, 'p_nom_min'] = n.links.loc[links_dc_b, 'p_nom']
        n.links.loc[links_dc_b, 'p_nom_extendable'] = True

        if lv != 'opt':
            line_volume = float(lv) * total_line_volume
            n.add('GlobalConstraint', 'lv_limit',
                  type='transmission_volume_expansion_limit',
                  sense='<=', constant=line_volume, carrier_attribute='AC, DC')
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
                                  clusters='5', ll='copt', opts='Co2L-24H')
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

    if 'Ep' in opts:
        add_emission_prices(n)

    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    if ll_type == 'v':
        set_line_volume_limit(n, factor, Nyears)
    elif ll_type == 'c':
        set_line_cost_limit(n, factor, Nyears)

    n.export_to_netcdf(snakemake.output[0])
