# coding: utf-8

import logging
logger = logging.getLogger(__name__)
import pandas as pd
idx = pd.IndexSlice

import numpy as np
import scipy as sp
import xarray as xr

import geopandas as gpd

import pypsa


def normed(s): return s/s.sum()

def add_co2limit(n, Nyears=1.):
    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=snakemake.config['electricity']['co2limit'] * Nyears)

def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']
    if exclude_co2: emission_prices.pop('co2')
    ep = (pd.Series(emission_prices).rename(lambda x: x+'_emissions') * n.carriers).sum(axis=1)
    n.generators['marginal_cost'] += n.generators.carrier.map(ep)
    n.storage_units['marginal_cost'] += n.storage_units.carrier.map(ep)

def set_line_volume_limit(n, lv):
    # Either line_volume cap or cost
    n.lines['capital_cost'] = 0.
    n.links['capital_cost'] = 0.

    lines_s_nom = n.lines.s_nom.where(
        n.lines.type == '',
        np.sqrt(3) * n.lines.num_parallel *
        n.lines.type.map(n.line_types.i_nom) *
        n.lines.bus0.map(n.buses.v_nom)
    )

    n.lines['s_nom_min'] = lines_s_nom
    n.links['p_nom_min'] = n.links['p_nom']

    n.lines['s_nom_extendable'] = True
    n.links['p_nom_extendable'] = True

    n.line_volume_limit = lv * ((lines_s_nom * n.lines['length']).sum() +
                                n.links.loc[n.links.carrier=='DC'].eval('p_nom * length').sum())

    return n

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('../config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.wildcards = Dict(clusters='37', lv='2', opts='Co2L')
        snakemake.input = ['../networks/elec_37.nc']
        snakemake.output = ['../networks/elec_37_lv2_Co2L.nc']

    logger.setLevel(snakemake.config['logging_level'])

    opts = snakemake.wildcards.opts.split('-')

    n = pypsa.Network(snakemake.input[0])
    Nyears = n.snapshot_weightings.sum()/8760.

    if 'Co2L' in opts:
        add_co2limit(n, Nyears)
        add_emission_prices(n, exclude_co2=True)

    if 'Ep' in opts:
        add_emission_prices(n)

    set_line_volume_limit(n, float(snakemake.wildcards.lv))

    n.export_to_netcdf(snakemake.output[0])
