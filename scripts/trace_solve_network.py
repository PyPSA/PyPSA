"""
Iteratively solves expansion problem like solve_network, but additionally
records intermediate branch capacity steps and values of the objective
"""

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from solve_network import patch_pyomo_tmpdir, prepare_network, solve_network

import pypsa


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict
        snakemake = MockSnakemake(
            wildcards=dict(network='elec', simpl='', clusters='45', lv='1.25', opts='Co2L-3H'),
            input=["networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc"],
            output=["results/networks/s{simpl}_{clusters}_lv{lv}_{opts}_trace.nc"],
            log=dict(python="logs/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_python_trace.log")
        )

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        patch_pyomo_tmpdir(tmpdir)

    logging.basicConfig(filename=snakemake.log.python,
                        level=snakemake.config['logging_level'])

    n = pypsa.Network(snakemake.input[0])

    solver_log = 'solver.log'
    config = snakemake.config['solving']
    opts = snakemake.wildcards.opts.split('-')

    def save_optimal_capacities(net, iteration, status):
        net.lines[f"s_nom_opt_{iteration}"] = net.lines["s_nom_opt"]
        net.links[f"p_nom_opt_{iteration}"] = net.links["p_nom_opt"]
        setattr(net, f"status_{iteration}", status)
        setattr(net, f"objective_{iteration}", net.objective)
        net.iteration = iteration

        net.export_to_netcdf(snakemake.output[0])

    config['options']['max_iterations'] = 12
    n = prepare_network(n, config['options'])
    n = solve_network(n, config, solver_log, opts, save_optimal_capacities)
    n.export_to_netcdf(snakemake.output[0])
