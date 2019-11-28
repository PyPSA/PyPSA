"""
Iteratively solves expansion problem like the rule :mod:`solve_network`, but additionally
records intermediate branch capacity steps and values of the objective function.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
        solver:
            name:
            (solveroptions):

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`solving_cf`

Inputs
------

- ``networks/{network}_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`

Outputs
-------

- ``results/networks/{network}_s{simpl}_{clusters}_ec_l{ll}_{opts}_trace.nc``: Solved PyPSA network including optimisation results (with trace)

Description
-----------

"""

import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging

from solve_network import patch_pyomo_tmpdir, prepare_network, solve_network

import pypsa

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake
        snakemake = MockSnakemake(
            wildcards=dict(network='elec', simpl='', clusters='45', lv='1.25', opts='Co2L-3H'),
            input=["networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc"],
            output=["results/networks/s{simpl}_{clusters}_lv{lv}_{opts}_trace.nc"],
            log=dict(python="logs/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_python_trace.log")
        )

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        patch_pyomo_tmpdir(tmpdir)

    configure_logging(snakemake)

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
