import numpy as np
import pandas as pd

import logging
logging.basicConfig(filename=snakemake.log.python, level=logging.INFO)

import pypsa

from _helpers import madd

if 'tmpdir' in snakemake.config['solving']:
    # PYOMO should write its lp files into tmp here
    tmpdir = snakemake.config['solving']['tmpdir']
    import os
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    from pyutilib.services import TempfileManager
    TempfileManager.tempdir = tmpdir

def prepare_network(n):
    solve_opts = snakemake.config['solving']['options']
    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('load_shedding'):
        n.add("Carrier", "Load")
        madd(n, "Generator", "Load",
             bus=n.buses.index,
             carrier='load',
             marginal_cost=1.0e5 * snakemake.config['costs']['EUR_to_ZAR'],
             # intersect between macroeconomic and surveybased
             # willingness to pay
             # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
             p_nom=1e6)

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components():
            #if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                t.df['marginal_cost'] += 1e-2 + 2e-3*(np.random.random(len(t.df)) - 0.5)

    if solve_opts.get('nhours'):
        nhours = solve_opts['nhours']
        n = n[:solve_opts['nhours'], :]
        n.snapshot_weightings[:] = 8760./nhours

    return n

def solve_network(n):
    def add_opts_constraints(n):
        opts = snakemake.wildcards.opts.split('-')

        if 'BAU' in opts:
            mincaps = snakemake.config['electricity']['BAU_mincapacities']
            def bau_mincapacities_rule(model, carrier):
                gens = n.generators.index[n.generators.p_nom_extendable & (n.generators.carrier == carrier)]
                return sum(model.generator_p_nom[gen] for gen in gens) >= mincaps[carrier]
            n.model.bau_mincapacities = pypsa.opt.Constraint(list(mincaps), rule=bau_mincapacities_rule)

        if 'SAFE' in opts:
            peakdemand = (1. + snakemake.config['electricity']['SAFE_reservemargin']) * n.loads_t.p_set.sum(axis=1).max()
            conv_techs = snakemake.config['plotting']['conv_techs']
            exist_conv_caps = n.generators.loc[n.generators.carrier.isin(conv_techs) & ~n.generators.p_nom_extendable, 'p_nom'].sum()
            ext_gens_i = n.generators.index[n.generators.carrier.isin(conv_techs) & n.generators.p_nom_extendable]
            n.model.safe_peakdemand = pypsa.opt.Constraint(expr=sum(n.model.generator_p_nom[gen] for gen in ext_gens_i) >= peakdemand - exist_conv_caps)

    def fix_lines(n, lines_i=None, links_i=None): # , fix=True):
        if lines_i is not None and len(lines_i) > 0:
            s_nom = n.lines.s_nom.where(
                n.lines.type == '',
                np.sqrt(3) * n.lines.type.map(n.line_types.i_nom) * n.lines.bus0.map(n.buses.v_nom) * n.lines.num_parallel
            )
            for l in lines_i:
                n.model.passive_branch_s_nom["Line", l].fix(s_nom.at[l])
                # n.model.passive_branch_s_nom[l].fixed = fix
            if isinstance(n.opt, pypsa.opf.PersistentSolver):
                n.opt.update_var(n.model.passive_branch_s_nom)

        if links_i is not None and len(links_i) > 0:
            for l in links_i:
                n.model.link_p_nom[l].fix(n.links.at[l, 'p_nom'])
                # n.model.link_p_nom[l].fixed = fix
            if isinstance(n.opt, pypsa.opf.PersistentSolver):
                n.opt.update_var(n.model.link_p_nom)

        # Not sure if this is needed
        # n.model.preprocess()

    solve_opts = snakemake.config['solving']['options']

    solver_options = snakemake.config['solving']['solver'].copy()
    solver_options['logfile'] = snakemake.log.gurobi
    solver_name = solver_options.pop('name')

    def run_lopf(n, allow_warning_status=False, fix_zero_lines=False):
        if not hasattr(n, 'opt') or not isinstance(n.opt, pypsa.opf.PersistentSolver):
            pypsa.opf.network_lopf_build_model(n, formulation=solve_opts['formulation'])
            add_opts_constraints(n)

            pypsa.opf.network_lopf_prepare_solver(n, solver_name=solver_name)

        if fix_zero_lines:
            fix_lines_b = (n.lines.s_nom_opt == 0.) & n.lines.s_nom_extendable
            n.lines.loc[fix_lines_b & (n.lines.type == ''), 's_nom'] = 0.
            n.lines.loc[fix_lines_b & (n.lines.type != ''), 'num_parallel'] = 0.

            fix_links_b = (n.links.p_nom_opt == 0.) & n.links.p_nom_extendable
            n.links.loc[fix_links_b, 'p_nom'] = 0.

            # WARNING: We are not unfixing these later
            fix_lines(n, lines_i=n.lines.index[fix_lines_b], links_i=n.links.index[fix_links_b])

        status, termination_condition = \
        pypsa.opf.network_lopf_solve(n,
                                     solver_options=solver_options,
                                     formulation=solve_opts['formulation'])

        assert status == "ok" or allow_warning_status and status == 'warning', \
            ("network_lopf did abort with status={} "
             "and termination_condition={}"
             .format(status, termination_condition))

        return status, termination_condition

    lines_ext_b = n.lines.s_nom_extendable
    if lines_ext_b.any():
        # puh: ok, we need to iterate, since there is a relation
        # between s/p_nom and r, x for branches.
        msq_threshold = 0.01
        lines = pd.DataFrame(n.lines[['r', 'x', 'type', 'num_parallel']])

        lines['s_nom'] = (
            np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) * n.lines.bus0.map(n.buses.v_nom) * n.lines.num_parallel
        ).where(n.lines.type != '', n.lines['s_nom'])

        lines_ext_typed_b = (n.lines.type != '') & lines_ext_b
        lines_ext_untyped_b = (n.lines.type == '') & lines_ext_b

        def update_line_parameters(n, zero_lines_below=10, fix_zero_lines=False):
            if zero_lines_below > 0:
                n.lines.loc[n.lines.s_nom_opt < zero_lines_below, 's_nom_opt'] = 0.
                n.links.loc[n.links.p_nom_opt < zero_lines_below, 'p_nom_opt'] = 0.

            if lines_ext_untyped_b.any():
                for attr in ('r', 'x'):
                    n.lines.loc[lines_ext_untyped_b, attr] = (
                        lines[attr].multiply(lines['s_nom']/n.lines['s_nom_opt'])
                    )

            if lines_ext_typed_b.any():
                n.lines.loc[lines_ext_typed_b, 'num_parallel'] = (
                    lines['num_parallel'].multiply(n.lines['s_nom_opt']/lines['s_nom'])
                )
                logger.debug("lines.num_parallel={}".format(n.lines.loc[lines_ext_typed_b, 'num_parallel']))

            if isinstance(n.opt, pypsa.opf.PersistentSolver):
                n.calculate_dependent_values()

                assert solve_opts['formulation'] == 'kirchhoff', \
                    "Updating persistent solvers has only been implemented for the kirchhoff formulation for now"

                n.opt.remove_constraint(n.model.cycle_constraints)
                del n.model.cycle_constraints_index
                del n.model.cycle_constraints_index_0
                del n.model.cycle_constraints_index_1
                del n.model.cycle_constraints

                pypsa.opf.define_passive_branch_flows_with_kirchhoff(n, n.snapshots, skip_vars=True)
                n.opt.add_constraint(n.model.cycle_constraints)

        iteration = 1

        lines['s_nom_opt'] = lines['s_nom']
        status, termination_condition = run_lopf(n, allow_warning_status=True)

        def msq_diff(n):
            lines_err = np.sqrt(((n.lines['s_nom_opt'] - lines['s_nom_opt'])**2).mean())/lines['s_nom_opt'].mean()
            logger.info("Mean square difference after iteration {} is {}".format(iteration, lines_err))
            return lines_err

        min_iterations = solve_opts.get('min_iterations', 2)
        max_iterations = solve_opts.get('max_iterations', 999)
        while msq_diff(n) > msq_threshold or iteration < min_iterations:
            if iteration >= max_iterations:
                logger.info("Iteration {} beyond max_iterations {}. Stopping ...".format(iteration, max_iterations))
                break

            update_line_parameters(n)
            lines['s_nom_opt'] = n.lines['s_nom_opt']
            iteration += 1

            # Not really needed, could also be taken out
            n.export_to_hdf5(snakemake.output[0])

            status, termination_condition = run_lopf(n, allow_warning_status=True)

        update_line_parameters(n, zero_lines_below=500)

    status, termination_condition = run_lopf(n, fix_zero_lines=True)

    # Drop zero lines from network
    zero_lines_i = n.lines.index[(n.lines.s_nom_opt == 0.) & n.lines.s_nom_extendable]
    if len(zero_lines_i):
        n.lines.drop(zero_lines_i, inplace=True)
    zero_links_i = n.links.index[(n.links.p_nom_opt == 0.) & n.links.p_nom_extendable]
    if len(zero_links_i):
        n.links.drop(zero_links_i, inplace=True)

    return n

if __name__ == "__main__":
    n = pypsa.Network()
    n.import_from_hdf5(snakemake.input[0])

    n = prepare_network(n)
    n = solve_network(n)

    n.export_to_hdf5(snakemake.output[0])
