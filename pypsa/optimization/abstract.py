#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build abstracted, extended optimisation problems from PyPSA networks with
Linopy.
"""
import logging
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.descriptors import nominal_attrs

logger = logging.getLogger(__name__)


def optimize_transmission_expansion_iteratively(
    n,
    snapshots=None,
    msq_threshold=0.05,
    min_iterations=1,
    max_iterations=100,
    track_iterations=False,
    **kwargs,
):
    """
    Iterative linear optimization updating the line parameters for passive AC
    and DC lines. This is helpful when line expansion is enabled. After each
    successful solving, line impedances and line resistance are recalculated
    based on the optimization result. If warmstart is possible, it uses the
    result from the previous iteration to fasten the optimization.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    msq_threshold: float, default 0.05
        Maximal mean square difference between optimized line capacity of
        the current and the previous iteration. As soon as this threshold is
        undercut, and the number of iterations is bigger than 'min_iterations'
        the iterative optimization stops
    min_iterations : integer, default 1
        Minimal number of iteration to run regardless whether the msq_threshold
        is already undercut
    max_iterations : integer, default 100
        Maximal number of iterations to run regardless whether msq_threshold
        is already undercut
    track_iterations: bool, default False
        If True, the intermediate branch capacities and values of the
        objective function are recorded for each iteration. The values of
        iteration 0 represent the initial state.
    **kwargs
        Keyword arguments of the `n.optimize` function which runs at each iteration
    """

    n.lines["carrier"] = n.lines.bus0.map(n.buses.carrier)
    ext_i = n.get_extendable_i("Line")
    typed_i = n.lines.query('type != ""').index
    ext_untyped_i = ext_i.difference(typed_i)
    ext_typed_i = ext_i.intersection(typed_i)
    base_s_nom = (
        np.sqrt(3)
        * n.lines["type"].map(n.line_types.i_nom)
        * n.lines.bus0.map(n.buses.v_nom)
    )
    n.lines.loc[ext_typed_i, "num_parallel"] = (n.lines.s_nom / base_s_nom)[ext_typed_i]

    def update_line_params(n, s_nom_prev):
        factor = n.lines.s_nom_opt / s_nom_prev
        for attr, carrier in (("x", "AC"), ("r", "DC")):
            ln_i = n.lines.query("carrier == @carrier").index.intersection(
                ext_untyped_i
            )
            n.lines.loc[ln_i, attr] /= factor[ln_i]
        ln_i = ext_i.intersection(typed_i)
        n.lines.loc[ln_i, "num_parallel"] = (n.lines.s_nom_opt / base_s_nom)[ln_i]

    def msq_diff(n, s_nom_prev):
        lines_err = (
            np.sqrt((s_nom_prev - n.lines.s_nom_opt).pow(2).mean())
            / n.lines["s_nom_opt"].mean()
        )
        logger.info(
            f"Mean square difference after iteration {iteration} is " f"{lines_err}"
        )
        return lines_err

    def save_optimal_capacities(n, iteration, status):
        for c, attr in pd.Series(nominal_attrs)[n.branch_components].items():
            n.df(c)[f"{attr}_opt_{iteration}"] = n.df(c)[f"{attr}_opt"]
        setattr(n, f"status_{iteration}", status)
        setattr(n, f"objective_{iteration}", n.objective)
        n.iteration = iteration
        n.global_constraints = n.global_constraints.rename(
            columns={"mu": f"mu_{iteration}"}
        )

    if track_iterations:
        for c, attr in pd.Series(nominal_attrs)[n.branch_components].items():
            n.df(c)[f"{attr}_opt_0"] = n.df(c)[f"{attr}"]

    iteration = 1
    kwargs["store_basis"] = True
    diff = msq_threshold
    while diff >= msq_threshold or iteration < min_iterations:
        if iteration > max_iterations:
            logger.info(
                f"Iteration {iteration} beyond max_iterations "
                f"{max_iterations}. Stopping ..."
            )
            break

        s_nom_prev = n.lines.s_nom_opt.copy() if iteration else n.lines.s_nom.copy()
        status, termination_condition = n.optimize(snapshots, **kwargs)
        assert status == "ok", (
            f"Optimization failed with status {status}"
            f"and termination {termination_condition}"
        )
        if track_iterations:
            save_optimal_capacities(n, iteration, status)
        update_line_params(n, s_nom_prev)
        diff = msq_diff(n, s_nom_prev)
        iteration += 1

    logger.info("Running last lopf with fixed branches (HVDC links and HVAC lines)")

    ext_dc_links_b = n.links.p_nom_extendable & (n.links.carrier == "DC")
    s_nom_orig = n.lines.s_nom.copy()
    p_nom_orig = n.links.p_nom.copy()

    n.lines.loc[ext_i, "s_nom"] = n.lines.loc[ext_i, "s_nom_opt"]
    n.lines.loc[ext_i, "s_nom_extendable"] = False

    n.links.loc[ext_dc_links_b, "p_nom"] = n.links.loc[ext_dc_links_b, "p_nom_opt"]
    n.links.loc[ext_dc_links_b, "p_nom_extendable"] = False

    n.optimize(snapshots, **kwargs)

    n.lines.loc[ext_i, "s_nom"] = s_nom_orig.loc[ext_i]
    n.lines.loc[ext_i, "s_nom_extendable"] = True

    n.links.loc[ext_dc_links_b, "p_nom"] = p_nom_orig.loc[ext_dc_links_b]
    n.links.loc[ext_dc_links_b, "p_nom_extendable"] = True

    ## add costs of additional infrastructure to objective value of last iteration
    obj_links = (
        n.links[ext_dc_links_b].eval("capital_cost * (p_nom_opt - p_nom_min)").sum()
    )
    obj_lines = n.lines.eval("capital_cost * (s_nom_opt - s_nom_min)").sum()
    n.objective += obj_links + obj_lines
    n.objective_constant -= obj_links + obj_lines


def optimize_security_constrained(
    n,
    snapshots=None,
    branch_outages=None,
    multi_investment_periods=False,
    model_kwargs={},
    **kwargs,
):
    """
    Computes Security-Constrained Linear Optimal Power Flow (SCLOPF).

    This ensures that no branch is overloaded even given the branch outages.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list-like, optional
        Set of snapshots to consider in the optimization. The default is None.
    branch_outages : list-like/pandas.Index/pandas.MultiIndex, optional
        Subset of passive branches to consider as possible outages. If a list
        or a pandas.Index is passed, it is assumed to identify lines. If a
        multiindex is passed, its first level has to contain the component names,
        the second the assets. The default None results in all passive branches
        to be considered.
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimise in multiple
        investment periods. Then, snapshots should be a ``pd.MultiIndex``.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,
        `problem_fn` or solver options directly passed to the solver.

    Returns
    -------
    None
    """
    all_passive_branches = n.passive_branches().index

    if branch_outages is None:
        branch_outages = all_passive_branches
    elif isinstance(branch_outages, (list, pd.Index)):
        branch_outages = pd.MultiIndex.from_product([("Line",), branch_outages])

        diff = set(branch_outages) - set(all_passive_branches)
        if diff:
            raise ValueError(
                f"The following passive branches are not in the network: {diff}"
            )

    if not len(all_passive_branches):
        return n.optimize(
            snapshots,
            multi_investment_periods=multi_investment_periods,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        **model_kwargs,
    )

    for sn in n.sub_networks.obj:

        branches_i = sn.branches_i()
        outages = branches_i.intersection(branch_outages)

        if outages.empty:
            continue

        sn.calculate_BODF()
        BODF = pd.DataFrame(sn.BODF, index=branches_i, columns=branches_i)
        BODF = (BODF - np.diagflat(np.diag(BODF)))[outages]

        for c_outage, c_affected in product(outages.unique(0), branches_i.unique(0)):

            c_outages = outages.get_loc_level(c_outage)[1]
            flow = m.variables[c_outage + "-s"].loc[:, c_outages]

            bodf = BODF.loc[c_affected, c_outage]
            bodf = xr.DataArray(bodf, dims=[c_affected + "-affected", c_outage])
            additional_flow = (bodf * flow).rename({c_outage: c_outage + "-outage"})

            for bound, kind in product(("lower", "upper"), ("fix", "ext")):

                constraint = c_affected + "-" + kind + "-s-" + bound
                if constraint not in m.constraints:
                    continue
                lhs = m.constraints[constraint].lhs
                sign = m.constraints[constraint].sign
                rhs = m.constraints[constraint].rhs
                rename = {c_affected + "-affected": c_affected + "-" + kind}
                lhs = lhs + additional_flow.rename(rename)
                m.add_constraints(lhs, sign, rhs, name=constraint + "-security")

    return n.optimize.solve_model(**kwargs)
