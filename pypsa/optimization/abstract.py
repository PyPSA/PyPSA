#!/usr/bin/env python3
"""
Build abstracted, extended optimisation problems from PyPSA networks with
Linopy.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from linopy import LinearExpression, QuadraticExpression, merge

from pypsa.descriptors import nominal_attrs

if TYPE_CHECKING:
    from pypsa import Network
logger = logging.getLogger(__name__)


def discretized_capacity(
    nom_opt: float,
    nom_max: float,
    unit_size: float,
    threshold: float,
    fractional_last_unit_size: bool,
    min_units: int | None = None,
) -> float:
    """
    Discretize a optimal capacity to a capacity that is either a multiple of a unit size
    or the maximum capacity, depending on the variable `fractional_last_unit_size`.

    This function checks if the optimal capacity is within the threshold of the unit
    size. If so, it returns the next multiple of the unit size - if not it returns the
    last multiple of the unit size.
    In the special case that the maximum capacity is not a multiple of the unit size,
    the variable `fractional_last_unit_size` determines if the returned capacity is the
    maximum capacity (True) or the last multiple of the unit size (False).
    In case the maximum capacity is lower than the unit size, the function returns the
    maximum capacity.

    Parameters
    ----------
    nom_opt : float
        The optimal capacity as returned by the optimization.
    nom_max : float
        The maximum capacity as defined in the network.
    unit_size : float
        The unit size for the capacity.
    threshold : float
        The threshold relative to the unit size for discretizing the capacity.
    fractional_last_unit_size : bool
        Whether only multiples of the unit size or the maximum capacity.
    min_units: int, default None
        The minimum number of units that should be installed.

        .. deprecated:: 0.31
            The `min_units` parameter is deprecated and will be removed in future versions.

    Returns
    -------
    float
        The discretized capacity.

    Examples
    --------
    >>> discretized_capacity(
    nom_opt = 7,
    nom_max = 25,
    unit_size = 5,
    threshold = 0.1,
    fractional_last_unit_size = False)
    10
    >>> discretized_capacity(
    nom_opt = 7,
    nom_max = 8,
    unit_size = 5,
    threshold = 0.1,
    fractional_last_unit_size = False)
    5
    >>> discretized_capacity(
    nom_opt = 7,
    nom_max = 8,
    unit_size = 5,
    threshold = 0.1,
    fractional_last_unit_size = True)
    8
    >>> discretized_capacity(
    nom_opt = 3,
    nom_max = 4,
    unit_size = 5,
    threshold = 0.1,
    fractional_last_unit_size = False)
    4
    """
    if min_units is not None:
        raise DeprecationWarning(
            "The `min_units` parameter is deprecated and will be removed in future "
            "versions."
        )
    units = nom_opt // unit_size + (nom_opt % unit_size >= threshold * unit_size)

    if min_units is not None:
        block_capacity = max(min_units, units) * unit_size
    else:
        block_capacity = units * unit_size
    if nom_max % unit_size == 0:
        return block_capacity

    else:
        if (nom_max - nom_opt) < unit_size:
            if (
                fractional_last_unit_size
                and ((nom_opt % unit_size) / (nom_max % unit_size)) >= threshold
            ):
                return nom_max
            elif nom_max < unit_size:
                return nom_max
            else:
                return (nom_opt // unit_size) * unit_size
        else:
            return block_capacity


def optimize_transmission_expansion_iteratively(
    n: Network,
    snapshots: Sequence | None = None,
    msq_threshold: float = 0.05,
    min_iterations: int = 1,
    max_iterations: int = 100,
    track_iterations: bool = False,
    line_unit_size: float | None = None,
    link_unit_size: dict | None = None,
    line_threshold: float | None = None,
    link_threshold: dict | None = None,
    fractional_last_unit_size: bool = False,
    **kwargs: Any,
) -> tuple[str, str]:
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
        n.snapshots, defaults to n.snapshots
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
    line_unit_size: float, default None
        The unit size for line components.
        Use None if no discretization is desired.
    link_unit_size: dict-like, default None
        A dictionary containing the unit sizes for link components,
        with carrier names as keys. Use None if no discretization is desired.
    line_threshold: float, default 0.3
        The threshold relative to the unit size for discretizing line components.
    link_threshold: dict-like, default 0.3 per carrier
        The threshold relative to the unit size for discretizing link components.
    fractional_last_unit_size: bool, default False
        Whether only multiples of the unit size or in case of a maximum capacity fractions of unit size is allowed.
    **kwargs
        Keyword arguments of the `n.optimize` function which runs at each iteration
    """

    n.lines["carrier"] = n.lines.bus0.map(n.buses.carrier)
    ext_i = n.get_extendable_i("Line").copy()
    typed_i = n.lines.query('type != ""').index
    ext_untyped_i = ext_i.difference(typed_i)
    ext_typed_i = ext_i.intersection(typed_i)
    base_s_nom = (
        np.sqrt(3)
        * n.lines["type"].map(n.line_types.i_nom)
        * n.lines.bus0.map(n.buses.v_nom)
    )
    n.lines.loc[ext_typed_i, "num_parallel"] = (n.lines.s_nom / base_s_nom)[ext_typed_i]

    def update_line_params(n: Network, s_nom_prev: float | pd.Series) -> None:
        factor = n.lines.s_nom_opt / s_nom_prev
        for attr, carrier in (("x", "AC"), ("r", "DC")):
            ln_i = n.lines.query("carrier == @carrier").index.intersection(
                ext_untyped_i
            )
            n.lines.loc[ln_i, attr] /= factor[ln_i]
        ln_i = ext_i.intersection(typed_i)
        n.lines.loc[ln_i, "num_parallel"] = (n.lines.s_nom_opt / base_s_nom)[ln_i]

    def msq_diff(n: Network, s_nom_prev: float | pd.Series) -> float:
        lines_err = (
            np.sqrt((s_nom_prev - n.lines.s_nom_opt).pow(2).mean())
            / n.lines["s_nom_opt"].mean()
        )
        logger.info(
            f"Mean square difference after iteration {iteration} is " f"{lines_err}"
        )
        return lines_err

    def save_optimal_capacities(n: Network, iteration: int, status: str) -> None:
        for c, attr in pd.Series(nominal_attrs)[list(n.branch_components)].items():
            n.static(c)[f"{attr}_opt_{iteration}"] = n.static(c)[f"{attr}_opt"]
        setattr(n, f"status_{iteration}", status)
        setattr(n, f"objective_{iteration}", n.objective)
        n.iteration = iteration
        n.global_constraints = n.global_constraints.rename(
            columns={"mu": f"mu_{iteration}"}
        )

    def discretize_branch_components(
        n: Network,
        line_unit_size: float | None,
        link_unit_size: dict | None,
        line_threshold: float | None,
        link_threshold: dict | None,
        fractional_last_unit_size: bool = False,
    ) -> None:
        """
        Discretizes the branch components of a network based on the specified
        unit sizes and thresholds.
        """
        # TODO: move default value definition to main function (unnest)
        line_threshold = line_threshold or 0.3
        link_threshold = link_threshold or {}

        if line_unit_size:
            n.lines["s_nom"] = n.lines.apply(
                lambda row: discretized_capacity(
                    nom_opt=row["s_nom_opt"],
                    nom_max=row["s_nom_max"],
                    unit_size=line_unit_size,
                    threshold=line_threshold,
                    fractional_last_unit_size=fractional_last_unit_size,
                ),
                axis=1,
            )

        if link_unit_size:
            for carrier in link_unit_size.keys() & n.links.carrier.unique():
                idx = n.links.carrier == carrier
                n.links.loc[idx, "p_nom"] = n.links.loc[idx].apply(
                    lambda row: discretized_capacity(
                        nom_opt=row["p_nom_opt"],
                        nom_max=row["p_nom_max"],
                        unit_size=link_unit_size[carrier],
                        threshold=link_threshold.get(carrier, 0.3),
                        fractional_last_unit_size=fractional_last_unit_size,
                    ),
                    axis=1,
                )

    if link_threshold is None:
        link_threshold = {}

    if track_iterations:
        for c, attr in pd.Series(nominal_attrs)[list(n.branch_components)].items():
            n.static(c)[f"{attr}_opt_0"] = n.static(c)[f"{attr}"]

    iteration = 1
    diff = msq_threshold
    while diff >= msq_threshold or iteration < min_iterations:
        if iteration > max_iterations:
            logger.info(
                f"Iteration {iteration} beyond max_iterations "
                f"{max_iterations}. Stopping ..."
            )
            break

        s_nom_prev = n.lines.s_nom_opt.copy() if iteration else n.lines.s_nom.copy()
        status, termination_condition = n.optimize(snapshots, **kwargs)  # type: ignore
        if status != "ok":
            msg = (
                f"Optimization failed with status {status} and termination "
                f"{termination_condition}"
            )
            raise RuntimeError(msg)
        if track_iterations:
            save_optimal_capacities(n, iteration, status)

        update_line_params(n, s_nom_prev)
        diff = msq_diff(n, s_nom_prev)
        iteration += 1

    logger.info(
        "Deleting model instance `n.model` from previour run to reclaim memory."
    )
    del n.model
    gc.collect()

    logger.info(
        "Preparing final iteration with fixed and potentially discretized branches (HVDC links and HVAC lines)."
    )

    link_carriers = {"DC"} if not link_unit_size else link_unit_size.keys() | {"DC"}
    ext_links_to_fix_b = n.links.p_nom_extendable & n.links.carrier.isin(link_carriers)
    s_nom_orig = n.lines.s_nom.copy()
    p_nom_orig = n.links.p_nom.copy()

    n.lines.loc[ext_i, "s_nom"] = n.lines.loc[ext_i, "s_nom_opt"]
    n.lines.loc[ext_i, "s_nom_extendable"] = False

    n.links.loc[ext_links_to_fix_b, "p_nom"] = n.links.loc[
        ext_links_to_fix_b, "p_nom_opt"
    ]
    n.links.loc[ext_links_to_fix_b, "p_nom_extendable"] = False

    discretize_branch_components(
        n,
        line_unit_size,
        link_unit_size,
        line_threshold,
        link_threshold,
        fractional_last_unit_size,
    )

    n.calculate_dependent_values()
    status, condition = n.optimize(snapshots, **kwargs)  # type: ignore

    n.lines.loc[ext_i, "s_nom"] = s_nom_orig.loc[ext_i]
    n.lines.loc[ext_i, "s_nom_extendable"] = True

    n.links.loc[ext_links_to_fix_b, "p_nom"] = p_nom_orig.loc[ext_links_to_fix_b]
    n.links.loc[ext_links_to_fix_b, "p_nom_extendable"] = True

    ## add costs of additional infrastructure to objective value of last iteration
    obj_links = (
        n.links[ext_links_to_fix_b].eval("capital_cost * (p_nom_opt - p_nom_min)").sum()
    )
    obj_lines = n.lines.eval("capital_cost * (s_nom_opt - s_nom_min)").sum()
    n.objective += obj_links + obj_lines
    n.objective_constant -= obj_links + obj_lines

    return status, condition


def optimize_security_constrained(
    n: Network,
    snapshots: Sequence | None = None,
    branch_outages: Sequence | pd.Index | pd.MultiIndex | None = None,
    multi_investment_periods: bool = False,
    model_kwargs: dict = {},
    **kwargs: Any,
) -> tuple[str, str]:
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

        if diff := set(branch_outages) - set(all_passive_branches):
            raise ValueError(
                f"The following passive branches are not in the network: {diff}"
            )

    if not len(all_passive_branches):
        return n.optimize(
            snapshots,  # type: ignore
            multi_investment_periods=multi_investment_periods,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        **model_kwargs,
    )

    for sub_network in n.sub_networks.obj:
        branches_i = sub_network.branches_i()
        outages = branches_i.intersection(branch_outages)

        if outages.empty:
            continue

        sub_network.calculate_BODF()
        BODF = pd.DataFrame(sub_network.BODF, index=branches_i, columns=branches_i)[
            outages
        ]

        for c_outage, c_affected in product(outages.unique(0), branches_i.unique(0)):
            c_outage_ = c_outage + "-outage"
            c_outages = outages.get_loc_level(c_outage)[1]
            flow_outage = m.variables[c_outage + "-s"].loc[:, c_outages]
            flow_outage = flow_outage.rename({c_outage: c_outage_})

            bodf = BODF.loc[c_affected, c_outage]
            bodf = xr.DataArray(bodf, dims=[c_affected, c_outage_])
            additional_flow = flow_outage * bodf
            for bound, kind in product(("lower", "upper"), ("fix", "ext")):
                coord = c_affected + "-" + kind
                constraint = coord + "-s-" + bound
                if constraint not in m.constraints:
                    continue
                rename = {c_affected: coord}
                added_flow = additional_flow.rename(rename)
                con = m.constraints[constraint]  # use this as a template
                # idx now contains fixed/extendable for the sub-network
                idx = con.lhs.indexes[coord].intersection(added_flow.indexes[coord])
                sel = {coord: idx}
                lhs = con.lhs.sel(sel) + added_flow.sel(sel)
                name = constraint + f"-security-for-{c_outage_}-in-{sub_network}"
                m.add_constraints(lhs, con.sign.sel(sel), con.rhs.sel(sel), name=name)

    return n.optimize.solve_model(**kwargs)


def optimize_with_rolling_horizon(
    n: Network,
    snapshots: Sequence | None = None,
    horizon: int = 100,
    overlap: int = 0,
    **kwargs: Any,
) -> Network:
    """
    Optimizes the network in a rolling horizon fashion.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list-like
        Set of snapshots to consider in the optimization. The default is None.
    horizon : int
        Number of snapshots to consider in each iteration. Defaults to 100.
    overlap : int
        Number of snapshots to overlap between two iterations. Defaults to 0.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

    Returns
    -------
    None
    """
    if snapshots is None:
        snapshots = n.snapshots

    if horizon <= overlap:
        raise ValueError("overlap must be smaller than horizon")

    starting_points = range(0, len(snapshots), horizon - overlap)
    for i, start in enumerate(starting_points):
        end = min(len(snapshots), start + horizon)
        sns = snapshots[start:end]
        logger.info(
            f"Optimizing network for snapshot horizon [{sns[0]}:{sns[-1]}] ({i+1}/{len(starting_points)})."
        )

        if i:
            if not n.stores.empty:
                n.stores.e_initial = n.stores_t.e.loc[snapshots[start - 1]]
            if not n.storage_units.empty:
                n.storage_units.state_of_charge_initial = (
                    n.storage_units_t.state_of_charge.loc[snapshots[start - 1]]
                )

        status, condition = n.optimize(sns, **kwargs)  # type: ignore
        if status != "ok":
            logger.warning(
                f"Optimization failed with status {status} and condition {condition}"
            )
    return n


def optimize_mga(
    n: Network,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    weights: dict | None = None,
    sense: str | int = "min",
    slack: float = 0.05,
    model_kwargs: dict = {},
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Run modelling-to-generate-alternatives (MGA) on network to find near-
    optimal solutions.

    Parameters
    ----------
    n : pypsa.Network snapshots : list-like
        Set of snapshots to consider in the optimization. The default is None.
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimize in
        multiple investment periods. Then, snapshots should be a
        ``pd.MultiIndex``.
    weights : dict-like
        Weights for alternate objective function. The default is None, which
        minimizes generation capacity. The weights dictionary should be keyed
        with the component and variable (see ``pypsa/variables.csv``), followed
        by a float, dict, pd.Series or pd.DataFrame for the coefficients of the
        objective function. Examples:

        >>> {"Generator": {"p_nom": 1}}
        >>> {"Generator": {"p_nom": pd.Series(1, index=n.generators.index)}}
        >>> {"Generator": {"p_nom": {"gas": 1, "coal": 2}}}
        >>> {"Generator": {"p": pd.Series(1, index=n.generators.index)}
        >>> {"Generator": {"p": pd.DataFrame(1, columns=n.generators.index, index=n.snapshots)}

        Weights for non-extendable components are ignored. The dictionary does
        not need to provide weights for all extendable components.
    sense : str|int
        Optimization sense of alternate objective function. Defaults to 'min'.
        Can also be 'max'.
    slack : float
        Cost slack for budget constraint. Defaults to 0.05.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or
        `chunk`.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

    Returns
    -------
    status : str
        The status of the optimization, either "ok" or one of the codes listed
        in https://linopy.readthedocs.io/en/latest/generated/linopy.constants.SolverStatus.html
    condition : str
        The termination condition of the optimization, either
        "optimal" or one of the codes listed in
        https://linopy.readthedocs.io/en/latest/generated/linopy.constants.TerminationCondition.html
    """
    if snapshots is None:
        snapshots = n.snapshots

    if weights is None:
        weights = dict(Generator=dict(p_nom=pd.Series(1, index=n.generators.index)))

    # check that network has been solved
    if not hasattr(n, "objective"):
        msg = "Network needs to be solved with `n.optimize()` before running MGA."
        raise ValueError(msg)

    # create basic model
    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        **model_kwargs,
    )

    # build budget constraint
    if not multi_investment_periods:
        optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
        fixed_cost = n.statistics.installed_capex().sum()
    else:
        w = n.investment_period_weightings.objective
        optimal_cost = (
            n.statistics.capex().sum() * w + n.statistics.opex().sum() * w
        ).sum()
        fixed_cost = (n.statistics.installed_capex().sum() * w).sum()

    objective = m.objective
    if not isinstance(objective, (LinearExpression, QuadraticExpression)):
        objective = objective.expression

    m.add_constraints(
        objective + fixed_cost <= (1 + slack) * optimal_cost, name="budget"
    )

    # parse optimization sense
    if (
        isinstance(sense, str)
        and sense.startswith("min")
        or isinstance(sense, int)
        and sense > 0
    ):
        sense = 1
    elif (
        isinstance(sense, str)
        and sense.startswith("max")
        or isinstance(sense, int)
        and sense < 0
    ):
        sense = -1
    else:
        raise ValueError(f"Could not parse optimization sense {sense}")

    # build alternate objective
    objective = []
    for c, attrs in weights.items():
        for attr, coeffs in attrs.items():
            if isinstance(coeffs, dict):
                coeffs = pd.Series(coeffs)
            if attr == nominal_attrs[c] and isinstance(coeffs, pd.Series):
                coeffs = coeffs.reindex(n.get_extendable_i(c))
                coeffs.index.name = ""
            elif isinstance(coeffs, pd.Series):
                coeffs = coeffs.reindex(columns=n.static(c).index)
            elif isinstance(coeffs, pd.DataFrame):
                coeffs = coeffs.reindex(columns=n.static(c).index, index=snapshots)
            objective.append(m[f"{c}-{attr}"] * coeffs * sense)

    m.objective = merge(objective)

    status, condition = n.optimize.solve_model(**kwargs)

    # write MGA coefficients into metadata
    n.meta["slack"] = slack
    n.meta["sense"] = sense

    def convert_to_dict(obj: Any) -> Any:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    n.meta["weights"] = convert_to_dict(weights)

    return status, condition


def optimize_and_run_non_linear_powerflow(
    n: Network,
    snapshots: Sequence | None = None,
    skip_pre: bool = False,
    x_tol: float = 1e-06,
    use_seed: bool = False,
    distribute_slack: bool = False,
    slack_weights: str = "p_set",
    **kwargs: Any,
) -> dict:
    """
    Optimizes the network and then performs a non-linear power flow for all snapshots.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA Network object to optimize and analyze.
    snapshots : Sequence | None, optional
        Set of snapshots to consider in the optimization and power flow.
        If None, uses all snapshots in the network.
    skip_pre : bool, optional
        Skip the preliminary steps of the power flow, by default False.
    x_tol : float, optional
        Power flow convergence tolerance, by default 1e-06.
    use_seed : bool, optional
        Use the last solution as initial guess, by default False.
    distribute_slack : bool, optional
        Distribute slack power across generators, by default False.
    slack_weights : str, optional
        How to distribute slack power, by default 'p_set'.
    **kwargs : Any
        Keyword arguments passed to the optimize function.

    Returns
    -------
    Tuple[str, str, Dict]
        A tuple containing:
        - optimization status
        - optimization condition
        - dictionary of power flow results for all snapshots
    """
    if snapshots is None:
        snapshots = n.snapshots

    # Step 1: Optimize the network
    status, condition = n.optimize(snapshots, **kwargs)  # type: ignore

    if status != "ok":
        logger.warning(
            f"Optimization failed with status {status} and condition {condition}"
        )
        return dict(status=status, terminantion_condition=condition)

    for c in n.one_port_components:
        n.dynamic(c)["p_set"] = n.dynamic(c)["p"]
    for c in {"Link"}:
        n.dynamic(c)["p_set"] = n.dynamic(c)["p0"]

    n.generators.control = "PV"
    for sub_network in n.sub_networks.obj:
        n.generators.loc[sub_network.slack_generator, "control"] = "Slack"
    # Need some PQ buses so that Jacobian doesn't break
    for sub_network in n.sub_networks.obj:
        generators = sub_network.generators_i()
        other_generators = generators.difference([sub_network.slack_generator])
        if not other_generators.empty:
            n.generators.loc[other_generators[0], "control"] = "PQ"

    # Step 2: Perform non-linear power flow for all snapshots
    logger.info("Running non-linear power flow iteratively...")

    # Run non-linear power flow
    res = n.pf(
        snapshots=snapshots,
        skip_pre=skip_pre,
        x_tol=x_tol,
        use_seed=use_seed,
        distribute_slack=distribute_slack,
        slack_weights=slack_weights,
    )

    return dict(status=status, terminantion_condition=condition, **res)
