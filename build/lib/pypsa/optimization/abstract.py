# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Build abstracted, extended optimisation problems from PyPSA networks with Linopy."""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa._options import options
from pypsa.descriptors import nominal_attrs
from pypsa.optimization.mga import OptimizationAbstractMGAMixin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network
logger = logging.getLogger(__name__)


def discretized_capacity(
    nom_opt: float,
    nom_max: float,
    unit_size: float,
    threshold: float,
    fractional_last_unit_size: bool,
) -> float:
    """Discretize a optimal capacity to a capacity that is either a multiple of a unit size or the maximum capacity.

    Depending on the variable `fractional_last_unit_size`. This function checks if the optimal capacity is within the threshold of the unit
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

    Returns
    -------
    float
        The discretized capacity.

    Examples
    --------
    >>> discretized_capacity(
    ... nom_opt = 7,
    ... nom_max = 25,
    ... unit_size = 5,
    ... threshold = 0.1,
    ... fractional_last_unit_size = False)
    10
    >>> discretized_capacity(
    ... nom_opt = 7,
    ... nom_max = 8,
    ... unit_size = 5,
    ... threshold = 0.1,
    ... fractional_last_unit_size = False)
    5
    >>> discretized_capacity(
    ... nom_opt = 7,
    ... nom_max = 8,
    ... unit_size = 5,
    ... threshold = 0.1,
    ... fractional_last_unit_size = True)
    8
    >>> discretized_capacity(
    ... nom_opt = 3,
    ... nom_max = 4,
    ... unit_size = 5,
    ... threshold = 0.1,
    ... fractional_last_unit_size = False)
    4

    """
    units = nom_opt // unit_size + (nom_opt % unit_size >= threshold * unit_size)

    block_capacity = units * unit_size
    if nom_max % unit_size == 0:
        return block_capacity

    if (nom_max - nom_opt) < unit_size:
        if (
            fractional_last_unit_size
            and ((nom_opt % unit_size) / (nom_max % unit_size)) >= threshold
        ):
            return nom_max
        if nom_max < unit_size:
            return nom_max
        return (nom_opt // unit_size) * unit_size
    return block_capacity


class OptimizationAbstractMixin(OptimizationAbstractMGAMixin):
    """Mixin class for additional optimization methods.

    Class inherits to [pypsa.optimization.OptimizationAccessor][]. All attributes and
    methods can be used within any Network instance via `n.optimize`.

    """

    _n: Network

    def optimize_transmission_expansion_iteratively(
        self,
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
        """Run iterative linear optimization.

        Updating the line parameters for passive AC
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
        n = self._n

        n.c.lines.static["carrier"] = n.c.lines.static.bus0.map(
            n.c.buses.static.carrier
        )
        ext_i = n.c.lines.extendables.difference(n.c.lines.inactive_assets)
        typed_i = n.c.lines.static.query('type != ""').index
        ext_untyped_i = ext_i.difference(typed_i)
        ext_typed_i = ext_i.intersection(typed_i)
        base_s_nom = (
            np.sqrt(3)
            * n.c.lines.static["type"].map(n.c.line_types.static.i_nom)
            * n.c.lines.static.bus0.map(n.c.buses.static.v_nom)
        )
        n.c.lines.static.loc[ext_typed_i, "num_parallel"] = (
            n.c.lines.static.s_nom / base_s_nom
        )[ext_typed_i]

        def update_line_params(n: Network, s_nom_prev: float | pd.Series) -> None:
            factor = n.c.lines.static.s_nom_opt / s_nom_prev
            for attr, carrier in (("x", "AC"), ("r", "DC")):  # noqa: B007
                ln_i = n.c.lines.static.query("carrier == @carrier").index.intersection(
                    ext_untyped_i
                )
                n.c.lines.static.loc[ln_i, attr] /= factor[ln_i]
            ln_i = ext_i.intersection(typed_i)
            n.c.lines.static.loc[ln_i, "num_parallel"] = (
                n.c.lines.static.s_nom_opt / base_s_nom
            )[ln_i]

        def msq_diff(n: Network, s_nom_prev: float | pd.Series) -> float:
            lines_err = (
                np.sqrt((s_nom_prev - n.c.lines.static.s_nom_opt).pow(2).mean())
                / n.c.lines.static["s_nom_opt"].mean()
            )
            logger.info(
                "Mean square difference after iteration %s is %s",
                iteration,
                lines_err,
            )
            return lines_err

        def save_optimal_capacities(n: Network, iteration: int, status: str) -> None:
            for c, attr in pd.Series(nominal_attrs)[list(n.branch_components)].items():
                n.c[c].static[f"{attr}_opt_{iteration}"] = n.c[c].static[f"{attr}_opt"]
            setattr(n, f"status_{iteration}", status)
            setattr(n, f"objective_{iteration}", n.objective)
            n.iteration = iteration
            n.c.global_constraints.static = n.c.global_constraints.static.rename(
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
            """Discretizes the branch components of a network based on the specified unit sizes and thresholds."""
            # TODO: move default value definition to main function (unnest)
            line_threshold = line_threshold or 0.3
            link_threshold = link_threshold or {}

            if line_unit_size:
                n.c.lines.static["s_nom"] = n.c.lines.static.apply(
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
                for carrier in (
                    link_unit_size.keys() & n.c.links.static.carrier.unique()
                ):
                    idx = n.c.links.static.carrier == carrier
                    n.c.links.static.loc[idx, "p_nom"] = n.c.links.static.loc[
                        idx
                    ].apply(
                        lambda row: discretized_capacity(
                            nom_opt=row["p_nom_opt"],
                            nom_max=row["p_nom_max"],
                            unit_size=link_unit_size[carrier],  # noqa: B023
                            threshold=link_threshold.get(carrier, 0.3),  # noqa: B023
                            fractional_last_unit_size=fractional_last_unit_size,
                        ),
                        axis=1,
                    )

        if link_threshold is None:
            link_threshold = {}

        if track_iterations:
            for c, attr in pd.Series(nominal_attrs)[list(n.branch_components)].items():
                n.c[c].static[f"{attr}_opt_0"] = n.c[c].static[f"{attr}"]

        iteration = 1
        diff = msq_threshold
        while diff >= msq_threshold or iteration < min_iterations:
            if iteration > max_iterations:
                logger.info(
                    "Iteration %s beyond max_iterations %s. Stopping ...",
                    iteration,
                    max_iterations,
                )
                break

            s_nom_prev = (
                n.c.lines.static.s_nom_opt.copy()
                if iteration
                else n.c.lines.static.s_nom.copy()
            )
            status, termination_condition = n.optimize(snapshots, **kwargs)
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
        ext_links_to_fix_b = (
            n.c.links.static.p_nom_extendable
            & n.c.links.static.carrier.isin(link_carriers)
        )
        s_nom_orig = n.c.lines.static.s_nom.copy()
        p_nom_orig = n.c.links.static.p_nom.copy()

        n.c.lines.static.loc[ext_i, "s_nom"] = n.c.lines.static.loc[ext_i, "s_nom_opt"]
        n.c.lines.static.loc[ext_i, "s_nom_extendable"] = False

        n.c.links.static.loc[ext_links_to_fix_b, "p_nom"] = n.c.links.static.loc[
            ext_links_to_fix_b, "p_nom_opt"
        ]
        n.c.links.static.loc[ext_links_to_fix_b, "p_nom_extendable"] = False

        discretize_branch_components(
            n,
            line_unit_size,
            link_unit_size,
            line_threshold,
            link_threshold,
            fractional_last_unit_size,
        )

        n.calculate_dependent_values()
        status, condition = n.optimize(snapshots, **kwargs)

        n.c.lines.static.loc[ext_i, "s_nom"] = s_nom_orig.loc[ext_i]
        n.c.lines.static.loc[ext_i, "s_nom_extendable"] = True

        n.c.links.static.loc[ext_links_to_fix_b, "p_nom"] = p_nom_orig.loc[
            ext_links_to_fix_b
        ]
        n.c.links.static.loc[ext_links_to_fix_b, "p_nom_extendable"] = True

        ## add costs of additional infrastructure to objective value of last iteration
        obj_links = (
            n.c.links.static[ext_links_to_fix_b]
            .eval("capital_cost * (p_nom_opt - p_nom_min)")
            .sum()
        )
        obj_lines = n.c.lines.static.eval(
            "capital_cost * (s_nom_opt - s_nom_min)"
        ).sum()
        n._objective += obj_links + obj_lines
        n._objective_constant -= obj_links + obj_lines

        return status, condition

    def optimize_security_constrained(
        self,
        snapshots: Sequence | None = None,
        branch_outages: Sequence | pd.Index | pd.MultiIndex | None = None,
        multi_investment_periods: bool = False,
        model_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Compute Security-Constrained Linear Optimal Power Flow (SCLOPF).

        This ensures that no branch is overloaded even given the branch outages.

        Parameters
        ----------
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
            investment periods. Then, snapshots should be a `pd.MultiIndex`.
        model_kwargs : dict, optional
            Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
            Defaults to module wide option (default: {}). See
            `https://go.pypsa.org/options-params` for more information.
        **kwargs:
            Keyword argument used by `linopy.Model.solve`, such as `solver_name`,
            `problem_fn` or solver options directly passed to the solver.

        """
        # Handle default parameters from options
        if model_kwargs is None:
            model_kwargs = options.params.optimize.model_kwargs.copy()

        n = self._n

        all_passive_branches = n.passive_branches().index

        if branch_outages is None:
            branch_outages = all_passive_branches
        elif isinstance(branch_outages, (list | pd.Index)):
            branch_outages = pd.MultiIndex.from_product([("Line",), branch_outages])

            if diff := set(branch_outages) - set(all_passive_branches):
                msg = f"The following passive branches are not in the network: {diff}"
                raise ValueError(msg)

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

        for sub_network in n.c.sub_networks.static.obj:
            branches_i = sub_network.branches_i()
            outages = branches_i.intersection(branch_outages)

            if outages.empty:
                continue

            sub_network.calculate_BODF()
            BODF = pd.DataFrame(sub_network.BODF, index=branches_i, columns=branches_i)[
                outages
            ]

            for c_outage, c_affected in product(
                outages.unique(0), branches_i.unique(0)
            ):
                c_outage_ = c_outage + "-outage"
                c_outages = outages.get_loc_level(c_outage)[1]
                flow_outage = m.variables[c_outage + "-s"].loc[:, c_outages]
                flow_outage = flow_outage.rename({"name": c_outage_})

                bodf = BODF.loc[c_affected, c_outage]
                bodf = xr.DataArray(bodf, dims=[c_affected, c_outage_])
                added_flow = flow_outage * bodf

                for bound, kind in product(("lower", "upper"), ("fix", "ext")):
                    constraint = c_affected + "-" + kind + "-s-" + bound
                    if constraint not in m.constraints:
                        continue

                    con = m.constraints[constraint]

                    idx = con.lhs.indexes["name"].intersection(
                        added_flow.indexes[c_affected]
                    )

                    added_flow_aligned = added_flow.sel({c_affected: idx}).rename(
                        {c_affected: "name"}
                    )
                    lhs = con.lhs.sel(name=idx) + added_flow_aligned

                    name = (
                        constraint
                        + f"-security-for-{c_outage_}-in-sub-network-{sub_network.name}"
                    )
                    m.add_constraints(
                        lhs, con.sign.sel(name=idx), con.rhs.sel(name=idx), name=name
                    )

        return n.optimize.solve_model(**kwargs)

    def optimize_with_rolling_horizon(
        self,
        snapshots: Sequence | None = None,
        horizon: int = 100,
        overlap: int = 0,
        **kwargs: Any,
    ) -> Network:
        """Optimizes the network in a rolling horizon fashion.

        Parameters
        ----------
        snapshots : list-like
            Set of snapshots to consider in the optimization. The default is None.
        horizon : int
            Number of snapshots to consider in each iteration. Defaults to 100.
        overlap : int
            Number of snapshots to overlap between two iterations. Defaults to 0.
        **kwargs:
            Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

        """
        n = self._n
        if snapshots is None:
            snapshots = n.snapshots

        if horizon <= overlap:
            msg = "overlap must be smaller than horizon"
            raise ValueError(msg)

        starting_points = range(0, len(snapshots), horizon - overlap)
        for i, start in enumerate(starting_points):
            end = min(len(snapshots), start + horizon)
            sns = snapshots[start:end]
            logger.info(
                "Optimizing network for snapshot horizon [%s:%s] (%s/%s).",
                sns[0],
                sns[-1],
                i + 1,
                len(starting_points),
            )

            if i:
                if not n.c.stores.static.empty:
                    n.c.stores.static.e_initial = n.c.stores.dynamic.e.loc[
                        snapshots[start - 1]
                    ]
                if not n.c.storage_units.static.empty:
                    n.c.storage_units.static.state_of_charge_initial = (
                        n.c.storage_units.dynamic.state_of_charge.loc[
                            snapshots[start - 1]
                        ]
                    )

            status, condition = n.optimize(sns, **kwargs)
            if status != "ok":
                logger.warning(
                    "Optimization failed with status %s and condition %s",
                    status,
                    condition,
                )
        return n

    def optimize_and_run_non_linear_powerflow(
        self,
        snapshots: Sequence | None = None,
        skip_pre: bool = False,
        x_tol: float = 1e-06,
        use_seed: bool = False,
        distribute_slack: bool = False,
        slack_weights: str = "p_set",
        **kwargs: Any,
    ) -> dict:
        """Optimizes the network and then performs a non-linear power flow for all snapshots.

        Parameters
        ----------
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
        n = self._n
        if snapshots is None:
            snapshots = n.snapshots

        n = self._n

        # Step 1: Optimize the network
        status, condition = n.optimize(snapshots, **kwargs)

        if status != "ok":
            logger.warning(
                "Optimization failed with status %s and condition %s",
                status,
                condition,
            )
            return {"status": status, "terminantion_condition": condition}

        for c in n.one_port_components:
            n.c[c].dynamic["p_set"] = n.c[c].dynamic["p"]
        for c in ("Link",):
            n.c[c].dynamic["p_set"] = n.c[c].dynamic["p0"]

        n.c.generators.static.control = "PV"
        for sub_network in n.c.sub_networks.static.obj:
            n.c.generators.static.loc[sub_network.slack_generator, "control"] = "Slack"
        # Need some PQ buses so that Jacobian doesn't break
        for sub_network in n.c.sub_networks.static.obj:
            generators = sub_network.c.generators.static.index
            other_generators = generators.difference([sub_network.slack_generator])
            if not other_generators.empty:
                n.c.generators.static.loc[other_generators[0], "control"] = "PQ"

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
