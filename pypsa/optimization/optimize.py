#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build optimisation problems from PyPSA networks with Linopy.
"""
import logging
import os
from functools import wraps

import numpy as np
import pandas as pd
from linopy import Model, merge
from linopy.expressions import LinearExpression, QuadraticExpression

from pypsa.descriptors import additional_linkports, get_committable_i
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import nominal_attrs
from pypsa.optimization.abstract import (
    optimize_mga,
    optimize_security_constrained,
    optimize_transmission_expansion_iteratively,
    optimize_with_rolling_horizon,
)
from pypsa.optimization.common import get_strongly_meshed_buses, set_from_frame
from pypsa.optimization.constraints import (
    define_fixed_nominal_constraints,
    define_fixed_operation_constraints,
    define_kirchhoff_voltage_constraints,
    define_loss_constraints,
    define_modular_constraints,
    define_nodal_balance_constraints,
    define_nominal_constraints_for_extendables,
    define_operational_constraints_for_committables,
    define_operational_constraints_for_extendables,
    define_operational_constraints_for_non_extendables,
    define_ramp_limit_constraints,
    define_storage_unit_constraints,
    define_store_constraints,
)
from pypsa.optimization.global_constraints import (
    define_growth_limit,
    define_nominal_constraints_per_bus_carrier,
    define_operational_limit,
    define_primary_energy_limit,
    define_tech_capacity_expansion_limit,
    define_transmission_expansion_cost_limit,
    define_transmission_volume_expansion_limit,
)
from pypsa.optimization.variables import (
    define_loss_variables,
    define_modular_variables,
    define_nominal_variables,
    define_operational_variables,
    define_shut_down_variables,
    define_spillage_variables,
    define_start_up_variables,
    define_status_variables,
)
from pypsa.pf import _as_snapshots

logger = logging.getLogger(__name__)


lookup = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "variables.csv"),
    index_col=["component", "variable"],
)


def define_objective(n, sns):
    """
    Defines and writes out the objective function.
    """
    m = n.model
    objective = []
    is_quadratic = False

    if n._multi_invest:
        periods = sns.unique("period")
        period_weighting = n.investment_period_weightings.objective[periods]

    # constant for already done investment
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = n.get_extendable_i(c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: n.get_active_assets(c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        constant += (cost * n.df(c)[attr][ext_i]).sum()

    n.objective_constant = constant
    if constant != 0:
        object_const = m.add_variables(constant, constant, name="objective_constant")
        objective.append(-1 * object_const)

    # marginal cost
    weighting = n.snapshot_weightings.objective
    if n._multi_invest:
        weighting = weighting.mul(period_weighting, level=0).loc[sns]
    else:
        weighting = weighting.loc[sns]

    for c, attr in lookup.query("marginal_cost").index:
        cost = (
            get_as_dense(n, c, "marginal_cost", sns)
            .loc[:, lambda ds: (ds != 0).any()]
            .mul(weighting, axis=0)
        )
        if cost.empty:
            continue
        operation = m[f"{c}-{attr}"].sel({"snapshot": sns, c: cost.columns})
        objective.append((operation * cost).sum())

    # marginal cost quadratic
    weighting = n.snapshot_weightings.objective
    if n._multi_invest:
        weighting = weighting.mul(period_weighting, level=0).loc[sns]
    else:
        weighting = weighting.loc[sns]

    for c, attr in lookup.query("marginal_cost").index:
        if "marginal_cost_quadratic" in n.df(c):
            cost = (
                get_as_dense(n, c, "marginal_cost_quadratic", sns)
                .loc[:, lambda ds: (ds != 0).any()]
                .mul(weighting, axis=0)
            )
            if cost.empty:
                continue
            operation = m[f"{c}-{attr}"].sel({"snapshot": sns, c: cost.columns})
            objective.append((operation * operation * cost).sum())
            is_quadratic = True

    # stand-by cost
    comps = {"Generator", "Link"}
    for c in comps:
        com_i = get_committable_i(n, c)

        if com_i.empty:
            continue

        stand_by_cost = (
            get_as_dense(n, c, "stand_by_cost", sns, com_i)
            .loc[:, lambda ds: (ds != 0).any()]
            .mul(weighting, axis=0)
        )
        stand_by_cost.columns.name = f"{c}-com"
        status = n.model.variables[f"{c}-status"].loc[:, stand_by_cost.columns]
        objective.append((status * stand_by_cost).sum())

    # investment
    for c, attr in nominal_attrs.items():
        ext_i = n.get_extendable_i(c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: n.get_active_assets(c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        caps = m[f"{c}-{attr}"]
        objective.append((caps * cost).sum())

    # unit commitment
    keys = ["start_up", "shut_down"]
    for c, attr in lookup.query("variable in @keys").index:
        com_i = n.get_committable_i(c)
        cost = n.df(c)[attr + "_cost"].reindex(com_i)

        if cost.sum():
            var = m[f"{c}-{attr}"]
            objective.append((var * cost).sum())

    if not len(objective):
        raise ValueError(
            "Objective function could not be created. "
            "Please make sure the components have assigned costs."
        )

    m.objective = sum(objective) if is_quadratic else merge(objective)


def create_model(
    n,
    snapshots=None,
    multi_investment_periods=False,
    transmission_losses=0,
    linearized_unit_commitment=False,
    **kwargs,
):
    """
    Create a linopy.Model instance from a pypsa network.

    The model is stored at `n.model`.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimize in multiple
        investment periods. Then, snapshots should be a ``pd.MultiIndex``.
    transmission_losses : int, default 0
    linearized_unit_commitment : bool, default False
        Whether to optimise using the linearised unit commitment formulation or not.
    **kwargs:
        Keyword arguments used by `linopy.Model()`, such as `solver_dir` or `chunk`.

    Returns
    -------
    linopy.model
    """
    sns = _as_snapshots(n, snapshots)
    n._linearized_uc = int(linearized_unit_commitment)
    n._multi_invest = int(multi_investment_periods)
    n.consistency_check()

    kwargs.setdefault("force_dim_names", True)
    n.model = Model(**kwargs)
    n.model.parameters = n.model.parameters.assign(snapshots=sns)

    # Define variables
    for c, attr in lookup.query("nominal").index:
        define_nominal_variables(n, c, attr)
        define_modular_variables(n, c, attr)

    for c, attr in lookup.query("not nominal and not handle_separately").index:
        define_operational_variables(n, sns, c, attr)
        define_status_variables(n, sns, c)
        define_start_up_variables(n, sns, c)
        define_shut_down_variables(n, sns, c)

    define_spillage_variables(n, sns)
    define_operational_variables(n, sns, "Store", "p")

    if transmission_losses:
        for c in n.passive_branch_components:
            define_loss_variables(n, sns, c)

    # Define constraints
    for c, attr in lookup.query("nominal").index:
        define_nominal_constraints_for_extendables(n, c, attr)
        define_fixed_nominal_constraints(n, c, attr)
        define_modular_constraints(n, c, attr)

    for c, attr in lookup.query("not nominal and not handle_separately").index:
        define_operational_constraints_for_non_extendables(
            n, sns, c, attr, transmission_losses
        )
        define_operational_constraints_for_extendables(
            n, sns, c, attr, transmission_losses
        )
        define_operational_constraints_for_committables(n, sns, c)
        define_ramp_limit_constraints(n, sns, c, attr)
        define_fixed_operation_constraints(n, sns, c, attr)

    meshed_buses = get_strongly_meshed_buses(n)
    weakly_meshed_buses = n.buses.index.difference(meshed_buses)
    if not meshed_buses.empty and not weakly_meshed_buses.empty:
        # Write constraint for buses many terms and for buses with a few terms
        # separately. This reduces memory usage for large networks.
        define_nodal_balance_constraints(
            n, sns, transmission_losses=transmission_losses, buses=weakly_meshed_buses
        )
        define_nodal_balance_constraints(
            n,
            sns,
            transmission_losses=transmission_losses,
            buses=meshed_buses,
            suffix="-meshed",
        )
    else:
        define_nodal_balance_constraints(
            n, sns, transmission_losses=transmission_losses
        )

    define_kirchhoff_voltage_constraints(n, sns)
    define_storage_unit_constraints(n, sns)
    define_store_constraints(n, sns)

    if transmission_losses:
        for c in n.passive_branch_components:
            define_loss_constraints(n, sns, c, transmission_losses)

    # Define global constraints
    define_primary_energy_limit(n, sns)
    define_transmission_expansion_cost_limit(n, sns)
    define_transmission_volume_expansion_limit(n, sns)
    define_tech_capacity_expansion_limit(n, sns)
    define_operational_limit(n, sns)
    define_nominal_constraints_per_bus_carrier(n, sns)
    define_growth_limit(n, sns)

    define_objective(n, sns)

    return n.model


def assign_solution(n):
    """
    Map solution to network components.
    """
    m = n.model
    sns = n.model.parameters.snapshots.to_index()

    for name, variable in m.variables.items():
        sol = variable.solution
        if name == "objective_constant":
            continue

        try:
            c, attr = name.split("-", 1)
            df = sol.to_pandas()
        except ValueError:
            continue

        if "snapshot" in sol.dims:
            if c in n.passive_branch_components and attr == "s":
                set_from_frame(n, c, "p0", df)
                set_from_frame(n, c, "p1", -df)

            elif c == "Link" and attr == "p":
                set_from_frame(n, c, "p0", df)

                for i in ["1"] + additional_linkports(n):
                    i_eff = "" if i == "1" else i
                    eff = get_as_dense(n, "Link", f"efficiency{i_eff}", sns)
                    set_from_frame(n, c, f"p{i}", -df * eff)
                    n.pnl(c)[f"p{i}"].loc[
                        sns, n.links.index[n.links[f"bus{i}"] == ""]
                    ] = float(n.component_attrs["Link"].loc[f"p{i}", "default"])

            else:
                set_from_frame(n, c, attr, df)
        elif attr != "n_mod":
            idx = df.index.intersection(n.df(c).index)
            n.df(c).loc[idx, attr + "_opt"] = df.loc[idx]

    # if nominal capacity was no variable set optimal value to nominal
    for c, attr in lookup.query("nominal").index:
        fix_i = n.get_non_extendable_i(c)
        if not fix_i.empty:
            n.df(c).loc[fix_i, f"{attr}_opt"] = n.df(c).loc[fix_i, attr]

    # recalculate storageunit net dispatch
    if not n.df("StorageUnit").empty:
        c = "StorageUnit"
        n.pnl(c)["p"] = n.pnl(c)["p_dispatch"] - n.pnl(c)["p_store"]

    n.objective = m.objective.value


def assign_duals(n, assign_all_duals=False):
    """
    Map dual values i.e. shadow prices to network components.

    Parameters
    ----------
    n : pypsa.Network
    assign_all_duals : bool, default False
        Whether to assign all dual values or only those that already
        have a designated place in the network.
    """
    m = n.model
    unassigned = []
    if all("dual" not in constraint for _, constraint in m.constraints.items()):
        logger.info("No shadow prices were assigned to the network.")
        return

    for name, constraint in m.constraints.items():
        dual = constraint.dual
        try:
            c, attr = name.split("-", 1)
        except ValueError:
            unassigned.append(name)
            continue

        if "snapshot" in dual.dims:
            try:
                df = dual.transpose("snapshot", ...).to_pandas()

                try:
                    spec = attr.rsplit("-", 1)[-1]
                except ValueError:
                    spec = attr

                if attr.endswith("nodal_balance"):
                    set_from_frame(n, c, "marginal_price", df)
                elif assign_all_duals or f"mu_{spec}" in n.df(c):
                    set_from_frame(n, c, "mu_" + spec, df)
                else:
                    unassigned.append(name)

            except:
                unassigned.append(name)

        elif (c == "GlobalConstraint") and (
            (assign_all_duals or attr in n.df(c).index)
        ):
            n.df(c).loc[attr, "mu"] = dual

    if unassigned:
        logger.info(
            f"The shadow-prices of the constraints {', '.join(unassigned)} were "
            "not assigned to the network."
        )


def post_processing(n):
    """
    Post-process the optimized network.

    This calculates quantities derived from the optimized values such as
    power injection per bus and snapshot, voltage angle.
    """
    sns = n.model.parameters.snapshots.to_index()

    # correct prices with objective weightings
    if n._multi_invest:
        period_weighting = n.investment_period_weightings.objective
        weightings = n.snapshot_weightings.objective.mul(
            period_weighting, level=0, axis=0
        ).loc[sns]
    else:
        weightings = n.snapshot_weightings.objective.loc[sns]

    n.buses_t.marginal_price.loc[sns] = n.buses_t.marginal_price.loc[sns].divide(
        weightings, axis=0
    )

    # load
    if len(n.loads):
        set_from_frame(n, "Load", "p", get_as_dense(n, "Load", "p_set", sns))

    # line losses
    if "Line-loss" in n.model.variables:
        losses = n.model["Line-loss"].solution.to_pandas()
        n.lines_t.p0 += losses / 2
        n.lines_t.p1 += losses / 2

    # recalculate injection
    ca = [
        ("Generator", "p", "bus"),
        ("Store", "p", "bus"),
        ("Load", "p", "bus"),
        ("StorageUnit", "p", "bus"),
        ("Link", "p0", "bus0"),
        ("Link", "p1", "bus1"),
    ]
    for i in additional_linkports(n):
        ca.append(("Link", f"p{i}", f"bus{i}"))

    def sign(c):
        return n.df(c).sign if "sign" in n.df(c) else -1  # sign for 'Link'

    n.buses_t.p = (
        pd.concat(
            [
                n.pnl(c)[attr].mul(sign(c)).rename(columns=n.df(c)[group])
                for c, attr, group in ca
            ],
            axis=1,
        )
        .T.groupby(level=0)
        .sum()
        .T.reindex(columns=n.buses.index, fill_value=0.0)
    )

    def v_ang_for_(sub):
        buses_i = sub.buses_o
        if len(buses_i) == 1:
            return pd.DataFrame(0, index=sns, columns=buses_i)
        sub.calculate_B_H(skip_pre=True)
        Z = pd.DataFrame(np.linalg.pinv((sub.B).todense()), buses_i, buses_i)
        Z -= Z[sub.slack_bus]
        return n.buses_t.p.reindex(columns=buses_i) @ Z

    # TODO: if multi investment optimization, the network topology is not the necessarily the same,
    # i.e. one has to iterate over the periods in order to get the correct angles.
    # Determine_network_topology is not necessarily called (only if KVL was assigned)
    if "obj" in n.sub_networks:
        n.buses_t.v_ang = pd.concat(
            [v_ang_for_(sub) for sub in n.sub_networks.obj], axis=1
        ).reindex(columns=n.buses.index, fill_value=0.0)


def optimize(
    n,
    snapshots=None,
    multi_investment_periods=False,
    transmission_losses=0,
    linearized_unit_commitment=False,
    model_kwargs={},
    extra_functionality=None,
    assign_all_duals=False,
    solver_name="glpk",
    solver_options={},
    **kwargs,
):
    """
    Optimize the pypsa network using linopy.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        n.snapshots, defaults to n.snapshots
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimise in multiple
        investment periods. Then, snapshots should be a ``pd.MultiIndex``.
    transmission_losses : int, default 0
        Whether an approximation of transmission losses should be included
        in the linearised power flow formulation. A passed number will denote
        the number of tangents used for the piecewise linear approximation.
        Defaults to 0, which ignores losses.
    linearized_unit_commitment : bool, default False
        Whether to optimise using the linearised unit commitment formulation or not.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
    extra_functionality : callable
        This function must take two arguments
        `extra_functionality(network, snapshots)` and is called after
        the model building is complete, but before it is sent to the
        solver. It allows the user to
        add/change constraints and add/change the objective function.
    assign_all_duals : bool, default False
        Whether to assign all dual values or only those that already
        have a designated place in the network.
    solver_name : str
        Name of the solver to use.
    solver_options : dict
        Keyword arguments used by the solver. Can also be passed via **kwargs.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,
        `problem_fn` or solver options directly passed to the solver.

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

    sns = _as_snapshots(n, snapshots)
    n._multi_invest = int(multi_investment_periods)
    n._linearized_uc = linearized_unit_commitment

    n.consistency_check()
    m = create_model(
        n,
        sns,
        multi_investment_periods,
        transmission_losses,
        linearized_unit_commitment,
        **model_kwargs,
    )
    if extra_functionality:
        extra_functionality(n, sns)
    status, condition = m.solve(solver_name=solver_name, **solver_options, **kwargs)

    if status == "ok":
        assign_solution(n)
        assign_duals(n, assign_all_duals)
        post_processing(n)

    return status, condition


class OptimizationAccessor:
    """
    Optimization accessor for building and solving models using linopy.
    """

    def __init__(self, network):
        self._parent = network

    @wraps(optimize)
    def __call__(self, *args, **kwargs):
        return optimize(self._parent, *args, **kwargs)

    @wraps(create_model)
    def create_model(self, *args, **kwargs):
        return create_model(self._parent, *args, **kwargs)

    def solve_model(
        self,
        extra_functionality=None,
        solver_name="glpk",
        solver_options={},
        assign_all_duals=False,
        **kwargs,
    ):
        """
        Solve an already created model and assign its solution to the network.

        Parameters
        ----------
        solver_name : str
            Name of the solver to use.
        solver_options : dict
            Keyword arguments used by the solver. Can also be passed via **kwargs.
        assign_all_duals : bool, default False
            Whether to assign all dual values or only those that already
            have a designated place in the network.
        **kwargs:
            Keyword argument used by `linopy.Model.solve`, such as `solver_name`,
            `problem_fn` or solver options directly passed to the solver.

        Returns
        -------
        status : str
            The status of the optimization, either "ok" or one of the
            codes listed in
            https://linopy.readthedocs.io/en/latest/generated/linopy.constants.SolverStatus.html
        condition : str
            The termination condition of the optimization, either
            "optimal" or one of the codes listed in
            https://linopy.readthedocs.io/en/latest/generated/linopy.constants.TerminationCondition.html
        """
        n = self._parent
        if extra_functionality:
            extra_functionality(n, n.snapshots)
        m = n.model
        status, condition = m.solve(solver_name=solver_name, **solver_options, **kwargs)

        if status == "ok":
            assign_solution(n)
            assign_duals(n, assign_all_duals)
            post_processing(n)

        return status, condition

    @wraps(assign_solution)
    def assign_solution(self, **kwargs):
        return assign_solution(self._parent, **kwargs)

    @wraps(assign_duals)
    def assign_duals(self, **kwargs):
        return assign_duals(self._parent, **kwargs)

    @wraps(post_processing)
    def post_processing(self, **kwargs):
        return post_processing(self._parent, **kwargs)

    @wraps(optimize_transmission_expansion_iteratively)
    def optimize_transmission_expansion_iteratively(self, *args, **kwargs):
        return optimize_transmission_expansion_iteratively(
            self._parent, *args, **kwargs
        )

    @wraps(optimize_security_constrained)
    def optimize_security_constrained(self, *args, **kwargs):
        return optimize_security_constrained(self._parent, *args, **kwargs)

    @wraps(optimize_with_rolling_horizon)
    def optimize_with_rolling_horizon(self, *args, **kwargs):
        return optimize_with_rolling_horizon(self._parent, *args, **kwargs)

    @wraps(optimize_mga)
    def optimize_mga(self, *args, **kwargs):
        return optimize_mga(self._parent, *args, **kwargs)

    def fix_optimal_capacities(self):
        """
        Fix capacities of extendable assets to optimized capacities.

        Use this function when a capacity expansion optimization was
        already performed and a operational optimization should be done
        afterwards.
        """
        n = self._parent
        for c, attr in nominal_attrs.items():
            ext_i = n.get_extendable_i(c)
            n.df(c).loc[ext_i, attr] = n.df(c).loc[ext_i, attr + "_opt"]
            n.df(c)[attr + "_extendable"] = False

    def fix_optimal_dispatch(self):
        """
        Fix dispatch of all assets to optimized values.

        Use this function when the optimal dispatch should be used as an
        starting point for power flow calculation (`Network.pf`).
        """
        n = self._parent
        for c in n.one_port_components:
            n.pnl(c).p_set = n.pnl(c).p
        for c in n.controllable_branch_components:
            n.pnl(c).p_set = n.pnl(c).p0

    def add_load_shedding(
        self,
        suffix=" load shedding",
        buses=None,
        sign=1e-3,
        marginal_cost=1e2,
        p_nom=1e9,
    ):
        """
        Add load shedding in form of generators to all or a subset of buses.

        For more information on load shedding see
        http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full

        Parameters
        ----------
        buses : pandas.Index, optional
            Subset of buses where load shedding should be available.
            Defaults to all buses.
        sign : float/Series, optional
            Scaling of the load shedding. This is used to scale the price of the
            load shedding. The default is 1e-3 which translates to a measure in kW instead
            of MW.
        marginal_cost : float/Series, optional
            Price of the load shedding. The default is 1e2.
        p_nom : float/Series, optional
            Maximal load shedding. The default is 1e9 (kW).
        """
        n = self._parent
        if "Load" not in n.carriers.index:
            n.add("Carrier", "Load")
        if buses is None:
            buses = n.buses.index

        return n.madd(
            "Generator",
            buses,
            suffix,
            bus=buses,
            carrier="load",
            sign=sign,
            marginal_cost=marginal_cost,
            p_nom=p_nom,
        )
