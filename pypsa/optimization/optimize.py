"""Build optimisation problems from PyPSA networks with Linopy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from deprecation import deprecated
from linopy import Model, merge
from linopy.solvers import available_solvers

from pypsa.common import as_index
from pypsa.descriptors import get_committable_i, nominal_attrs
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.optimization.abstract import OptimizationAbstractMixin
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
    define_total_supply_constraints,
)
from pypsa.optimization.expressions import StatisticExpressionsAccessor
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

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pypsa import Network, SubNetwork
logger = logging.getLogger(__name__)


lookup = pd.read_csv(
    Path(__file__).parent / ".." / "data" / "variables.csv",
    index_col=["component", "variable"],
)


def define_objective(n: Network, sns: pd.Index) -> None:
    """Define and write out the objective function."""
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
        cost = n.static(c)["capital_cost"][ext_i]
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
        else:
            active = n.get_active_assets(c)[ext_i]
            cost = cost[active]

        constant += (cost * n.static(c)[attr][ext_i]).sum()

    n._objective_constant = constant
    if constant != 0:
        object_const = m.add_variables(constant, constant, name="objective_constant")
        objective.append(-1 * object_const)

    # Weightings
    weighting = n.snapshot_weightings.objective
    if n._multi_invest:
        weighting = weighting.mul(period_weighting, level=0).loc[sns]
    else:
        weighting = weighting.loc[sns]

    # marginal costs, marginal storage cost, and spill cost
    for cost_type in ["marginal_cost", "marginal_cost_storage", "spill_cost"]:
        for c, attr in lookup.query(cost_type).index:
            cost = (
                get_as_dense(n, c, cost_type, sns)
                .loc[:, lambda ds: (ds != 0).any()]
                .mul(weighting, axis=0)
            )
            if cost.empty:
                continue
            operation = m[f"{c}-{attr}"].sel({"snapshot": sns, c: cost.columns})
            objective.append((operation * cost).sum())

    # marginal cost quadratic
    for c, attr in lookup.query("marginal_cost_quadratic").index:
        if "marginal_cost_quadratic" in n.static(c):
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
        cost = n.static(c)["capital_cost"][ext_i]
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
        else:
            active = n.get_active_assets(c)[ext_i]
            cost = cost[active]

        caps = m[f"{c}-{attr}"]
        objective.append((caps * cost).sum())

    # unit commitment
    keys = ["start_up", "shut_down"]  # noqa: F841
    for c, attr in lookup.query("variable in @keys").index:
        com_i = n.get_committable_i(c)
        cost = n.static(c)[attr + "_cost"].reindex(com_i)

        if cost.sum():
            var = m[f"{c}-{attr}"]
            objective.append((var * cost).sum())

    if not objective:
        msg = (
            "Objective function could not be created. "
            "Please make sure the components have assigned costs."
        )
        raise ValueError(msg)

    m.objective = sum(objective) if is_quadratic else merge(objective)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.optimize.create_model` instead.",
)
def create_model(
    n: Network,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    transmission_losses: int = 0,
    linearized_unit_commitment: bool = False,
    consistency_check: bool = True,
    **kwargs: Any,
) -> Model:
    """Use `n.optimize.create_model` instead."""
    return n.optimize.create_model(
        snapshots,
        multi_investment_periods,
        transmission_losses,
        linearized_unit_commitment,
        consistency_check,
        **kwargs,
    )


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.optimize.assign_solution` instead.",
)
def assign_solution(n: Network) -> None:
    """Use `n.optimize.assign_solution` instead."""
    n.optimize.assign_solution()


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.optimize.assign_duals` instead.",
)
def assign_duals(n: Network, assign_all_duals: bool = False) -> None:
    """Use `n.optimize.assign_duals` instead."""
    n.optimize.assign_duals(assign_all_duals)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.optimize.post_processing` instead.",
)
def post_processing(n: Network) -> None:
    """Use `n.optimize.post_processing` instead."""
    n.optimize.post_processing()


@deprecated(
    deprecated_in="0.35", removed_in="1.0", details="Use `n.optimize()` instead."
)
def optimize(
    n: Network,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    transmission_losses: int = 0,
    linearized_unit_commitment: bool = False,
    model_kwargs: dict | None = None,
    extra_functionality: Callable | None = None,
    assign_all_duals: bool = False,
    solver_name: str = "highs",
    solver_options: dict | None = None,
    compute_infeasibilities: bool = False,
    **kwargs: Any,
) -> tuple[str, str]:
    """Use `n.optimize()` instead."""
    return n.optimize(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        transmission_losses=transmission_losses,
        linearized_unit_commitment=linearized_unit_commitment,
        model_kwargs=model_kwargs,
        extra_functionality=extra_functionality,
        assign_all_duals=assign_all_duals,
        solver_name=solver_name,
        solver_options=solver_options,
        compute_infeasibilities=compute_infeasibilities,
        **kwargs,
    )


class OptimizationAccessor(OptimizationAbstractMixin):
    """Optimization accessor for building and solving models using linopy."""

    def __init__(self, n: Network) -> None:
        self._n = n
        self.expressions = StatisticExpressionsAccessor(self._n)

    def __call__(
        self,
        snapshots: Sequence | None = None,
        multi_investment_periods: bool = False,
        transmission_losses: int = 0,
        linearized_unit_commitment: bool = False,
        model_kwargs: dict | None = None,
        extra_functionality: Callable | None = None,
        assign_all_duals: bool = False,
        solver_name: str = "highs",
        solver_options: dict | None = None,
        compute_infeasibilities: bool = False,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Optimize the pypsa network using linopy.

        Parameters
        ----------
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
            `extra_functionality(n, snapshots)` and is called after
            the model building is complete, but before it is sent to the
            solver. It allows the user to
            add/change constraints and add/change the objective function.
        assign_all_duals : bool, default False
            Whether to assign all dual values or only those that already
            have a designated place in the network.
        solver_name : str
            Name of the solver to use.
        solver_options : dict
            Keyword arguments used by the solver. Can also be passed via `**kwargs`.
        compute_infeasibilities : bool, default False
            Whether to compute and print Irreducible Inconsistent Subsystem (IIS) in case
            of an infeasible solution. Requires Gurobi.
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
        if model_kwargs is None:
            model_kwargs = {}

        if solver_options is None:
            solver_options = {}

        sns = as_index(self._n, snapshots, "snapshots")
        self._n._multi_invest = int(multi_investment_periods)
        self._n._linearized_uc = linearized_unit_commitment

        self._n.consistency_check(strict=["unknown_buses"])
        m = self._n.optimize.create_model(
            sns,
            multi_investment_periods,
            transmission_losses,
            linearized_unit_commitment,
            consistency_check=False,
            **model_kwargs,
        )
        if extra_functionality:
            extra_functionality(self._n, sns)
        status, condition = m.solve(solver_name=solver_name, **solver_options, **kwargs)

        if status == "ok":
            self._n.optimize.assign_solution()
            self._n.optimize.assign_duals(assign_all_duals)
            self._n.optimize.post_processing()

        if (
            condition == "infeasible"
            and compute_infeasibilities
            and "gurobi" in available_solvers
        ):
            self._n.model.print_infeasibilities()

        return status, condition

    def create_model(
        self,
        snapshots: Sequence | None = None,
        multi_investment_periods: bool = False,
        transmission_losses: int = 0,
        linearized_unit_commitment: bool = False,
        consistency_check: bool = True,
        **kwargs: Any,
    ) -> Model:
        """Create a linopy.Model instance from a pypsa network.

        The model is stored at `n.model`.

        Parameters
        ----------
        snapshots : list or index slice
            A list of snapshots to optimise, must be a subset of
            n.snapshots, defaults to n.snapshots
        multi_investment_periods : bool, default False
            Whether to optimise as a single investment period or to optimize in multiple
            investment periods. Then, snapshots should be a ``pd.MultiIndex``.
        transmission_losses : int, default 0
        linearized_unit_commitment : bool, default False
            Whether to optimise using the linearised unit commitment formulation or not.
        consistency_check : bool, default True
            Whether to run the consistency check before building the model.
        **kwargs:
            Keyword arguments used by `linopy.Model()`, such as `solver_dir` or `chunk`.

        Returns
        -------
        linopy.model

        """
        sns = as_index(self._n, snapshots, "snapshots")
        self._n._linearized_uc = int(linearized_unit_commitment)
        self._n._multi_invest = int(multi_investment_periods)
        if consistency_check:
            self._n.consistency_check()

        kwargs.setdefault("force_dim_names", True)
        self._n._model = Model(**kwargs)
        self._n.model.parameters = self._n.model.parameters.assign(snapshots=sns)

        # Define variables
        for c, attr in lookup.query("nominal").index:
            define_nominal_variables(self._n, c, attr)
            define_modular_variables(self._n, c, attr)

        for c, attr in lookup.query("not nominal and not handle_separately").index:
            define_operational_variables(self._n, sns, c, attr)
            define_status_variables(self._n, sns, c)
            define_start_up_variables(self._n, sns, c)
            define_shut_down_variables(self._n, sns, c)

        define_spillage_variables(self._n, sns)
        define_operational_variables(self._n, sns, "Store", "p")

        if transmission_losses:
            for c in self._n.passive_branch_components:
                define_loss_variables(self._n, sns, c)

        # Define constraints
        for c, attr in lookup.query("nominal").index:
            define_nominal_constraints_for_extendables(self._n, c, attr)
            define_fixed_nominal_constraints(self._n, c, attr)
            define_modular_constraints(self._n, c, attr)

        for c, attr in lookup.query("not nominal and not handle_separately").index:
            define_operational_constraints_for_non_extendables(
                self._n, sns, c, attr, transmission_losses
            )
            define_operational_constraints_for_extendables(
                self._n, sns, c, attr, transmission_losses
            )
            define_operational_constraints_for_committables(self._n, sns, c)
            define_ramp_limit_constraints(self._n, sns, c, attr)
            define_fixed_operation_constraints(self._n, sns, c, attr)

        meshed_threshold = kwargs.get("meshed_threshold", 45)
        meshed_buses = get_strongly_meshed_buses(self._n, threshold=meshed_threshold)
        weakly_meshed_buses = self._n.buses.index.difference(meshed_buses)
        if not meshed_buses.empty and not weakly_meshed_buses.empty:
            # Write constraint for buses many terms and for buses with a few terms
            # separately. This reduces memory usage for large networks.
            define_nodal_balance_constraints(
                self._n,
                sns,
                transmission_losses=transmission_losses,
                buses=weakly_meshed_buses,
            )
            define_nodal_balance_constraints(
                self._n,
                sns,
                transmission_losses=transmission_losses,
                buses=meshed_buses,
                suffix="-meshed",
            )
        else:
            define_nodal_balance_constraints(
                self._n, sns, transmission_losses=transmission_losses
            )

        define_kirchhoff_voltage_constraints(self._n, sns)
        define_storage_unit_constraints(self._n, sns)
        define_store_constraints(self._n, sns)
        define_total_supply_constraints(self._n, sns)

        if transmission_losses:
            for c in self._n.passive_branch_components:
                define_loss_constraints(self._n, sns, c, transmission_losses)

        # Define global constraints
        define_primary_energy_limit(self._n, sns)
        define_transmission_expansion_cost_limit(self._n, sns)
        define_transmission_volume_expansion_limit(self._n, sns)
        define_tech_capacity_expansion_limit(self._n, sns)
        define_operational_limit(self._n, sns)
        define_nominal_constraints_per_bus_carrier(self._n, sns)
        define_growth_limit(self._n, sns)

        define_objective(self._n, sns)

        return self._n.model

    def solve_model(
        self,
        extra_functionality: Callable | None = None,
        solver_name: str = "highs",
        solver_options: dict | None = None,
        assign_all_duals: bool = False,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Solve an already created model and assign its solution to the network.

        Parameters
        ----------
        solver_name : str
            Name of the solver to use.
        solver_options : dict
            Keyword arguments used by the solver. Can also be passed via `**kwargs`.
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
        if solver_options is None:
            solver_options = {}

        n = self._n
        if extra_functionality:
            extra_functionality(n, n.snapshots)
        m = n.model
        status, condition = m.solve(solver_name=solver_name, **solver_options, **kwargs)

        if status == "ok":
            self._n.optimize.assign_solution()
            self._n.optimize.assign_duals(assign_all_duals)
            self._n.optimize.post_processing()

        return status, condition

    def assign_solution(self) -> None:
        """Map solution to network components."""
        m = self._n.model
        sns = self._n.model.parameters.snapshots.to_index()

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
                if c in self._n.passive_branch_components and attr == "s":
                    set_from_frame(self._n, c, "p0", df)
                    set_from_frame(self._n, c, "p1", -df)

                elif c == "Link" and attr == "p":
                    set_from_frame(self._n, c, "p0", df)

                    for i in ["1"] + self._n.c.links.additional_ports:
                        i_eff = "" if i == "1" else i
                        eff = get_as_dense(self._n, "Link", f"efficiency{i_eff}", sns)
                        set_from_frame(self._n, c, f"p{i}", -df * eff)
                        self._n.dynamic(c)[f"p{i}"].loc[
                            sns, self._n.links.index[self._n.links[f"bus{i}"] == ""]
                        ] = float(
                            self._n.components["Link"]["attrs"].loc[f"p{i}", "default"]
                        )

                else:
                    set_from_frame(self._n, c, attr, df)
            elif attr != "n_mod":
                idx = df.index.intersection(self._n.static(c).index)
                self._n.static(c).loc[idx, attr + "_opt"] = df.loc[idx]

        # if nominal capacity was no variable set optimal value to nominal
        for c, attr in lookup.query("nominal").index:
            fix_i = self._n.get_non_extendable_i(c)
            if not fix_i.empty:
                self._n.static(c).loc[fix_i, f"{attr}_opt"] = self._n.static(c).loc[
                    fix_i, attr
                ]

        # recalculate storageunit net dispatch
        if not self._n.static("StorageUnit").empty:
            c = "StorageUnit"
            self._n.dynamic(c)["p"] = (
                self._n.dynamic(c)["p_dispatch"] - self._n.dynamic(c)["p_store"]
            )

        self._n._objective = m.objective.value

    def assign_duals(self, assign_all_duals: bool = False) -> None:
        """Map dual values i.e. shadow prices to network components.

        Parameters
        ----------
        assign_all_duals : bool, default False
            Whether to assign all dual values or only those that already
            have a designated place in the network.

        """
        m = self._n.model
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
                        set_from_frame(self._n, c, "marginal_price", df)
                    elif assign_all_duals or f"mu_{spec}" in self._n.static(c):
                        set_from_frame(self._n, c, "mu_" + spec, df)
                    else:
                        unassigned.append(name)

                except:  # noqa: E722 # TODO: specify exception
                    unassigned.append(name)

            elif (c == "GlobalConstraint") and (
                assign_all_duals or attr in self._n.static(c).index
            ):
                self._n.static(c).loc[attr, "mu"] = dual

        if unassigned:
            logger.info(
                "The shadow-prices of the constraints %s were not assigned to the network.",
                ", ".join(unassigned),
            )

    def post_processing(self) -> None:
        """Post-process the optimized network.

        This calculates quantities derived from the optimized values such as
        power injection per bus and snapshot, voltage angle.
        """
        sns = self._n.model.parameters.snapshots.to_index()

        # correct prices with objective weightings
        if self._n._multi_invest:
            period_weighting = self._n.investment_period_weightings.objective
            weightings = self._n.snapshot_weightings.objective.mul(
                period_weighting, level=0, axis=0
            ).loc[sns]
        else:
            weightings = self._n.snapshot_weightings.objective.loc[sns]

        self._n.buses_t.marginal_price.loc[sns] = self._n.buses_t.marginal_price.loc[
            sns
        ].divide(weightings, axis=0)

        # load
        if len(self._n.loads):
            set_from_frame(
                self._n, "Load", "p", get_as_dense(self._n, "Load", "p_set", sns)
            )

        # line losses
        if "Line-loss" in self._n.model.variables:
            losses = self._n.model["Line-loss"].solution.to_pandas()
            self._n.lines_t.p0 += losses / 2
            self._n.lines_t.p1 += losses / 2

        # recalculate injection
        ca = [
            ("Generator", "p", "bus"),
            ("Store", "p", "bus"),
            ("Load", "p", "bus"),
            ("StorageUnit", "p", "bus"),
            ("Link", "p0", "bus0"),
            ("Link", "p1", "bus1"),
        ]
        ca.extend(
            [("Link", f"p{i}", f"bus{i}") for i in self._n.c.links.additional_ports]
        )

        def sign(c: str) -> int:
            return (
                self._n.static(c).sign if "sign" in self._n.static(c) else -1
            )  # sign for 'Link'

        self._n.buses_t.p = (
            pd.concat(
                [
                    self._n.dynamic(c)[attr]
                    .mul(sign(c))
                    .rename(columns=self._n.static(c)[group])
                    for c, attr, group in ca
                ],
                axis=1,
            )
            .T.groupby(level=0)
            .sum()
            .T.reindex(columns=self._n.buses.index, fill_value=0.0)
        )

        def v_ang_for_(sub: SubNetwork) -> pd.DataFrame:
            buses_i = sub.buses_o
            if len(buses_i) == 1:
                return pd.DataFrame(0, index=sns, columns=buses_i)
            sub.calculate_B_H(skip_pre=True)
            Z = pd.DataFrame(np.linalg.pinv((sub.B).todense()), buses_i, buses_i)
            Z -= Z[sub.slack_bus]
            return self._n.buses_t.p.reindex(columns=buses_i) @ Z

        # TODO: if multi investment optimization, the network topology is not the necessarily the same,
        # i.e. one has to iterate over the periods in order to get the correct angles.
        # Determine_network_topology is not necessarily called (only if KVL was assigned)
        if "obj" in self._n.sub_networks:
            self._n.buses_t.v_ang = pd.concat(
                [v_ang_for_(sub) for sub in self._n.sub_networks.obj], axis=1
            ).reindex(columns=self._n.buses.index, fill_value=0.0)

    def fix_optimal_capacities(self) -> None:
        """Fix capacities of extendable assets to optimized capacities.

        Use this function when a capacity expansion optimization was
        already performed and a operational optimization should be done
        afterwards.
        """
        for c, attr in nominal_attrs.items():
            ext_i = self._n.get_extendable_i(c)
            self._n.static(c).loc[ext_i, attr] = self._n.static(c).loc[
                ext_i, attr + "_opt"
            ]
            self._n.static(c)[attr + "_extendable"] = False

    def fix_optimal_dispatch(self) -> None:
        """Fix dispatch of all assets to optimized values.

        Use this function when the optimal dispatch should be used as an
        starting point for power flow calculation (`Network.pf`).
        """
        for c in self._n.one_port_components:
            self._n.dynamic(c).p_set = self._n.dynamic(c).p
        for c in self._n.controllable_branch_components:
            self._n.dynamic(c).p_set = self._n.dynamic(c).p0

    def add_load_shedding(
        self,
        suffix: str = " load shedding",
        buses: pd.Index | None = None,
        sign: float | pd.Series = 1e-3,
        marginal_cost: float | pd.Series = 1e2,
        p_nom: float | pd.Series = 1e9,
    ) -> pd.Index:
        """Add load shedding in form of generators to all or a subset of buses.

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
        if "Load" not in self._n.carriers.index:
            self._n.add("Carrier", "Load")
        if buses is None:
            buses = self._n.buses.index

        return self._n.add(
            "Generator",
            buses,
            suffix,
            bus=buses,
            carrier="load",
            sign=sign,
            marginal_cost=marginal_cost,
            p_nom=p_nom,
        )
