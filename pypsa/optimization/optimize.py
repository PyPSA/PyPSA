"""Build optimisation problems from PyPSA networks with Linopy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Model, merge
from linopy.solvers import available_solvers

from pypsa.common import UnexpectedError, as_index, list_as_string
from pypsa.components.common import as_components
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import nominal_attrs
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
    """Define and write the optimization objective function.

    Builds the (linear or quadratic) objective by assembling the following terms:

    1. **Constant term** for already-built capacity
       Calculates capex of existing assets and stores it in `n.objective_constant`.
    2. **Operating costs**
       Marginal generation costs, storage operation costs, and spill costs weighted by snapshot durations.
    3. **Quadratic costs**
       If present, adds second-order marginal cost terms to convex quadratic objective.
    4. **Stand-by costs**
       Fixed costs for committed assets (e.g. generators and links) when online.
    5. **Investment costs**
       Capex for new capacity, weighted by investment periods if `n._multi_invest` is True.
    6. **Unit-commitment costs**
       Start-up and shut-down costs for committable components.

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the Linopy model and component data.
    sns : pandas.Index
        Snapshots (and, for multi-investment, periods) over which to build the objective.

    Returns
    -------
    None

    Notes
    -----
    - The final objective expression is assigned to `n.model.objective`.
    - Applies snapshot and investment-period weightings to operational and capex terms.
    - For a stochastic problem, scenario probabilities are applied as weightings to all cost (includes *both* investment terms).

    """
    weighted_cost: xr.DataArray | int
    m = n.model
    objective = []
    is_quadratic = False

    if n._multi_invest:
        periods = sns.unique("period")
        period_weighting = n.investment_period_weightings.objective[periods]

    # constant for already done investment
    nom_attr = nominal_attrs.items()
    constant: xr.DataArray | float = 0
    terms = []

    for c_name, attr in nom_attr:
        c = as_components(n, c_name)
        ext_i = c.extendables

        if ext_i.empty:
            continue

        capital_cost = c.da.capital_cost.sel(name=ext_i)
        if capital_cost.size == 0:
            continue

        nominal = c.da[attr].sel(name=ext_i)

        # only charge capex for already-existing assets
        if n._multi_invest:
            weighted_cost = 0
            for i, period in enumerate(periods):
                # collapse time axis via any() so capex value isn't broadcasted
                active = c.da.active.sel(period=period, name=ext_i).any(dim="timestep")
                weighted_cost += capital_cost * active * period_weighting.iloc[i]
        else:
            # collapse time axis via any() so capex value isn’t broadcasted
            active = c.da.active.sel(name=ext_i).any(dim="snapshot")
            weighted_cost = capital_cost * active

        terms.append((weighted_cost * nominal).sum(dim=["name"]))

    constant += sum(terms)

    # Handle constant for stochastic vs deterministic networks
    if n.has_scenarios and isinstance(constant, xr.DataArray):
        # For stochastic networks, weight constant by scenario probabilities
        weighted_constant = sum(
            constant.sel(scenario=s) * n.scenario_weightings.loc[s, "weight"]
            for s in n.scenarios
        )
        n._objective_constant = float(weighted_constant)
        has_const = (constant != 0).any().item()
    else:
        n._objective_constant = float(constant)
        has_const = constant != 0
    if has_const:
        object_const = m.add_variables(constant, constant, name="objective_constant")
        objective.append(-1 * object_const)

    # Weightings
    weighting = n.snapshot_weightings.objective
    if n._multi_invest:
        weighting = weighting.mul(period_weighting, level=0).loc[sns]
    else:
        weighting = weighting.loc[sns]
    weight = xr.DataArray(weighting.values, coords={"snapshot": sns}, dims=["snapshot"])

    # marginal costs, marginal storage cost, and spill cost
    for cost_type in ["marginal_cost", "marginal_cost_storage", "spill_cost"]:
        for c_name, attr in lookup.query(cost_type).index:
            c = as_components(n, c_name)

            if c.static.empty:
                continue

            var_name = f"{c.name}-{attr}"
            if var_name not in m.variables and cost_type == "spill_cost":
                continue

            cost = c.da[cost_type].sel(snapshot=sns)
            if cost.size == 0 or (cost == 0).all():
                continue

            cost = cost * weight

            operation = m[var_name].sel(snapshot=sns, name=cost.coords["name"].values)
            objective.append((operation * cost).sum(dim=["name", "snapshot"]))

    # marginal cost quadratic
    for c_name, attr in lookup.query("marginal_cost_quadratic").index:
        c = as_components(n, c_name)

        if c.static.empty or "marginal_cost_quadratic" not in c.static.columns:
            continue

        cost = c.da.marginal_cost_quadratic.sel(snapshot=sns)
        if cost.size == 0 or (cost == 0).all():
            continue

        cost = cost * weight

        operation = m[f"{c.name}-{attr}"].sel(
            snapshot=sns, name=cost.coords["name"].values
        )
        objective.append((operation * operation * cost).sum(dim=["name", "snapshot"]))
        is_quadratic = True

    # stand-by cost
    for c_name in ["Generator", "Link"]:
        c = as_components(n, c_name)
        com_i = c.committables

        if com_i.empty:
            continue

        stand_by_cost = c.da.stand_by_cost.sel(name=com_i, snapshot=sns)
        if stand_by_cost.size == 0 or (stand_by_cost == 0).all():
            continue

        stand_by_cost = stand_by_cost * weight

        status = m[f"{c.name}-status"].sel(
            snapshot=sns, name=stand_by_cost.coords["name"].values
        )
        objective.append((status * stand_by_cost).sum(dim=["name", "snapshot"]))

    # investment
    for c_name, attr in nominal_attrs.items():
        c = as_components(n, c_name)
        ext_i = c.extendables

        if ext_i.empty:
            continue

        capital_cost = c.da.capital_cost.sel(name=ext_i)
        if capital_cost.size == 0 or (capital_cost == 0).all():
            continue

        # only charge capex for already-existing assets
        if n._multi_invest:
            weighted_cost = 0
            for i, period in enumerate(periods):
                # collapse time axis via any() so capex value isn't broadcasted
                active = c.da.active.sel(period=period, name=ext_i).any(dim="timestep")
                weighted_cost += capital_cost * active * period_weighting.iloc[i]
        else:
            # collapse time axis via any() so capex value isn't broadcasted
            active = c.da.active.sel(name=ext_i).any(dim="snapshot")
            weighted_cost = capital_cost * active

        caps = m[f"{c.name}-{attr}"].sel(name=ext_i)
        objective.append((caps * weighted_cost).sum(dim=["name"]))

    # unit commitment
    keys = ["start_up", "shut_down"]  # noqa: F841
    for c_name, attr in lookup.query("variable in @keys").index:
        c = as_components(n, c_name)
        com_i = c.committables

        if com_i.empty:
            continue

        cost = c.da[attr + "_cost"].sel(name=com_i)

        if cost.size == 0 or cost.sum().item() == 0:
            continue

        var = m[f"{c.name}-{attr}"].sel(name=com_i)
        objective.append((var * cost).sum(dim=["name", "snapshot"]))

    if not objective:
        msg = (
            "Objective function could not be created. "
            "Please make sure the components have assigned costs."
        )
        raise ValueError(msg)

    terms = []
    if n.has_scenarios:
        # Apply scenario probabilities as weights to the objective
        for s, p in n.scenario_weightings["weight"].items():
            selected = [e.sel(scenario=s) for e in objective]
            merged = merge(selected)
            terms.append(merged * p)
    else:
        terms = objective

    # Ensure we're returning the correct expression type (MGA compatibility)
    m.objective = sum(terms) if is_quadratic else merge(terms)


def from_xarray(da: xr.DataArray) -> pd.DataFrame | pd.Series:
    """# TODO move."""
    # Get available dimensions
    dims = set(da.dims)

    if dims in ({"name"}, {"snapshot", "name"}, {"snapshot"}):
        return da.to_pandas()

    elif dims == {"name", "snapshot", "scenario"}:
        df = (
            da.transpose("name", "scenario", "snapshot")
            .stack(combined=("scenario", "name"))
            .to_pandas()
        )

        df.columns.name = None
        return df

    # Handle auxiliary dimensions (e.g. from security constrained optimization)
    elif len(dims) > 2:
        # Find auxiliary dimensions
        contingency_dims = [
            d for d in dims if d not in {"snapshot", "name", "scenario"}
        ]

        if contingency_dims:
            # Stack auxiliary dimensions with component dimension to create combined index
            if "scenario" in dims:
                stack_dims = ["name", "scenario"] + contingency_dims
            else:
                stack_dims = ["name"] + contingency_dims

            combined_name = "combined"
            df = da.stack({combined_name: stack_dims}).to_pandas()

            if hasattr(df, "columns"):
                df.columns.name = None

            return df

    # Handle cases with auxiliary dimensions but no component dimension (e.g. GlobalConstraint with cycle)
    elif len(dims) == 2 and "snapshot" in dims:
        # For 2D cases like ('snapshot', 'cycle'), just use to_pandas() directly
        return da.to_pandas()

    # Handle other cases
    available_dims = list_as_string(dims)
    msg = (
        f"Unexpected combination of dimensions: {available_dims}. "
        f"Expected some combination of 'snapshot', 'name', and 'scenario'."
    )
    raise UnexpectedError(msg)


class OptimizationAccessor(OptimizationAbstractMixin):
    """Optimization accessor for building and solving models using linopy."""

    def __init__(self, n: Network) -> None:
        """Initialize the optimization accessor."""
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

        n = self._n
        sns = as_index(n, snapshots, "snapshots")
        n._multi_invest = int(multi_investment_periods)
        n._linearized_uc = linearized_unit_commitment

        n.consistency_check(strict=["unknown_buses"])
        m = n.optimize.create_model(
            sns,
            multi_investment_periods,
            transmission_losses,
            linearized_unit_commitment,
            consistency_check=False,
            **model_kwargs,
        )
        if extra_functionality:
            extra_functionality(n, sns)
        status, condition = m.solve(solver_name=solver_name, **solver_options, **kwargs)

        if status == "ok":
            n.optimize.assign_solution()
            n.optimize.assign_duals(assign_all_duals)
            n.optimize.post_processing()

        if (
            condition == "infeasible"
            and compute_infeasibilities
            and "gurobi" in available_solvers
        ):
            n.model.print_infeasibilities()

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
        multi_investment_periods : bool, default: False
            Whether to optimise as a single investment period or to optimize in multiple
            investment periods. Then, snapshots should be a ``pd.MultiIndex``.
        transmission_losses : int, default: 0
            Whether an approximation of transmission losses should be included
            in the linearised power flow formulation.
        linearized_unit_commitment : bool, default: False
            Whether to optimise using the linearised unit commitment formulation or not.
        consistency_check : bool, default: True
            Whether to run the consistency check before building the model.
        **kwargs:
            Keyword arguments used by `linopy.Model()`, such as `solver_dir` or `chunk`.

        Returns
        -------
        linopy.model

        """
        n = self._n
        sns = as_index(n, snapshots, "snapshots")
        n._linearized_uc = int(linearized_unit_commitment)
        n._multi_invest = int(multi_investment_periods)
        if consistency_check:
            n.consistency_check()

        kwargs.setdefault("force_dim_names", True)
        n._model = Model(**kwargs)
        n.model.parameters = n.model.parameters.assign(snapshots=sns)

        # Define variables
        for c, attr in lookup.query("nominal").index:
            define_nominal_variables(n, c, attr)
            define_modular_variables(n, c, attr)

        for c, attr in lookup.query("not nominal and not handle_separately").index:
            define_operational_variables(n, sns, c, attr)
            define_status_variables(n, sns, c, linearized_unit_commitment)
            define_start_up_variables(n, sns, c, linearized_unit_commitment)
            define_shut_down_variables(n, sns, c, linearized_unit_commitment)

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

        meshed_threshold = kwargs.get("meshed_threshold", 45)
        meshed_buses = get_strongly_meshed_buses(n, threshold=meshed_threshold)

        if isinstance(n.buses.index, pd.MultiIndex):
            bus_names = n.buses.index.get_level_values(1)
            weakly_meshed_buses = pd.Index(
                [b for b in bus_names if b not in meshed_buses], name="Bus"
            )
        else:
            weakly_meshed_buses = n.buses.index.difference(meshed_buses)

        if not meshed_buses.empty and not weakly_meshed_buses.empty:
            # Write constraint for buses many terms and for buses with a few terms
            # separately. This reduces memory usage for large networks.
            define_nodal_balance_constraints(
                n,
                sns,
                transmission_losses=transmission_losses,
                buses=weakly_meshed_buses,
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
        define_total_supply_constraints(n, sns)

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
        extra_functionality : callable
            This function must take two arguments
            `extra_functionality(n, snapshots)` and is called after
            the model building is complete, but before it is sent to the
            solver. It allows the user to
            add/change constraints and add/change the objective function.
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
        n = self._n
        m = n.model
        sns = n.model.parameters.snapshots.to_index()

        for name, variable in m.variables.items():
            sol = variable.solution
            if name == "objective_constant":
                continue

            try:
                c, attr = name.split("-", 1)
                df = from_xarray(sol)
            except ValueError:
                # TODO Why is this needed?
                continue

            if "snapshot" in sol.dims:
                if c in n.passive_branch_components and attr == "s":
                    set_from_frame(n, c, "p0", df)
                    set_from_frame(n, c, "p1", -df)

                elif c == "Link" and attr == "p":
                    set_from_frame(n, c, "p0", df)

                    for i in ["1"] + n.components.links.additional_ports:
                        i_eff = "" if i == "1" else i
                        eff = get_as_dense(n, "Link", f"efficiency{i_eff}", sns)
                        set_from_frame(n, c, f"p{i}", -df * eff)
                        n.dynamic(c)[f"p{i}"].loc[
                            sns, n.links.index[n.links[f"bus{i}"] == ""]
                        ] = float(n.components["Link"]["attrs"].loc[f"p{i}", "default"])

                else:
                    set_from_frame(n, c, attr, df)
            elif attr != "n_mod":
                idx = df.index.intersection(n.components[c].component_names)
                static = n.components[c].static
                static.loc[:, attr + "_opt"] = static.index.get_level_values(
                    "name"
                ).map(df.loc[idx])

        # if nominal capacity was no variable set optimal value to nominal
        for c, attr in lookup.query("nominal").index:
            fix_i = n.components[c].fixed
            if not fix_i.empty:
                n.static(c).loc[fix_i, f"{attr}_opt"] = n.static(c).loc[fix_i, attr]

        # recalculate storageunit net dispatch
        if not n.static("StorageUnit").empty:
            c = "StorageUnit"
            n.dynamic(c)["p"] = n.dynamic(c)["p_dispatch"] - n.dynamic(c)["p_store"]

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
        unassigned_constraints = []

        # Early return if no dual values are available
        if all("dual" not in constraint for _, constraint in m.constraints.items()):
            logger.info("No shadow prices were assigned to the network.")
            return

        # Process each constraint and its dual values
        for constraint_name, constraint in m.constraints.items():
            dual_values = constraint.dual

            # Parse constraint name into component and attribute
            try:
                component_name, attribute_name = constraint_name.split("-", 1)
            except ValueError:
                unassigned_constraints.append(constraint_name)
                continue

            # TIME-VARYING DUALS (constraints with snapshot dimension)
            if "snapshot" in dual_values.dims:
                try:
                    # Use from_xarray for all constraints (now handles GlobalConstraint cases too)
                    dual_df = from_xarray(dual_values.transpose("snapshot", ...))

                    # Determine what the dual variable will be called (e.g., "mu_<spec>")
                    if "security" in attribute_name:
                        # Security constraints: preserve more information to avoid conflicts
                        # e.g., "fix-s-lower-security-for-Line-outage-in-SubNetwork-0"
                        attr_parts = attribute_name.split("-")
                        dual_spec = (
                            "-".join(attr_parts[1:])
                            if len(attr_parts) >= 3
                            else attribute_name
                        )
                    elif component_name == "GlobalConstraint":
                        dual_spec = attribute_name
                    else:
                        # Standard components: extract last part after final dash
                        # e.g., "Line-s-upper" -> "upper", "Generator-p-lower" -> "lower"
                        try:
                            dual_spec = attribute_name.rsplit("-", 1)[-1]
                        except ValueError:
                            dual_spec = attribute_name

                    # Assign dual values to appropriate network attribute
                    if attribute_name.endswith("nodal_balance"):
                        # Special case: nodal balance duals become marginal prices
                        set_from_frame(
                            self._n, component_name, "marginal_price", dual_df
                        )
                    elif assign_all_duals or f"mu_{dual_spec}" in self._n.static(
                        component_name
                    ):
                        # Standard case: assign as "mu_<spec>" (e.g., "mu_upper", "mu_generation_limit_dynamic")
                        set_from_frame(
                            self._n, component_name, "mu_" + dual_spec, dual_df
                        )
                    else:
                        # Dual variable doesn't have a designated place and assign_all_duals=False
                        unassigned_constraints.append(constraint_name)

                except (KeyError, ValueError):
                    unassigned_constraints.append(constraint_name)

            # SCALAR DUALS (constraints without snapshot dimension)
            elif component_name == "GlobalConstraint" and (
                assign_all_duals
                or attribute_name in self._n.static(component_name).index
            ):
                # GlobalConstraint scalar duals: assign directly to the "mu" column
                self._n.static(component_name).loc[attribute_name, "mu"] = dual_values

        if unassigned_constraints:
            logger.info(
                "The shadow-prices of the constraints %s were not assigned to the network.",
                ", ".join(unassigned_constraints),
            )

    def post_processing(self) -> None:
        """Post-process the optimized network.

        This calculates quantities derived from the optimized values such as
        power injection per bus and snapshot, voltage angle.
        """
        n = self._n
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
        ca.extend(
            [("Link", f"p{i}", f"bus{i}") for i in n.components.links.additional_ports]
        )

        def sign(c: str) -> int:
            return n.static(c).sign if "sign" in n.static(c) else -1  # sign for 'Link'

        n.buses_t.p = (
            pd.concat(
                [
                    n.dynamic(c)[attr]
                    .mul(sign(c))
                    .rename(columns=n.static(c)[group], level="name")
                    for c, attr, group in ca
                ],
                axis=1,
            )
            .T.groupby(level=0)
            .sum()
            .T.reindex(columns=n.buses.index, fill_value=0.0)
        )

        if not n.has_scenarios:

            def v_ang_for_(sub: SubNetwork) -> pd.DataFrame:
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
            if n.sub_networks.empty:
                n.determine_network_topology()

            # Calculate voltage angles (only needed for power flow)
            if "obj" in n.sub_networks:
                n.buses_t.v_ang = pd.concat(
                    [v_ang_for_(sub) for sub in n.sub_networks.obj], axis=1
                ).reindex(columns=n.buses.index, fill_value=0.0)

    def fix_optimal_capacities(self) -> None:
        """Fix capacities of extendable assets to optimized capacities.

        Use this function when a capacity expansion optimization was
        already performed and a operational optimization should be done
        afterwards.
        """
        for c, attr in nominal_attrs.items():
            ext_i = self._n.components[c].extendables
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
        suffix : str, default: " load shedding"
            Suffix of the load shedding generators. See suffix parameter of
            [pypsa.Network.add].
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
