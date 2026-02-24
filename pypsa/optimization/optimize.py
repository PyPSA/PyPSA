# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Build optimisation problems from PyPSA networks with Linopy."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Model, merge
from linopy.solvers import available_solvers

from pypsa._options import options
from pypsa.common import UnexpectedError, as_index
from pypsa.components._types.links import Links
from pypsa.components.array import _from_xarray
from pypsa.components.common import as_components
from pypsa.consistency import check_big_m_exceeded, check_no_modular_committables
from pypsa.descriptors import nominal_attrs
from pypsa.guards import _assert_data_integrity
from pypsa.optimization.abstract import OptimizationAbstractMixin
from pypsa.optimization.common import _set_dynamic_data, get_strongly_meshed_buses
from pypsa.optimization.constraints import (
    define_committability_variables_constraints_with_fixed_upper_limit,
    define_committability_variables_constraints_with_variable_upper_limit,
    define_fixed_nominal_constraints,
    define_fixed_operation_constraints,
    define_kirchhoff_voltage_constraints,
    define_modular_constraints,
    define_nodal_balance_constraints,
    define_nominal_constraints_for_extendables,
    define_operational_constraints_for_committables,
    define_operational_constraints_for_extendables,
    define_operational_constraints_for_non_extendables,
    define_ramp_limit_constraints,
    define_secant_loss_constraints,
    define_storage_unit_constraints,
    define_store_constraints,
    define_tangent_loss_constraints,
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
    define_cvar_variables,
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


def _resolve_include_objective_constant(
    value: bool | None, stacklevel: int = 3
) -> bool:
    """Resolve include_objective_constant from explicit value or options.

    Raises FutureWarning if neither is set.
    """
    if value is None:
        value = options.params.optimize.include_objective_constant
    if value is None:
        warnings.warn(
            "The default value of `include_objective_constant` will change from "
            "True to False in version 2.0. Set `include_objective_constant` "
            "explicitly to suppress this warning. Using False improves LP numerical "
            "conditioning by not including the objective constant as a variable.",
            FutureWarning,
            stacklevel=stacklevel,
        )
        value = True
    return value


def define_objective(
    n: Network, sns: pd.Index, include_objective_constant: bool
) -> None:
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
    7. **Conditional CVaR terms**
        Define auxiliary CVaR constraints for stochastic risk-averse optimization.

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the Linopy model and component data.
    sns : pandas.Index
        Snapshots (and, for multi-investment, periods) over which to build the objective.
    include_objective_constant : bool
        Whether to include the objective constant as a variable in the objective function.

    Notes
    -----
    - The final objective expression is assigned to `n.model.objective`.
    - Applies snapshot and investment-period weightings to operational and capex terms.
    - For a stochastic problem, scenario probabilities are applied as weightings to all cost (includes *both* investment terms).

    """
    weighted_cost: xr.DataArray | int
    m = n.model
    # Separate lists to distinguish CAPEX and OPEX terms
    capex_terms = []
    opex_terms = []
    is_quadratic = False

    if n._multi_invest:
        periods = sns.unique("period")
        period_weighting = n.investment_period_weightings.objective[periods]

    # constant for already done investment
    if include_objective_constant:
        nom_attr = nominal_attrs.items()
        constant: xr.DataArray | float = 0
        terms = []

        for c_name, attr in nom_attr:
            c = as_components(n, c_name)
            ext_i = c.extendables.difference(c.inactive_assets)

            if ext_i.empty:
                continue

            periodic_cost = c.periodized_cost.sel(name=ext_i)
            if periodic_cost.size == 0:
                continue

            nominal = c.da[attr].sel(name=ext_i)

            if n._multi_invest:
                weighted_cost = 0
                for period in periods:
                    active = c.da.active.sel(period=period, name=ext_i).any(
                        dim="timestep"
                    )
                    weighted_cost += (
                        active * periodic_cost * period_weighting.loc[period]
                    )
            else:
                active = c.da.active.sel(name=ext_i).any(dim="snapshot")
                weighted_cost = active * periodic_cost

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
            object_const = m.add_variables(
                constant, constant, name="objective_constant"
            )
            capex_terms.append(-1 * object_const)
    else:
        n._objective_constant = 0.0

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

            cost = c.da[cost_type].sel(snapshot=sns, name=c.active_assets)
            if cost.size == 0 or (cost == 0).all():
                continue

            cost = cost * weight

            operation = m[var_name].sel(snapshot=sns, name=cost.coords["name"].values)
            opex_terms.append((operation * cost).sum(dim=["name", "snapshot"]))

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
        opex_terms.append((operation * operation * cost).sum(dim=["name", "snapshot"]))
        is_quadratic = True

    # stand-by cost
    for c_name in ["Generator", "Link"]:
        c = as_components(n, c_name)
        com_i = c.committables.difference(c.inactive_assets)

        if com_i.empty:
            continue

        stand_by_cost = c.da.stand_by_cost.sel(name=com_i, snapshot=sns)
        if stand_by_cost.size == 0 or (stand_by_cost == 0).all():
            continue

        stand_by_cost = stand_by_cost * weight

        status = m[f"{c.name}-status"].sel(
            snapshot=sns, name=stand_by_cost.coords["name"].values
        )
        opex_terms.append((status * stand_by_cost).sum(dim=["name", "snapshot"]))

    # investment
    for c_name, attr in nominal_attrs.items():
        c = as_components(n, c_name)
        ext_i = c.extendables.difference(c.inactive_assets)

        if ext_i.empty:
            continue

        periodic_cost = c.periodized_cost.sel(name=ext_i)
        if periodic_cost.size == 0 or (periodic_cost == 0).all():
            continue

        if n._multi_invest:
            weighted_cost = 0
            for period in periods:
                active = c.da.active.sel(period=period, name=ext_i).any(dim="timestep")
                weighted_cost += active * periodic_cost * period_weighting.loc[period]
        else:
            active = c.da.active.sel(name=ext_i).any(dim="snapshot")
            weighted_cost = active * periodic_cost

        caps = m[f"{c.name}-{attr}"].sel(name=ext_i)
        capex_terms.append((caps * weighted_cost).sum(dim=["name"]))

    # unit commitment
    keys = ["start_up", "shut_down"]  # noqa: F841
    for c_name, attr in lookup.query("variable in @keys").index:
        c = as_components(n, c_name)
        com_i = c.committables.difference(c.inactive_assets)

        if com_i.empty:
            continue

        cost = c.da[attr + "_cost"].sel(name=com_i)

        if cost.size == 0 or cost.sum().item() == 0:
            continue

        var = m[f"{c.name}-{attr}"].sel(name=com_i)
        opex_terms.append((var * cost).sum(dim=["name", "snapshot"]))

    if not (capex_terms or opex_terms):
        msg = (
            "Objective function could not be created. "
            "Please make sure the components have assigned costs."
        )
        raise ValueError(msg)

    # Build expected CAPEX and expected OPEX (scenario-weighted if stochastic)
    def _expected(exprs: list) -> Any:
        if not exprs:
            return 0
        if n.has_scenarios:
            terms = []
            for s, p in n.scenario_weightings["weight"].items():
                selected = [e.sel(scenario=s) for e in exprs]
                # If quadratic terms exist, avoid merge (which is linear-only) and sum instead
                merged = sum(selected) if is_quadratic else merge(selected)
                terms.append(merged * p)
            return sum(terms) if is_quadratic else merge(terms)
        return sum(exprs) if is_quadratic else merge(exprs)

    expected_capex = _expected(capex_terms)
    expected_opex = _expected(opex_terms)

    # CVaR augmentation if enabled
    if n.has_risk_preference:
        rp = n.risk_preference
        if rp is None:  # mypy type guard
            msg = "risk_preference is None when has_risk_preference is True"
            raise UnexpectedError(msg)
        alpha = rp["alpha"]
        omega = rp["omega"]

        # Guard: quadratic OPEX would make CVaR constraints quadratic
        if is_quadratic:
            msg_q = (
                "CVaR with quadratic operational costs yields quadratic constraints. "
                "So a(s) >= OPEX(s) - theta becomes a quadratic inequality. "
                "Remove/approximate quadratic costs (e.g. set 'marginal_cost_quadratic=0' "
                "or use a piecewise-linear approximation)."
            )
            raise ValueError(msg_q)

        # Create per-scenario OPEX expressions to use in constraints
        scen_opex_exprs: dict[Any, Any] = {}
        for s in n.scenarios:
            scen_selected = [e.sel(scenario=s) for e in opex_terms]
            scen_opex_exprs[s] = (
                (sum(scen_selected) if is_quadratic else merge(scen_selected))
                if scen_selected
                else 0
            )

        # Retrieve CVaR auxiliary variables
        a = m["CVaR-a"]
        theta = m["CVaR-theta"]
        cvar = m["CVaR"]

        for s in n.scenarios:
            lhs = a.sel(scenario=s) - scen_opex_exprs[s] + theta
            m.add_constraints(lhs, ">=", 0, name=f"CVaR-excess-{s}")

        inv_tail = 1.0 / (1.0 - alpha)
        weighted_a = None
        for s, p in n.scenario_weightings["weight"].items():
            term = a.sel(scenario=s) * float(p)
            weighted_a = term if weighted_a is None else weighted_a + term
        if weighted_a is None:  # mypy type guard
            msg = "No scenarios found in scenario_weightings"
            raise UnexpectedError(msg)
        m.add_constraints(theta + inv_tail * weighted_a, "<=", cvar, name="CVaR-def")

        # Final objective: CAPEX + (1-omega) * E[OPEX] + omega * CVaR
        obj_expr = expected_capex + (1 - omega) * expected_opex + omega * cvar
    else:
        # Deterministic or no risk: CAPEX + OPEX
        obj_expr = expected_capex + expected_opex

    # Set objective
    m.objective = obj_expr


class OptimizationAccessor(OptimizationAbstractMixin):
    """Optimization accessor for building and solving models using linopy.

    <!-- md:guide network-optimization.md -->
    """

    def __init__(self, n: Network) -> None:
        """Initialize the optimization accessor."""
        self._n = n
        self.expressions = StatisticExpressionsAccessor(self._n)

    def __call__(
        self,
        snapshots: Sequence | None = None,
        multi_investment_periods: bool = False,
        transmission_losses: bool | int | dict = False,
        linearized_unit_commitment: bool = False,
        model_kwargs: dict | None = None,
        extra_functionality: Callable | None = None,
        assign_all_duals: bool = False,
        solver_name: str | None = None,
        solver_options: dict | None = None,
        log_to_console: bool | None = None,
        compute_infeasibilities: bool = False,
        include_objective_constant: bool | None = None,
        committable_big_m: float | None = None,
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
            investment periods. Then, snapshots should be a `pd.MultiIndex`.
        transmission_losses : bool | int | dict, default False
            Include piecewise linear approximation of transmission losses for
            passive branches:

            - ``True``: secant-based approximation with default tolerances
            - ``int``: *(deprecated)* tangent-based with that many segments
            - ``dict``: explicit config with key ``mode`` ("secants" or
              "tangents") and mode-specific options. Secant options:
              ``atol`` (default 1), ``rtol`` (default 0.1),
              ``max_segments`` (default 20). Tangent options: ``segments``
              (required).

            See `https://go.pypsa.org/transmission-losses` for details.
        linearized_unit_commitment : bool, default False
            Whether to optimise using the linearised unit commitment formulation or not.
        model_kwargs : dict, optional
            Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
            Defaults to module wide option (default: {}). See
            `https://go.pypsa.org/options-params` for more information.
        extra_functionality : callable
            This function must take two arguments
            `extra_functionality(n, snapshots)` and is called after
            the model building is complete, but before it is sent to the
            solver. It allows the user to
            add/change constraints and add/change the objective function.
        assign_all_duals : bool, default False
            Whether to assign all dual values or only those that already
            have a designated place in the network.
        solver_name : str, optional
            Name of the solver to use. Defaults to module wide option
            (default: 'highs'). See `https://go.pypsa.org/options-params` for more
            information.
        solver_options : dict, optional
            Keyword arguments used by the solver. Can also be passed via `**kwargs`.
            Defaults to module wide option (default: {}). See
            `https://go.pypsa.org/options-params` for more information.
        log_to_console : bool, optional
            Whether the solver prints its progress to the console. Passed as a
            solver option to linopy's `Model.solve()` method. When None,
            solver default behavior is used. Note: not all solvers support
            this option (e.g. HiGHS does, CPLEX does not). See
            `https://go.pypsa.org/options-params` for more information.
        compute_infeasibilities : bool, default False
            Whether to compute and print Irreducible Inconsistent Subsystem (IIS) in case
            of an infeasible solution. Requires Gurobi.
        include_objective_constant : bool | None, default None
            Whether to include the objective constant (capital costs of existing
            infrastructure) as a variable in the objective function. Setting to False
            improves LP numerical conditioning. Defaults to module wide option. See
            `pypsa.options.params.optimize.describe()` for more information.
        committable_big_m : float | None, default None
            Big-M value for committable+extendable constraints. If None, PyPSA infers
            a scale from the network (e.g. peak load). Otherwise this numeric bound
            is used when no component-specific limit (p_nom_max) is available.
        **kwargs:
            Keyword argument used by `linopy.Model.solve`, such as `solver_name`,
            `problem_fn` or solver options directly passed to the solver.

        Returns
        -------
        status : str
            The status of the optimization, either "ok" or one of the codes listed
            in [linopy.constants.SolverStatus](https://linopy.readthedocs.io/en/latest/generated/linopy.constants.SolverStatus.html)
        condition : str
            The termination condition of the optimization, either
            "optimal" or one of the codes listed in
            [linopy.constants.TerminationCondition](https://linopy.readthedocs.io/en/latest/generated/linopy.constants.TerminationCondition.html)

        """
        # Handle default parameters from options
        if model_kwargs is None:
            model_kwargs = options.params.optimize.model_kwargs.copy()
        if solver_name is None:
            solver_name = options.params.optimize.solver_name
        if solver_options is None:
            solver_options = options.params.optimize.solver_options.copy()
        if log_to_console is None:
            log_to_console = options.params.optimize.log_to_console

        include_objective_constant = _resolve_include_objective_constant(
            include_objective_constant
        )

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
            include_objective_constant=include_objective_constant,
            committable_big_m=committable_big_m,
            **model_kwargs,
        )
        if extra_functionality:
            extra_functionality(n, sns)
        if log_to_console is not None:
            kwargs["log_to_console"] = log_to_console
        status, condition = m.solve(
            solver_name=solver_name,
            **solver_options,
            **kwargs,
        )

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
        transmission_losses: int | dict | bool = False,
        linearized_unit_commitment: bool = False,
        consistency_check: bool = True,
        include_objective_constant: bool | None = None,
        committable_big_m: float | None = None,
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
            investment periods. Then, snapshots should be a `pd.MultiIndex`.
        transmission_losses : bool | int | dict, default False
            Include piecewise linear approximation of transmission losses for
            passive branches:
            - ``True``: secant-based approximation with default tolerances
            - ``int``: *(deprecated)* tangent-based with that many segments
            - ``dict``: explicit config with key ``mode`` ("secants" or
              "tangents") and mode-specific options. Secant options:
              ``atol`` (default 1), ``rtol`` (default 0.1),
              ``max_segments`` (default 20). Tangent options: ``segments``
              (required).

            See `https://go.pypsa.org/transmission-losses` for details.
        linearized_unit_commitment : bool, default: False
            Whether to optimise using the linearised unit commitment formulation or not.
        consistency_check : bool, default: True
            Whether to run the consistency check before building the model.
        include_objective_constant : bool | None, default: None
            Whether to include the objective constant (capital costs of existing
            infrastructure) as a variable in the objective function. Setting to False
            improves LP numerical conditioning. Defaults to module wide option. See
            `pypsa.options.params.optimize.describe()` for more information.
        committable_big_m : float | None, default: None
            Big-M value for committable+extendable constraints. If None, PyPSA infers
            a scale from the network (e.g. peak load). Otherwise this numeric bound
            is used when no component-specific limit (p_nom_max) is available.
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
        n._committable_big_m = committable_big_m

        if linearized_unit_commitment:
            check_no_modular_committables(n)

        if consistency_check:
            n.consistency_check()

        include_objective_constant = _resolve_include_objective_constant(
            include_objective_constant
        )

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
            define_committability_variables_constraints_with_fixed_upper_limit(
                n, sns, c, attr
            )

        define_spillage_variables(n, sns)
        define_operational_variables(n, sns, "Store", "p")

        # CVaR auxiliary variables (only when stochastic + risk preference is set)
        define_cvar_variables(n)

        if transmission_losses:
            for c in n.passive_branch_components:
                define_loss_variables(n, sns, c)

        # Define constraints
        for c, attr in lookup.query("nominal").index:
            define_nominal_constraints_for_extendables(n, c, attr)
            define_fixed_nominal_constraints(n, c, attr)
            define_modular_constraints(n, c, attr)
            define_committability_variables_constraints_with_variable_upper_limit(
                n, sns, c, attr
            )

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

        # Handle StorageUnit p_set separately (fixes p_dispatch - p_store = p_set)
        define_fixed_operation_constraints(n, sns, "StorageUnit", "p")

        meshed_threshold = kwargs.get("meshed_threshold", 45)
        strongly_meshed_buses = get_strongly_meshed_buses(n, threshold=meshed_threshold)
        weakly_meshed_buses = n.c.buses.names.difference(strongly_meshed_buses)

        if not strongly_meshed_buses.empty and not weakly_meshed_buses.empty:
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
                buses=strongly_meshed_buses,
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
            if isinstance(transmission_losses, bool):
                transmission_losses = {"mode": "secants"}
            elif isinstance(transmission_losses, int):
                equivalent = {"mode": "tangents", "segments": transmission_losses}
                warnings.warn(
                    "Passing an int for `transmission_losses` is deprecated "
                    "and will be removed in PyPSA 2.0. Explicitly pass "
                    f"{equivalent} (current behavior) or use the new "
                    "secant-based losses via `transmission_losses=True`.",
                    FutureWarning,
                    stacklevel=2,
                )
                transmission_losses = {
                    "mode": "tangents",
                    "segments": transmission_losses,
                }
            # Don't mutate passed dict
            transmission_losses = dict(transmission_losses)
            mode = transmission_losses.pop("mode", "secants")
            if mode == "tangents" and "segments" not in transmission_losses:
                msg = (
                    "The 'tangents' mode requires a 'segments' key, e.g. "
                    "transmission_losses={'mode': 'tangents', 'segments': 3}"
                )
                raise ValueError(msg)

            for c in n.passive_branch_components:
                if mode == "secants":
                    define_secant_loss_constraints(n, sns, c, **transmission_losses)
                elif mode == "tangents":
                    define_tangent_loss_constraints(n, sns, c, **transmission_losses)
                else:
                    msg = f"Unknown transmission_losses mode: {mode!r}"
                    raise ValueError(msg)

        # Define global constraints
        define_primary_energy_limit(n, sns)
        define_transmission_expansion_cost_limit(n, sns)
        define_transmission_volume_expansion_limit(n, sns)
        define_tech_capacity_expansion_limit(n, sns)
        define_operational_limit(n, sns)
        define_nominal_constraints_per_bus_carrier(n, sns)
        define_growth_limit(n, sns)

        define_objective(n, sns, include_objective_constant)

        return n.model

    def solve_model(
        self,
        extra_functionality: Callable | None = None,
        solver_name: str | None = None,
        solver_options: dict | None = None,
        log_to_console: bool | None = None,
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
        solver_name : str | None, default=None
            Name of the solver to use. Defaults to module wide option
            (default: 'highs'). See `https://go.pypsa.org/options-params` for more
            information.
        solver_options : dict | None, default=None
            Keyword arguments used by the solver. Defaults to module wide option
            (default: {}). Can also be passed via `**kwargs`. See
            `https://go.pypsa.org/options-params` for more information.
        log_to_console : bool, optional
            Whether the solver prints its progress to the console. Passed as a
            solver option to linopy's `Model.solve()` method. When None,
            solver default behavior is used. Note: not all solvers support
            this option (e.g. HiGHS does, CPLEX does not). See
            `https://go.pypsa.org/options-params` for more information.
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
            [linopy.constants.SolverStatus](https://linopy.readthedocs.io/en/latest/generated/linopy.constants.SolverStatus.html)
        condition : str
            The termination condition of the optimization, either
            "optimal" or one of the codes listed in
            [linopy.constants.TerminationCondition](https://linopy.readthedocs.io/en/latest/generated/linopy.constants.TerminationCondition.html)

        """
        # Handle default parameters from options
        if solver_options is None:
            solver_options = options.params.optimize.solver_options.copy()
        if solver_name is None:
            solver_name = options.params.optimize.solver_name
        if log_to_console is None:
            log_to_console = options.params.optimize.log_to_console

        n = self._n
        if extra_functionality:
            extra_functionality(n, n.snapshots)
        m = n.model
        if log_to_console is not None:
            kwargs["log_to_console"] = log_to_console
        status, condition = m.solve(
            solver_name=solver_name,
            **solver_options,
            **kwargs,
        )

        if status == "ok":
            self._n.optimize.assign_solution()
            self._n.optimize.assign_duals(assign_all_duals)
            self._n.optimize.post_processing()

        # Optional runtime verification
        if options.debug.runtime_verification:
            _assert_data_integrity(self._n)

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

            # Skip auxiliary CVaR variables
            if name.startswith("CVaR"):
                continue

            # Log variables without component-attribute naming
            if "-" not in name:
                # Custom variables might not contain a dash
                logger.info(
                    "The variable '%s' could not be mapped to the network component because it does not include the symbol '-'.",
                    name,
                )
                continue

            _c_name, attr = name.split("-", 1)
            if not hasattr(n.c, _c_name):
                # Custom variables might correspond to a designated component
                logger.info(
                    "The variable '%s' could not be mapped to the network component because the component '%s' does not exist.",
                    name,
                    _c_name,
                )
                continue
            c = n.c[_c_name]
            df = _from_xarray(sol, c)

            if "snapshot" in sol.dims:
                if c.name in n.passive_branch_components and attr == "s":
                    _set_dynamic_data(n, c.name, "p0", df)
                    _set_dynamic_data(n, c.name, "p1", -df)

                elif c.name == "Link" and attr == "p":
                    _set_dynamic_data(n, c.name, "p0", df)

                    for i in ["1"] + n.c.links.additional_ports:
                        i_suffix = "" if i == "1" else i
                        eff = n.get_switchable_as_dense(
                            "Link", f"efficiency{i_suffix}", sns
                        )
                        port_df = -df * eff
                        # For delayed links, time shift the p variable so that output at
                        # snapshot t uses the input from the source snapshot s(t)
                        delay_weightings = n.snapshot_weightings.generators.loc[sns]
                        delay_col = f"delay{i_suffix}"
                        cyclic_col = f"cyclic_delay{i_suffix}"
                        link_static = c.static
                        if delay_col in link_static.columns:
                            delayed = link_static[link_static[delay_col] > 0]
                            if cyclic_col in link_static.columns:
                                grp_cols = [delay_col, cyclic_col]
                            else:
                                delayed = delayed.assign(_cyclic=True)
                                grp_cols = [delay_col, "_cyclic"]
                            for (d, cyc), grp in delayed.groupby(grp_cols):
                                cols = grp.index
                                src_snapshot_pos, valid = (
                                    Links.get_delay_source_indexer(
                                        sns,
                                        delay_weightings,
                                        int(d),
                                        bool(cyc),
                                    )
                                )
                                delayed_values = port_df[cols].to_numpy()[
                                    src_snapshot_pos, :
                                ]
                                delayed_values[~valid, :] = 0.0
                                port_df[cols] = delayed_values
                        _set_dynamic_data(n, c.name, f"p{i}", port_df)
                        c.dynamic[f"p{i}"].loc[
                            sns, c.static.index[c.static[f"bus{i}"] == ""]
                        ] = float(c.defaults.loc[f"p{i}", "default"])

                else:
                    _set_dynamic_data(n, c.name, attr, df)
            # Ignore `n_mod`
            elif attr == "n_mod":
                pass
            else:
                c.static.update(df.rename(attr + "_opt"), overwrite=True)

        # If nominal capacity was no variable set optimal value to nominal
        for c_name, attr in lookup.query("nominal").index:
            c = n.components[c_name]
            fix_i = c.fixed
            if n.has_scenarios:
                fix_i = pd.MultiIndex.from_product([n.scenarios, fix_i])
            if not fix_i.empty:
                c.static.loc[fix_i, f"{attr}_opt"] = c.static.loc[fix_i, attr]

        # Recalculate storageunit net dispatch
        storage_units = n.c.storage_units
        if not storage_units.empty:
            storage_units.dynamic["p"] = (
                storage_units.dynamic["p_dispatch"] - storage_units.dynamic["p_store"]
            )

        n._objective = m.objective.value

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
            # Parse constraint name into component and attribute
            # Dual variable doesn't have a designated component are ignored

            # Split constraint name into component and attribute
            # GlobalConstraints refer instead to the component name, e.g. "GlobalConstraint-X"
            try:
                prefix, suffix = constraint_name.split("-", 1)
                c = self._n.components[prefix]
            except (ValueError, KeyError):
                unassigned_constraints.append(constraint_name)
                continue

            # Add placeholder for custom constraints, marked as GlobalConstraint
            # TODO This should go to an actual custom constraint
            if (
                c.name == "GlobalConstraint"
                and suffix not in c.static.index
                and assign_all_duals
            ):
                if c.has_scenarios:
                    msg = (
                        "Dual values for custom constraints with scenarios are not "
                        "yet supported for stochastic optimization."
                    )
                    raise NotImplementedError(msg)
                else:
                    c.static.loc[suffix] = None

            # Dynamic duals (constraints with snapshot dimension)
            if "snapshot" in constraint.dual.dims:
                # Get dual from constraint as formatted pandas DataFrame
                dual_df = _from_xarray(constraint.dual, c)

                # Standard components: extract last part after final dash
                # e.g., "Line-s-upper" -> "upper", "Generator-p-lower" -> "lower"
                try:
                    dual_spec = suffix.rsplit("-", 1)[-1]
                except ValueError:
                    dual_spec = suffix
                # Don't try to split GlobalConstraint names, since they refer to
                # component name instead of attribute name
                if c.name == "GlobalConstraint":
                    dual_spec = suffix

                # Handle special cases for dual attribute name

                # 1. Nodal balance duals become marginal prices
                if suffix.endswith("nodal_balance"):
                    dual_attr_name = "marginal_price"

                # Standard case: assign as "mu_<spec>"
                # (e.g., "mu_upper", "mu_generation_limit_dynamic")
                elif assign_all_duals or f"mu_{dual_spec}" in c.static:
                    dual_attr_name = f"mu_{dual_spec}"

                # Duals that don't have a placeholder are ignored
                else:
                    unassigned_constraints.append(constraint_name)
                    continue

                # Assign dynamic duals to component
                _set_dynamic_data(self._n, c.name, dual_attr_name, dual_df)

            # SCALAR DUALS (constraints without snapshot dimension)
            # else:
            elif c.name == "GlobalConstraint" and suffix in c.static.index:
                if c.has_scenarios:
                    raise NotImplementedError()

                c.static.loc[suffix, "mu"] = constraint.dual

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

        check_big_m_exceeded(n)

        # correct prices with objective weightings
        if n._multi_invest:
            period_weighting = n.investment_period_weightings.objective
            weightings = n.snapshot_weightings.objective.mul(
                period_weighting, level=0, axis=0
            ).loc[sns]
        else:
            weightings = n.snapshot_weightings.objective.loc[sns]

        n.c.buses.dynamic.marginal_price.loc[sns] = (
            n.c.buses.dynamic.marginal_price.loc[sns].divide(weightings, axis=0)
        )

        # load
        if len(n.loads):
            _set_dynamic_data(
                n, "Load", "p", n.get_switchable_as_dense("Load", "p_set", sns)
            )

        # line losses
        if "Line-loss" in n.model.variables:
            losses = n.model["Line-loss"].solution.to_pandas()
            n.c.lines.dynamic.p0 += losses / 2
            n.c.lines.dynamic.p1 += losses / 2

        # recalculate injection
        ca = [
            ("Generator", "p", "bus"),
            ("Store", "p", "bus"),
            ("Load", "p", "bus"),
            ("StorageUnit", "p", "bus"),
            ("Link", "p0", "bus0"),
            ("Link", "p1", "bus1"),
        ]
        ca.extend([("Link", f"p{i}", f"bus{i}") for i in n.c.links.additional_ports])

        def sign(c: str) -> int:
            return n.c[c].static.get("sign", -1)  # -1 is the sign for 'Link'

        n.c.buses.dynamic.p = (
            pd.concat(
                [
                    n.c[c]
                    .dynamic[attr]
                    .mul(sign(c))
                    .rename(columns=n.c[c].static[group], level="name")
                    for c, attr, group in ca
                ],
                axis=1,
            )
            .T.groupby(level=0)
            .sum()
            .T.reindex(columns=n.c.buses.static.index, fill_value=0.0)
        )

        if not n.has_scenarios:

            def v_ang_for_(sub: SubNetwork) -> pd.DataFrame:
                buses_i = sub.buses_o
                if len(buses_i) == 1:
                    return pd.DataFrame(0, index=sns, columns=buses_i)
                sub.calculate_B_H(skip_pre=True)
                Z = pd.DataFrame(np.linalg.pinv((sub.B).todense()), buses_i, buses_i)
                Z -= Z[sub.slack_bus]
                return n.c.buses.dynamic.p.reindex(columns=buses_i) @ Z

            # TODO: if multi investment optimization, the network topology is not the necessarily the same,
            # i.e. one has to iterate over the periods in order to get the correct angles.

            # Determine_network_topology is not necessarily called (only if KVL was assigned)
            if n.c.sub_networks.static.empty:
                n.determine_network_topology()

            # Calculate voltage angles (only needed for power flow)
            if "obj" in n.c.sub_networks.static:
                n.c.buses.dynamic.v_ang = pd.concat(
                    [v_ang_for_(sub) for sub in n.c.sub_networks.static.obj], axis=1
                ).reindex(columns=n.c.buses.static.index, fill_value=0.0)

    def fix_optimal_capacities(self) -> None:
        """Fix capacities of extendable assets to optimized capacities.

        Use this function when a capacity expansion optimization was
        already performed and a operational optimization should be done
        afterwards.
        """
        n = self._n
        for c, attr in nominal_attrs.items():
            c = n.components[c]
            ext_i = c.extendables.difference(c.inactive_assets)
            c.static.loc[ext_i, attr] = c.static.loc[ext_i, attr + "_opt"]
            c.static[attr + "_extendable"] = False

    def fix_optimal_dispatch(self) -> None:
        """Fix dispatch of all assets to optimized values.

        Use this function when the optimal dispatch should be used as an
        starting point for power flow calculation (`Network.pf`).
        """
        for c in self._n.one_port_components:
            self._n.components[c].dynamic.p_set = self._n.components[c].dynamic.p
        for c in self._n.controllable_branch_components:
            self._n.components[c].dynamic.p_set = self._n.components[c].dynamic.p0

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
        if "Load" not in self._n.c.carriers.static.index:
            self._n.add("Carrier", "Load")
        if buses is None:
            buses = self._n.c.buses.static.index

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
