# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Explain the names that the declarative formulation removed in PyPSA 1.5.0."""

from collections.abc import Callable
from typing import NoReturn

REMOVED_IN = "1.5.0"
GUIDE = "https://docs.pypsa.org/latest/user-guide/optimization/custom-constraints/#overriding-the-formulation"
CONSTANT = "This internal constant has no replacement."

REPLACEMENTS: dict[str, dict[str, str]] = {
    "pypsa.optimization.constraints": {
        "define_operational_constraints_for_non_extendables": "Its constraints are `<Component>_fix_<attr>_lower` and `_upper`, as `Generator_fix_p_upper` or `Line_fix_s_upper`.",
        "define_operational_constraints_for_extendables": "Its constraints are `<Component>_ext_<attr>_lower` and `_upper`, as `Generator_ext_p_upper` or `Line_ext_s_upper`.",
        "define_operational_constraints_for_committables": "Its constraints are `<Component>_com_*`, as `Generator_com_p_upper` (traits/commitment.yaml).",
        "define_maintenance_constraints": "Its constraints are `<Component>_maint*`, as `Generator_maint_window` (traits/maintenance.yaml).",
        "define_nominal_constraints_for_extendables": "Its constraints are `<Component>_ext_<nom>_lower` and `_upper`, as `Generator_ext_p_nom_upper` (traits/capacity.yaml).",
        "define_ramp_limit_constraints": "Its constraints are `<Component>_p_ramp_limit_*`, as `Generator_p_ramp_limit_up` (traits/ramping.yaml).",
        "define_nodal_balance_constraints": "Its constraints are `Bus_nodal_balance` (system.yaml).",
        "define_kirchhoff_voltage_constraints": "Its constraints are `Kirchhoff_Voltage_Law` (system.yaml).",
        "define_voltage_angle_constraints": "Its constraints are `Line_v_ang_lower`, `Line_v_ang_upper` and the `Transformer_v_ang_*` entries (components/line.yaml, components/transformer.yaml).",
        "define_fixed_nominal_constraints": "Its constraints are `<Component>_<nom>_set`, as `Generator_p_nom_set` (traits/capacity.yaml).",
        "define_modular_constraints": "Its constraints are `<Component>_<nom>_modularity`, as `Generator_p_nom_modularity` (traits/capacity.yaml).",
        "define_committability_variables_constraints_with_fixed_upper_limit": "Its constraints are `<Component>_status_p_fixed_upper`, `_start_up_p_fixed_upper` and `_shut_down_p_fixed_upper` (traits/commitment.yaml).",
        "define_committability_variables_constraints_with_variable_upper_limit": "Its constraints are `<Component>_status_<nom>_variable_upper`, `_start_up_<nom>_variable_upper` and `_shut_down_<nom>_variable_upper` (traits/commitment.yaml).",
        "define_fixed_operation_constraints": "Its constraints are `<Component>_<attr>_set`, as `Generator_p_set` or `Store_e_set`.",
        "define_storage_unit_constraints": "Its constraints are `StorageUnit_energy_balance` (components/storage_unit.yaml).",
        "define_store_constraints": "Its constraints are `Store_energy_balance` (components/store.yaml).",
        "define_tangent_loss_constraints": "Its constraints are `<Component>_loss_upper`, `_loss_tangents_pos` and `_loss_tangents_neg`, as `Line_loss_upper` (traits/passive_branch.yaml).",
        "define_secant_loss_constraints": "Its constraints are `<Component>_loss_upper`, `_loss_tangents_pos` and `_loss_tangents_neg`, built from secants, as `Line_loss_upper` (traits/passive_branch.yaml).",
        "define_total_supply_constraints": "Its constraints are `<Component>_e_sum_min` and `_e_sum_max`, as `Generator_e_sum_max` (traits/dispatch.yaml).",
        "DEG2RAD": CONSTANT,
        "PHASE_SHIFT": CONSTANT,
        "lookup": CONSTANT,
    },
    "pypsa.optimization.variables": {
        "define_operational_variables": "Its variables are `<Component>_<attr>`, as `Generator_p`, `Line_s` or `Store_e`.",
        "define_status_variables": "Its variables are `<Component>_status` (traits/commitment.yaml).",
        "define_start_up_variables": "Its variables are `<Component>_start_up` (traits/commitment.yaml).",
        "define_shut_down_variables": "Its variables are `<Component>_shut_down` (traits/commitment.yaml).",
        "define_maintenance_variables": "Its variables are `<Component>_maintenance` (traits/maintenance.yaml).",
        "define_maintenance_start_variables": "Its variables are `<Component>_maintenance_start` (traits/maintenance.yaml).",
        "define_maintenance_capacity_variables": "Its variables are `<Component>_maintenance_capacity` (traits/maintenance.yaml).",
        "define_maintenance_status_variables": "Its variables are `<Component>_maintenance_status` (traits/maintenance.yaml).",
        "define_nominal_variables": "Its variables are `<Component>_<nom>_ext`, as `Generator_p_nom_ext` (traits/capacity.yaml).",
        "define_modular_variables": "Its variables are `<Component>_n_mod` (traits/capacity.yaml).",
        "define_spillage_variables": "Its variables are `<Component>_spill`, as `StorageUnit_spill` (traits/storage.yaml).",
        "define_phase_shift_variables": "Its variables are `Transformer_phase_shift` (components/transformer.yaml).",
        "define_loss_variables": "Its variables are `<Component>_loss`, as `Line_loss` (traits/passive_branch.yaml).",
        "define_cvar_variables": "Its variables are `CVaR`, `CVaR_a` and `CVaR_theta` (system.yaml).",
    },
    "pypsa.optimization.global_constraints": {
        "define_tech_capacity_expansion_limit": "Its constraints are part of `GlobalConstraint_upper`, `_lower` and `_fix`, which sum the expression `tech_capacity_expansion` (system.yaml).",
        "define_growth_limit": "Its constraints are `Carrier_growth_limit` (system.yaml).",
        "define_primary_energy_limit": "Its constraints are part of `GlobalConstraint_upper`, `_lower` and `_fix`, which sum the expression `primary_energy` (system.yaml).",
        "define_operational_limit": "Its constraints are part of `GlobalConstraint_upper`, `_lower` and `_fix`, which sum the expression `operational_limit` (system.yaml).",
        "define_transmission_volume_expansion_limit": "Its constraints are part of `GlobalConstraint_upper`, `_lower` and `_fix`, which sum the expression `transmission_volume_expansion` (system.yaml).",
        "define_transmission_expansion_cost_limit": "Its constraints are part of `GlobalConstraint_upper`, `_lower` and `_fix`, which sum the expression `transmission_expansion_cost` (system.yaml).",
    },
    "pypsa.optimization.piecewise": {
        "define_piecewise": "Each curve is a variable `<Component>_<attr>_piecewise`, as `Generator_marginal_cost_piecewise`; pass `piecewise_options` to `n.optimize()` to tune it.",
    },
    "pypsa.optimization.optimize": {
        "define_objective": "It is the section `objective`, which minimises the expression `total_cost` (system.yaml).",
    },
}


def removed_names(module: str) -> Callable[[str], NoReturn]:
    """Return a module `__getattr__` that explains the removed names of `module`.

    Parameters
    ----------
    module : str
        Full name of the module, as `pypsa.optimization.constraints`.

    Returns
    -------
    Callable[[str], NoReturn]
        Raises `ImportError` with a migration hint for a removed name and the
        plain `AttributeError` for any other name.

    """
    replacements = REPLACEMENTS[module]

    def __getattr__(name: str) -> NoReturn:
        if name not in replacements:
            msg = f"module {module!r} has no attribute {name!r}"
            raise AttributeError(msg, name=name)
        msg = (
            f"`{module}.{name}` was removed in PyPSA {REMOVED_IN}. "
            "`n.optimize()` now builds the model from the specification in "
            f"`pypsa/optimization/spec/`. {replacements[name]} To change the model, "
            f"pass `overrides` to `n.optimize()`, see {GUIDE}"
        )
        raise ImportError(msg, name=module)

    return __getattr__
