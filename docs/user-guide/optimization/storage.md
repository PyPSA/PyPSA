<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

Two components are available for modelling storage: [`Store`][pypsa.components.Stores] and [`StorageUnit`][pypsa.components.StorageUnits]. See
the [Store Component](../components/stores.md) and [Storage Unit Component](../components/storage-units.md) descriptions for an overview of
the conceptual differences.


## Stores

Stores have three time-dependent variables, the net dispatch $h_{n,s,t}$ in MW (positive when discharging), the charge $h_{n,s,t}^+$ in MW and the energy level $e_{n,s,t}$ in MWh. The discharge $h_{n,s,t}^- = h_{n,s,t} + h_{n,s,t}^+$ is derived. Only $h_{n,s,t}$ enters the nodal balance.

The energy level $e_{n,s,t}$ is constrained by maximum and minimum energy levels.

For **non-extendable** stores (`e_nom_extendable=False`):

| Constraint | Dual Variable | Name |
|------------|---------------|------|
| $e_{n,s,t} \geq \underline{e}_{n,s,t} \hat{e}_{n,s}$ | `n.stores_t.mu_lower` | `Store-fix-e-lower` |
| $e_{n,s,t} \leq \bar{e}_{n,s,t} \hat{e}_{n,s}$ | `n.stores_t.mu_upper` | `Store-fix-e-upper` |
| $h_{n,s,t}^- \geq 0$ | only in `n.model` | `Store-fix-p-lower` |
| $h_{n,s,t}^- \leq \bar{h}_{n,s,t} \hat{e}_{n,s} / r_{n,s}$ | only in `n.model` | `Store-fix-p-upper` |
| $h_{n,s,t}^+ \geq 0$ | only in `n.model` | `Store-fix-p_store-lower` |
| $h_{n,s,t}^+ \leq - \underline{h}_{n,s,t} \hat{e}_{n,s} / r_{n,s}$ | only in `n.model` | `Store-fix-p_store-upper` |

where $\hat{e}_{n,s}$ is the nominal energy capacity, $\underline{e}_{n,s,t}$ and $\bar{e}_{n,s,t}$ are time-dependent restrictions on the energy level given per unit of nominal capacity, $r_{n,s}$ is the number of hours at full power that fill the store (`max_hours`), and $\bar{h}_{n,s,t}$ and $\underline{h}_{n,s,t}$ restrict discharging and charging per unit of the power capacity $\hat{e}_{n,s} / r_{n,s}$ (usually $\underline{h}_{n,s,t}$ is negative, -1 by default). The power constraints are only built where $r_{n,s}$ is finite; by default ($r_{n,s} = \infty$) charging and discharging power are unconstrained. For stores with unit efficiencies and no direction-specific marginal costs, $h_{n,s,t}^+$ is fixed to zero and the charging bound applies to $h_{n,s,t}$ directly, which avoids a degenerate pair of variables; the outputs `p_dispatch` and `p_store` are then split by sign.

These constraints are set in the function `define_operational_constraints_for_non_extendables()`.

For **extendable** stores (`e_nom_extendable=True`):

| Constraint | Dual Variable | Name |
|------------|---------------|------|
| $e_{n,s,t} \geq \underline{e}_{n,s,t} E_{n,s}$ | `n.stores_t.mu_lower` | `Store-ext-e-lower` |
| $e_{n,s,t} \leq \bar{e}_{n,s,t} E_{n,s}$ | `n.stores_t.mu_upper` | `Store-ext-e-upper` |
| $h_{n,s,t}^- \geq 0$ | only in `n.model` | `Store-ext-p-lower` |
| $h_{n,s,t}^- \leq \bar{h}_{n,s,t} E_{n,s} / r_{n,s}$ | only in `n.model` | `Store-ext-p-upper` |
| $h_{n,s,t}^+ \geq 0$ | only in `n.model` | `Store-ext-p_store-lower` |
| $h_{n,s,t}^+ \leq - \underline{h}_{n,s,t} E_{n,s} / r_{n,s}$ | only in `n.model` | `Store-ext-p_store-upper` |

where $E_{n,s}$ is the energy capacity to be optimised.

These constraints are set in the function `define_operational_constraints_for_extendables()`.

!!! note "Capacity limits of [`Store`][pypsa.components.Stores] components"

    For handling of capacity limits, see the <!-- md:guide optimization/capacity-limits.md --> section.

The variables can also be fixed to certain values:

| Constraint | Dual Variable | Name |
|------------|---------------|------|
| $e_{n,s,t} = \tilde{e}_{n,s,t}$ | only in `n.model` | `Store-e_set` |
| $h_{n,s,t} = \tilde{h}_{n,s,t}$ | only in `n.model` | `Store-p_set` |

These constraints are set in the function `define_fixed_operation_constraints()`.

The power and energy variables are related by the **storage consistency** equation (`Store-energy_balance`):

$$\begin{gather*}e_{n,s,t} = \eta_{\textrm{stand},n,s}^{w_t^s} e_{n,s,t-1} - \eta^{-1}_{\textrm{dispatch};n,s} w_t^s h_{n,s,t} + \left(\eta_{\textrm{store};n,s} - \eta^{-1}_{\textrm{dispatch};n,s}\right) w_t^s h_{n,s,t}^+ \\ + w_t^s\textrm{inflow}_{n,s,t} - w_t^s\textrm{spillage}_{n,s,t} \quad \leftrightarrow \quad \lambda_{n,s,t}^\textrm{MSV} \end{gather*}$$

which is equivalent to $\eta_{\textrm{store}} h^+ - \eta^{-1}_{\textrm{dispatch}} h^-$ in terms of charge and discharge.

where $\eta_{\textrm{stand},n,s}$ represents the storage efficiency after accounting for standing losses (e.g. thermal losses in thermal storage) per hour, $\eta_{\textrm{store};n,s}$ and $\eta_{\textrm{dispatch};n,s}$ are the efficiencies for power going into and out of the store, and $w_t^s$ is the snapshot weighting (defaulting to 1 hour). The spillage variable $0 \leq \textrm{spillage}_{n,s,t} \leq \textrm{inflow}_{n,s,t}$ is only created for stores with positive inflow. The dual variable $\lambda_{n,s,t}^\textrm{MSV}$ represents the marginal storage value (also known as water value in the hydro-electricity literature).

These constraints are set in the function `define_store_constraints()`.

The marginal costs enter the objective as $o_{n,s,t} h_{n,s,t} + o^-_{n,s,t} h_{n,s,t}^- + o^+_{n,s,t} h_{n,s,t}^+$, i.e. `marginal_cost` applies to the net dispatch while `marginal_cost_dispatch` and `marginal_cost_store` apply to one direction each. See the [Store component](../components/stores.md#marginal-costs) description.

Furthermore, there are two options for specifying the initial energy level $e_{n,s,t=-1}$:

1. Set `e_cyclic=False` (default) and the value of `e_initial` in MWh.

$$e_{n,s,t=-1} = e_{n,s,\textrm{initial}}$$

2. Set `e_cyclic=True` and the optimisation sets the initial energy to be equal to the final energy level.

$$e_{n,s,t=-1} = e_{n,s,t=|T|-1}$$

!!! note "[`Store`][pypsa.components.Stores] cyclicity constraints with multiple investment periods"

    For how storage cyclicity constraints are handled with multiple investment periods, see the [Pathway Planning](pathway-planning.md) section.

??? note "Mapping of symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $h_{n,s,t}$ | `n.stores_t.p` | Decision Variable |
    | $h_{n,s,t}^+$ | `n.stores_t.p_store` | Decision Variable |
    | $h_{n,s,t}^-$ | `n.stores_t.p_dispatch` | Derived Output |
    | $e_{n,s,t}$ | `n.stores_t.e` | Decision Variable |
    | $\textrm{spillage}_{n,s,t}$ | `n.stores_t.spill` | Decision Variable |
    | $E_{n,s}$ | `n.stores.e_nom_opt` | Decision Variable |
    | $\lambda_{n,s,t}^\textrm{MSV}$ | `n.stores_t.mu_energy_balance` | Dual Variable |
    | $\hat{e}_{n,s}$ | `n.stores.e_nom` | Parameter |
    | $r_{n,s}$ | `n.stores.max_hours` | Parameter |
    | $\underline{e}_{n,s,t}$ | `n.stores_t.e_min_pu` | Parameter |
    | $\bar{e}_{n,s,t}$ | `n.stores_t.e_max_pu` | Parameter |
    | $\underline{h}_{n,s,t}$ | `n.stores_t.p_min_pu` | Parameter |
    | $\bar{h}_{n,s,t}$ | `n.stores_t.p_max_pu` | Parameter |
    | $\tilde{e}_{n,s,t}$ | `n.stores_t.e_set` | Parameter |
    | $\tilde{h}_{n,s,t}$ | `n.stores_t.p_set` | Parameter |
    | $\textrm{inflow}_{n,s,t}$ | `n.stores_t.inflow` | Parameter |
    | $\eta_{\textrm{stand};n,s}$ | `n.stores.standing_loss` | Parameter |
    | $\eta_{\textrm{store};n,s}$ | `n.stores.efficiency_store` | Parameter |
    | $\eta_{\textrm{dispatch};n,s}$ | `n.stores.efficiency_dispatch` | Parameter |
    | $o_{n,s,t}$ | `n.stores_t.marginal_cost` | Parameter |
    | $o^-_{n,s,t}$ | `n.stores_t.marginal_cost_dispatch` | Parameter |
    | $o^+_{n,s,t}$ | `n.stores_t.marginal_cost_store` | Parameter |
    | $w_t^s$ | `n.snapshot_weightings.stores` | Parameter |

## Storage Units

Storage units have three time-dependent variables, the discharge $h_{n,s,t}^-$ in MW, the charge $h_{n,s,t}^+$ in MW and the state of charge $soc_{n,s,t}$ in MWh.

With a storage unit the nominal state of charge may not be independently optimised from the nominal power output (they are linked by the maximum hours parameter `max_hours`) and the nominal charging power is linked to the maximum discharging power.

For **non-extendable** storage units (`p_nom_extendable=False`), the charge ($h_{n,s,t}^+$), discharge ($h_{n,s,t}^-$), and state of charge ($soc_{n,s,t}$) variables are constrained by:

| Constraint | Dual Variable | Name |
|------------|---------------|------|
| $h_{n,s,t}^- \geq 0$ | only in `n.model` | `StorageUnit-p_dispatch-lower` |
| $h_{n,s,t}^- \leq \bar{h}_{n,s,t} \hat{h}_{n,s}$ | only in `n.model` | `StorageUnit-p_dispatch-upper` |
| $h_{n,s,t}^+ \geq 0$ | only in `n.model` | `StorageUnit-p_store-lower` |
| $h_{n,s,t}^+ \leq - \underline{h}_{n,s,t} \hat{h}_{n,s}$ | only in `n.model` | `StorageUnit-p_store-upper` |
| $soc_{n,s,t} \geq 0$ | only in `n.model` | `StorageUnit-state_of_charge-lower` |
| $soc_{n,s,t} \leq r_{n,s} \hat{h}_{n,s}$ | only in `n.model` | `StorageUnit-state_of_charge-upper` |


where $\hat{h}_{n,s}$ is the nominal power capacity, $\bar{h}_{n,s,t}$ is a time-dependent restriction on the discharge per unit of nominal capacity, $\underline{h}_{n,s,t}$ is a time-dependent restriction on the maximum charge per unit of nominal capacity (usually negative, -1 by default), and $r_{n,s}$ is the number of hours at nominal power that fill the state of charge.

These constraints are set in the function `define_operational_constraints_for_non_extendables()`.

For **extendable** storage units (`p_nom_extendable=True`), the charge and discharge variables are constrained by:

| Constraint | Dual Variable | Name |
|------------|---------------|------|
| $h_{n,s,t}^- \geq 0$ | only in `n.model` | `StorageUnit-ext-p_dispatch-lower` |
| $h_{n,s,t}^- \leq \bar{h}_{n,s,t}H_{n,s}$ | only in `n.model` | `StorageUnit-ext-p_dispatch-upper` |
| $h_{n,s,t}^+ \geq 0$ | only in `n.model` | `StorageUnit-ext-p_store-lower` |
| $h_{n,s,t}^+ \leq - \underline{h}_{n,s,t} H_{n,s}$ | only in `n.model` | `StorageUnit-ext-p_store-upper` |

where $H_{n,s}$ is the power capacity to be optimised.

These constraints are set in the function `define_operational_constraints_for_extendables()`.

!!! note "Capacity limits of [`StorageUnit`][pypsa.components.StorageUnits] components"

    For handling of capacity limits, see the <!-- md:guide optimization/capacity-limits.md --> section.

All three variables can also be fixed to certain values $\tilde{h}_{n,s,t}^-$, $\tilde{h}_{n,s,t}^+$, and $\tilde{soc}_{n,s,t}$:

| Constraint | Dual Variable | Name |
|------------|---------------|------|
| $h_{n,s,t}^- = \tilde{h}_{n,s,t}^-$ | only in `n.model` | `StorageUnit-p_dispatch_set` |
| $h_{n,s,t}^+ = \tilde{h}_{n,s,t}^+$ | only in `n.model` | `StorageUnit-p_store_set` |
| $soc_{n,s,t} = \tilde{soc}_{n,s,t}$ | `n.storage_units_t.mu_state_of_charge_set` |`StorageUnit-state_of_charge_set` |

An example use case would be if a storage unit must be empty and/or full every day.

These constraints are set in the function `define_fixed_operation_constraints()`.

The charging, discharging and state of charge variables are related by (`StorageUnit-energy_balance`):

$$\begin{gather*}soc_{n,s,t} = \eta_{\textrm{stand};n,s}^{w_t^s} soc_{n,s,t-1}\\
+ \eta_{\textrm{store};n,s} w_t^s h_{n,s,t}^+ -  \eta^{-1}_{\textrm{dispatch};n,s} w_t^s h_{n,s,t}^- \\ + w_t^s\textrm{inflow}_{n,s,t} - w_t^s\textrm{spillage}_{n,s,t} \quad \leftrightarrow \quad \lambda_{n,s,t}^\textrm{MSV} \end{gather*}$$

$\eta_{\textrm{stand};n,s}$ is the standing efficiency (e.g. due to thermal losses for thermal storage). $\eta_{\textrm{store};n,s}$ and $\eta_{\textrm{dispatch};n,s}$ are the efficiencies for power going into and out of the storage unit, and $w_t^s$ is the snapshot weighting for stores (e.g. 1 hour).
The dual variable $\lambda_{n,s,t}^\textrm{MSV}$ represents the marginal storage value (also known as water value in the hydro-electricity literature).

These constraints are set in the function `define_storage_unit_constraints()`.

Furthermore, there are two options for specifying the initial state of charge $soc_{n,s,t=-1}$:

1. Set `cyclic_state_of_charge=False` (default) and the value of `state_of_charge_initial` in MWh.

2. Set `cyclic_state_of_charge=True` and the optimisation set the initial state of charge to the final state of charge.

$$soc_{n,s,t=-1} = soc_{n,s,t=|T|-1}$$

!!! note "[`StorageUnit`][pypsa.components.StorageUnits] cyclicity constraints with multiple investment periods"

    For how store cyclicity constraints are handled with multiple investment periods, see the [Pathway Planning](pathway-planning.md) section.


??? note "Mapping of symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $h_{n,s,t}^+$ | `n.stores_t.p_dispatch` | Decision Variable |
    | $h_{n,s,t}^-$ | `n.stores_t.p_store` | Decision Variable |
    | $soc_{n,s,t}$ | `n.storage_units_t.state_of_charge` | Decision Variable |
    | $H_{n,s}$ | `n.storage_units.p_nom_opt` | Decision Variable |
    | $\textrm{inflow}_{n,s,t}$ | `n.storage_units_t.inflow` | Decision Variable |
    | $\textrm{spillage}_{n,s,t}$ | `n.storage_units_t.spillage` | Decision Variable |
    | $\lambda_{n,s,t}^\textrm{MSV}$ | `n.storage_units_t.mu_energy_balance` | Dual Variable |
    | $\bar{h}_{n,s}$ | `n.storage_units.p_nom` | Parameter |
    | $\underline{h}_{n,s,t}$ | `n.storage_units_t.p_min_pu` | Parameter |
    | $\bar{h}_{n,s,t}$ | `n.storage_units_t.p_max_pu` | Parameter |
    | $r_{n,s}$ | `n.storage_units.max_hours` | Parameter |
    | $\tilde{h}_{n,s,t}^-$ | `n.storage_units_t.p_dispatch_set` | Parameter |
    | $\tilde{h}_{n,s,t}^+$ | `n.storage_units_t.p_store_set` | Parameter |
    | $\tilde{soc}_{n,s,t}$ | `n.storage_units_t.state_of_charge_set` | Parameter |
    | $\eta_{\textrm{stand};n,s}$ | `n.storage_units.standing_loss` | Parameter |
    | $\eta_{\textrm{store};n,s}$ | `n.storage_units.store_efficiency` | Parameter |
    | $\eta_{\textrm{dispatch};n,s}$ | `n.storage_units.dispatch_efficiency` | Parameter |
    | $w_t^s$ | `n.snapshot_weightings.stores` | Parameter |

## Examples


<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Storage Units as Links & Stores**

    ---

    Shows how storage units can be replaced by more fundamental links and stores.

    [:octicons-arrow-right-24: Go to example](../../examples/replace-generator-storage-units-with-store.ipynb)

</div>
