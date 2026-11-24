<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

Two components are available for modelling storage: [`Store`][pypsa.components.Stores] and [`StorageUnit`][pypsa.components.StorageUnits]. See
the [Store Component](../components/stores.md) and [Storage Unit Component](../components/storage-units.md) descriptions for an overview of
the conceptual differences. The [trait `storage`](formulation/traits/storage.md) states the formulation shared by both.


## Stores

Stores $y$ have three time-dependent variables, the net dispatch $h_{s,t,y}$ in MW (positive when discharging), the charge $h^{+}_{s,t,y}$ in MW and the energy level $e_{s,t,y}$ in MWh. The discharge $h_{s,t,y} + h^{+}_{s,t,y}$ is derived. Only $h_{s,t,y}$ enters the nodal balance.

The energy level $e_{s,t,y}$ is constrained by maximum and minimum energy levels.

For **non-extendable** stores (`e_nom_extendable=False`):

<!-- md:spec-table Store_fix_e_lower Store_fix_e_upper Store_fix_p_lower Store_fix_p_upper Store_fix_p_store_lower Store_fix_p_store_upper -->

where $\mathrm{Store\_e\_nom\_rated}_{s,y}$ is the nominal energy capacity $\hat{e}_{s,y}$ (`e_nom`), or zero where `e_nom` is infinite, $\underline{e}_{s,t,y}$ and $\bar{e}_{s,t,y}$ are time-dependent restrictions on the energy level given per unit of nominal capacity, $\mathrm{Store\_max\_hours}_{s,y}$ (`max_hours`) is the number of hours at full power that fill the store, and $\bar{h}_{s,t,y}$ and $\underline{h}_{s,t,y}$ restrict discharging and charging per unit of the power capacity $\mathrm{Store\_e\_nom\_rated}_{s,y} / \mathrm{Store\_max\_hours}_{s,y}$ (usually $\underline{h}_{s,t,y}$ is negative, -1 by default). The power constraints are only built where `max_hours` is finite; by default (`max_hours` infinite) charging and discharging power are unconstrained. For stores with unit efficiencies, no direction-specific marginal costs and an infinite `cycles_max`, $h^{+}_{s,t,y}$ is fixed to zero ($\mathrm{Store\_charge\_cap\_pu}_{s,t,y} = 0$) and the charging bound applies to $h_{s,t,y}$ directly ($\mathrm{Store\_discharge\_floor\_pu}_{s,t,y} = \underline{h}_{s,t,y} / \mathrm{Store\_max\_hours}_{s,y}$), which avoids a degenerate pair of variables; the outputs `p_dispatch` and `p_store` are then split by sign. Otherwise the discharge floor is zero and the charging bound applies to $h^{+}_{s,t,y}$ ($\mathrm{Store\_charge\_cap\_pu}_{s,t,y} = -\underline{h}_{s,t,y} / \mathrm{Store\_max\_hours}_{s,y}$).

The dual variables of the energy level bounds are stored in `n.stores_t.mu_lower` and `n.stores_t.mu_upper`. The dual variables of the power bounds are only stored in `n.model`.

For **extendable** stores (`e_nom_extendable=True`):

<!-- md:spec-table Store_ext_e_lower Store_ext_e_upper Store_ext_p_lower Store_ext_p_upper Store_ext_p_store_lower Store_ext_p_store_upper -->

where $E_{y}$ is the energy capacity to be optimised.

!!! note "Capacity limits of [`Store`][pypsa.components.Stores] components"

    For handling of capacity limits, see the <!-- md:guide optimization/capacity-limits.md --> section.

The variables can also be fixed to certain values:

<!-- md:spec-table Store_e_set Store_p_set -->

The power and energy variables are related by the **storage consistency** equation:

<!-- md:spec Store_energy_balance -->

which is equivalent to $\eta^{\mathrm{store}} h^{+} - (h + h^{+}) / \eta^{\mathrm{dispatch}}$ in terms of charge and discharge.

Here $\eta^{\mathrm{stand}}_{s,t,y}$ is the share of energy kept over the snapshot after standing losses (e.g. thermal losses in thermal storage), $(1 - \text{standing\_loss})^{w^{\mathrm{s}}_t}$, $\eta^{\mathrm{store}}_{s,t,y}$ and $\eta^{\mathrm{dispatch}}_{s,t,y}$ are the efficiencies for power going into and out of the store, and $w^{\mathrm{s}}_t$ is the snapshot weighting (defaulting to 1 hour). The spillage variable is only created for stores with positive inflow. The dual variable of the storage consistency equation represents the marginal storage value (also known as water value in the hydro-electricity literature) and is stored in `n.stores_t.mu_energy_balance`.

The marginal costs enter the objective as $o_{s,t,y}\, h_{s,t,y} + o^{-}_{s,t,y} (h_{s,t,y} + h^{+}_{s,t,y}) + o^{+}_{s,t,y}\, h^{+}_{s,t,y}$, i.e. `marginal_cost` applies to the net dispatch while `marginal_cost_dispatch` and `marginal_cost_store` apply to one direction each. See the [Store component](../components/stores.md#marginal-costs) description.

Furthermore, there are two options for specifying the energy level before the first snapshot:

1. Set `e_cyclic=False` (default) and the value of `e_initial` in MWh.
2. Set `e_cyclic=True` and the optimisation sets the initial energy to be equal to the final energy level.

<!-- md:spec Store_level_carried_in Store_level_before -->

!!! note "[`Store`][pypsa.components.Stores] cyclicity constraints with multiple investment periods"

    For how storage cyclicity constraints are handled with multiple investment periods, see the [Pathway Planning](pathway-planning.md) section.

### Cycle Limits

Stores may wear out with use. The attribute `cycles_max` limits the number of equivalent full cycles over all snapshots of the optimisation, including all investment periods. One cycle is a discharge equal to the energy capacity. The limit stays linear for extendable capacities. The formulation follows eq. (22) in Cardoso et al. (2018).[^1]

<!-- md:spec-table Store_cycles_max -->

Stores with a finite `cycles_max` always get the separate charging variable $h^{+}_{s,t,y}$, so that the discharge $h_{s,t,y} + h^{+}_{s,t,y}$ is well defined. Where unit efficiencies and no direction-specific marginal costs make simultaneous charging and discharging free, the reported `p_store` and `p_dispatch` are netted and split by sign of $h_{s,t,y}$.

In a rolling horizon optimisation, the cycles of previous windows count against `cycles_max`.

A cycle life $N^\textrm{life}$ over a technical lifetime $L^\textrm{life}$ translates into `cycles_max` $= N^\textrm{life} \cdot Y / L^\textrm{life}$ for an optimisation over $Y$ years.

??? note "Symbols"

    <!-- md:spec-legend Store_fix_e_lower Store_fix_e_upper Store_fix_p_lower Store_fix_p_upper Store_fix_p_store_lower Store_fix_p_store_upper Store_ext_e_lower Store_ext_e_upper Store_ext_p_lower Store_ext_p_upper Store_ext_p_store_lower Store_ext_p_store_upper Store_e_set Store_p_set Store_energy_balance Store_level_carried_in Store_level_before Store_cycles_max -->

## Storage Units

Storage units $v$ have three time-dependent variables, the discharge $h^{-}_{s,t,v}$ in MW, the charge $h^{+}_{s,t,v}$ in MW and the state of charge $\mathit{soc}_{s,t,v}$ in MWh.

With a storage unit the nominal state of charge may not be independently optimised from the nominal power output (they are linked by the maximum hours parameter `max_hours`) and the nominal charging power is linked to the maximum discharging power.

For **non-extendable** storage units (`p_nom_extendable=False`), the charge ($h^{+}_{s,t,v}$), discharge ($h^{-}_{s,t,v}$), and state of charge ($\mathit{soc}_{s,t,v}$) variables are constrained by:

<!-- md:spec-table StorageUnit_fix_p_dispatch_lower StorageUnit_fix_p_dispatch_upper StorageUnit_fix_p_store_lower StorageUnit_fix_p_store_upper StorageUnit_fix_state_of_charge_lower StorageUnit_fix_state_of_charge_upper -->

where the rated capacity `StorageUnit_p_nom_finite` is the nominal power capacity $\hat{h}_{s,v}$, $\bar{h}_{s,t,v}$ is a time-dependent restriction on the discharge per unit of nominal capacity, $\underline{h}_{s,t,v}$ is a time-dependent restriction on the maximum charge per unit of nominal capacity (usually negative, -1 by default), and `max_hours` is the number of hours at nominal power that fill the state of charge.

For **extendable** storage units (`p_nom_extendable=True`), the charge and discharge variables are constrained by:

<!-- md:spec-table StorageUnit_ext_p_dispatch_lower StorageUnit_ext_p_dispatch_upper StorageUnit_ext_p_store_lower StorageUnit_ext_p_store_upper StorageUnit_ext_state_of_charge_lower StorageUnit_ext_state_of_charge_upper -->

where $H_{v}$ is the power capacity to be optimised. The dual variables of these bounds are only stored in `n.model`.

!!! note "Capacity limits of [`StorageUnit`][pypsa.components.StorageUnits] components"

    For handling of capacity limits, see the <!-- md:guide optimization/capacity-limits.md --> section.

All three variables can also be fixed to certain values $\tilde{h}^{-}_{s,t,v}$, $\tilde{h}^{+}_{s,t,v}$, and $\tilde{\mathit{soc}}_{s,t,v}$:

<!-- md:spec-table StorageUnit_p_dispatch_set StorageUnit_p_store_set StorageUnit_state_of_charge_set -->

The dual variable of the fixed state of charge is stored in `n.storage_units_t.mu_state_of_charge_set`. An example use case would be if a storage unit must be empty and/or full every day.

The charging, discharging and state of charge variables are related by:

<!-- md:spec StorageUnit_energy_balance -->

$\eta^{\mathrm{stand}}_{s,t,v}$ is the share of energy kept over the snapshot after standing losses (e.g. thermal losses for thermal storage), $(1 - \text{standing\_loss})^{w^{\mathrm{s}}_t}$. $\eta^{\mathrm{store}}_{s,t,v}$ and $\eta^{\mathrm{dispatch}}_{s,t,v}$ are the efficiencies for power going into and out of the storage unit, and $w^{\mathrm{s}}_t$ is the snapshot weighting for stores (e.g. 1 hour).
The dual variable represents the marginal storage value (also known as water value in the hydro-electricity literature) and is stored in `n.storage_units_t.mu_energy_balance`.

Furthermore, there are two options for specifying the state of charge before the first snapshot:

1. Set `cyclic_state_of_charge=False` (default) and the value of `state_of_charge_initial` in MWh.
2. Set `cyclic_state_of_charge=True` and the optimisation set the initial state of charge to the final state of charge.

<!-- md:spec StorageUnit_level_carried_in StorageUnit_level_before -->

!!! note "[`StorageUnit`][pypsa.components.StorageUnits] cyclicity constraints with multiple investment periods"

    For how store cyclicity constraints are handled with multiple investment periods, see the [Pathway Planning](pathway-planning.md) section.


??? note "Symbols"

    <!-- md:spec-legend StorageUnit_fix_p_dispatch_lower StorageUnit_fix_p_dispatch_upper StorageUnit_fix_p_store_lower StorageUnit_fix_p_store_upper StorageUnit_fix_state_of_charge_lower StorageUnit_fix_state_of_charge_upper StorageUnit_ext_p_dispatch_lower StorageUnit_ext_p_dispatch_upper StorageUnit_ext_p_store_lower StorageUnit_ext_p_store_upper StorageUnit_ext_state_of_charge_lower StorageUnit_ext_state_of_charge_upper StorageUnit_p_dispatch_set StorageUnit_p_store_set StorageUnit_state_of_charge_set StorageUnit_energy_balance StorageUnit_level_carried_in StorageUnit_level_before -->

## Examples


<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Storage Units as Links & Stores**

    ---

    Shows how storage units can be replaced by more fundamental links and stores.

    [:octicons-arrow-right-24: Go to example](../../examples/replace-generator-storage-units-with-store.ipynb)

</div>

[^1]: G. Cardoso, T. Brouhard, N. DeForest, D. Wang, M. Heleno, L. Kotzur (2018), [Battery aging in multi-energy microgrid design using mixed integer linear programming](https://doi.org/10.1016/j.apenergy.2018.09.185), Applied Energy, 231, 1059-1069.
