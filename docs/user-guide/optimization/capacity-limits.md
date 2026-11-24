<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Capacity Limits

!!! note "Process formulation"

    Unless stated otherwise, all formulations shown for [`Link`][pypsa.components.Links] apply analogously to [`Process`][pypsa.components.Processes] by replacing $(f,\hat f,F,k)$ with $(r,\hat r,R,m)$ and `Link-*` with `Process-*` in the constraint names.

## Upper and Lower Bounds

If the nominal capacity of a component is also the subject of optimisation (e.g. with decision variable $G_{i}$ for generators), limits on the installable capacity may also be introduced (e.g. $\underline{G}_{i}$ and $\bar{G}_{i}$). The [trait `extendable`](formulation/traits/extendable.md) states them:

<!-- md:spec-table Generator_ext_p_nom_lower Generator_ext_p_nom_upper Link_ext_p_nom_lower Link_ext_p_nom_upper Process_ext_p_nom_lower Process_ext_p_nom_upper Line_ext_s_nom_lower Line_ext_s_nom_upper Transformer_ext_s_nom_lower Transformer_ext_s_nom_upper Store_ext_e_nom_lower Store_ext_e_nom_upper StorageUnit_ext_p_nom_lower StorageUnit_ext_p_nom_upper -->

The dual variables of the lower and upper bounds are only stored in `n.model`, e.g. `n.model.constraints["Generator-ext-p_nom-upper"].dual`.

In a [stochastic network](stochastic.md), all scenarios share the built capacity. The bounds may differ by scenario, but only the tightest applies: the largest `p_nom_min` and the smallest `p_nom_max` over the scenarios. A fixed capacity `p_nom_set` must be the same in all scenarios.

## Committable and Extendable Components

When components are both **committable** (`committable=True`) and **extendable** (e.g. `p_nom_extendable=True`), the optimizer co-optimizes both capacity expansion and operational unit commitment. The challenge is that the upper bound on dispatch becomes nonlinear: $g_{i,s,t} \leq u_{i,s,t} \cdot G_{i}$ (the product of binary status $u$ and continuous capacity $G$). To maintain a Mixed-Integer Linear Programme (MILP), PyPSA uses a **big-M formulation** that replaces this nonlinear constraint with linear constraints:

=== "Generator"

    <!-- md:spec-table Generator_com_ext_p_upper_bigM Generator_com_ext_p_upper_cap Generator_com_ext_p_lower Generator_com_ext_p_lower_nonneg -->

=== "Link"

    <!-- md:spec-table Link_com_ext_p_upper_bigM Link_com_ext_p_upper_cap Link_com_ext_p_lower Link_com_ext_p_lower_nonneg -->

=== "Process"

    <!-- md:spec-table Process_com_ext_p_upper_bigM Process_com_ext_p_upper_cap Process_com_ext_p_lower Process_com_ext_p_lower_nonneg -->

where $M_{i,s}$ is a sufficiently large constant (the "big-M") and $\bar{g}$, $\bar{f}$, $\bar{r}$ are the maximum per-unit dispatch limits (`p_max_pu`, default 1). When the status $u = 0$, the big-M constraint forces dispatch to zero. When $u = 1$, the lower bound becomes $g \geq \underline{g} \cdot G$ (minimum part-load) and the upper bound is $g \leq \bar{g} \cdot G$ (capacity limit).

### Big-M Parameter Configuration

The big-M constant must be large enough to not constrain the optimization, but not so large as to cause numerical issues. PyPSA automatically infers an appropriate value based on the network's peak load:

!!! note "Background"

    $$M = 10 \cdot \max_t \sum_{j \in \mathcal{D}} d_{j,t}$$

    where $d_{j,t}$ is the demand of load $j$ at snapshot $t$. The factor of 10 provides a safety margin.

The big-M value can be manually overridden using the `committable_big_m` parameter:

```python
n.optimize(committable_big_m=value)
```

!!! warning "Big-M Size Warning"

    If the optimized capacity $G_{i}$, $F_{k}$ or $R_{m}$ exceeds the big-M value, PyPSA will issue a warning. In this case, increase the big-M value manually to ensure the formulation remains valid.

### Ramp Constraints for Extendable Committable Components

For components that are both committable and extendable with ramp limits, a big-M formulation is used to handle the interaction between ramp limits and the commitment status. Four constraints are added — two for ramping up (during normal operation and during start-up) and two for ramping down (during normal operation and during shut-down):

=== "Generator"

    <!-- md:spec-table Generator_p_ramp_limit_up_run_bigM Generator_p_ramp_limit_up_start_bigM Generator_p_ramp_limit_down_run_bigM Generator_p_ramp_limit_down_shut_bigM -->

=== "Link"

    <!-- md:spec-table Link_p_ramp_limit_up_run_bigM Link_p_ramp_limit_up_start_bigM Link_p_ramp_limit_down_run_bigM Link_p_ramp_limit_down_shut_bigM -->

=== "Process"

    <!-- md:spec-table Process_p_ramp_limit_up_run_bigM Process_p_ramp_limit_up_start_bigM Process_p_ramp_limit_down_run_bigM Process_p_ramp_limit_down_shut_bigM -->

When the unit was running in the previous snapshot ($u_{i,s,t-1} = 1$), the big-M term vanishes and the normal ramp-up limit `ramp_limit_up` applies. When $u_{i,s,t-1} = 0$, the big-M relaxes the constraint, allowing unrestricted ramp-up. The start-up ramp limit `ramp_limit_start_up` is enforced only during start-up events ($su_{i,s,t} = 1$). The same logic applies symmetrically to ramp-down and shut-down constraints.

??? note "Symbols"

    === "Generator"

        <!-- md:spec-legend Generator_com_ext_p_upper_bigM Generator_com_ext_p_upper_cap Generator_com_ext_p_lower Generator_com_ext_p_lower_nonneg Generator_p_ramp_limit_up_run_bigM Generator_p_ramp_limit_up_start_bigM Generator_p_ramp_limit_down_run_bigM Generator_p_ramp_limit_down_shut_bigM -->

    === "Link"

        <!-- md:spec-legend Link_com_ext_p_upper_bigM Link_com_ext_p_upper_cap Link_com_ext_p_lower Link_com_ext_p_lower_nonneg Link_p_ramp_limit_up_run_bigM Link_p_ramp_limit_up_start_bigM Link_p_ramp_limit_down_run_bigM Link_p_ramp_limit_down_shut_bigM -->


## Modularity Constraints

The capacity expansion can be further constrained to be a whole number of modules (e.g. $\mathit{Generator\_n\_mod}_{i} \in \mathbb{N}$) of a modular capacity `p_nom_mod` to represent fixed block sizes of added components (e.g. fixed block size of a nuclear power plant or a fixed capacity of a new circuit).

If `{p,s,e}_nom_mod>0`, the nominal capacity is given by:

<!-- md:spec-table Generator_p_nom_modularity Link_p_nom_modularity Process_p_nom_modularity Line_s_nom_modularity Transformer_s_nom_modularity Store_e_nom_modularity StorageUnit_p_nom_modularity -->

### Modular and Committable Components

When extendable components additionally have **modular capacities** activated (`p_nom_mod > 0`) and are **committable** (`committable=True`), the formulation differs from the big-M approach above. Instead of a binary on/off status variable, the status variable becomes an **integer** representing the **number of committed modules** inside the component.

#### Module-Level Commitment Formulation

For modular committable components, two integer variables are introduced:

- $\mathit{Generator\_n\_mod}_{i}$: Number of modules built (determines total capacity)
- $u_{i,s,t}$: Number of modules committed at snapshot $t$ (determines operational state)

The capacity is constrained to be a multiple of the module size:

=== "Generator"

    <!-- md:spec-table Generator_p_nom_modularity Generator_status_p_nom_variable_upper -->

=== "Link"

    <!-- md:spec-table Link_p_nom_modularity Link_status_p_nom_variable_upper -->

The dispatch constraints enforce that power output respects the number of committed modules:

=== "Generator"

    <!-- md:spec-table Generator_com_mod_p_lower Generator_com_mod_p_upper -->

=== "Link"

    <!-- md:spec-table Link_com_mod_p_lower Link_com_mod_p_upper -->

Note that the minimum and maximum part-load parameters ($\underline{g}$ and $\bar{g}$) apply **per committed module**, not to the total capacity.

#### Start-up and Shut-down for Modular Components

Start-up and shut-down variables track changes in the number of committed modules:

=== "Generator"

    <!-- md:spec-table Generator_com_transition_start_up Generator_com_transition_shut_down -->

=== "Link"

    <!-- md:spec-table Link_com_transition_start_up Link_com_transition_shut_down -->

The start-up and shut-down cost terms in the objective function are multiplied by the number of modules being started or stopped.

!!! warning "Initial status affects start-up costs"

    The `status` attribute defaults to 1, which is used as $u_{i,s,t-1}$ for the first snapshot. For modular committable components, this means **one module is assumed to be already committed** at the start of the optimization. For example, if the optimizer invests in 5 modules and commits all 5 in the first snapshot, only 4 start-up events are counted ($su_{i,s,t} \geq 5 - 1 = 4$), and start-up costs are only charged for those 4 transitions. To charge start-up costs for all modules, set the initial status to 0 via `n.generators.loc[name, "status"] = 0`. This also applies to the non-modular (big-M) committable formulation.

??? note "Symbols"

    === "Generator"

        <!-- md:spec-legend Generator_p_nom_modularity Generator_status_p_nom_variable_upper Generator_com_mod_p_lower Generator_com_mod_p_upper Generator_com_transition_start_up Generator_com_transition_shut_down -->

    === "Link"

        <!-- md:spec-legend Link_p_nom_modularity Link_status_p_nom_variable_upper Link_com_mod_p_lower Link_com_mod_p_upper Link_com_transition_start_up Link_com_transition_shut_down -->


## Compatibility of Capacity Expansion with Unit Commitment Features

The following table summarizes which unit commitment features are compatible with the two formulations:

| Feature | Committable + Extendable (Big-M) | Modular + Committable |
|---------|----------------------------------|----------------------|
| Start-up costs | ✓ | ✓ |
| Shut-down costs | ✓ | ✓ |
| Minimum part-load (`p_min_pu`) | ✓ | ✓ (per module)¹ |
| Ramp limits (`ramp_limit_up/down`) | ✓ | ✓² |
| Stand-by costs | ✓ | ✓ |
| Minimum up time (`min_up_time`) | ✓ | ✗ |
| Minimum down time (`min_down_time`) | ✓ | ✓ |
| Up time before (`up_time_before`) | ✓ | ✗ |
| Down time before (`down_time_before`) | ✓ | ✗ |

---

¹ For modular components, `p_min_pu` and `p_max_pu` apply to **each committed module**. The minimum/maximum power is calculated as `p_min_pu × p_nom_mod × status` where `status` is the number of committed modules.

² For modular + committable components, ramp limits apply to **each committed module**, like the part-load constraints. The ramp-up constraint is `p_t - p_{t-1} ≤ ramp_limit_up × p_nom_mod × status_{t-1} + ramp_limit_start_up × p_nom_mod × (status_t - status_{t-1})`, where `status` is the number of committed modules. This holds for fixed and extendable modular committable components alike.


## Fixed Capacity

Additionally, the nominal capacity can be fixed to a certain value $\tilde{G}_{i}$ for generators, $\tilde{F}_{k}$ for links, $\tilde{R}_{m}$ for processes, $\tilde{P}_{l}$ for lines and transformers, $\tilde{E}_{y}$ for stores, and $\tilde{H}_{v}$ for storage units. In this case, the nominal capacity is given by:

<!-- md:spec-table Generator_p_nom_set Link_p_nom_set Process_p_nom_set Line_s_nom_set Transformer_s_nom_set Store_e_nom_set StorageUnit_p_nom_set -->

The dual variables of these constraints are only stored in `n.model`.

!!! note "Why not just set `p_nom_extendable=False`?"

    Using `p_nom_extendable=False` means the capacity is fixed and not optimized. However, sometimes we need to fix the capacity to a specific value while still keeping track of the dual variables associated with capacity constraints. Setting `{p,s,e}_nom_set` allows for this while maintaining `p_nom_extendable=True`.

??? note "Symbols"

    === "Generator"

        <!-- md:spec-legend Generator_ext_p_nom_lower Generator_ext_p_nom_upper Generator_p_nom_modularity Generator_p_nom_set -->

    === "Link"

        <!-- md:spec-legend Link_ext_p_nom_lower Link_ext_p_nom_upper Link_p_nom_modularity Link_p_nom_set -->

    === "Process"

        <!-- md:spec-legend Process_ext_p_nom_lower Process_ext_p_nom_upper Process_p_nom_modularity Process_p_nom_set -->

    === "Line"

        <!-- md:spec-legend Line_ext_s_nom_lower Line_ext_s_nom_upper Line_s_nom_modularity Line_s_nom_set -->

    === "Transformer"

        <!-- md:spec-legend Transformer_ext_s_nom_lower Transformer_ext_s_nom_upper Transformer_s_nom_modularity Transformer_s_nom_set -->

    === "Store"

        <!-- md:spec-legend Store_ext_e_nom_lower Store_ext_e_nom_upper Store_e_nom_modularity Store_e_nom_set -->

    === "StorageUnit"

        <!-- md:spec-legend StorageUnit_ext_p_nom_lower StorageUnit_ext_p_nom_upper StorageUnit_p_nom_modularity StorageUnit_p_nom_set -->


## Examples


<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Modular Capacity Expansion**

    ---

    Models discrete capacity additions with integer constraints on investment
    decisions considering predefined unit sizes.

    [:octicons-arrow-right-24: Go to example](../../examples/modular-expansion.ipynb)

-   :material-notebook:{ .lg .middle } **Committable and Extendable Components**

    ---

    Co-optimize capacity expansion and unit commitment using big-M linearization.
    Demonstrates continuous capacity decisions with start-up/shut-down costs,
    ramp limits, and minimum load constraints.

    [:octicons-arrow-right-24: Go to example](../../examples/committable-extendable.ipynb)

-   :material-notebook:{ .lg .middle } **Modular and Committable Components**

    ---

    Model discrete capacity blocks with unit commitment where status represents
    the number of committed modules. Shows modular gas turbines, HVDC links,
    and multi-module operational dynamics.

    [:octicons-arrow-right-24: Go to example](../../examples/modular-committable.ipynb)

</div>
