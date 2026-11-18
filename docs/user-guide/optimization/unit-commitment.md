<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Unit Commitment

Unit commitment constraints are implemented for the [`Generator`][pypsa.components.Generators], [`Link`][pypsa.components.Links] and [`Process`][pypsa.components.Processes]
components. They are used to model the start-up and shut-down constraints, as
well as ramping constraints. The implementation is based on Taylor (2015)[^1],
and is supplemented with work by Hua et al. (2017)[^2] for a tightened linear
relaxation.

Unit commitment is not limited to supply. A committable generator with a high
marginal cost models load shedding in blocks, and one with `sign=-1` models a
demand that switches on and off. The [unit commitment example](../../examples/unit-commitment.ipynb)
shows both.

!!! note "Process formulation"

    Unless stated otherwise, all formulations shown for [`Link`][pypsa.components.Links] apply analogously to [`Process`][pypsa.components.Processes] by replacing $(f,\hat f,F,k)$ with $(r,\hat r,R,m)$ and `Link-*` with `Process-*` in the constraint names.

## Start-Up and Shut-down

For components marked as `committable=True`, new binary status variables
$u_{i,s,t}$ are introduced, which indicate whether the component is
running (1) or not (0) in snapshot $t$. This turns the model into a mixed-integer
linear programme (MILP):

<!-- md:spec Generator_status -->

The restrictions on the dispatch now enforce that the
dispatch $g_{i,s,t}$, $f_{k,s,t}$ or $r_{m,s,t}$ is only non-zero if the component is running:

=== "Generator"

    <!-- md:spec-table Generator_com_p_lower Generator_com_p_upper -->

=== "Link"

    <!-- md:spec-table Link_com_p_lower Link_com_p_upper -->

=== "Process"

    <!-- md:spec-table Process_com_p_lower Process_com_p_upper -->

!!! note | Combined with Capacity Expansion

Unit commitment can be combined with capacity expansion for co-optimized investment and operational decisions:

- [Committable and Extendable Components](capacity-limits.md#committable-and-extendable-components) - Continuous capacity expansion with unit commitment using big-M formulation
- [Modular and Committable Components](capacity-limits.md#modular-and-committable-components) - Discrete capacity blocks with integer commitment variables

///

If the **minimum up time** `min_up_time` is set, a component stays up after a start-up. The sum of its start-ups over the last `min_up_time` snapshots, including the current one, may not exceed its status:

=== "Generator"

    <!-- md:spec-table Generator_com_up_time -->

=== "Link"

    <!-- md:spec-table Link_com_up_time -->

=== "Process"

    <!-- md:spec-table Process_com_up_time -->

The component may have been up for some periods before the optimisation period (`n.optimize(snapshots=snapshots)`). Start-ups before the start of `snapshots` count towards this sum, so a component that started up less than `min_up_time` snapshots before `snapshots` is forced to remain up for the difference. If `snapshots` starts at the beginning of `n.snapshots` or no statuses have been computed before, the time before is reconstructed from the input attributes: the component was up for `up_time_before` snapshots, or, if `down_time_before` is positive, down for `down_time_before` snapshots (`down_time_before` takes precedence). If `snapshots` starts in the middle of `n.snapshots`, the statuses before `snapshots` are read from `n.c.generators.dynamic.status` as set by previous runs, and `up_time_before`/`down_time_before` apply to the time before `n.snapshots`. All snapshots before `snapshots` count as solved once `n.c.generators.dynamic.status` is set. Successive runs must therefore solve the snapshots in order, starting from `n.snapshots[0]`, as [`n.optimize.optimize_with_rolling_horizon()`][pypsa.optimization.OptimizationAccessor.optimize_with_rolling_horizon] does. The start-ups before `snapshots` are the positive differences of these statuses, which may be fractional with the [linearized unit commitment](#linearization).

At the end of `snapshots` the sum covers fewer snapshots, if fewer than `min_up_time` snapshots remain.

If the **minimum down time** `min_down_time` is set, a component stays down after a shut-down. The sum of its shut-downs over the last `min_down_time` snapshots, including the current one, may not exceed the part of its capacity that is off (one minus its status):

=== "Generator"

    <!-- md:spec-table Generator_com_down_time -->

=== "Link"

    <!-- md:spec-table Link_com_down_time -->

=== "Process"

    <!-- md:spec-table Process_com_down_time -->

The component may have been down for some periods before the optimisation period. Shut-downs before the start of `snapshots` count towards this sum in the same way as start-ups do for the minimum up-time.

Furthermore, two **state transition variables** for start-up ($su$) and shut-down ($sd$) are introduced to associate them with start-up and shut-down cost terms in the objective function. Start-up and shut-down costs can be static or time series varying per snapshot. The constraints are set so that the start-up variable is only non-zero if the component has just started up, and the shut-down variable is only non-zero if the component has just shut down. The previous status is the status of the snapshot before, or the status carried in before the first snapshot:

=== "Generator"

    <!-- md:spec-table Generator_com_transition_start_up Generator_com_transition_shut_down -->

=== "Link"

    <!-- md:spec-table Link_com_transition_start_up Link_com_transition_shut_down -->

=== "Process"

    <!-- md:spec-table Process_com_transition_start_up Process_com_transition_shut_down -->

??? note "Symbols"

    === "Generator"

        <!-- md:spec-legend Generator_status Generator_start_up Generator_shut_down Generator_com_p_lower Generator_com_p_upper Generator_com_transition_start_up Generator_com_transition_shut_down Generator_com_up_time Generator_com_down_time Generator_min_up_time Generator_min_down_time Generator_up_time_brought_in Generator_down_time_brought_in -->

    === "Link"

        <!-- md:spec-legend Link_status Link_start_up Link_shut_down Link_com_p_lower Link_com_p_upper Link_com_transition_start_up Link_com_transition_shut_down Link_com_up_time Link_com_down_time Link_min_up_time Link_min_down_time Link_up_time_brought_in Link_down_time_brought_in -->

    === "Process"

        <!-- md:spec-legend Process_status Process_start_up Process_shut_down Process_com_p_lower Process_com_p_upper Process_com_transition_start_up Process_com_transition_shut_down -->

    The results are stored in `n.{generators,links,processes}_t.{status,start_up,shut_down}`.

## Ramping

Ramp rate limits can be defined for increasing output (`ramp_limit_up`) and decreasing output (`ramp_limit_down`). By default these are null and ignored. They must be provided per unit of the nominal capacity.

!!! note

    When provided, ramping limits are also considered if they are not
    committable (`committable=False`), i.e. the component is not subject to start-up and shut-down constraints.

A unified ramp constraint formulation is used for fixed, extendable, and committable components. For committable components, additional ramp limits at start-up (`ramp_limit_start_up`) and shut-down (`ramp_limit_shut_down`) can be specified. The change of dispatch from the previous snapshot is limited by an allowance:

=== "Generator"

    <!-- md:spec-table Generator_p_ramp_limit_up Generator_p_ramp_limit_down -->

=== "Link"

    <!-- md:spec-table Link_p_ramp_limit_up Link_p_ramp_limit_down -->

=== "Process"

    <!-- md:spec-table Process_p_ramp_limit_up Process_p_ramp_limit_down -->

The allowance depends on the commitment. For non-committable components it is the ramp rate times the capacity, which is the capacity variable for extendable components:

<!-- md:spec Generator_ramp_up_allowance Generator_ramp_down_allowance -->

Committable components that are also extendable use a [big-M formulation](capacity-limits.md#committable-and-extendable-components) instead.

### Initial Conditions

The ramp constraints require knowledge of the previous snapshot's dispatch. At the first snapshot of the optimization horizon, this is handled as follows:

- **Rolling horizon optimization**: When `n.optimize(snapshots=...)` starts after the first snapshot in `n.snapshots` and previous runs have set the dispatch, the dispatch and the commitment status from the previous snapshot are used automatically. With the [linearized unit commitment](#linearization) the previous status may be fractional.
- **First snapshot of `n.snapshots` or no previous run**: The `p_init` attribute specifies the initial dispatch level. For committable components, the initial status is determined by `up_time_before > 0` (or `down_time_before > 0`, which takes precedence) and the initial dispatch is zero if the component was off. If `p_init` is not set (NaN), no ramp constraint is applied at the first snapshot. Non-committable components are always on; `up_time_before` and `down_time_before` are ignored for them.

??? note "Symbols"

    === "Generator"

        <!-- md:spec-legend Generator_p_ramp_limit_up Generator_p_ramp_limit_down Generator_ramp_up_allowance Generator_ramp_down_allowance -->

    === "Link"

        <!-- md:spec-legend Link_p_ramp_limit_up Link_p_ramp_limit_down Link_ramp_up_allowance Link_ramp_down_allowance -->

    === "Process"

        <!-- md:spec-legend Process_p_ramp_limit_up Process_p_ramp_limit_down Process_ramp_up_allowance Process_ramp_down_allowance -->

    The dual variables are stored in `n.{generators,links,processes}_t.mu_ramp_limit_up` and `mu_ramp_limit_down`.

## Linearization

The implementation is based on Hua et al. (2017)[^2] and relaxes the binary unit commitment variables $u$, $su$ and $sd$ to continuous variables between 0 and 1:

<!-- md:spec linearized Generator_status Generator_start_up Generator_shut_down -->

This allows for partial commitment states (generators can be partially on/off), making the problem more computationally tractable while losing some accuracy. To enable this, use:

``` py
n.optimize(linearized_unit_commitment=True)
```

!!! warning "Not compatible with modular components"

    Linearized unit commitment cannot be used with [modular committable components](capacity-limits.md#modular-and-committable-components) and will result in a `ValueError`.

To tighten the relaxation, additional constraints are introduced that improve capturing the relationship between commitment status, ramping, and dispatch. At the first snapshot they use the previous dispatch and status as described in [Initial Conditions](#initial-conditions) and are skipped if the previous dispatch is unknown. The tightening is only applied to units whose start-up and shut-down costs are equal in every snapshot; for the other units the unit commitment variables are purely relaxed and a warning is logged. The added constraints limit the dispatch during partial start-up and shut-down, as well as ramping during partial commitment:

=== "Generator"

    <!-- md:spec-table linearized Generator_com_p_before Generator_com_p_current Generator_com_partly_start_up Generator_com_partly_shut_down -->

=== "Link"

    <!-- md:spec-table linearized Link_com_p_before Link_com_p_current Link_com_partly_start_up Link_com_partly_shut_down -->

=== "Process"

    <!-- md:spec-table linearized Process_com_p_before Process_com_p_current Process_com_partly_start_up Process_com_partly_shut_down -->

??? note "Symbols"

    === "Generator"

        <!-- md:spec-legend linearized Generator_com_p_before Generator_com_p_current Generator_com_partly_start_up Generator_com_partly_shut_down -->

    === "Link"

        <!-- md:spec-legend linearized Link_com_p_before Link_com_p_current Link_com_partly_start_up Link_com_partly_shut_down -->

    === "Process"

        <!-- md:spec-legend linearized Process_com_p_before Process_com_p_current Process_com_partly_start_up Process_com_partly_shut_down -->

## Examples


<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Unit Commitment**

    ---

    Models generator unit commitment with start-up and shut-down costs, ramping limits, minimum part loads, up and down times using binary variables. Also applies it to demand: load shedding in blocks and flexible demand with start-up cost.

    [:octicons-arrow-right-24: Go to example](../../examples/unit-commitment.ipynb)

-   :material-notebook:{ .lg .middle } **Negative Prices in Linearized Unit Commitment**

    ---

    Demonstrates how negative electricity prices emerge in linearized unit commitment when generators face high cycling costs and prefer to stay online at negative prices.

    [:octicons-arrow-right-24: Go to example](../../examples/uc-prices.ipynb)

</div>

[^1]: J.A. Taylor (2015), [Convex Optimization of Power Systems](http://www.cambridge.org/de/academic/subjects/engineering/control-systems-and-optimization/convex-optimization-power-systems), Cambridge University Press, Chapter 4.3.

[^2]: B. Hua, R. Baldick and J. Wang (2018), [Representing Operational Flexibility in Generation Expansion Planning Through Convex Relaxation of Unit Commitment](https://doi.org/10.1109/TPWRS.2017.2735026), IEEE Transactions on Power Systems, 33, 2, 2272-2281, doi:10.1109/TPWRS.2017.2735026, equations (21-24).
