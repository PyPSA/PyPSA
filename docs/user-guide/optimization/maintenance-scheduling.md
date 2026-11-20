<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Maintenance Scheduling

Maintenance scheduling constraints are implemented for the [`Generator`][pypsa.components.Generators], [`Link`][pypsa.components.Links]
and [`Process`][pypsa.components.Processes] components. The `Process` component follows the same formulation as the `Link`,
applied to its internal dispatch $r_{m,s,t}$. For components marked as `maintainable=True`, two new variables are
introduced: binary maintenance start variables (`maintenance_start`), indicating
that a maintenance event begins in snapshot $t$, and continuous maintenance status
variables (`maintenance`) between 0 and 1, indicating whether the component is undergoing
maintenance (1) or available (0):

<!-- md:spec Generator_maintenance_start Generator_maintenance -->

The integrality of the maintenance status is implied through the
window coverage equality below, so only the start variable enters the model as a binary. This
turns the model into a mixed-integer linear programme (MILP). This page shows the constraints of the generator. Links and processes have the same constraints with the prefixes `Link` and `Process`.

Scheduling generator maintenance as an integer program dates back to Dopazo and
Merrill (1975).[^1] The event-based formulation here follows the tradition of
scheduling a given number of maintenance events of fixed duration[^2]; for a
survey of alternative formulations and the tightness of their linear relaxations,
see Andrade (2025).[^3] A broader review of maintenance scheduling in the
electricity industry is given by Froger et al. (2016).[^4]

## Event Count

The total number of maintenance start events must equal the specified number of events (`maintenance_events`):

<!-- md:spec Generator_maint_event_count -->

## Maintenance Windows

The duration per event (`maintenance_duration`) is given in elapsed time. For
each generator $i$, scenario $s$ and potential start snapshot $t'$, the coverage window $\text{cov}(t')$ is the
minimal run of consecutive snapshots whose weightings $w^{\mathrm{g}}_t$ accumulate to at
least the duration:

!!! note "Background"

    $$\text{cov}(t') = \{t', \dots, \tau(t')\}, \quad \tau(t') = \min\Big\{\tau : \sum_{t=t'}^{\tau} w^{\mathrm{g}}_t \ge \mathrm{Generator\_maintenance\_duration}_{i,s}\Big\}$$

    The relation `Generator_maintenance_cover` holds the pairs $(t, t')$ with $t \in \text{cov}(t')$.

The maintenance status equals the sum of all start events whose window covers the snapshot:

<!-- md:spec Generator_maint_window -->

Together with the upper bound of 1 on the status, this single equality enforces contiguous maintenance
blocks, forbids overlapping events and implies the integrality of the status.

!!! note "Round-up semantics"

    Each event lasts *at least* `maintenance_duration` elapsed hours, overshooting by less than the
    weighting of the last covered snapshot. For uniform hourly snapshots and
    integer durations, the duration is exact.

## Start Validity

Maintenance cannot start where the coverage window does not fit, i.e. where the
remaining weighted horizon is shorter than the duration or where the window would span
snapshots in which the component is inactive (`maintenance_start_blocked`):

<!-- md:spec Generator_maint_start_horizon -->

## Effect on Dispatch Limits

During maintenance, the available capacity is reduced by the fraction `maintenance_pu`. When `maintenance_pu=1` (default), the component is fully unavailable during maintenance.

### Non-Extendable Components

=== "Generator"

    <!-- md:spec-table Generator_fix_p_lower Generator_fix_p_upper -->

=== "Link"

    <!-- md:spec-table Link_fix_p_lower Link_fix_p_upper -->

=== "Process"

    <!-- md:spec-table Process_fix_p_lower Process_fix_p_upper -->

### Extendable Components

For extendable components, the dispatch coupling involves the bilinear product of the optimised capacity (e.g. $G_i$) and the maintenance status. This product is linearised using a McCormick envelope with the auxiliary variable `maintenance_capacity`, which equals the capacity times the maintenance status:

=== "Generator"

    <!-- md:spec-table Generator_ext_p_lower Generator_ext_p_upper -->

=== "Link"

    <!-- md:spec-table Link_ext_p_lower Link_ext_p_upper -->

=== "Process"

    <!-- md:spec-table Process_ext_p_lower Process_ext_p_upper -->

The McCormick envelope bounds on the auxiliary variable are:

=== "Generator"

    <!-- md:spec-table Generator_maintcap_upper Generator_maintcap_upper_nommax Generator_maintcap_lower_nommax Generator_maintcap_lower_nommin -->

=== "Link"

    <!-- md:spec-table Link_maintcap_upper Link_maintcap_upper_nommax Link_maintcap_lower_nommax Link_maintcap_lower_nommin -->

=== "Process"

    <!-- md:spec-table Process_maintcap_upper Process_maintcap_upper_nommax Process_maintcap_lower_nommax Process_maintcap_lower_nommin -->

Together with its lower bound of zero, these form the convex hull of the product for a binary status and a capacity between `p_nom_min` and `p_nom_max`. Since the status only takes integral values in any feasible solution, the linearisation is exact. It requires a finite `p_nom_max`. The auxiliary variable is internal and not written to the network outputs.

### Committable Components

When combined with unit commitment (`committable=True`), the dispatch bounds scale the available capacity by the commitment status $u$ minus `maintenance_pu` times the auxiliary variable `maintenance_status`, the product of the commitment status and the maintenance status. This couples the two decisions exactly: at full maintenance (`maintenance_pu=1`) the unit can be shut down ($u = 0$), while fractional maintenance leaves the remaining capacity available if committed.

=== "Fixed capacity"

    === "Generator"

        <!-- md:spec-table Generator_com_p_lower Generator_com_p_upper Generator_maint_status_le_status Generator_maint_status_le_maint Generator_maint_status_lb -->

    === "Link"

        <!-- md:spec-table Link_com_p_lower Link_com_p_upper Link_maint_status_le_status Link_maint_status_le_maint Link_maint_status_lb -->

    === "Process"

        <!-- md:spec-table Process_com_p_lower Process_com_p_upper Process_maint_status_le_status Process_maint_status_le_maint Process_maint_status_lb -->

    Together with its lower bound of zero, these McCormick inequalities give the product exactly, since the maintenance status is binary. This holds for binary status (standard unit commitment) and for the relaxed continuous status of `linearized_unit_commitment=True`. The auxiliary variable is internal and not written to the network outputs.

=== "Modular capacity"

    For modular committables the status $u$ is the integer
    number of committed modules and `p_nom_mod` the module size.
    The same product scales the committed capacity,
    with `maintenance_modules` bounding the module count.

    === "Generator"

        <!-- md:spec-table Generator_com_mod_p_lower Generator_com_mod_p_upper Generator_maint_modstatus_le_status Generator_maint_modstatus_le_maint Generator_maint_modstatus_lb -->

    === "Link"

        <!-- md:spec-table Link_com_mod_p_lower Link_com_mod_p_upper Link_maint_modstatus_le_status Link_maint_modstatus_le_maint Link_maint_modstatus_lb -->

    === "Process"

        <!-- md:spec-table Process_com_mod_p_lower Process_com_mod_p_upper Process_maint_modstatus_le_status Process_maint_modstatus_le_maint Process_maint_modstatus_lb -->

=== "Extendable capacity"

    === "Generator"

        <!-- md:spec-table Generator_com_ext_p_lower Generator_com_ext_p_upper_big_m Generator_com_ext_p_upper_cap -->

    === "Link"

        <!-- md:spec-table Link_com_ext_p_lower Link_com_ext_p_upper_big_m Link_com_ext_p_upper_cap -->

    === "Process"

        <!-- md:spec-table Process_com_ext_p_lower Process_com_ext_p_upper_big_m Process_com_ext_p_upper_cap -->

    Non-modular extendable committables keep the big-M status formulation with the
    McCormick capacity auxiliary `maintenance_capacity`; modular committables use the
    integer-status product `maintenance_status` above instead.

!!! note "Combination with other features"

    Maintenance scheduling can be combined with `committable=True` (unit commitment) and `p_nom_extendable=True` (capacity expansion). When all three are active, dispatch bounds use the big-M formulation with the McCormick auxiliary variable `maintenance_capacity`.

!!! warning "Caveats"

    - For committable components, a unit may be shut down ($u = 0$) during
      maintenance. If it stays committed (e.g. forced by minimum up time),
      start-up costs and minimum up/down times interact with maintenance events.
    - In rolling-horizon optimisation, the event count applies per horizon chunk
      and in-progress events are not carried over across chunk boundaries.

??? note "Symbols"

    === "Generator"

        <!-- md:spec-legend Generator_maintenance Generator_maintenance_start Generator_maint_event_count Generator_maint_window Generator_maint_start_horizon Generator_fix_p_upper Generator_ext_p_upper Generator_maintcap_upper Generator_maintcap_upper_nommax Generator_com_p_upper Generator_maint_status_lb Generator_com_mod_p_upper Generator_maint_modstatus_lb Generator_com_ext_p_lower -->

    === "Link"

        <!-- md:spec-legend Link_maintenance Link_maintenance_start Link_maint_event_count Link_maint_window Link_maint_start_horizon Link_fix_p_upper Link_ext_p_upper Link_maintcap_upper Link_maintcap_upper_nommax Link_com_p_upper Link_maint_status_lb Link_com_mod_p_upper Link_maint_modstatus_lb Link_com_ext_p_lower -->

    === "Process"

        <!-- md:spec-legend Process_maintenance Process_maintenance_start Process_maint_event_count Process_maint_window Process_maint_start_horizon Process_fix_p_upper Process_ext_p_upper Process_maintcap_upper Process_maintcap_upper_nommax Process_com_p_upper Process_maint_status_lb Process_com_mod_p_upper Process_maint_modstatus_lb Process_com_ext_p_lower -->

    The maintenance status and start are written to `n.{generators,links,processes}_t.maintenance` and `maintenance_start`. The auxiliary variables are internal.

## Examples

<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Maintenance Scheduling**

    ---

    Schedules optimal maintenance windows for generators with contiguous downtime blocks, partial outages, and multiple events.

    [:octicons-arrow-right-24: Go to example](../../examples/maintenance-scheduling.ipynb)

</div>

[^1]: J. F. Dopazo and H. M. Merrill (1975), [Optimal Generator Maintenance Scheduling Using Integer Programming](https://doi.org/10.1109/T-PAS.1975.31996), IEEE Transactions on Power Apparatus and Systems, 94, 5, 1537-1545, doi:10.1109/T-PAS.1975.31996.
[^2]: F. Fourcade, E. Johnson, M. Bara, P. Cortey-Dumont (1997), [Optimizing nuclear power plant refueling with mixed-integer programming](https://doi.org/10.1016/S0377-2217(96)00197-X), European Journal of Operational Research, 97, 2, 269-280, doi:10.1016/S0377-2217(96)00197-X.
[^3]: T. Andrade (2025), [On the tightness of linear relaxations of alternative mixed integer programming formulations for the generator maintenance scheduling problem](https://arxiv.org/abs/2502.08855), arXiv:2502.08855.
[^4]: A. Froger, M. Gendreau, J. E. Mendoza, É. Pinson, L.-M. Rousseau (2016), [Maintenance scheduling in the electricity industry: A literature review](https://doi.org/10.1016/j.ejor.2015.08.045), European Journal of Operational Research, 251, 3, 695-706, doi:10.1016/j.ejor.2015.08.045.
