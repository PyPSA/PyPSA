<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Maintenance Scheduling

Maintenance scheduling constraints are implemented for the [`Generator`][pypsa.components.Generators], [`Link`][pypsa.components.Links]
and [`Process`][pypsa.components.Processes] components. The `Process` component follows the same formulation as the `Link`,
applied to its internal power $p$. For components marked as `maintainable=True`, two new variables are
introduced: binary maintenance start variables $ms_{*,t} \in \{0,1\}$, indicating
that a maintenance event begins in snapshot $t$, and continuous maintenance status
variables $m_{*,t} \in [0,1]$, indicating whether the component is undergoing
maintenance (1) or available (0). The integrality of $m$ is implied through the
window coverage equality below, so only $ms$ enters the model as a binary. This
turns the model into a mixed-integer linear programme (MILP).

## Dispatch Coupling

During maintenance, the available capacity is reduced by a fraction $\alpha$ (`maintenance_pu`). When $\alpha = 1$ (default), the component is fully unavailable during maintenance.

### Non-Extendable Components

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $g_{n,s,t} \geq \underline{g}_{n,s,t} \cdot \hat{g}_{n,s} \cdot (1 - \alpha_{n,s} \cdot m_{n,s,t})$ | `Generator-fix-p-lower` |
    | $g_{n,s,t} \leq \bar{g}_{n,s,t} \cdot \hat{g}_{n,s} \cdot (1 - \alpha_{n,s} \cdot m_{n,s,t})$ | `Generator-fix-p-upper` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $f_{l,t} \geq \underline{f}_{l,t} \cdot \hat{f}_{l} \cdot (1 - \alpha_{l} \cdot m_{l,t})$ | `Link-fix-p-lower` |
    | $f_{l,t} \leq \bar{f}_{l,t} \cdot \hat{f}_{l} \cdot (1 - \alpha_{l} \cdot m_{l,t})$ | `Link-fix-p-upper` |

### Extendable Components

For extendable components, the dispatch coupling involves the bilinear product $\hat{g}_{n,s} \cdot m_{n,s,t}$. Since $\hat{g}_{n,s}$ is a continuous variable (the optimised capacity), this product is linearised using a McCormick envelope with the auxiliary variable $z_{n,s,t} = \hat{g}_{n,s} \cdot m_{n,s,t}$:

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $g_{n,s,t} \geq \underline{g}_{n,s,t} \cdot \hat{g}_{n,s} - \underline{g}_{n,s,t} \cdot \alpha_{n,s} \cdot z_{n,s,t}$ | `Generator-ext-p-lower` |
    | $g_{n,s,t} \leq \bar{g}_{n,s,t} \cdot \hat{g}_{n,s} - \bar{g}_{n,s,t} \cdot \alpha_{n,s} \cdot z_{n,s,t}$ | `Generator-ext-p-upper` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $f_{l,t} \geq \underline{f}_{l,t} \cdot \hat{f}_{l} - \underline{f}_{l,t} \cdot \alpha_{l} \cdot z_{l,t}$ | `Link-ext-p-lower` |
    | $f_{l,t} \leq \bar{f}_{l,t} \cdot \hat{f}_{l} - \bar{f}_{l,t} \cdot \alpha_{l} \cdot z_{l,t}$ | `Link-ext-p-upper` |

The McCormick envelope bounds on $z$ are:

| Constraint | Name |
|-------------------|------------------|
| $z_{*,t} \leq \hat{g}_{*} - \hat{g}^{\min}_{*} \cdot (1 - m_{*,t})$ | `*-maintcap_upper` |
| $z_{*,t} \leq \hat{g}^{\max}_{*} \cdot m_{*,t}$ | `*-maintcap_upper_nommax` |
| $z_{*,t} \geq \hat{g}_{*} + \hat{g}^{\max}_{*} \cdot (m_{*,t} - 1)$ | `*-maintcap_lower_nommax` |
| $z_{*,t} \geq \hat{g}^{\min}_{*} \cdot m_{*,t}$ | `*-maintcap_lower_nommin` (only added where $\hat{g}^{\min} > 0$) |

Together with $z \geq 0$, these form the convex hull of $z = \hat{g} \cdot m$ for $m \in \{0,1\}$ and $\hat{g}^{\min} \leq \hat{g} \leq \hat{g}^{\max}$. Since $m$ only takes integral values in any feasible solution, the linearisation is exact. It requires a finite `p_nom_max`. The auxiliary variable $z$ is internal and not written to the network outputs.

### Committable Components

When combined with unit commitment (`committable=True`), the dispatch bounds scale the available capacity by $u_{n,s,t} - \alpha_{n,s} \cdot w_{n,s,t}$, where $w = u \cdot m$ is the product of the commitment status and the maintenance status. This couples the two decisions exactly: at full maintenance ($\alpha = 1$) the unit can be shut down ($u = 0$), while fractional maintenance leaves the remaining capacity available if committed.

=== "Generator (fixed capacity)"

    | Constraint | Name |
    |-------------------|------------------|
    | $g_{n,s,t} \geq \underline{g}_{n,s,t} \cdot \hat{g}_{n,s} \cdot (u_{n,s,t} - \alpha_{n,s} \cdot w_{n,s,t})$ | `Generator-com-p-lower` |
    | $g_{n,s,t} \leq \bar{g}_{n,s,t} \cdot \hat{g}_{n,s} \cdot (u_{n,s,t} - \alpha_{n,s} \cdot w_{n,s,t})$ | `Generator-com-p-upper` |
    | $w_{n,s,t} \leq u_{n,s,t}$ | `Generator-maint-status-le-status` |
    | $w_{n,s,t} \leq m_{n,s,t}$ | `Generator-maint-status-le-maint` |
    | $w_{n,s,t} \geq u_{n,s,t} + m_{n,s,t} - 1$ | `Generator-maint-status-lb` |

    Together with $w \geq 0$, these McCormick inequalities give $w = u \cdot m$ exactly, since $m$ is binary. This holds for binary status (standard unit commitment) and for the relaxed continuous status of `linearized_unit_commitment=True`. The auxiliary variable $w$ is internal and not written to the network outputs.

=== "Generator (modular capacity)"

    For modular committables the status $u^{\mathrm{mod}}_{n,s,t}$ is the integer
    number of committed modules and $\hat{g}^{\mathrm{mod}}_{n,s}$ the module size.
    The same product $w = u^{\mathrm{mod}} \cdot m$ scales the committed capacity,
    with $U_{n,s} = \hat{g}^{\max}_{n,s} / \hat{g}^{\mathrm{mod}}_{n,s}$ bounding the
    module count.

    | Constraint | Name |
    |-------------------|------------------|
    | $g_{n,s,t} \geq \underline{g}_{n,s,t} \cdot \hat{g}^{\mathrm{mod}}_{n,s} \cdot (u^{\mathrm{mod}}_{n,s,t} - \alpha_{n,s} \cdot w_{n,s,t})$ | `Generator-com-mod-p-lower` |
    | $g_{n,s,t} \leq \bar{g}_{n,s,t} \cdot \hat{g}^{\mathrm{mod}}_{n,s} \cdot (u^{\mathrm{mod}}_{n,s,t} - \alpha_{n,s} \cdot w_{n,s,t})$ | `Generator-com-mod-p-upper` |
    | $w_{n,s,t} \leq u^{\mathrm{mod}}_{n,s,t}$ | `Generator-maint-modstatus-le-status` |
    | $w_{n,s,t} \leq U_{n,s} \cdot m_{n,s,t}$ | `Generator-maint-modstatus-le-maint` |
    | $w_{n,s,t} \geq u^{\mathrm{mod}}_{n,s,t} - U_{n,s} \cdot (1 - m_{n,s,t})$ | `Generator-maint-modstatus-lb` |

=== "Generator (extendable capacity)"

    | Constraint | Name |
    |-------------------|------------------|
    | $g_{n,s,t} - \underline{g}_{n,s,t} \cdot \hat{g}_{n,s} - M \cdot u_{n,s,t} + \underline{g}_{n,s,t} \cdot \alpha_{n,s} \cdot z_{n,s,t} \geq -M$ | `Generator-com-ext-p-lower` |
    | $g_{n,s,t} - M \cdot u_{n,s,t} \leq 0$ | `Generator-com-ext-p-upper-bigM` |
    | $g_{n,s,t} - \bar{g}_{n,s,t} \cdot \hat{g}_{n,s} + \bar{g}_{n,s,t} \cdot \alpha_{n,s} \cdot z_{n,s,t} \leq 0$ | `Generator-com-ext-p-upper` |

    Non-modular extendable committables keep the big-M status formulation with the
    McCormick capacity auxiliary $z = \hat{g} \cdot m$; modular committables use the
    integer-status product $w$ above instead.


## Event Count

The total number of maintenance start events must equal the specified number of events $E$:

$$\sum_{t} ms_{*,t} = E$$

Constraint name: `*-maint-event-count`

## Maintenance Windows

The duration per event $d$ (`maintenance_duration`) is given in elapsed time. For
each potential start snapshot $t'$, the coverage window $\text{cov}(t')$ is the
minimal run of consecutive snapshots whose weightings $w_t$ accumulate to at
least $d$:

$$\text{cov}(t') = \{t', \dots, k(t')\}, \quad k(t') = \min\Big\{k : \sum_{t=t'}^{k} w_t \ge d\Big\}$$

The maintenance status equals the sum of all start events whose window covers the snapshot:

$$m_{*,t} = \sum_{t' :\, t \in \text{cov}(t')} ms_{*,t'}$$

Together with $m \leq 1$, this single equality enforces contiguous maintenance
blocks, forbids overlapping events and implies the integrality of $m$.

Constraint name: `*-maint-window`

!!! note "Round-up semantics"

    Each event lasts *at least* $d$ elapsed hours, overshooting by less than the
    weighting of the last covered snapshot. For uniform hourly snapshots and
    integer $d$, the duration is exact.

## Start Validity

Maintenance cannot start where the coverage window does not fit, i.e. where the
remaining weighted horizon is shorter than $d$ or where the window would span
snapshots in which the component is inactive:

$$ms_{*,t'} = 0 \quad \forall t' : \text{cov}(t') \text{ incomplete or partially inactive}$$

Constraint name: `*-maint-start-horizon`

These constraints are defined in the functions `define_maintenance_variables()`, `define_maintenance_start_variables()`, `define_maintenance_capacity_variables()`, and `define_maintenance_constraints()`.

!!! note "Combination with other features"

    Maintenance scheduling can be combined with `committable=True` (unit commitment) and `p_nom_extendable=True` (capacity expansion). When all three are active, dispatch bounds use the big-M formulation with the McCormick auxiliary variable $z$.

!!! warning "Caveats"

    - For committable components, a unit may be shut down ($u_{*,t} = 0$) during
      maintenance. If it stays committed (e.g. forced by minimum up time),
      start-up costs and minimum up/down times interact with maintenance events.
    - In rolling-horizon optimisation, the event count applies per horizon chunk
      and in-progress events are not carried over across chunk boundaries.

??? note "Mapping of symbols to component attributes"

    === "Generator"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $g_{n,s,t}$       | `n.generators_t.p` | Decision variable |
        | $m_{n,s,t}$       | `n.generators_t.maintenance` | Decision variable |
        | $ms_{n,s,t}$      | `n.generators_t.maintenance_start` | Decision variable |
        | $w_{n,s,t}$       | internal, not written to outputs | Decision variable |
        | $z_{n,s,t}$       | internal, not written to outputs | Decision variable |
        | $\hat{g}_{n,s}$   | `n.generators.p_nom` | Parameter |
        | $\hat{g}^{\min}_{n,s}$ | `n.generators.p_nom_min` | Parameter |
        | $\hat{g}^{\max}_{n,s}$ | `n.generators.p_nom_max` | Parameter |
        | $\underline{g}_{n,s,t}$ | `n.generators_t.p_min_pu` | Parameter |
        | $\bar{g}_{n,s,t}$ | `n.generators_t.p_max_pu` | Parameter |
        | $\alpha_{n,s}$    | `n.generators.maintenance_pu` | Parameter |
        | $d$               | `n.generators.maintenance_duration` | Parameter |
        | $E$               | `n.generators.maintenance_events` | Parameter |

    === "Link"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $f_{l,t}$         | `n.links_t.p`  | Decision variable |
        | $m_{l,t}$         | `n.links_t.maintenance`  | Decision variable |
        | $ms_{l,t}$        | `n.links_t.maintenance_start`  | Decision variable |
        | $w_{l,t}$         | internal, not written to outputs  | Decision variable |
        | $z_{l,t}$         | internal, not written to outputs  | Decision variable |
        | $\hat{f}_{l}$     | `n.links.p_nom`  | Parameter |
        | $\hat{f}^{\min}_{l}$ | `n.links.p_nom_min` | Parameter |
        | $\hat{f}^{\max}_{l}$ | `n.links.p_nom_max` | Parameter |
        | $\underline{f}_{l,t}$| `n.links_t.p_min_pu`  | Parameter |
        | $\bar{f}_{l,t}$   | `n.links_t.p_max_pu`  | Parameter |
        | $\alpha_{l}$      | `n.links.maintenance_pu`  | Parameter |
        | $d$               | `n.links.maintenance_duration`  | Parameter |
        | $E$               | `n.links.maintenance_events`  | Parameter |

## Examples

<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Maintenance Scheduling**

    ---

    Schedules optimal maintenance windows for generators with contiguous downtime blocks, partial outages, and multiple events.

    [:octicons-arrow-right-24: Go to example](../../examples/maintenance-scheduling.ipynb)

</div>
