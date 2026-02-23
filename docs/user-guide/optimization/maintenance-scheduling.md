<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Maintenance Scheduling

Maintenance scheduling constraints are implemented for the [`Generator`][pypsa.components.Generators] and [`Link`][pypsa.components.Links]
components. For components marked as `maintainable=True`, new binary maintenance variables
$m_{*,t} \in \{0,1\}$ are introduced, which indicate whether the component is
undergoing maintenance (1) or available (0) in period $t$. This turns the model into a mixed-integer
linear programme (MILP).

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

For extendable components, the dispatch coupling involves the bilinear product $\hat{g}_{n,s} \cdot m_{n,s,t}$. Since $\hat{g}_{n,s}$ is a continuous variable (the optimised capacity), this product is linearised using a McCormick envelope with the auxiliary variable $z_{n,s,t} \approx \hat{g}_{n,s} \cdot m_{n,s,t}$:

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
| $z_{*,t} \leq \hat{g}_{*}$ | `*-maintcap_upper` |
| $z_{*,t} \leq \hat{g}^{\max}_{*} \cdot m_{*,t}$ | `*-maintcap_upper_nommax` |
| $z_{*,t} \geq \hat{g}_{*} + \hat{g}^{\max}_{*} \cdot m_{*,t} - \hat{g}^{\max}_{*}$ | `*-maintcap_lower_nommax` |

Together with $z \geq 0$, these form the convex hull of $z = \hat{g} \cdot m$ for $m \in \{0,1\}$ and $0 \leq \hat{g} \leq \hat{g}^{\max}$.

### Committable Components

When combined with unit commitment (`committable=True`), the dispatch bounds incorporate both the commitment status $u$ and the maintenance status $m$:

=== "Generator (fixed capacity)"

    | Constraint | Name |
    |-------------------|------------------|
    | $g_{n,s,t} \geq \underline{g}_{n,s,t} \cdot \hat{g}_{n,s} \cdot u_{n,s,t} - \underline{g}_{n,s,t} \cdot \hat{g}_{n,s} \cdot \alpha_{n,s} \cdot m_{n,s,t}$ | `Generator-com-p-lower` |
    | $g_{n,s,t} \leq \bar{g}_{n,s,t} \cdot \hat{g}_{n,s} \cdot u_{n,s,t} - \bar{g}_{n,s,t} \cdot \hat{g}_{n,s} \cdot \alpha_{n,s} \cdot m_{n,s,t}$ | `Generator-com-p-upper` |

=== "Generator (extendable capacity)"

    | Constraint | Name |
    |-------------------|------------------|
    | $g_{n,s,t} - \underline{g}_{n,s,t} \cdot \hat{g}_{n,s} - M \cdot u_{n,s,t} + \underline{g}_{n,s,t} \cdot \alpha_{n,s} \cdot z_{n,s,t} \geq -M$ | `Generator-com-ext-p-lower` |
    | $g_{n,s,t} - M \cdot u_{n,s,t} \leq 0$ | `Generator-com-ext-p-upper-bigM` |
    | $g_{n,s,t} - \bar{g}_{n,s,t} \cdot \hat{g}_{n,s} + \bar{g}_{n,s,t} \cdot \alpha_{n,s} \cdot z_{n,s,t} \leq 0$ | `Generator-com-ext-p-upper` |


## Event Count

The total number of maintenance start events must equal the specified number of events $E$:

$$\sum_{t} ms_{*,t} = E$$

Constraint name: `*-maint-event-count`

## Total Duration

The total number of snapshots in maintenance equals the duration per event $d$ times the number of events $E$:

$$\sum_{t} m_{*,t} = d \cdot E$$

Constraint name: `*-maint-total-duration`

## Contiguity

Each maintenance snapshot must be covered by a maintenance start event within the previous $d$ snapshots, enforcing contiguous maintenance blocks:

$$m_{*,t} \leq \sum_{t'=t-d+1}^{t} ms_{*,t'}$$

Constraint name: `*-maint-duration`

## Horizon Restriction

Maintenance cannot start in the last $d-1$ snapshots (to ensure there are enough remaining snapshots to complete the maintenance block):

$$\sum_{t \in \text{last}(d-1)} ms_{*,t} \leq 0$$

Constraint name: `*-maint-start-horizon`

These constraints are defined in the functions `define_maintenance_variables()`, `define_maintenance_start_variables()`, `define_maintenance_capacity_variables()`, and `define_maintenance_constraints()`.

!!! note "Combination with other features"

    Maintenance scheduling can be combined with `committable=True` (unit commitment) and `p_nom_extendable=True` (capacity expansion). When all three are active, dispatch bounds use the big-M formulation with the McCormick auxiliary variable $z$.

??? note "Mapping of symbols to component attributes"

    === "Generator"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $g_{n,s,t}$       | `n.generators_t.p` | Decision variable |
        | $m_{n,s,t}$       | `n.generators_t.maintenance` | Decision variable |
        | $ms_{n,s,t}$      | `n.generators_t.maintenance_start` | Decision variable |
        | $z_{n,s,t}$       | `n.generators_t.maintenance_capacity` | Decision variable |
        | $\hat{g}_{n,s}$   | `n.generators.p_nom` | Parameter |
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
        | $z_{l,t}$         | `n.links_t.maintenance_capacity`  | Decision variable |
        | $\hat{f}_{l}$     | `n.links.p_nom`  | Parameter |
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
