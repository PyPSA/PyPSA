<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Capacity Limits

!!! note "Process formulation"

    Unless stated otherwise, all formulations shown for [`Link`][pypsa.components.Links] apply analogously to [`Process`][pypsa.components.Processes] by replacing $(f,\hat f,F,l)$ with $(r,\hat r,R,m)$ and `Link-*` with `Process-*` in the constraint names.

## Upper and Lower Bounds

If the nominal capacity of a components is also the subject of optimisation (e.g. with decision variable $G_{n,s}$ for generators), limits on the installable capacity may also be introduced (e.g. $\underline{G}_{n,s}$ and $\bar{G}_{n,s}$):

| Constraint | Dual Variable | Name |
|-------------------|------------------|------------------|
| $G_{n,s} \geq \underline{G}_{n,s}$ | `n.generators.mu_lower` | `Generator-ext-p_nom-lower` |
| $G_{n,s} \leq \bar{G}_{n,s}$ | `n.generators.mu_upper` | `Generator-ext-p_nom-upper` |
| $F_{l} \geq \underline{F}_{l}$ | `n.links.mu_lower` | `Link-ext-p_nom-lower` |
| $F_{l} \leq \bar{F}_{l}$ | `n.links.mu_upper` | `Link-ext-p_nom-upper` |
| $R_{m} \geq \underline{R}_{m}$ | `n.processes.mu_lower` | `Process-ext-p_nom-lower` |
| $R_{m} \leq \bar{R}_{m}$ | `n.processes.mu_upper` | `Process-ext-p_nom-upper` |
| $P_{l} \geq \underline{P}_{l}$ | `n.{lines,transformers}.mu_lower` | `{Line,Transformer}-ext-s_nom-lower` |
| $P_{l} \leq \bar{P}_{l}$ | `n.{lines, transformers}.mu_upper` | `{Line,Transformer}-ext-s_nom-upper` |
| $E_{n,s} \geq \underline{E}_{n,s}$ | `n.stores.mu_lower` | `Store-ext-e_nom-lower` |
| $E_{n,s} \leq \bar{E}_{n,s}$ | `n.stores.mu_upper` | `Store-ext-e_nom-upper` |
| $H_{n,s} \geq \underline{H}_{n,s}$ | `n.storage_units.mu_lower` | `StorageUnit-ext-p_nom-lower` |

These constraints are set in the function `define_nominal_constraints_for_extendables`.

## Committable and Extendable Components

When components are both **committable** (`committable=True`) and **extendable** (e.g. `p_nom_extendable=True`), the optimizer co-optimizes both capacity expansion and operational unit commitment. The challenge is that the upper bound on dispatch becomes nonlinear: $g_{n,s,t} \leq u_{n,s,t} \cdot G_{n,s}$ (the product of binary status $u$ and continuous capacity $G$). To maintain a Mixed-Integer Linear Programme (MILP), PyPSA uses a **big-M formulation** that replaces this nonlinear constraint with two linear constraints:

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $g_{n,s,t} \leq u_{n,s,t} \cdot M$ | `Generator-com-ext-p-upper-bigM` |
    | $g_{n,s,t} \leq \bar{g}_{n,s,t} \cdot G_{n,s}$ | `Generator-com-ext-p-upper-cap` |
    | $g_{n,s,t} \geq \underline{g}_{n,s,t} \cdot G_{n,s} - M \cdot (1 - u_{n,s,t})$ | `Generator-com-ext-p-lower` |
    | $g_{n,s,t} \geq 0$ | `Generator-com-ext-p-lower-nonneg` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $f_{l,t} \leq u_{l,t} \cdot M$ | `Link-com-ext-p-upper-bigM` |
    | $f_{l,t} \leq \bar{f}_{l,t} \cdot F_{l}$ | `Link-com-ext-p-upper-cap` |
    | $f_{l,t} \geq \underline{f}_{l,t} \cdot F_{l} - M \cdot (1 - u_{l,t})$ | `Link-com-ext-p-lower` |
    | $f_{l,t} \geq 0$ | `Link-com-ext-p-lower-nonneg` |

=== "Process"

    | Constraint | Name |
    |-------------------|------------------|
    | $r_{m,t} \leq u_{m,t} \cdot M$ | `Process-com-ext-p-upper-bigM` |
    | $r_{m,t} \leq \bar{r}_{m,t} \cdot R_{m}$ | `Process-com-ext-p-upper-cap` |
    | $r_{m,t} \geq \underline{r}_{m,t} \cdot R_{m} - M \cdot (1 - u_{m,t})$ | `Process-com-ext-p-lower` |
    | $r_{m,t} \geq 0$ | `Process-com-ext-p-lower-nonneg` |

where $M$ is a sufficiently large constant (the "big-M") and $\bar{g}$, $\bar{f}$, $\bar{r}$ are the maximum per-unit dispatch limits (`p_max_pu`, default 1). When the status $u = 0$, the big-M constraint forces dispatch to zero. When $u = 1$, the lower bound becomes $g \geq \underline{g} \cdot G$ (minimum part-load) and the upper bound is $g \leq \bar{g} \cdot G$ (capacity limit).

### Big-M Parameter Configuration

The big-M constant must be large enough to not constrain the optimization, but not so large as to cause numerical issues. PyPSA automatically infers an appropriate value based on the network's peak load:

$$M = 10 \times \max_t \left( \sum_n L_{n,t} \right)$$

where $L_{n,t}$ is the load at bus $n$ and time $t$. The factor of 10 provides a safety margin.

The big-M value can be manually overridden using the `committable_big_m` parameter:

```python
n.optimize(committable_big_m=value)
```

!!! warning "Big-M Size Warning"

    If the optimized capacity $G_{n,s}$, $F_{l}$ or $R_{m}$ exceeds the big-M value, PyPSA will issue a warning. In this case, increase the big-M value manually to ensure the formulation remains valid.

### Ramp Constraints for Extendable Committable Components

For components that are both committable and extendable with ramp limits, a big-M formulation is used to handle the interaction between ramp limits and the commitment status. Four constraints are added — two for ramping up (during normal operation and during start-up) and two for ramping down (during normal operation and during shut-down):

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $(g_{n,s,t} - g_{n,s,t-1}) \leq ru_{n,s} \cdot G_{n,s} + M \cdot (1 - u_{n,s,t-1})$ | `Generator-p-ramp_limit_up-run-bigM` |
    | $(g_{n,s,t} - g_{n,s,t-1}) \leq rs_{n,s} \cdot G_{n,s} + M \cdot (1 - su_{n,s,t})$ | `Generator-p-ramp_limit_up-start-bigM` |
    | $(g_{n,s,t} - g_{n,s,t-1}) \geq -rd_{n,s} \cdot G_{n,s} - M \cdot (1 - u_{n,s,t})$ | `Generator-p-ramp_limit_down-run-bigM` |
    | $(g_{n,s,t} - g_{n,s,t-1}) \geq -rsd_{n,s} \cdot G_{n,s} - M \cdot (1 - sd_{n,s,t})$ | `Generator-p-ramp_limit_down-shut-bigM` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $(f_{l,t} - f_{l,t-1}) \leq ru_{l} \cdot F_{l} + M \cdot (1 - u_{l,t-1})$ | `Link-p-ramp_limit_up-run-bigM` |
    | $(f_{l,t} - f_{l,t-1}) \leq rs_{l} \cdot F_{l} + M \cdot (1 - su_{l,t})$ | `Link-p-ramp_limit_up-start-bigM` |
    | $(f_{l,t} - f_{l,t-1}) \geq -rd_{l} \cdot F_{l} - M \cdot (1 - u_{l,t})$ | `Link-p-ramp_limit_down-run-bigM` |
    | $(f_{l,t} - f_{l,t-1}) \geq -rsd_{l} \cdot F_{l} - M \cdot (1 - sd_{l,t})$ | `Link-p-ramp_limit_down-shut-bigM` |

=== "Process"

    | Constraint | Name |
    |-------------------|------------------|
    | $(r_{m,t} - r_{m,t-1}) \leq ru_{m} \cdot R_{m} + M \cdot (1 - u_{m,t-1})$ | `Process-p-ramp_limit_up-run-bigM` |
    | $(r_{m,t} - r_{m,t-1}) \leq rs_{m} \cdot R_{m} + M \cdot (1 - su_{m,t})$ | `Process-p-ramp_limit_up-start-bigM` |
    | $(r_{m,t} - r_{m,t-1}) \geq -rd_{m} \cdot R_{m} - M \cdot (1 - u_{m,t})$ | `Process-p-ramp_limit_down-run-bigM` |
    | $(r_{m,t} - r_{m,t-1}) \geq -rsd_{m} \cdot R_{m} - M \cdot (1 - sd_{m,t})$ | `Process-p-ramp_limit_down-shut-bigM` |

When the unit was running in the previous timestep ($u_{t-1} = 1$), the big-M term vanishes and the normal ramp-up limit $ru$ applies. When $u_{t-1} = 0$, the big-M relaxes the constraint, allowing unrestricted ramp-up. The start-up ramp limit $rs$ is enforced only during start-up events ($su_t = 1$). The same logic applies symmetrically to ramp-down and shut-down constraints.

These constraints are defined in the function `define_ramp_limit_constraints()`.

??? note "Mapping of symbols to component attributes"

    === "Generator"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $g_{n,s,t}$       | `n.generators_t.p` | Decision variable |
        | $G_{n,s}$         | `n.generators.p_nom_opt` | Decision variable |
        | $u_{n,s,t}$       | `n.generators_t.status` | Decision variable |
        | $su_{n,s,t}$      | `n.generators_t.start_up` | Decision variable |
        | $sd_{n,s,t}$      | `n.generators_t.shut_down` | Decision variable |
        | $M$               | auto-inferred or `committable_big_m` parameter | Parameter |
        | $ru_{n,s}$        | `n.generators.ramp_limit_up` | Parameter |
        | $rd_{n,s}$        | `n.generators.ramp_limit_down` | Parameter |
        | $rs_{n,s}$        | `n.generators.ramp_limit_start_up` | Parameter |
        | $rsd_{n,s}$       | `n.generators.ramp_limit_shut_down` | Parameter |

    === "Link"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $f_{l,t}$         | `n.links_t.p` | Decision variable |
        | $F_{l}$           | `n.links.p_nom_opt` | Decision variable |
        | $u_{l,t}$         | `n.links_t.status` | Decision variable |
        | $su_{l,t}$        | `n.links_t.start_up` | Decision variable |
        | $sd_{l,t}$        | `n.links_t.shut_down` | Decision variable |
        | $M$               | auto-inferred or `committable_big_m` parameter | Parameter |
        | $ru_{l}$          | `n.links.ramp_limit_up` | Parameter |
        | $rd_{l}$          | `n.links.ramp_limit_down` | Parameter |
        | $rs_{l}$          | `n.links.ramp_limit_start_up` | Parameter |
        | $rsd_{l}$         | `n.links.ramp_limit_shut_down` | Parameter |


## Modularity Constraints

The capacity expansion can be decided in **modules**, represented by an integer number of modules $n^{\textrm{mod}}_{*} \in \mathbb{N}$ per asset.
A component is modular if `modular=True` or if it has a positive module size `{p,s,e}_nom_mod>0`, which implies `modular=True`.
Modularity comes in two flavours:

| `modular` | `{p,s,e}_nom_mod` | Capacity of each module | Typical use |
|-----------|-------------------|-------------------------|-------------|
| any | $> 0$ | Fixed to the module size | Fixed block sizes, e.g. a nuclear power plant or a new circuit |
| `True` | $0$ | Continuous decision | Capacity-independent costs, e.g. grid connection, land acquisition or project development |

Each module incurs the capacity-independent `module_cost` in the objective (see [Module Costs](objective.md#module-costs)).
The module variables are available as `n.model.variables['{Component}-n_mod']` and the optimised number of modules is written back to the static output `n_mod_opt` (e.g. `n.generators.n_mod_opt`), which is `NaN` for assets that are not modular or not extendable.
The module decision carries neither a snapshot nor a scenario dimension.

### Fixed Module Sizes

If `{p,s,e}_nom_mod>0`, the nominal capacity is given by:

| Constraint | Dual Variable | Name |
|-------------------|------------------|------------------|
| $G_{n,s} = G^{\textrm{mod}}_{n,s} \cdot \tilde{G}_{n,s}$ | N/A | `Generator-p_nom_modularity` |
| $F_{l} = F^{\textrm{mod}}_{l} \cdot \tilde{F}_{l}$ | N/A | `Link-p_nom_modularity` |
| $R_{m} = R^{\textrm{mod}}_{m} \cdot \tilde{R}_{m}$ | N/A | `Process-p_nom_modularity` |
| $P_{l} = P^{\textrm{mod}}_{l} \cdot \tilde{P}_{l}$ | N/A | `{Line,Transformer}-s_nom_modularity` |
| $E_{n,s} = E^{\textrm{mod}}_{n,s} \cdot \tilde{E}_{n,s}$ | N/A | `Store-e_nom_modularity` |
| $H_{n,s} = H^{\textrm{mod}}_{n,s} \cdot \tilde{H}_{n,s}$ | N/A | `StorageUnit-p_nom_modularity` |

### Continuous Module Capacity

If `modular=True` and `{p,s,e}_nom_mod=0`, the total capacity of the modules is a continuous decision, which requires at least one module to be built:

| Constraint | Dual Variable | Name |
|-------------------|------------------|------------------|
| $G_{n,s} \leq \bar{G}_{n,s} \cdot G^{\textrm{mod}}_{n,s}$ | N/A | `Generator-p_nom_modularity_upper` |
| $F_{l} \leq \bar{F}_{l} \cdot F^{\textrm{mod}}_{l}$ | N/A | `Link-p_nom_modularity_upper` |
| $R_{m} \leq \bar{R}_{m} \cdot R^{\textrm{mod}}_{m}$ | N/A | `Process-p_nom_modularity_upper` |
| $P_{l} \leq \bar{P}_{l} \cdot P^{\textrm{mod}}_{l}$ | N/A | `{Line,Transformer}-s_nom_modularity_upper` |
| $E_{n,s} \leq \bar{E}_{n,s} \cdot E^{\textrm{mod}}_{n,s}$ | N/A | `Store-e_nom_modularity_upper` |
| $H_{n,s} \leq \bar{H}_{n,s} \cdot H^{\textrm{mod}}_{n,s}$ | N/A | `StorageUnit-p_nom_modularity_upper` |

Where the maximum capacity is infinite, the [big-M constant](#big-m-parameter-configuration) is used in its place.
The minimum and maximum capacities `{p,s,e}_nom_{min,max}` always bound the **total** capacity of all modules.

By default, a continuous modular asset consists of a single module ($n^{\textrm{mod}} \in \{0, 1\}$), i.e. the module decides *whether* the asset is built, separately from *how much* capacity is built.
In that case, the lower capacity bound only has to be respected if the asset is built at all:

| Constraint | Dual Variable | Name |
|-------------------|------------------|------------------|
| $G_{n,s} \geq \underline{G}_{n,s} \cdot G^{\textrm{mod}}_{n,s}$ | N/A | `Generator-ext-p_nom-lower-modular` |

This replaces the plain lower bound described in [Upper and Lower Bounds](#upper-and-lower-bounds) and is set in `define_nominal_constraints_for_extendables()`.
If more than one module is allowed, the plain lower bound applies, as for fixed module sizes.

### Number of Modules

The number of modules is bounded by `n_mod_min` (default 0) and `n_mod_max`:

$$\underline{n}^{\textrm{mod}}_{*} \leq n^{\textrm{mod}}_{*} \leq \bar{n}^{\textrm{mod}}_{*}$$

If `n_mod_max` is not set (`NaN`), there is no upper limit for fixed module sizes and a single module (or `n_mod_min` modules, if larger) for continuous module capacities.
Setting `n_mod_min` forces a minimum number of modules to be built, e.g. to reproduce a known number of turbines whose total capacity is still optimised.

All modularity constraints are set in the function `define_modular_constraints()`.

### Modular and Committable Components

When extendable components additionally have **fixed module sizes** (`p_nom_mod > 0`) and are **committable** (`committable=True`), the formulation differs from the big-M approach above. Instead of a binary on/off status variable, the status variable becomes an **integer** representing the **number of committed modules** inside the component.

#### Module-Level Commitment Formulation

For modular committable components, two integer variables are introduced:

- $n^{\textrm{mod}}_{*}$: Number of modules built (determines total capacity)
- $u_{*,t}$: Number of modules committed at time $t$ (determines operational state)

The capacity is constrained to be a multiple of the module size:

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $G_{n,s} = n^{\textrm{mod}}_{n,s} \cdot \tilde{G}_{n,s}$ | `Generator-p_nom_modularity` |
    | $0 \leq u_{n,s,t} \leq n^{\textrm{mod}}_{n,s}$ | `Generator-status-p_nom-variable-upper` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $F_{l} = n^{\textrm{mod}}_{l} \cdot \tilde{F}_{l}$ | `Link-p_nom_modularity` |
    | $0 \leq u_{l,t} \leq n^{\textrm{mod}}_{l}$ | `Link-status-p_nom-variable-upper` |

The dispatch constraints enforce that power output respects the number of committed modules:

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $g_{n,s,t} \geq u_{n,s,t} \cdot \underline{g}_{n,s,t} \cdot \tilde{G}_{n,s}$ | `Generator-com-mod-p-lower` |
    | $g_{n,s,t} \leq u_{n,s,t} \cdot \bar{g}_{n,s,t} \cdot \tilde{G}_{n,s}$ | `Generator-com-mod-p-upper` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $f_{l,t} \geq u_{l,t} \cdot \underline{f}_{l,t} \cdot \tilde{F}_{l}$ | `Link-com-mod-p-lower` |
    | $f_{l,t} \leq u_{l,t} \cdot \bar{f}_{l,t} \cdot \tilde{F}_{l}$ | `Link-com-mod-p-upper` |

Note that the minimum and maximum part-load parameters ($\underline{g}$ and $\bar{g}$) apply **per committed module**, not to the total capacity.

#### Start-up and Shut-down for Modular Components

Start-up and shut-down variables track changes in the number of committed modules:

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $su_{n,s,t} \geq u_{n,s,t} - u_{n,s,t-1}$ | `Generator-com-transition-start-up` |
    | $sd_{n,s,t} \geq u_{n,s,t-1} - u_{n,s,t}$ | `Generator-com-transition-shut-down` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $su_{l,t} \geq u_{l,t} - u_{l,t-1}$ | `Link-com-transition-start-up` |
    | $sd_{l,t} \geq u_{l,t-1} - u_{l,t}$ | `Link-com-transition-shut-down` |

The start-up and shut-down cost terms in the objective function are multiplied by the number of modules being started or stopped.

!!! warning "Initial status affects start-up costs"

    The `status` attribute defaults to 1, which is used as $u_{t-1}$ for the first snapshot. For modular committable components, this means **one module is assumed to be already committed** at the start of the optimization. For example, if the optimizer invests in 5 modules and commits all 5 in the first timestep, only 4 start-up events are counted ($su_0 \geq 5 - 1 = 4$), and start-up costs are only charged for those 4 transitions. To charge start-up costs for all modules, set the initial status to 0 via `n.generators.loc[name, "status"] = 0`. This also applies to the non-modular (big-M) committable formulation.

These constraints are defined in the function `define_operational_constraints_for_committables()`.

??? note "Mapping of symbols to component attributes"

    === "Generator"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $g_{n,s,t}$       | `n.generators_t.p` | Decision variable |
        | $G_{n,s}$         | `n.generators.p_nom_opt` | Decision variable |
        | $n^{\textrm{mod}}_{n,s}$ | `n.model.variables['Generator-n_mod']` | Decision variable |
        | $u_{n,s,t}$       | `n.generators_t.status` | Decision variable (integer) |
        | $su_{n,s,t}$      | `n.generators_t.start_up` | Decision variable (integer) |
        | $sd_{n,s,t}$      | `n.generators_t.shut_down` | Decision variable (integer) |
        | $\tilde{G}_{n,s}$ | `n.generators.p_nom_mod` | Parameter |
        | $\underline{g}_{n,s,t}$ | `n.generators_t.p_min_pu` | Parameter |
        | $\bar{g}_{n,s,t}$ | `n.generators_t.p_max_pu` | Parameter |

    === "Link"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $f_{l,t}$         | `n.links_t.p` | Decision variable |
        | $F_{l}$           | `n.links.p_nom_opt` | Decision variable |
        | $n^{\textrm{mod}}_{l}$ | `n.model.variables['Link-n_mod']` | Decision variable |
        | $u_{l,t}$         | `n.links_t.status` | Decision variable (integer) |
        | $su_{l,t}$        | `n.links_t.start_up` | Decision variable (integer) |
        | $sd_{l,t}$        | `n.links_t.shut_down` | Decision variable (integer) |
        | $\tilde{F}_{l}$   | `n.links.p_nom_mod` | Parameter |
        | $\underline{f}_{l,t}$ | `n.links_t.p_min_pu` | Parameter |
        | $\bar{f}_{l,t}$   | `n.links_t.p_max_pu` | Parameter |


### Continuous Modules and Unit Commitment

Committable components with a single continuous module cannot use the big-M dispatch formulation of [Committable and Extendable Components](#committable-and-extendable-components) directly, because the dispatch bounds would depend on the product of the capacity, the commitment status and the module decision.
Instead, an auxiliary variable
$A_{n,s,t}$ (the **capacity available** in each snapshot, i.e. the product $G_{n,s} \cdot u_{n,s,t}$ of capacity and commitment status) is introduced and linearised against the module decision:

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $u_{n,s,t} \leq n^{\textrm{mod}}_{n,s}$ | `Generator-status-p_nom-variable-upper` |
    | $A_{n,s,t} \leq G_{n,s}$ | `Generator-p_nom_available_continuous` |
    | $A_{n,s,t} \leq \bar{G}_{n,s} \cdot u_{n,s,t}$ | `Generator-p_nom_available_binary` |
    | $A_{n,s,t} \geq G_{n,s} + (u_{n,s,t} - n^{\textrm{mod}}_{n,s}) \cdot \bar{G}_{n,s}$ | `Generator-p_nom_available_switch` |
    | $g_{n,s,t} \geq \underline{g}_{n,s,t} \cdot A_{n,s,t}$ | `Generator-com-mod-continuous-p-lower` |
    | $g_{n,s,t} \leq \bar{g}_{n,s,t} \cdot A_{n,s,t}$ | `Generator-com-mod-continuous-p-upper` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $u_{l,t} \leq n^{\textrm{mod}}_{l}$ | `Link-status-p_nom-variable-upper` |
    | $A_{l,t} \leq F_{l}$ | `Link-p_nom_available_continuous` |
    | $A_{l,t} \leq \bar{F}_{l} \cdot u_{l,t}$ | `Link-p_nom_available_binary` |
    | $A_{l,t} \geq F_{l} + (u_{l,t} - n^{\textrm{mod}}_{l}) \cdot \bar{F}_{l}$ | `Link-p_nom_available_switch` |
    | $f_{l,t} \geq \underline{f}_{l,t} \cdot A_{l,t}$ | `Link-com-mod-continuous-p-lower` |
    | $f_{l,t} \leq \bar{f}_{l,t} \cdot A_{l,t}$ | `Link-com-mod-continuous-p-upper` |

An asset without a module can never be committed ($u \leq n^{\textrm{mod}}$), so $A = 0$ and the dispatch is forced to zero.
An asset that is built and committed has $A_{n,s,t} = G_{n,s}$ and behaves exactly as an ordinary committable component.
When it is built but not committed, the availability constraint drives $A_{n,s,t}$ to zero without affecting the installed capacity.
The status, start-up and shut-down variables stay binary.

Like other modular committable components, they cannot be used with `n.optimize(linearized_unit_commitment=True)`, because relaxing the integrality of the module decision is not meaningful.
Doing so raises a `ValueError`.

!!! note "Initial status of modular committables"

    For [modular committable](#modular-and-committable-components) assets with fixed module sizes the `status` attribute defaults to 1.
    That is, one module is assumed to be committed before the optimisation horizon starts, which implicitly forces at least one module to be built.
    Set `status=0` and `up_time_before=0` to leave the module decision entirely to the optimiser.

!!! warning "Current limitations"

    - Modular components without a module size are only supported for extendable components.
      Setting `modular=True` on a component with a fixed capacity and `{p,s,e}_nom_mod=0` raises a `ValueError` before the model is built.
    - A committable component without a module size must consist of a single module (`n_mod_max <= 1`), because the capacity available from a subset of several continuous modules is not linear.
    - A committable component without a module size cannot also be maintainable.
    - A one-off cost for the asset as a whole cannot be combined with fixed-size modules, since `module_cost` is charged per module.

??? note "Mapping of symbols to component attributes"

    | Symbol | Attribute | Type |
    |-------------------|-----------|-------------|
    | $G^{\textrm{mod}}_{n,s}$ | `n.generators.n_mod_opt` | Decision variable (integer) |
    | $F^{\textrm{mod}}_{l}$ | `n.links.n_mod_opt` | Decision variable (integer) |
    | $A_{n,s,t}$       | `n.model.variables['Generator-available_p_nom']` | Decision variable |
    | $u_{n,s,t}$       | `n.generators_t.status` | Decision variable (binary) |
    | $\tilde{G}_{n,s}$ | `n.generators.p_nom_mod` | Parameter |
    | $\underline{G}_{n,s}$ | `n.generators.p_nom_min` | Parameter |
    | $\bar{G}_{n,s}$   | `n.generators.p_nom_max` | Parameter |
    | $\underline{n}^{\textrm{mod}}_{*}$ | `n.{generators,...}.n_mod_min` | Parameter |
    | $\bar{n}^{\textrm{mod}}_{*}$ | `n.{generators,...}.n_mod_max` | Parameter |
    | $M$               | auto-inferred or `committable_big_m` parameter | Parameter |

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
| Minimum down time (`min_down_time`) | ✓ | ✗ |
| Up time before (`up_time_before`) | ✓ | ✗ |
| Down time before (`down_time_before`) | ✓ | ✗ |

---

¹ For modular components, `p_min_pu` and `p_max_pu` apply to **each committed module**. The minimum/maximum power is calculated as `p_min_pu × p_nom_mod × status` where `status` is the number of committed modules.

² For modular + committable components, ramp limits are applied to the **total installed capacity** (similar to standard committable components), not per module. The ramp constraint is `p_t - p_{t-1} ≤ ramp_limit_up × p_nom_opt`, where `p_nom_opt` is the total capacity of all installed modules. This differs from the per-module behavior of minimum/maximum part-load constraints.


## Fixed Capacity

Additionally, the nominal capacity can be fixed to a certain value $\tilde{G}_{n,s}$ for generators, $\tilde{F}_{l}$ for links, $\tilde{R}_{m}$ for processes, $\tilde{P}_{l}$ for lines and transformers, and $\tilde{E}_{n,s}$ for stores, and $\tilde{H}_{n,s}$ for storage units. In this case, the nominal capacity is given by:

| Constraint | Dual Variable | Name |
|-------------------|------------------|------------------|
| $G_{n,s} = \tilde{G}_{n,s}$ | only in `n.model` | `Generator-p_nom_set` |
| $F_{l} = \tilde{F}_{l}$ | only in `n.model` | `Link-p_nom_set` |
| $R_{m} = \tilde{R}_{m}$ | only in `n.model` | `Process-p_nom_set` |
| $P_{l} = \tilde{P}_{l}$ | only in `n.model` | `{Line,Transformer}-s_nom_set` |
| $E_{n,s} = \tilde{E}_{n,s}$ | only in `n.model` | `Store-e_nom_set` |
| $H_{n,s} = \tilde{H}_{n,s}$ | only in `n.model` | `StorageUnit-p_nom_set` |

These constraints are set in the function `define_fixed_nominal_constraints()`.

!!! note "Why not just set `p_nom_extendable=False`?"

    Using `p_nom_extendable=False` means the capacity is fixed and not optimized. However, sometimes we need to fix the capacity to a specific value while still keeping track of the dual variables associated with capacity constraints. Setting `{p,s,e}_nom_set` allows for this while maintaining `p_nom_extendable=True`.


??? note "Mapping of symbols to component attributes"

    === "Generator"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $G_{n,s}$         | `n.generators.p_nom_opt` | Decision variable |
        | $G^{\textrm{mod}}_{n,s}$ | not stored | Decision variable |
        | $\underline{G}_{n,s}$ | `n.generators.p_nom_min` | Parameter |
        | $\bar{G}_{n,s}$   | `n.generators.p_nom_max` | Parameter |
        | $\tilde{G}_{n,s}$ | `n.generators.p_nom_mod` | Parameter |
        | $\hat{G}_{n,s}$   | `n.generators.p_nom_set` | Parameter |


    === "Link"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $F_{l}$           | `n.links.p_nom_opt` | Decision variable |
        | $F^{\textrm{mod}}_{l}$   | not stored | Decision variable |
        | $\underline{F}_{l}$ | `n.links.p_nom_min` | Parameter |
        | $\bar{F}_{l}$     | `n.links.p_nom_max` | Parameter |
        | $\tilde{F}_{l}$   | `n.links.p_nom_mod` | Parameter |
        | $\hat{F}_{l}$     | `n.links.p_nom_set` | Parameter |

    === "Process"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $R_{m}$           | `n.processes.p_nom_opt` | Decision variable |
        | $R^{\textrm{mod}}_{m}$   | not stored | Decision variable |
        | $\underline{R}_{m}$ | `n.processes.p_nom_min` | Parameter |
        | $\bar{R}_{m}$     | `n.processes.p_nom_max` | Parameter |
        | $\tilde{R}_{m}$   | `n.processes.p_nom_mod` | Parameter |
        | $\hat{R}_{m}$     | `n.processes.p_nom_set` | Parameter |

    === "Line"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $P_{l}$           | `n.lines.s_nom_opt` | Decision variable |
        | $P^{\textrm{mod}}_{l}$   | not stored | Decision variable |
        | $\underline{P}_{l}$ | `n.lines.s_nom_min` | Parameter |
        | $\bar{P}_{l}$     | `n.lines.s_nom_max` | Parameter |
        | $\tilde{P}_{l}$   | `n.lines.s_nom_mod` | Parameter |
        | $\hat{P}_{l}$     | `n.lines.s_nom_set` | Parameter |

    === "Transformer"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $P_{l}$           | `n.transformers.s_nom_opt` | Decision variable |
        | $P^{\textrm{mod}}_{l}$   | not stored | Decision variable |
        | $\underline{P}_{l}$ | `n.transformers.s_nom_min` | Parameter |
        | $\bar{P}_{l}$     | `n.transformers.s_nom_max` | Parameter |
        | $\tilde{P}_{l}$   | `n.transformers.s_nom_mod` | Parameter |
        | $\hat{P}_{l}$     | `n.transformers.s_nom_set` | Parameter |

    === "Store"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $E_{n,s}$         | `n.stores.e_nom_opt` | Decision variable |
        | $E^{\textrm{mod}}_{n,s}$   | not stored | Decision variable |
        | $\underline{E}_{n,s}$ | `n.stores.e_nom_min` | Parameter |
        | $\bar{E}_{n,s}$   | `n.stores.e_nom_max` | Parameter |
        | $\tilde{E}_{n,s}$ | `n.stores.e_nom_mod` | Parameter |
        | $\hat{E}_{n,s}$   | `n.stores.e_nom_set` | Parameter |

    === "Storage Unit"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $H_{n,s}$         | `n.storage_units.p_nom_opt` | Decision variable |
        | $H^{\textrm{mod}}_{n,s}$   | not stored | Decision variable |
        | $\underline{H}_{n,s}$ | `n.storage_units.p_nom_min` | Parameter |
        | $\bar{H}_{n,s}$   | `n.storage_units.p_nom_max` | Parameter |
        | $\tilde{H}_{n,s}$ | `n.storage_units.p_nom_mod` | Parameter |
        | $\hat{H}_{n,s}$   | `n.storage_units.p_nom_set` | Parameter |


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
