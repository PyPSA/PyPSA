<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Dispatch Limits

Each [`Generator`][pypsa.components.Generators] has a dispatch variable $g_{n,s,t}$ where $n$ labels the bus, $s$ labels the particular generator at the bus and $t$ labels the snapshot. Each [`Link`][pypsa.components.Links] has a dispatch variable $f_{l,t}$ where $l$ labels the link and $t$ labels the snapshot. Each [`Line`][pypsa.components.Lines] and [`Transformer`][pypsa.components.Transformers] has a dispatch variable $p_{l,t}$ where $l$ labels the line/transformer and $t$ labels the snapshot.

!!! note "Dispatch limits of [`Store`][pypsa.components.Stores] and [`StorageUnit`][pypsa.components.StorageUnits]"

    Dispatch limits of stores and storage units are described together with their
    storage consistency equations in the [Storage](storage.md) section.

## Non-extendable Components

For non-extendable components (`{p,s}_nom_extendable=False`), the dispatch is limited by:

| Constraint | Dual Variable | | Name |
|------------|---------------|--|------|
| $g_{n,s,t} \geq \underline{g}_{n,s,t} \cdot \hat{g}_{n,s}$ | $w_t^o \underline{\mu}_{n,s,t}$ | `n.generators_t.mu_lower` | `Generator-fix-p-lower` |
| $g_{n,s,t} \leq \bar{g}_{n,s,t} \cdot \hat{g}_{n,s}$ | $w_t^o \bar{\mu}_{n,s,t}$ | `n.generators_t.mu_upper` | `Generator-fix-p-upper` |
| $f_{l,t} \geq \underline{f}_{l,t} \cdot \hat{f}_{l}$ | $w_t^o \underline{\mu}_{l,t}$ | `n.links_t.mu_lower` | `Link-fix-p-lower` |
| $f_{l,t} \leq \bar{f}_{l,t} \cdot \hat{f}_{l}$ | $w_t^o \bar{\mu}_{l,t}$ | `n.links_t.mu_upper` | `Link-fix-p-upper` |
| $p_{l,t} \geq - \bar{p}_{l,t} \cdot \hat{p}_{l}$ | $w_t^o \underline{\mu}_{l,t}$ | `n.{lines,transformers}_t.mu_lower` | `Line-fix-p-lower` |
| $p_{l,t} \leq \bar{p}_{l,t} \cdot \hat{p}_{l}$ | $w_t^o \bar{\mu}_{l,t}$ | `n.{lines,transformers}_t.mu_upper` | `Line-fix-p-upper` |

where $\hat{g}_{n,s}$, $\hat{f}_{l}$, and $\hat{p}_{l}$ are the nominal capacities; $\underline{g}_{n,s,t}$, $\underline{f}_{l,t}$, $\bar{g}_{n,s,t}$, and $\bar{f}_{l,t}$ and $\bar{p}_{l,t}$ are time-dependent restrictions on the dispatch given per unit of nominal capacity (e.g. due to wind availability for generators or dynamic line rating and security margins for lines). 

These constraints are set in the function `define_operational_constraints_for_non_extendables()`.

## Extendable Components

For extendable components  (`{p,s}_nom_extendable=True`), the dispatch is limited by:

Constraint | Dual Variable | | Name |
|------------|---------------|--|------|
|  $g_{n,s,t} \geq \underline{g}_{n,s,t} \cdot G_{n,s}$ | $w_t^o \underline{\mu}_{n,s,t}$ | `n.generators_t.mu_lower` | `Generator-ext-p-lower` |
| $g_{n,s,t} \leq \bar{g}_{n,s,t} \cdot G_{n,s}$ | $w_t^o \bar{\mu}_{n,s,t}$ | `n.generators_t.mu_upper` | `Generator-ext-p-upper` |
| $f_{l,t} \geq \underline{f}_{l,t} \cdot F_{l}$ | $w_t^o \underline{\mu}_{l,t}$ | `n.links_t.mu_lower` | `Link-ext-p-lower` |
| $f_{l,t} \leq \bar{f}_{l,t} \cdot F_{l}$ | $w_t^o \bar{\mu}_{l,t}$ | `n.links_t.mu_upper` | `Link-ext-p-upper` |
| $p_{l,t} \geq - \bar{p}_{l,t} \cdot P_{l}$ | $w_t^o \underline{\mu}_{l,t}$ | `n.{lines,transformers}_t.mu_lower` | `Line-ext-p-lower` |
| $p_{l,t} \leq \bar{p}_{l,t} \cdot P_{l}$ | $w_t^o \bar{\mu}_{l,t}$ | `n.{lines,transformers}_t.mu_upper` | `Line-ext-p-upper` |

where $G_{n,s}$, $F_{l}$, and $P_{l}$ are the nominal capacities to be optimised. 

These constraints are set in the function `define_operational_constraints_for_extendables()`.

## Fixed Dispatch

Additionally, the dispatch can be fixed to a certain value $\tilde{g}_{n,s,t}$ for generators and $\tilde{f}_{l,t}$ for links. In this case, the dispatch is given by:


Constraint | Dual Variable | | Name |
|------------|---------------|--|------|
| $g_{n,s,t} = \tilde{g}_{n,s,t}$ | $w_t^o  \tilde{\mu}_{n,s,t}$ | `n.generators_t.mu_p_set` | `Generator-p_set` |
| $f_{l,t} = \tilde{f}_{l,t}$ | $w_t^o  \tilde{\mu}_{l,t}$ | `n.links_t.mu_p_set` | `Link-p_set` |

These constraints are set in the function `define_fixed_operation_constraints()`.

## Volume Limits

Generators and links can also have volume limits, i.e. the total dispatch over all snapshots must be above a minimum $\underline{e}_{*}$ or below a maximum $\bar{e}_{*}$.

| Constraint | Dual Variable | Name |
|-------------------|------------------|------------------|
| $\sum_t w_t^g g_{n,s,t} \geq \underline{e}_{n,s} \quad \forall n,s$ | only in `n.model` | `Generator-e_sum_min` |
| $\sum_t w_t^g g_{n,s,t} \leq \bar{e}_{n,s} \quad \forall n,s$ | only in `n.model` | `Generator-e_sum_max` |
| $\sum_t w_t^g f_{l,t} \geq \underline{e}_{l} \quad \forall l$ | only in `n.model` | `Link-e_sum_min` |
| $\sum_t w_t^g f_{l,t} \leq \bar{e}_{l} \quad \forall l$ | only in `n.model` | `Link-e_sum_max` |

These constraints are set in the function `define_total_supply_constraints()`.


!!! note "Mapping of symbols to attributes"

    === "Generator"

        | Symbol | Attribute | Type | 
        |-------------------|-----------|-------------|
        | $g_{n,s,t}$       | `n.generators_t.p` | Decision variable |
        | $G_{n,s}$         | `n.generators.p_nom_opt` | Decision variable |
        | $\hat{g}_{n,s}$   | `n.generators.p_nom` | Parameter |
        | $\underline{g}_{n,s,t}$ | `n.generators_t.p_min_pu` | Parameter |
        | $\bar{g}_{n,s,t}$ | `n.generators_t.p_max_pu` | Parameter |
        | $\tilde{g}_{n,s,t}$ | `n.generators_t.p_set` | Parameter |
        | $\underline{e}_{n,s}$        | `n.generators.e_sum_min` | Parameter |
        | $\bar{e}_{n,s}$        | `n.generators.e_sum_max` | Parameter |
        | $w_t^g$           | `n.snapshots.weightings.generators` | Parameter |
        | $w_t^o$             | `n.snapshots.weightings.objective` | Parameter |

    === "Link"

        
        | Symbol | Attribute | Type | 
        |-------------------|-----------|-------------|
        | $f_{l,t}$         | `n.links_t.p` | Decision variable |
        | $F_{l}$           | `n.links.p_nom_opt` | Decision variable |
        | $\hat{f}_{l}$     | `n.links.p_nom` | Parameter |
        | $\underline{f}_{l,t}$ | `n.links_t.p_min_pu` | Parameter |
        | $\bar{f}_{l,t}$   | `n.links_t.p_max_pu` | Parameter |
        | $\tilde{f}_{l,t}$ | `n.links_t.p_set` | Parameter |
        | $\underline{e}_{l}$          | `n.links.e_sum_min` | Parameter |
        | $\bar{e}_{l}$          | `n.links.e_sum_max` | Parameter |
        | $w_t^g$           | `n.snapshots.weightings.generators` | Parameter |
        | $w_t^o$             | `n.snapshots.weightings.objective` | Parameter |

    === "Line"

        | Symbol | Attribute | Type | 
        |-------------------|-----------|-------------|
        | $p_{l,t}$         | `n.{lines,transformers}_t.p0` | Decision variable |
        | $P_{l}$           | `n.{lines,transformers}.s_nom_opt` | Decision variable |
        | $\hat{p}_l$       | `n.{lines,transformers}.s_nom` | Parameter |
        | $\bar{p}_{l,t}$   | `n.{lines,transformers}_t.s_max_pu` | Parameter |
        | $w_t^g$           | `n.snapshots.weightings.generators` | Parameter |
        | $w_t^o$             | `n.snapshots.weightings.objective` | Parameter |


