<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Capacity Limits

## Upper and Lower Bounds

If the nominal capacity of a components is also the subject of optimisation (e.g. with decision variable $G_{n,s}$ for generators), limits on the installable capacity may also be introduced (e.g. $\underline{G}_{n,s}$ and $\bar{G}_{n,s}$):

| Constraint | Dual Variable | Name |
|-------------------|------------------|------------------|
| $G_{n,s} \geq \underline{G}_{n,s}$ | `n.generators.mu_lower` | `Generator-ext-p_nom-lower` |
| $G_{n,s} \leq \bar{G}_{n,s}$ | `n.generators.mu_upper` | `Generator-ext-p_nom-upper` |
| $F_{l} \geq \underline{F}_{l}$ | `n.links.mu_lower` | `Link-ext-p_nom-lower` |
| $F_{l} \leq \bar{F}_{l}$ | `n.links.mu_upper` | `Link-ext-p_nom-upper` |
| $P_{l} \geq \underline{P}_{l}$ | `n.{lines,transformers}.mu_lower` | `{Line,Transformer}-ext-s_nom-lower` |
| $P_{l} \leq \bar{P}_{l}$ | `n.{lines, transformers}.mu_upper` | `{Line,Transformer}-ext-s_nom-upper` |
| $E_{n,s} \geq \underline{E}_{n,s}$ | `n.stores.mu_lower` | `Store-ext-e_nom-lower` |
| $E_{n,s} \leq \bar{E}_{n,s}$ | `n.stores.mu_upper` | `Store-ext-e_nom-upper` |
| $H_{n,s} \geq \underline{H}_{n,s}$ | `n.storage_units.mu_lower` | `StorageUnit-ext-p_nom-lower` |

These constraints are set in the function `define_nominal_constraints_for_extendables`.

## Modularity Constraints

The capacity expansion can be further constrained to be a multiple (e.g. $G^{\textrm{mod}}_{n,s} \in \mathbb{N}$) of a modular capacity (e.g. $\tilde{G}_{n,s}$) to represent fixed block sizes of added components (e.g. fixed block size of a nuclear power plant or a fixed capacity of a new circuit).

If `{p,s,e}_nom_mod>0`, the nominal capacity is given by:

| Constraint | Dual Variable | Name |
|-------------------|------------------|------------------|
| $G_{n,s} = G^{\textrm{mod}}_{n,s} \cdot \tilde{G}_{n,s}$ | N/A | `Generator-p_nom_modularity` |
| $F_{l} = F^{\textrm{mod}}_{l} \cdot \tilde{F}_{l}$ | N/A | `Link-p_nom_modularity` |
| $P_{l} = P^{\textrm{mod}}_{l} \cdot \tilde{P}_{l}$ | N/A | `{Line,Transformer}-s_nom_modularity` |
| $E_{n,s} = E^{\textrm{mod}}_{n,s} \cdot \tilde{E}_{n,s}$ | N/A | `Store-e_nom_modularity` |
| $H_{n,s} = H^{\textrm{mod}}_{n,s} \cdot \tilde{H}_{n,s}$ | N/A | `StorageUnit-p_nom_modularity` |

These constraints are set in the function `define_modular_constraints()`.


## Fixed Capacity

Additionally, the nominal capacity can be fixed to a certain value $\tilde{G}_{n,s}$ for generators, $\tilde{F}_{l}$ for links, $\tilde{P}_{l}$ for lines and transformers, and $\tilde{E}_{n,s}$ for stores, and $\tilde{H}_{n,s}$ for storage units. In this case, the nominal capacity is given by:

| Constraint | Dual Variable | Name |
|-------------------|------------------|------------------|
| $G_{n,s} = \tilde{G}_{n,s}$ | only in `n.model` | `Generator-p_nom_set` |
| $F_{l} = \tilde{F}_{l}$ | only in `n.model` | `Link-p_nom_set` |
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

</div>

