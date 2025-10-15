<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Objective

The objective is to **minimise total system costs** (i.e. the sum of all costs) subject to
meeting all constraints of the optimisation problem (e.g. energy balance, dispatch limits, etc.) for the selected snapshots.

The objective function is composed of investment costs and multiple forms of operational costs, such as marginal costs, quadratic marginal costs, marginal storage costs, spillage costs, start-up, shut-down and stand-by costs.

Which cost components are included in the objective function depends on the data provided for the network components. For instance, if no extendable components are present (`{p,s,e}_nom_extendable=False`), the investment cost terms are skipped. Or, if no components are marked as committable (`committable=False`), the start-up, shut-down and stand-by costs are skipped. All cost coefficients default to zero, so that the objective function is empty if no costs are defined.

## Investment Costs

For extendable components (`{p,s,e}_nom_extendable=True`), the investment costs of expanding their capacity are given by

$$\sum_{n,s} c_{n,s} G_{n,s} + \sum_{n,s} c_{n,s} H_{n,s} + \sum_{n,s} c_{n,s} E_{n,s} + \sum_{l} c_{l} F_l + \sum_{l} c_{l} P_l$$

where $c_{*}$ are the capital costs per unit of nominal capacity for generators ($G_{n,s} \in \mathbb{R}$), storage units ($H_{n,s} \in \mathbb{R}$), stores ($E_{n,s} \in \mathbb{R}$), links ($F_l \in \mathbb{R}$), and lines and transformers ($P_l \in \mathbb{R}$). The subscript $n$ labels the bus, $s$ labels the particular generator/storage type at the bus, and $l$ labels the branch.
The decision variables $G_{n,s}$, $H_{n,s}$, $F_l$, and $P_l$ are the nominal power capacities, whereas $E_{n,s}$ is the nominal energy capacity of the store. The capital costs are given in currency/MW for generators, links, lines and transformers, and in currency/MWh for stores.

To minimise **long-run annual system costs** (currency/a), capital costs for components should be set to annualised investment costs (i.e. currency/MW/a and currency/MWh/a ), marginal costs for dispatch in currency/MWh, and the weightings (h/a) are chosen such that $\sum_t w_t^o = 8760$ hours per annum.

If no extendable components are present, only the dispatch of the components is optimised as in a **short-run market model**.


??? note "Mapping of symbols to attributes"

    | Symbol | Attribute | Type |
    |-------------------|-----------|-------------|
    | $G_{n,s}$         | `n.generators.p_nom_opt` | Decision variable |
    | $H_{n,s}$         | `n.storage_units.p_nom_opt` | Decision variable |
    | $E_{n,s}$         | `n.stores.p_nom_opt` | Decision variable |
    | $F_l$             | `n.links.p_nom_opt` | Decision variable |
    | $P_l$             | `n.lines.s_nom_opt` | Decision variable |
    | $c_{n,s}$         | `n.{generators,storage_units,stores}.capital_cost` | Parameter |
    | $c_{l}$           | `n.{links,lines,transformers}.capital_cost` | Parameter |

## Marginal Costs

The marginal costs of dispatch are given by

$$+ \sum_{t} w_t^o \left( \sum_{n,s} o_{n,s,t} g_{n,s,t} + \sum_{n,s} o_{n,s,t} h_{n,s,t}^- + \sum_{n,s} o_{n,s,t} h_{n,s,t} + \sum_{l} o_{l,t} f_{l,t} \right)$$

where $o_{*}$ are the marginal costs per unit of power for generator ($g_{n,s,t} \in \mathbb{R}$), storage unit ($h_{n,s,t}^- \in \mathbb{R}$), store ($h_{n,s,t} \in \mathbb{R}$) and link ($f_{l,t} \in \mathbb{R}$) dispatch. Note here the difference between storage unit dispatch $h_{n,s,t}^-$ (where costs are only incurred for discharging) and store dispatch $h_{n,s,t}$ (where marginal costs incur a cost for discharging and a revenue for charging). The subscript $t$ indicates the snapshot.

??? note "Mapping of symbols to attributes"

    | Symbol | Attribute | Type |
    |-------------------|-----------|-------------|
    | $g_{n,s,t}$       | `n.generators_t.p` | Decision variable |
    | $h_{n,s,t}^-$     | `n.storage_units_t.p_dispatch` | Decision variable |
    | $h_{n,s,t}$       | `n.stores_t.p` | Decision variable |
    | $f_{l,t}$         | `n.links_t.p` | Decision variable |
    | $o_{n,s,t}$       | `n.{generators,storage_units,stores}_t.marginal_cost` | Parameter |
    | $o_{l,t}$         | `n.links_t.marginal_cost` | Parameter |
    | $w_t^o$           | `n.snapshots.weightings.objective` | Parameter |


## Quadratic Marginal Costs

Quadratic marginal costs can be included, which turn the problem into a quadratic program (QP):

$$+ \sum_{t} w_t^o \left( \sum_{n,s} qmc_{n,s,t} g_{n,s,t}^2 + \sum_{n,s} qmc_{n,s,t} {h_{n,s,t}^-}^2 + \sum_{n,s} qmc_{n,s,t} h_{n,s,t}^2  + \sum_{l} qmc_{l,t} f_{l,t}^2 \right)$$

where $qmc_{*}$ are the quadratic marginal costs per unit of dispatch for generators ($g_{n,s,t}$), storage units ($h_{n,s,t}^-$), stores ($h_{n,s,t}$) and links ($f_{l,t}$), weighted by the objective snapshot weightings $w_t^o$.

??? note "Mapping of symbols to attributes"

    | Symbol | Attribute | Type |
    |-------------------|-----------|-------------|
    | $g_{n,s,t}$       | `n.generators_t.p` | Decision variable |
    | $h_{n,s,t}^-$     | `n.storage_units_t.p_dispatch` | Decision variable |
    | $h_{n,s,t}$       | `n.stores_t.p` | Decision variable |
    | $f_{l,t}$         | `n.links_t.p` | Decision variable |
    | $qmc_{n,s,t}$     | `n.{generators,storage_units,stores}_t.marginal_cost_quadratic` | Parameter |
    | $qmc_{l,t}$       | `n.links_t.marginal_cost_quadratic` | Parameter |
    | $w_t^o$           | `n.snapshots.weightings.objective` | Parameter |

## Marginal Storage Costs

Marginal storage costs can also be applied to storage units and stores, which represent the cost of holding energy in storage. The objective function includes these costs as follows:

$$+ \sum_{t} w_t^o \left( \sum_{n,s} mcs_{n,s,t} soc_{n,s,t} + \sum_{n,s} mcs_{n,s,t} e_{n,s,t} \right)$$

where $mcs_{*}$ are the marginal storage costs per unit of energy for storage units ($soc_{n,s,t} \in \mathbb{R}$) and stores ($e_{n,s,t} \in \mathbb{R}$), weighted by the objective snapshot weightings $w_t^o$.

??? note "Mapping of symbols to attributes"

    | Symbol | Attribute | Type |
    |-------------------|-----------|-------------|
    | $soc_{n,s,t}$     | `n.storage_units_t.state_of_charge` | Decision variable |
    | $e_{n,s,t}$       | `n.stores_t.e` | Decision variable |
    | $mcs_{n,s,t}$     | `n.storage_units_t.marginal_cost_storage` | Parameter |
    | $mcs_{n,s,t}$     | `n.stores_t.marginal_cost_storage` | Parameter |
    | $w_t^o$           | `n.snapshots.weightings.objective` | Parameter |

## Spillage Costs

Spillage costs, e.g. sending water in a hydro reservoir over a spillway without generating electricity, can be given for storage units by

$$
+ \sum_{n,s,t} w_t^o sc_{n,s,t} \textrm{spillage}_{n,s,t}
$$

where $sc_{n,s,t}$ are the spillage costs per unit of spillage ($\textrm{spillage}_{n,s,t} \in \mathbb{R}$), weighted by the objective snapshot weightings $w_t^o$.

??? note "Mapping of symbols to attributes"

    | Symbol | Attribute | Type |
    |-------------------|-----------|-------------|
    | $\textrm{spillage}_{n,s,t}$ | `n.storage_units_t.spillage` | Decision variable |
    | $sc_{n,s,t}$      | `n.storage_units_t.spill_cost` | Parameter |
    | $w_t^o$           | `n.snapshots.weightings.objective` | Parameter |

## Start-Up, Shut-Down, Stand-By Costs

For generators and links with unit commitment (`committable=True`), stand-by costs $sbc_{*,t}$, start-up costs $suc_{*,t}$ and shut-down costs $sdc_{*,t}$ are given by

=== "Generator"

    $$+ \sum_{n,s,t} w_t^o sbc_{n,s,t} u_{n,s,t} + \sum_{n,s} suc_{n,s} su_{n,s,t} + \sum_{n,s} sdc_{n,s} sd_{n,s,t}$$

=== "Link"

    $$+ \sum_{l,t} w_t^o sbc_{l,t} u_{l,t} + \sum_{l,t} suc_{l,t} su_{l,t} + \sum_{l,t} sdc_{l,t} sd_{l,t}$$

where $sbc_{*,t}$, $suc_{*,t}$, and $sdc_{*,t}$ are the stand-by, start-up, and shut-down costs linked to the status ($u_{*,t} \in \mathbb{B}$), start-up ($su_{*,t} \in \mathbb{B}$), and shut-down ($sd_{*,t} \in \mathbb{B}$) unit commitment variables. Only the stand-by costs are weighted by the objective snapshot weightings $w_t^o$. Components with unit commitment constraints turn the problem into a mixed-integer linear program (MILP).

??? note "Mapping of symbols to attributes"

    === "Generator"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $u_{n,s,t}$       | `n.generators_t.status` | Decision variable |
        | $su_{n,s,t}$      | `n.generators_t.start_up` | Decision variable |
        | $sd_{n,s,t}$      | `n.generators_t.shut_down` | Decision variable |
        | $sbc_{n,s,t}$     | `n.generators_t.stand_by_cost` | Parameter |
        | $suc_{n,s}$       | `n.generators.start_up_cost` | Parameter |
        | $sdc_{n,s}$       | `n.generators.shut_down_cost` | Parameter |
        | $w_t^o$           | `n.snapshots.weightings.objective` | Parameter |

    === "Link"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $u_{l,t}$         | `n.links_t.status` | Decision variable |
        | $su_{l,t}$        | `n.links_t.start_up` | Decision variable |
        | $sd_{l,t}$        | `n.links_t.shut_down` | Decision variable |
        | $sbc_{l,t}$       | `n.links_t.stand_by_cost` | Parameter |
        | $suc_{l,t}$       | `n.links.start_up_cost` | Parameter |
        | $sdc_{l,t}$       | `n.links.shut_down_cost` | Parameter |
        | $w_t^o$           | `n.snapshots.weightings.objective` | Parameter |

Some decision variables do not show up in the objective function, such as the power flow on lines and transformers ($p_{l,t} \in \mathbb{R}$) and the storage unit charging ($h_{n,s,t}^+ \in \mathbb{R}$). They are only used to enforce constraints, e.g. the power flow on lines and transformers.

The objective function is defined in the function `define_objective()`.
