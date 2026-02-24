<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Energy Balances

The energy balance equations are the most important constraints, which enforces that incoming and outgoing energy flows balance out at each bus $n$ for each time $t$ (like Kirchhoff's current law for electrical buses, see [Linearised Power Flow](../optimization/power-flow.md)). Considering all components, the balance constraint is given by

$$\begin{gather*}\sum_{s} g_{n,s,t} + \sum_{s} \left(h_{n,s,t}^- - h_{n,s,t}^+ \right) + \sum_{s} h_{n,s,t}\\
+ \sum_{l} L_{n,l,t} f_{l,t} + \sum_{l} K_{n,l} p_{l,t} = \sum_{s} d_{n,s,t} \quad \leftrightarrow  \quad w_t^o\lambda_{n,t}\end{gather*}$$

where the **decision variables** are represented by:

- $d_{n,s,t}$ for the demand of [`Load`][pypsa.components.Loads] components
- $g_{n,s,t}$ for the dispatch of [`Generator`][pypsa.components.Generators] components
- $h_{n,s,t}^-$ for the discharging of [`StorageUnit`][pypsa.components.StorageUnits] components
- $h_{n,s,t}^+$ for the charging of [`StorageUnit`][pypsa.components.StorageUnits] components
- $f_{l,t}$ for the flow on the link [`Link`][pypsa.components.Links] at `bus0`
- $p_{l,t}$ for the power flow on the [`Line`][pypsa.components.Lines] and [`Transformer`][pypsa.components.Transformers] components at `bus0`

!!! note

    The orientation of each dispatch variable can be inverted or rescaled through the `sign` attribute of the component.

The **incidence matrices** $K_{n,l}$ for the [`Line`][pypsa.components.Lines] and [`Transformer`][pypsa.components.Transformers] components and $L_{n,l,t}$ for the [`Link`][pypsa.components.Links] components govern flows between buses. The incidence matrix $K_{n,l}$ takes non-zero values $-1$ if the line or transformer $l$ starts at bus $n$ and $1$ if it ends at
bus $n$. If $p_{l,t}>0$ it withdraws from the starting bus. The time-varying incidence matrix $L_{n,l,t}$ takes non-zero values $-1$ if the link $l$ starts at bus $n$ and efficiency $\eta_{n,l,t}$ if it ends at bus $n$. If $f_{l,t}>0$ it withdraws from `bus0` and feeds in $\eta_{n,l,t} f_{l,t}$ to `bus1`. For a link with more than two outputs (e.g. a combined heat and power plant), the incidence matrix $L_{n,l,t}$ has more than two non-zero entries with efficiencies $\eta_{n,l,t}$ for `bus2`, `bus3`, etc.. The entries may also be negative to denote additional inputs rather than multiple outputs.

When a link has a non-zero `delay` attribute, the output flow at snapshot $t$ corresponds to the input at an earlier source snapshot $s(t)$ rather than at $t$ itself. The source snapshot $s(t)$ is the latest snapshot such that $\tau(s) \leq \tau(t) - \delta$, where $\tau$ denotes the cumulative start time derived from `snapshot_weightings.generators` and $\delta$ is the delay value. The delayed contribution to the energy balance at output bus $n$ becomes:

$$L_{n,l,t} \cdot f_{l,s(t)}$$

With `cyclic_delay=True` (default), the source time wraps cyclically modulo the total horizon length, so energy sent in the last snapshots arrives at the first snapshots. With `cyclic_delay=False`, snapshots whose source time falls before the start of the horizon receive no flow ($f_{l,s(t)} = 0$), and energy sent in the tail snapshots that would arrive beyond the horizon is lost. See the [Link component page](../components/links.md#time-delayed-energy-delivery) for usage details.

The dual variable $\lambda_{n,t}$ represents the shadow price of the constraint (e.g. market clearing price, dynamic locational marginal prices) and is scaled by the snapshot weighting $w_t^o$ to yield units of currency per unit of energy regardless of the time resolution.

The energy balance constraints are set in the function `define_nodal_balance_constraints()` and is called `Bus-nodal_balance`.

??? note "Mapping of symbols to component attributes"

    | Symbol | Attribute | Type |
    |-------------------|-----------|-------------|
    | $g_{n,s,t}$       | `n.generators_t.p` | Decision variable |
    | $h_{n,s,t}^-$     | `n.storage_units_t.p_dispatch` | Decision variable |
    | $h_{n,s,t}^+$     | `n.storage_units_t.p_store` | Decision variable |
    | $h_{n,s,t}$       | `n.stores_t.p` | Decision variable |
    | $f_{l,t}$         | `n.links_t.p0` | Decision variable |
    | $p_{l,t}$         | `n.lines_t.p0` or `n.transformers_t.p0` | Decision variable |
    | $d_{n,s,t}$     | `n.loads_t.p_set` | Decision variable |
    | $\lambda_{n,t}$  | `n.buses_t.marginal_price` | Dual variable |
    | $K_{n,l}$ | Calculated internally by `n.incidence_matrix()` | Parameter |
    | $L_{n,l,t}$ | Calculated internally from `efficiency{i}` attributes | Parameter |
    | $\delta$ | `delay`, `delay2`, ... on `n.links` | Parameter |
    | $s(t)$ | Source snapshot mapping from `get_delay_source_indexer()` | Parameter |
    | $w_t^o$          | `n.snapshot_weightings.objective` | Parameter |
