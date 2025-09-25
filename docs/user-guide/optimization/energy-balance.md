# Energy Balances

The energy balance equations are the most important constraints, which enforces that incoming and outgoing energy flows balance out at each bus $n$ for each time $t$ (like Kirchhoff's current law for electrical buses, see [Linearised Power Flow](power-flow.md)). Considering all components, the balance constraint is given by

$$\begin{gather*}\sum_{s} g_{n,s,t} + \sum_{s} \left(h_{n,s,t}^- - h_{n,s,t}^+ \right) + \sum_{s} h_{n,s,t}\\
+ \sum_{l} L_{n,l,t} f_{l,t} + \sum_{l} K_{n,l} p_{l,t} = \sum_{s} d_{n,s,t} \quad \leftrightarrow  \quad w_t^o\lambda_{n,t}\end{gather*}$$

where the **decision variables** are represented by:

- $d_{n,s,t}$ for the demand of [`Load`](/api/components/types/loads) components
- $g_{n,s,t}$ for the dispatch of [`Generator`](/api/components/types/generators) components
- $h_{n,s,t}^-$ for the discharging of [`StorageUnit`](/api/components/types/storage_units) components
- $h_{n,s,t}^+$ for the charging of [`StorageUnit`](/api/components/types/storage_units) components
- $f_{l,t}$ for the flow on the link [`Link`](/api/components/types/links) at `bus0`
- $p_{l,t}$ for the power flow on the [`Line`](/api/components/types/lines) and [`Transformer`](/api/components/types/transformers) components at `bus0`

!!! note

    The orientation of each dispatch variable can be inverted or rescaled through the `sign` attribute of the component.

The **incidence matrices** $K_{n,l}$ for the [`Line`](/api/components/types/lines) and [`Transformer`](/api/components/types/transformers) components and $L_{n,l,t}$ for the [`Link`](/api/components/types/links) components govern flows between buses. The incidence matrix $K_{n,l}$ takes non-zero values $-1$ if the line or transformer $l$ starts at bus $n$ and $1$ if it ends at
bus $n$. If $p_{l,t}>0$ it withdraws from the starting bus. The time-varying incidence matrix $L_{n,l,t}$ takes non-zero values $-1$ if the link $l$ starts at bus $n$ and efficiency $\eta_{n,l,t}$ if it ends at bus $n$. If $f_{l,t}>0$ it withdraws from `bus0` and feeds in $\eta_{n,l,t} f_{l,t}$ to `bus1`. For a link with more than two outputs (e.g. a combined heat and power plant), the incidence matrix $L_{n,l,t}$ has more than two non-zero entries with efficiencies $\eta_{n,l,t}$ for `bus2`, `bus3`, etc.. The entries may also be negative to denote additional inputs rather than multiple outputs.

The dual variable $\lambda_{n,t}$ represents the shadow price of the constraint (e.g. market clearing price, dynamic locational marginal prices) and is scaled by the snapshot weighting $w_t^o$ to yield units of currency per unit of energy regardless of the time resolution.

The energy balance constraints are set in the function `define_nodal_balance_constraints()` and is called `Bus-nodal_balance`.

!!! note "Mapping of symbols to component attributes"

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
    | $w_t^o$          | `n.snapshot_weightings.objective` | Parameter |
