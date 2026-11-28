<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Energy Balances

The energy balance equations are the most important constraints. They enforce that incoming and outgoing energy flows balance out at each bus $n$, in each scenario $s$ and at each snapshot $t$ (like Kirchhoff's current law for electrical buses, see [Linearised Power Flow](../optimization/power-flow.md)). The balance holds at each bus that has a component attached:

<!-- md:spec Bus_nodal_balance Bus_injection -->

Each component adds its injection at its bus. A positive term feeds energy in, a negative term withdraws it:

<!-- md:spec Generator_injection Load_injection StorageUnit_injection Store_injection Link_injection Process_injection Line_injection Transformer_injection -->

The dispatch $g_{i,s,t}$ of a [`Generator`][pypsa.components.Generators] $i$, the discharge $h^{-}_{s,t,v}$ and charge $h^{+}_{s,t,v}$ of a [`StorageUnit`][pypsa.components.StorageUnits] $v$, the dispatch $h_{s,t,y}$ of a [`Store`][pypsa.components.Stores] $y$ and the demand $d_{j,s,t}$ of a [`Load`][pypsa.components.Loads] $j$ enter at the bus of the component.

!!! note

    The orientation of each dispatch variable can be inverted or rescaled through the `sign` attribute of the component.

The flow $p_{l,s,t}$ on a [`Line`][pypsa.components.Lines] $l$ (and on a [`Transformer`][pypsa.components.Transformers] $q$) withdraws from `bus0` and feeds in at `bus1` if $p_{l,s,t}>0$. With [lossy transmission](../optimization/power-flow.md#linear-loss-approximation), each end loses half of the loss $\psi_{l,s,t}$.

The flow $f_{k,s,t}$ on a [`Link`][pypsa.components.Links] $k$ withdraws from `bus0`: port 0 has the efficiency $-1$. Each further port $\beta$ (`bus1`, `bus2`, ...) receives what arrives there: $f_{k,s,t}$ times the efficiency $\eta_{\beta,s,t}$ of the port. For a link with more than two outputs (e.g. a combined heat and power plant), there are more ports with their own efficiencies. The efficiencies may also be negative to denote additional inputs rather than multiple outputs.

For processes, the port of each `bus{i}` receives the internal dispatch $r_{m,s,t}$ of a [`Process`][pypsa.components.Processes] $m$ times the `rate{i}` attribute of the port. A negative rate denotes an input.

When a link has a non-zero `delay` attribute, the output flow at snapshot $t$ corresponds to the input at an earlier source snapshot rather than at $t$ itself. The source snapshot is the latest snapshot whose cumulative start time, derived from `snapshot_weightings.generators`, is at most the start time of $t$ minus the delay:

<!-- md:spec Link_port_arrival Link_port_linear_arrival -->

With `cyclic_delay=True` (default), the source time wraps cyclically modulo the total horizon length, so energy sent in the last snapshots arrives at the first snapshots. With `cyclic_delay=False`, snapshots whose source time falls before the start of the horizon receive no flow, and energy sent in the tail snapshots that would arrive beyond the horizon is lost. See the [Link component page](../components/links.md) for usage details. Processes delay their ports in the same way.

The dual variable $\lambda_{n,s,t}$ of the balance represents the shadow price of the constraint (e.g. market clearing price, dynamic locational marginal prices). PyPSA scales it by the snapshot weighting $w^{\mathrm{o}}_t$ to yield units of currency per unit of energy regardless of the time resolution, and stores it in `n.buses_t.marginal_price`.

The energy balance constraints are called `Bus-nodal_balance`.

??? note "Symbols"

    <!-- md:spec-legend Bus_nodal_balance Generator_injection Load_injection StorageUnit_injection Store_injection Link_injection Link_port_linear_arrival Process_injection Line_injection Transformer_injection -->
