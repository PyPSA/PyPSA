<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Flow-based market coupling

Flow-based market coupling (FBMC) replaces the transmission grid constraints between market zones by a compact set of linear constraints on the zones' net positions. Instead of resolving every node and line, the market treats each zone as a copper plate and trades against a few linear limits that capture how zonal exchanges load the critical grid elements. This is the capacity-allocation method used in parts of the European day-ahead market, notably the Core region.[^core-ccm] For an introduction to the method and its parameters, see Van den Bergh et al. (2016)[^vandenbergh] and Schönheit et al. (2021)[^schonheit]. The [`FlowBasedConstraint`](../components/flow-based-constraints.md) component stores this domain of feasible net positions.

!!! info "See Also"

    - [:material-notebook: Flow-based market coupling example](../../examples/flow-based-market-coupling.ipynb)
    - [`FlowBasedConstraint` component](../components/flow-based-constraints.md)

!!! note "Naming Convention"

    The net-position variable is `FlowBasedConstraint-net_position`, the domain half-spaces are `FlowBasedConstraint-domain` , and the zero-sum balance is `FlowBasedConstraint-balance`.

## Net position

The net position $NP_{z,t}$ of a market zone $z$ at snapshot $t$ is its net export. It is added as a decision variable and injected directly into the [nodal balance](energy-balance.md) of its zone bus,

$$g_{z,t} - d_{z,t} - NP_{z,t} = 0 .$$

The solved net positions are written to `n.buses_t.net_position`.

## Domain constraints

Each critical network element (CNEC) $c$ — one component row, a monitored grid constraint such as a line under a given outage — contributes one half-space that bounds a weighted sum of net positions by its remaining available margin $\text{RAM}_{c,t}$, the headroom left on that element for cross-zonal trade:

$$\sum_{z} \text{PTDF}_{c,z,t}\, NP_{z,t} \;\le\; \text{RAM}_{c,t} \quad \leftrightarrow \quad \mu_{c,t}$$

The sensitivities $\text{PTDF}_{c,z,t}$ are the zonal power transfer distribution factors (`zonal_ptdf`); both they and the RAM may be static or vary by snapshot, and the constraint broadcasts over $t$ either way. The shadow price $\mu_{c,t}$ is written to `mu_domain` when optimising with `assign_all_duals=True`. A single zero-sum balance closes the copper plate across the zones,

$$\sum_{z} NP_{z,t} = 0.$$

These constraints are added in `define_flow_based_constraints()`; the net-position injection is added in `define_nodal_balance_constraints()`.

## Controllable link flows (AHC and EvFB)

A domain column may name a [`Link`](../components/links.md) instead of a zone [`Bus`](../components/buses.md), e.g. representing a controllable HVDC corridor. These hybrid-coupling schemes extend the flow-based domain to borders that are otherwise not flow-based; for their market-design background see Estermann et al. (2025)[^estermann]. Two cases arise, distinguished only by the link's endpoints:

- **Advanced hybrid coupling (AHC):** a border from a flow-based zone bus to an *external* virtual hub bus $v$.
- **Evolved flow-based coupling (EvFB):** an HVDC *between two flow-based zones*.

The flow through the link loads its CNECs through its own column in the zonal PTDF, so the domain constraint gains a link-flow term with $f_{\ell,t}$ the link flow (`Link-p`, in its `bus0 -> bus1` direction):

$$\sum_{z} \text{PTDF}_{c,z,t}\, NP_{z,t} + \sum_{\ell} \text{PTDF}_{c,\ell,t}\, f_{\ell,t} \;\le\; \text{RAM}_{c,t} \quad \leftrightarrow \quad \mu_{c,t}$$

### Keeping net positions consistent

A `Link` is a physical component, so PyPSA's nodal balance already adds $-f_{\ell,t}$ at `bus0` and $+\eta_\ell f_{\ell,t}$ at `bus1`. Left alone, that flow would enter the adjacent zone's net position *and* affect the CNEC through its column, double-counting the contribution. To prevent it, the corridor's contribution to its flow-based-side bus balance is cancelled, defining the cut

$$\kappa_{z,t} = \sum_{\ell:\, \text{bus0}_\ell = z} f_{\ell,t} \; - \sum_{\ell:\, \text{bus1}_\ell = z} \eta_\ell\, f_{\ell,t},$$

which restores $NP_{z,t} = g_{z,t} - d_{z,t}$ at every zone. The zero-sum balance becomes

$$\sum_{z} NP_{z,t} - \sum_{z} \kappa_{z,t} = 0.$$

For an EvFB link (both ends are flow-based zone buses, lossless) the two cut terms cancel. This is consistent with the methodology, where the corridor's two virtual hubs have a combined net position of zero. For an AHC link (one end external) only the flow-based-side term survives, and $-\sum_z \kappa_{z,t}$ is the net position $NP_{v,t}$ of the external virtual hub, so the balance reads

$$\sum_z NP_{z,t} + \sum_v NP_{v,t} = 0$$

In this case, the flow-based region need not be internally balanced when it exchanges over AHC borders.

The cut is built in `flow_based_balance_terms()` (nodal balance) and `define_flow_based_constraints()` (plate).

## Deriving a zonal PTDF from a nodal grid

If a nodal network is available, the zonal PTDF follows from the nodal PTDF and a *generation shift key* (GSK). A zone's net-position change is distributed to its buses by the GSK $\text{GSK}_{b,z}$ (a bus by zone matrix whose columns sum to one), so with the nodal $F = \text{PTDF}\cdot P$ and $P = \text{GSK}\cdot NP$,

$$\text{PTDF}^{\text{zonal}} = \text{PTDF}^{\text{nodal}} \cdot \text{GSK}.$$

The GSK choice (uniform, weighted by installed capacity, ...) shapes the resulting domain; see Schönheit et al. (2020)[^gsk] for a comparison of strategies. The zonal PTDF can be computed per sub-network by [`calculate_zonal_PTDF`][pypsa.SubNetwork.calculate_zonal_PTDF]; see the [component page](../components/flow-based-constraints.md#deriving-a-zonal-ptdf) for usage.

## Symbols

| Symbol | Attribute | Type |
|--------|-----------|------|
| $NP_{z,t}$ | `n.buses_t.net_position` | Decision Variable |
| $f_{\ell,t}$ | `n.links_t.p0` | Decision Variable |
| $\mu_{c,t}$ | `n.c.flow_based_constraints.mu_domain` | Dual Variable |
| $\text{PTDF}_{c,z,t}$ | `n.c.flow_based_constraints.zonal_ptdf` | Parameter |
| $\text{RAM}_{c,t}$ | `n.c.flow_based_constraints.ram` | Parameter |
| $\eta_\ell$ | `n.links.efficiency` | Parameter |

[^vandenbergh]: K. Van den Bergh, J. Boury and E. Delarue (2016), [The Flow-Based Market Coupling in Central Western Europe: Concepts and definitions](https://doi.org/10.1016/j.tej.2015.12.004), The Electricity Journal, 29(1), 24-29, doi:10.1016/j.tej.2015.12.004.

[^schonheit]: D. Schönheit, M. Kenis, L. Lorenz, D. Möst, E. Delarue and K. Bruninx (2021), [Toward a fundamental understanding of flow-based market coupling for cross-border electricity trading](https://doi.org/10.1016/j.adapen.2021.100027), Advances in Applied Energy, 2, 100027, doi:10.1016/j.adapen.2021.100027.

[^gsk]: D. Schönheit, R. Weinhold and C. Dierstein (2020), [The impact of different strategies for generation shift keys (GSKs) on the flow-based market coupling domain: A model-based analysis of Central Western Europe](https://doi.org/10.1016/j.apenergy.2019.114067), Applied Energy, 258, 114067, doi:10.1016/j.apenergy.2019.114067.

[^estermann]: A. Estermann, M. Schrade and L. Anderson (eds.) (2025), [European Electricity Market Coupling: A Practitioner's Guide](https://doi.org/10.1007/978-3-031-86315-8), Springer, doi:10.1007/978-3-031-86315-8.

[^core-ccm]: ACER (2019), Day-ahead capacity calculation methodology of the Core capacity calculation region, in accordance with [Commission Regulation (EU) 2015/1222 (CACM)](https://eur-lex.europa.eu/eli/reg/2015/1222/oj).
