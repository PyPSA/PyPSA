<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Global Constraints

Global constraints apply to more than one component at once and are stored under a unique `name` in `n.global_constraints`. Several pre-defined types of global constraints are available (`type`).
For instance, for defining emission limits, transmission expansion limits, or technology capacity expansion limits using the `sense` and `constant` attributes.

By default, global constraints span across all investment periods. For models with multiple investment periods, global constraints can be limited to affect only single investment period (e.g. an emission limit for a specific year) by specifying the in the attribute `investment_period`. The shadow price of each global constraint is stored in the attribute `mu`.

!!! note "Naming Convention"

    Global constraints carry the name scheme `GlobalConstraint-{name}` in the Linopy model.

## Primary Energy

The primary energy constraints (`type="primary_energy"`) are used to limit byproducts of primary energy consumption of a carrier (e.g. CO~2~ emissions) in generators, storage units and stores. They depend on the generator efficiency and carrier-specific attributes (e.g. `co2_emissions`).

Suppose there is a global constraint defined for CO~2~ emissions (`carrier_attribute`) with sense "<=" (`sense`) and a limit $\Gamma$ (`constant`). Emissions can come from generators whose energy carriers have CO~2~ emissions, and from stores and storage units whose storage medium releases or absorbs CO~2~ when it is converted. Only stores and storage units with non-cyclic state of charge that is different at the start ($t=-1$) and end ($t=|T|-1$) of the optimisation period contribute.

If the specific emissions of carrier $s$ is $\rho_s$ (`n.carriers.co2_emissions`) and the generator with carrier $s$ at node $n$ and snapshot $t$ has efficiency $\eta_{n,s,t}$ then the CO~2~ constraint is

$$\begin{gather*}\sum_{n,s,t}  w_t^g \cdot \eta_{n,s,t}^{-1} \cdot g_{n,s,t}\cdot \rho_s + \sum_{n,s}\left(e_{n,s,t=-1} - e_{n,s,t=|T|-1}\right) \cdot \rho_s\\
+ \sum_{n,s}\left(soc_{n,s,t=-1} - soc_{n,s,t=|T|-1}\right) \cdot \rho_s \leq  \Gamma  \quad \leftrightarrow  \quad \mu\end{gather*}$$

The first sum is over generators; the second sum is over stores; the third over storage units. The shadow price $\mu$ would represent the CO~2~ price in this case.

This global constraint is added in the function `define_primary_energy_limit()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $g_{n,s,t}$ | `n.generators_t.p` | Decision Variable |
    | $e_{n,s,t}$ | `n.stores_t.e` | Decision Variable |
    | $soc_{n,s,t}$ | `n.storage_units_t.soc` | Decision Variable |
    | $\mu$ | `n.global_constraints.mu` | Dual Variable |
    | $w_t^g$ | `n.snapshot_weightings.generators` | Parameter |
    | $\eta_{n,s,t}$ | `n.generators.efficiency` | Parameter |
    | $\rho_s$ | `n.carriers.co2_emissions` | Parameter |
    | $\Gamma$ | `n.global_constraints.constant` | Parameter |


## Operational Limit

The operational constraints can limit the net production of a carrier taking
into account generator, storage units and stores (`type="operational_limit"`).
For example, this can be used to limit the usage of gas in the system to a
certain amount $\Gamma$ (`constant` in MWh). With sense "<=" (`sense`), the
constraint would be given by

$$\begin{gather*}\sum_{n,s,t}  w_t^g \cdot g_{n,s,t}+ \sum_{n,s}\left(e_{n,s,t=-1} - e_{n,s,t=|T|-1}\right) \\
+ \sum_{n,s}\left(soc_{n,s,t=-1} - soc_{n,s,t=|T|-1}\right) \leq  \Gamma  \quad \leftrightarrow  \quad \mu\end{gather*}$$

The first sum is over generators; the second sum is over stores; the third over
storage units. Structurally, it is similar to the primary energy limit, but
without the consideration of specific emissions and efficiencies. The shadow
price $\mu$ (in currency/MWh) would represent the reduction in system costs if
the operational limit were relaxed by one unit.

This global constraint is added in the function `define_operational_limit()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $g_{n,s,t}$ | `n.generators_t.p` | Decision Variable |
    | $e_{n,s,t}$ | `n.stores_t.e` | Decision Variable |
    | $soc_{n,s,t}$ | `n.storage_units_t.soc` | Decision Variable |
    | $\mu$ | `n.global_constraints.mu` | Dual Variable |
    | $w_t^g$ | `n.snapshot_weightings.generators` | Parameter |
    | $\Gamma$ | `n.global_constraints.constant` | Parameter |

## Volume Limit on Transmission Expansion

This global constraint can be used to limit the expansion volume in MWkm of transmission lines and links (`type="transmission_volume_expansion_limit"`). The `carrier_attribute` specifies the subset of carriers to consider. These can be individual carriers or concatenated by commas, e.g. "AC", "DC", "AC,DC", or a [`Link`][pypsa.components.Links] carrier such as "H2 pipeline". With `sense="<="`, the constraint is defined as

$$\sum_{l\in L_{\textrm{carriers}}} d_{l} F_{l} \leq \Gamma \quad \leftrightarrow  \quad \mu$$

where $L_{\textrm{carriers}}$ is the set of lines and links with the specified carriers, $\Gamma$ is the maximum allowed volume expansion in MWkm, $d_{l}$ is the distance of line or link $l$ in km and $F_{l}$ is the capacity of line or link $l$ in MW. The shadow price $\mu$ represents the marginal benefit of expanding the transmission capacity in currency/MWkm/a.

This global constraint is added in the function `define_transmission_volume_expansion_limit()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $F_l$  | `n.{lines,links}.p_nom_opt` | Decision Variable |
    | $\mu$  | `n.global_constraints.mu` | Dual Variable |
    | $d_l$  | `n.{lines,links}.length` | Parameter |
    | $\Gamma$ | `n.global_constraints.constant` | Parameter |

## Cost Limit on Transmission Expansion

This global constraint can be used to limit the total investment cost in currency/a of transmission lines and links (`type="transmission_expansion_cost_limit"`). The `carrier_attribute` specifies the subset of carriers to consider. These can be individual carriers or concatenated by commas, e.g. "AC", "DC", "AC,DC", or a [`Link`][pypsa.components.Links] carrier such as "H2 pipeline". With `sense="<="`, the constraint is defined as

$$\sum_{l\in L_{\textrm{carriers}}} c_{l} F_{l} \leq \Gamma \quad \leftrightarrow  \quad \mu$$

where $L_{\textrm{carriers}}$ is the set of lines and links with the specified carriers, $c_{l}$ is the capital cost of line or link $l$ in currency/MW/a, $F_{l}$ is the capacity of line or link $l$ in MW and $\Gamma$ is the maximum allowed cost of line expansion in currency/a. The shadow price $mu$ represents how much the total system cost could be reduced if the spending limit was increased by one currency/a.

This global constraint is added in the function `define_transmission_expansion_cost_limit()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $F_l$  | `n.{lines,links}.p_nom_opt` | Decision Variable |
    | $\mu$  | `n.global_constraints.mu` | Dual Variable |
    | $c_l$  | `n.{lines,links}.capital_cost` | Parameter |
    | $\Gamma$ | `n.global_constraints.constant` | Parameter |

## Expansion Limit

This global constraint can be used to limit the total capacity of components of a carrier (`type=tech_capacity_expansion_limit`). This global constraint can be specific to an investment period by setting the `investment_period` attribute and specific to a bus by setting the `bus` attribute. This constraint is mainly used for networks with multiple investment periods, where land usage and building rate restrictions need to be applied for a range of active components of a particular carrier (`carrier_attribute`) in a certain region.

!!! warning

    Currently, only the capacities of extendable components are considered, i.e. generators, storage units and stores with `extendable=True`. The capacities of non-extendable components are not considered in this constraint.

For example, the capacities of all onshore wind generators (`carrier_attribute="onshore wind"`) at a certain bus (`bus="DE"`) should be smaller (`sense="<="`) than a hypothetical technical potential of 200 GW for onshore wind in the specific region (`constant=200e3`). Then the technology capacity expansion constraint across all *active* components is given by

$$\sum_{s | b_s<=a<b_s+L_s} G_{n,s} \leq  \Gamma \quad a \in A \quad \leftrightarrow \quad \mu$$

where $A$ are the investment periods, $s$ are all extendable generators of the specified carrier, $b_s$ is the build year of an asset $s$ with lifetime $L_s$. In this example, the shadow price $\mu$ would represent the marginal benefit of expanding the capacity of onshore wind in currency/MW/a.

In general, the constraint would iterate over all investment variables for generators $G_{n,s}$, lines and transformers $P_{l}$, links $F_{l}$, stores $E_{n,s}$ and storage units $H_{n,s}$ for the specified carrier and bus. For components connecting two buses, the bus selection is done by `bus0`.

This global constraint is added in the function `define_tech_capacity_expansion_limit()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $G_{n,s}$ | `n.generators.p_nom_opt` | Decision Variable |
    | $H_{n,s}$ | `n.storage_units.p_nom_opt` | Decision Variable |
    | $E_{n,s}$ | `n.stores.e_nom_opt` | Decision Variable |
    | $F_l$  | `n.links.p_nom_opt` | Decision Variable |
    | $P_l$  | `n.{lines,transformers}.s_nom_opt` | Decision Variable |
    | $\mu$  | `n.global_constraints.mu` | Dual Variable |
    | $b_s$  | `n.{<component>}.build_year` | Parameter |
    | $L_s$  | `n.{<component>}.lifetime` | Parameter |
    | $a$, $A$    | `n.investment_periods` | Parameter |
    | $\Gamma$ | `n.global_constraints.constant` | Parameter |


## Growth Limit per Carrier

This carrier-specific constraint type implements absolute and relative growth limits per carrier which constrains new installed capacities for each investment period. It can be defined by providing the attributes `n.carriers.max_growth` and `n.carriers.max_relative_growth`.

Suppose the absolute growth limit for a specific carrier $s$ is $\Gamma_s$ (`max_growth`) and the relative growth limit is $\gamma_s$ (`max_relative_growth`). With the growth limit constraint, for each investment period $a$, the new installed capacity $G_{a,s}$ of all components with carrier $s$ is limited by:

$$G_{a,s} \leq \gamma_s \cdot G_{a-1,s} + \Gamma_s \quad \forall a, s \quad \leftrightarrow \quad \mu_{a,s}$$

where $G_{a,s}$ represents the sum of all newly built nominal capacities of extendable components with carrier $s$ in period $a$. The relative growth limit $\gamma_s$ allows the new capacity in period $a$ to be proportional to the capacity added in the previous period $a-1$.

This constraint only applies to networks with multiple investment periods and only considers components that are newly activated in each period. For the constraint to take effect, at least one of the attributes `max_growth` or `max_relative_growth` must be set to a finite value for the carrier.

!!! warning

    The relative and abolute growth limits are additive, i.e. the absolute growth limit applies in addition to the relative growth limit.

In general, the constraint would iterate over all investment variables for generators $G_{n,s}$, lines and transformers $P_{l}$, links $F_{l}$, stores $E_{n,s}$ and storage units $H_{n,s}$ for the specified carrier. For components connecting two buses, the bus selection is done by `bus0`.

This global constraint is added in the function `define_growth_limit()` and carries the name `Carrier-growth_limit`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $G_{n,s}$ | `n.generators.p_nom_opt` | Decision Variable |
    | $H_{n,s}$ | `n.storage_units.p_nom_opt` | Decision Variable |
    | $E_{n,s}$ | `n.stores.e_nom_opt` | Decision Variable |
    | $F_l$  | `n.links.p_nom_opt` | Decision Variable |
    | $P_l$  | `n.{lines,transformers}.s_nom_opt` | Decision Variable |#
    | $\gamma_s$ | `n.carriers.max_relative_growth` | Parameter |
    | $\Gamma_s$ | `n.carriers.max_growth` | Parameter |

## Flow-Based Market Coupling

Flow-based market coupling (FBMC) replaces the transmission grid constraints between market zones by a compact set of linear constraints on the zones' net positions. Instead of resolving every node and line, the market treats each zone as a copper plate and trades against a few linear limits that capture how zonal exchanges load the critical grid elements. This is the capacity-allocation method used in parts of the European day-ahead market, notably the Core region.[^core-ccm] For an introduction to the method and its parameters, see Van den Bergh et al. (2016)[^vandenbergh] and Schönheit et al. (2021)[^schonheit]. The domain of feasible net positions is stored as global constraints of `type="flow_based"`, see the [component page](../components/global-constraints.md#flow-based-domain).

The net-position variable is `Bus-net_position`, the domain half-spaces are `GlobalConstraint-flow_based`, and the zero-sum balance is `GlobalConstraint-flow_based_balance`; see also the [example notebook](../../examples/flow-based-market-coupling.ipynb).

### Net position

The net position $NP_{z,t}$ of a market zone $z$ at snapshot $t$ is the power it feeds into the flow-based region. It is added as a decision variable and injected directly into the [nodal balance](energy-balance.md) of its zone bus,

$$g_{z,t} - d_{z,t} - NP_{z,t} = 0 ,$$

so it equals the bus's net injection and is read from `n.buses_t.p` after solving.

### Domain constraints

Each critical network element (CNEC) $c$ — one global constraint row, a monitored grid constraint such as a line under a given outage — contributes one half-space that bounds a weighted sum of net positions by its remaining available margin $\text{RAM}_{c,t}$, the headroom left on that element for cross-zonal trade:

$$\sum_{z} \text{PTDF}_{c,z,t}\, NP_{z,t} \;\le\; \text{RAM}_{c,t} \quad \leftrightarrow \quad \mu_{c,t}$$

The sensitivities $\text{PTDF}_{c,z,t}$ are the zonal power transfer distribution factors (attributes `ptdf_<zone>`, see the [component page](../components/global-constraints.md#zonal-ptdf)); both they and the RAM (`constant`) may be static or vary by snapshot, and the constraint broadcasts over $t$ either way. A row with an `investment_period` only applies in that period's snapshots. The shadow price $\mu_{c,t}$ is written to `n.global_constraints_t.mu`. A single zero-sum balance closes the copper plate across the zones,

$$\sum_{z} NP_{z,t} = 0.$$

These constraints are added in `define_flow_based_constraints()`; the net-position injection is added in `define_nodal_balance_constraints()` via `flow_based_balance_terms()`.

### Controllable link flows (AHC and EvFB)

A domain column may name a [`Link`](../components/links.md) instead of a zone [`Bus`](../components/buses.md). This extends the domain to controllable corridors, such as HVDC links;[^estermann] PyPSA tells the two cases apart by the link's ends:

- **Advanced hybrid coupling (AHC):** one end is a zone, the other a bus outside the flow-based region.
- **Evolved flow-based (EvFB):** both ends are flow-based zones.

The link stays in the nodal balances of its buses like any other link, so a zone's net position now includes the corridor inflow,

$$NP_{z,t} = g_{z,t} - d_{z,t} + \sum_{\ell} a_{z,\ell}\, f_{\ell,t},$$

with $f_{\ell,t}$ the link flow (`Link-p`, in the `bus0 -> bus1` direction) and $a_{z,\ell}$ the link's incidence at $z$ ($-1$ at `bus0`, $+\eta_\ell$ at `bus1`, else $0$). Each link adds a term to every CNEC:

$$\sum_{z} \text{PTDF}_{c,z,t}\, NP_{z,t} + \sum_{\ell} \text{PTDF}_{c,\ell,t}\, f_{\ell,t} \;\le\; \text{RAM}_{c,t} \quad \leftrightarrow \quad \mu_{c,t}$$

The link column $\text{PTDF}_{c,\ell}$ is the sensitivity of CNEC $c$ to the link flow **with all zone net positions held fixed**: the loading the corridor causes beyond what its zones' PTDFs already explain. Let $h_{c,\ell,z}$ be the sensitivity of CNEC $c$ to an injection at the node where link $\ell$ lands in zone $z$ (its hub sensitivity). Then

$$\text{PTDF}_{c,\ell} = h_{c,\ell} - \sum_{z} a_{z,\ell}\, \text{PTDF}_{c,z},$$

which is $h - \text{PTDF}_z$ for an AHC link into zone $z$, and $(h_B - h_A) - (\text{PTDF}_B - \text{PTDF}_A)$ for an EvFB link from $A$ to $B$. If the corridor lands where the zone's GSK puts its power anyway, the column is zero. The zero-sum balance is unchanged, $\sum_z NP_{z,t} = 0$, because the corridor imports are already inside the net positions.

!!! note "Two equivalent conventions"

    Published domains use one of two conventions for corridors. ERAA publishes AHC columns as above (`PTDF*_AHC = PTDF_AHC - PTDF_SZ`). The Core methodology, JAO and TSO files instead keep $NP_z = g_z - d_z$, treat each corridor end as a *virtual hub* with the raw hub sensitivity $h$ and count the hub's flow in the zero-sum balance.[^core-ccm] Both describe the same feasible set: moving the link term out of the zone's balance shifts its column by $-\sum_z a_{z,\ell}\text{PTDF}_{c,z}$. PyPSA uses the first because the link then needs no special treatment in the nodal balance; the [importers](../components/global-constraints.md#importing-published-domains) convert hub sensitivities accordingly. The commercial net position over all borders, $g_z - d_z$, is the net position minus the corridor inflow.

    | | PyPSA (ERAA) | Virtual hub (Core, JAO, TSO) |
    |---|---|---|
    | zone net position | $g_z - d_z + \sum_\ell a_{z,\ell} f_\ell$ | $g_z - d_z$ |
    | link column | $h - \sum_z a_{z,\ell}\text{PTDF}_z$ | $h$ |
    | zero-sum balance | $\sum_z NP_z = 0$ | $\sum_z NP_z + \sum_{\text{hubs}} f = 0$ |

These terms are built in `define_flow_based_constraints()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $NP_{z,t}$ | `n.buses_t.p` (variable `Bus-net_position`) | Decision Variable |
    | $f_{\ell,t}$ | `n.links_t.p0` | Decision Variable |
    | $\mu_{c,t}$ | `n.global_constraints_t.mu` | Dual Variable |
    | $\text{PTDF}_{c,z,t}$ | `n.c.global_constraints.zonal_ptdf` | Parameter |
    | $\text{RAM}_{c,t}$ | `n.c.global_constraints.constant` | Parameter |
    | $h_{c,\ell}$ | hub sensitivity of a published corridor column | Data |
    | $\eta_\ell$ | `n.links.efficiency` | Parameter |

[^vandenbergh]: K. Van den Bergh, J. Boury and E. Delarue (2016), [The Flow-Based Market Coupling in Central Western Europe: Concepts and definitions](https://doi.org/10.1016/j.tej.2015.12.004), The Electricity Journal, 29(1), 24-29, doi:10.1016/j.tej.2015.12.004.

[^schonheit]: D. Schönheit, M. Kenis, L. Lorenz, D. Möst, E. Delarue and K. Bruninx (2021), [Toward a fundamental understanding of flow-based market coupling for cross-border electricity trading](https://doi.org/10.1016/j.adapen.2021.100027), Advances in Applied Energy, 2, 100027, doi:10.1016/j.adapen.2021.100027.

[^estermann]: A. Estermann, M. Schrade and L. Anderson (eds.) (2025), [European Electricity Market Coupling: A Practitioner's Guide](https://doi.org/10.1007/978-3-031-86315-8), Springer, doi:10.1007/978-3-031-86315-8.

[^core-ccm]: ACER (2019), Day-ahead capacity calculation methodology of the Core capacity calculation region, in accordance with [Commission Regulation (EU) 2015/1222 (CACM)](https://eur-lex.europa.eu/eli/reg/2015/1222/oj).
