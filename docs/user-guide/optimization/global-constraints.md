<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Global Constraints

Global constraints apply to more than one component at once and are stored under a unique `name` in `n.global_constraints`. Several pre-defined types of global constraints are available (`type`).
For instance, for defining emission limits, transmission expansion limits, or technology capacity expansion limits using the `sense` and `constant` attributes.

By default, global constraints span across all investment periods. For models with multiple investment periods, global constraints can be limited to affect only single investment period (e.g. an emission limit for a specific year) by specifying the attribute `investment_period`. The shadow price of each global constraint is stored in the attribute `mu`. Flow-based rows store theirs per snapshot in `mu_ram` instead.

!!! note "Naming Convention"

    Each global constraint `{name}` of every type but `flow_based` carries the name `GlobalConstraint-{name}` in the Linopy model, as `GlobalConstraint-co2_limit`. All `flow_based` rows share the one constraint `GlobalConstraint-ram`, indexed by `name`. Their net positions and the balance of the net positions are `FlowBased-net_position` and `FlowBased-balance`.

Each global constraint $\gamma$ bounds the sum of its type by its `constant` $\Gamma_{\gamma,s}$. The `sense` selects one of three rows: `GlobalConstraint_upper` for `"<="`, `GlobalConstraint_lower` for `">="` and `GlobalConstraint_fix` for `"=="`. The [formulation reference](formulation/index.md#GlobalConstraint_upper) lists all of them. The sections below show the sum of each type.

<!-- md:spec GlobalConstraint_upper GlobalConstraint_total -->

## Primary Energy

The primary energy constraints (`type="primary_energy"`) are used to limit byproducts of primary energy consumption of a carrier (e.g. CO~2~ emissions) in generators, storage units and stores. They depend on the generator efficiency and carrier-specific attributes (e.g. `co2_emissions`).

Suppose there is a global constraint defined for CO~2~ emissions (`carrier_attribute`) with sense "<=" (`sense`) and a limit $\Gamma$ (`constant`). Emissions can come from generators whose energy carriers have CO~2~ emissions, and from stores and storage units whose storage medium releases or absorbs CO~2~ when it is converted:

<!-- md:spec primary_energy GlobalConstraint_energy_weight -->

=== "Generator"

    <!-- md:spec Generator_primary_energy Generator_primary_input -->

=== "StorageUnit"

    <!-- md:spec StorageUnit_primary_energy StorageUnit_withdrawal -->

=== "Store"

    <!-- md:spec Store_primary_energy Store_withdrawal -->

The weight `Generator_primary_energy_weight` is the carrier attribute (here `n.carriers.co2_emissions`) of the generator's carrier divided by the generator efficiency. The weights of storage units and stores are the carrier attribute of their carrier. The energy weight counts each snapshot with its generator weighting $w^{\mathrm{g}}_t$ (and, with investment periods, the years of the period $W^{\mathrm{y}}$) if the snapshot falls into the constraint's `investment_period`.

A store or storage unit contributes the net energy withdrawn from it (discharging minus charging): its energy level at the start minus its level at the end of the counted snapshots, plus its net inflow minus spillage. This follows from the sum of the [storage energy balance](storage.md) over the counted snapshots and includes standing losses. For a constraint with an `investment_period`, the start level is the level that the storage carries into that period. With investment periods, the level change and the net inflow of each period are weighted by the years of the period. The level term is zero for storage that is cyclic within the counted snapshots.

The shadow price $\mu$ of the constraint would represent the CO~2~ price in this case. It is stored in `n.global_constraints.mu`.

??? note "Symbols"

    <!-- md:spec-legend GlobalConstraint_upper GlobalConstraint_energy_weight Generator_primary_energy Generator_primary_input StorageUnit_withdrawal Store_withdrawal -->

## Operational Limit

The operational constraints can limit the net production of a carrier taking
into account generator, storage units and stores (`type="operational_limit"`).
For example, this can be used to limit the usage of gas in the system to a
certain amount $\Gamma$ (`constant` in MWh). With sense "<=" (`sense`), the
constraint is given by

<!-- md:spec operational_limit Generator_operational_limit StorageUnit_operational_limit Store_operational_limit -->

The weights `*_operational_limit_weight` are one for the components of the constraint's carrier (and bus, if given) and zero otherwise. The withdrawal of stores and storage units is the same as for the [primary energy limit](#primary-energy). Structurally, the constraint is similar to the primary energy limit, but
without the consideration of specific emissions and efficiencies. The shadow
price $\mu$ (in currency/MWh) would represent the reduction in system costs if
the operational limit were relaxed by one unit.

??? note "Symbols"

    <!-- md:spec-legend GlobalConstraint_upper Generator_operational_limit StorageUnit_operational_limit Store_operational_limit -->

## Volume Limit on Transmission Expansion

This global constraint can be used to limit the expansion volume in MWkm of transmission lines and links (`type="transmission_volume_expansion_limit"`). The `carrier_attribute` specifies the subset of carriers to consider. These can be individual carriers or concatenated by commas, e.g. "AC", "DC", "AC,DC", or a [`Link`][pypsa.components.Links] carrier such as "H2 pipeline". With `sense="<="`, the constraint is defined as

<!-- md:spec transmission_volume_expansion Line_transmission_volume_expansion Link_transmission_volume_expansion -->

where the weights `Line_volume_weight` and `Link_volume_weight` are the `length` in km of the extendable lines and links with the specified carriers (zero for all others), $P_l$ and $F_k$ are their capacities in MW and $\Gamma$ is the maximum allowed volume expansion in MWkm. The shadow price $\mu$ represents the marginal benefit of expanding the transmission capacity in currency/MWkm/a.

??? note "Symbols"

    <!-- md:spec-legend GlobalConstraint_upper Line_transmission_volume_expansion Link_transmission_volume_expansion -->

## Cost Limit on Transmission Expansion

This global constraint can be used to limit the total investment cost in currency/a of transmission lines and links (`type="transmission_expansion_cost_limit"`). The `carrier_attribute` specifies the subset of carriers to consider. These can be individual carriers or concatenated by commas, e.g. "AC", "DC", "AC,DC", or a [`Link`][pypsa.components.Links] carrier such as "H2 pipeline". With `sense="<="`, the constraint is defined as

<!-- md:spec transmission_expansion_cost Line_transmission_expansion_cost Link_transmission_expansion_cost -->

where the weights `Line_expansion_cost_weight` and `Link_expansion_cost_weight` are the capital costs in currency/MW/a of the extendable lines and links with the specified carriers (zero for all others), and $\Gamma$ is the maximum allowed cost of line expansion in currency/a. The shadow price $\mu$ represents how much the total system cost could be reduced if the spending limit was increased by one currency/a.

??? note "Symbols"

    <!-- md:spec-legend GlobalConstraint_upper Line_transmission_expansion_cost Link_transmission_expansion_cost -->

## Expansion Limit

This global constraint can be used to limit the total capacity of components of a carrier (`type=tech_capacity_expansion_limit`). This global constraint can be specific to an investment period by setting the `investment_period` attribute and specific to a bus by setting the `bus` attribute. This constraint is mainly used for networks with multiple investment periods, where land usage and building rate restrictions need to be applied for a range of active components of a particular carrier (`carrier_attribute`) in a certain region.

!!! warning

    Currently, only the capacities of extendable components are considered, i.e. generators, storage units and stores with `extendable=True`. The capacities of non-extendable components are not considered in this constraint.

For example, the capacities of all onshore wind generators (`carrier_attribute="onshore wind"`) at a certain bus (`bus="DE"`) should be smaller (`sense="<="`) than a hypothetical technical potential of 200 GW for onshore wind in the specific region (`constant=200e3`). Then the technology capacity expansion constraint across all *active* components is given by

<!-- md:spec tech_capacity_expansion -->

=== "Generator"

    <!-- md:spec Generator_tech_capacity_expansion -->

=== "StorageUnit"

    <!-- md:spec StorageUnit_tech_capacity_expansion -->

=== "Store"

    <!-- md:spec Store_tech_capacity_expansion -->

=== "Link"

    <!-- md:spec Link_tech_capacity_expansion -->

=== "Process"

    <!-- md:spec Process_tech_capacity_expansion -->

=== "Line"

    <!-- md:spec Line_tech_capacity_expansion -->

The weight `*_tech_capacity_weight` is one for an extendable component of the specified carrier and bus that is active in the constraint's investment period, i.e. whose `build_year` is at most the period and whose `build_year` plus `lifetime` is after it. It is zero otherwise. In this example, the shadow price $\mu$ would represent the marginal benefit of expanding the capacity of onshore wind in currency/MW/a. For components connecting two buses, the bus selection is done by `bus0`.

??? note "Symbols"

    <!-- md:spec-legend GlobalConstraint_upper Generator_tech_capacity_expansion StorageUnit_tech_capacity_expansion Store_tech_capacity_expansion Link_tech_capacity_expansion Process_tech_capacity_expansion Line_tech_capacity_expansion -->

## Growth Limit per Carrier

This carrier-specific constraint type implements absolute and relative growth limits per carrier which constrains new installed capacities for each investment period. It can be defined by providing the attributes `n.carriers.max_growth` and `n.carriers.max_relative_growth`.

With the growth limit constraint, for each carrier $\sigma$ and investment period $a$, the new installed capacity of all components with carrier $\sigma$ is limited by the absolute growth limit `max_growth` plus the relative growth limit `max_relative_growth` times the capacity added in the previous period:

<!-- md:spec Carrier_growth_limit Carrier_additions Carrier_max_relative_growth_nonneg Generator_additions -->

The other components add their newly built capacities in the same way as `Generator_additions`. This constraint only applies to networks with multiple investment periods and only considers components that are newly activated in each period. For the constraint to take effect, at least one of the attributes `max_growth` or `max_relative_growth` must be set to a finite value for the carrier.

!!! warning

    The relative and absolute growth limits are additive, i.e. the absolute growth limit applies in addition to the relative growth limit.

For components connecting two buses, the bus selection is done by `bus0`.

This global constraint carries the name `Carrier-growth_limit`.

??? note "Symbols"

    <!-- md:spec-legend Carrier_growth_limit Carrier_additions Generator_additions -->

## Flow-Based Market Coupling

Flow-based market coupling (FBMC) replaces the transmission grid constraints between market zones by a compact set of linear constraints on the zones' net positions. Instead of resolving every node and line, the market treats each zone as a copper plate and trades against a few linear limits that capture how zonal exchanges load the critical grid elements. This is the capacity-allocation method used in parts of the European day-ahead market, notably the Core region.[^core-ccm] For an introduction to the method and its parameters, see Van den Bergh et al. (2016)[^vandenbergh] and Schönheit et al. (2021)[^schonheit]. The domain of feasible net positions is stored as global constraints of `type="flow_based"`, see the [component page](../components/global-constraints.md#flow-based-domain).

The net-position variable is `FlowBased-net_position`, the domain half-spaces are `GlobalConstraint-ram`, and the zero-sum balance is `FlowBased-balance`; see also the [example notebook](../../examples/fbmc-simple-example.ipynb).

### Net position

The net position of a market zone (a bus with `flow_based_zone`) $n$ at snapshot $t$ is the power it feeds into the flow-based region. It is added as a decision variable and withdrawn directly in the [nodal balance](energy-balance.md) of its zone bus,

<!-- md:spec FlowBased_net_position FlowBased_injection -->

so it equals the bus's net injection and is read from `n.buses_t.p` after solving. The output `n.buses_t.net_position` holds the generation minus load of each zone bus (0 for other buses), which is the net position itself unless corridor links are present (see below).

### Domain constraints

Each critical network element (CNEC) $\gamma$ is one global constraint row, a monitored grid constraint such as a line under a given outage. It contributes one half-space that bounds a weighted sum of net positions (and of controllable link flows, see below) by its remaining available margin `ram`, the headroom left on that element for cross-zonal trade:

<!-- md:spec GlobalConstraint_ram -->

The sensitivities `zonal_ptdf` are the zonal power transfer distribution factors (attributes `ptdf_<zone>`, see the [component page](../components/global-constraints.md#zonal-ptdf)); both they and the RAM (`ram`) may be static or vary by snapshot, and the constraint broadcasts over $t$ either way. A row with an `investment_period` only applies in that period's snapshots, and a row with a NaN `ram` is dropped in that snapshot. The shadow price of each row is written to `n.global_constraints_t.mu_ram`. A zero-sum balance closes the copper plate across the zones of each domain $z$:

<!-- md:spec FlowBased_balance -->

By default all zones form one domain. Rows that name their `domain`, e.g. Core and Nordic, give one balance per named domain. A link between two domains then acts as an AHC corridor in each. The rows of one domain see neither the zones nor the hubs of the other (their zonal PTDFs and hub sensitivities are 0 there), so the EvFB and AHC formulas below give the same column.

### Controllable link flows (AHC and EvFB)

A domain column may name a [`Link`](../components/links.md) instead of a zone [`Bus`](../components/buses.md). This extends the domain to controllable corridors, such as HVDC links;[^estermann] PyPSA tells the two cases apart by the link's ends:

- **Advanced hybrid coupling (AHC):** one end is a zone, the other a bus outside the flow-based region.
- **Evolved flow-based (EvFB):** both ends are flow-based zones.

The link stays in the nodal balances of its buses like any other link, so a zone's net position now includes the corridor inflow $\sum_{k} K_{n,k} f_{k,s,t}$, with $f_{k,s,t}$ the link flow (`Link-p`, in the `bus0 -> bus1` direction) and $K_{n,k}$ the link's incidence at zone $n$ ($-1$ at `bus0`, $+\eta_k$ at `bus1`, else $0$). Each link adds its flow, weighted by its column `link_ptdf`, to every CNEC, as in the [domain constraint](#domain-constraints) above.

The link column `link_ptdf` is the sensitivity of CNEC $\gamma$ to the link flow **with all zone net positions held fixed**: the loading the corridor causes beyond what its zones' PTDFs already explain.

!!! note "Background"

    Let $\mathrm{PTDF}^{\mathrm{hub}}_{\gamma,k}$ be the sensitivity of CNEC $\gamma$ to an injection at the node where link $k$ lands in a zone (its hub sensitivity), and $\mathrm{PTDF}_{\gamma,n}$ the zonal PTDF of zone $n$. Then

    $$\mathrm{PTDF}_{\gamma,k} = \mathrm{PTDF}^{\mathrm{hub}}_{\gamma,k} - \sum_{n} K_{n,k}\, \mathrm{PTDF}_{\gamma,n},$$

    which is $\mathrm{PTDF}^{\mathrm{hub}} - \mathrm{PTDF}_n$ for an AHC link into zone $n$, and $(\mathrm{PTDF}^{\mathrm{hub}}_B - \mathrm{PTDF}^{\mathrm{hub}}_A) - (\mathrm{PTDF}_B - \mathrm{PTDF}_A)$ for an EvFB link from $A$ to $B$.

If the corridor lands where the zone's GSK puts its power anyway, the column is zero. The zero-sum balance is unchanged, because the corridor imports are already inside the net positions.

!!! note "Two equivalent net-position conventions"

    Published domains use one of two conventions for corridors. Both describe the same feasible set and give the same CNEC flows; they only split each flow differently between the zone terms and the link terms.

    - **PyPSA:** the net position includes the corridor flow. The zone PTDF already spreads that flow as if the GSK placed it, so the link column keeps only the correction from the GSK to the landing node. ERAA publishes its AHC columns in this form (`PTDF*_AHC = PTDF_AHC - PTDF_SZ`).
    - **Virtual hub:** the net position is generation minus load, $NP_n = g_n - d_n$. Each corridor end acts as a hub with its raw hub sensitivity $\mathrm{PTDF}^{\mathrm{hub}}$, and the hub flows join the zero-sum balance. The Core methodology,[^core-ccm] JAO and TSO files use this form, and so do ERAA's EvFB columns.

    The conversion is one rule: what PyPSA adds to the net positions, it takes out of the link column,

    !!! note "Background"

        $$NP_n = (g_n - d_n) + \sum_k K_{n,k}\, f_k \qquad \Longleftrightarrow \qquad \mathrm{PTDF}_{\gamma,k} = \mathrm{PTDF}^{\mathrm{hub}}_{\gamma,k} - \sum_n K_{n,k}\, \mathrm{PTDF}_{\gamma,n} .$$

    | | PyPSA | Virtual hub |
    |---|---|---|
    | zone net position | $g_n - d_n + \sum_k K_{n,k} f_k$ (`n.buses_t.p`) | $g_n - d_n$ (`n.buses_t.net_position`) |
    | AHC column, $X \to A$ | $\mathrm{PTDF}^{\mathrm{hub}}_A - \mathrm{PTDF}_A$ | $\mathrm{PTDF}^{\mathrm{hub}}_A$ |
    | EvFB column, $A \to B$ | $(\mathrm{PTDF}^{\mathrm{hub}}_B - \mathrm{PTDF}^{\mathrm{hub}}_A) - (\mathrm{PTDF}_B - \mathrm{PTDF}_A)$ | $\mathrm{PTDF}^{\mathrm{hub}}_B - \mathrm{PTDF}^{\mathrm{hub}}_A$ |
    | zero-sum balance | $\sum_n NP_n = 0$ | $\sum_n NP_n + \sum_{\text{hubs}} f = 0$ |

    PyPSA uses the first form because the link then needs no special treatment in the nodal balance. After solving, `n.buses_t.net_position` reports the virtual-hub net position $g_n - d_n$ of each zone bus, i.e. `n.buses_t.p` without the flows of its corridor links.

??? note "Symbols"

    <!-- md:spec-legend FlowBased_net_position FlowBased_injection GlobalConstraint_ram FlowBased_balance -->

    The dual of each domain row is stored in `n.global_constraints_t.mu_ram`. The virtual-hub net position $g_n - d_n$ is reported in `n.buses_t.net_position`.

[^vandenbergh]: K. Van den Bergh, J. Boury and E. Delarue (2016), [The Flow-Based Market Coupling in Central Western Europe: Concepts and definitions](https://doi.org/10.1016/j.tej.2015.12.004), The Electricity Journal, 29(1), 24-29, doi:10.1016/j.tej.2015.12.004.

[^schonheit]: D. Schönheit, M. Kenis, L. Lorenz, D. Möst, E. Delarue and K. Bruninx (2021), [Toward a fundamental understanding of flow-based market coupling for cross-border electricity trading](https://doi.org/10.1016/j.adapen.2021.100027), Advances in Applied Energy, 2, 100027, doi:10.1016/j.adapen.2021.100027.

[^estermann]: A. Estermann, M. Schrade and L. Anderson (eds.) (2025), [European Electricity Market Coupling: A Practitioner's Guide](https://doi.org/10.1007/978-3-031-86315-8), Springer, doi:10.1007/978-3-031-86315-8.

[^core-ccm]: ACER (2019), Day-ahead capacity calculation methodology of the Core capacity calculation region, in accordance with [Commission Regulation (EU) 2015/1222 (CACM)](https://eur-lex.europa.eu/eli/reg/2015/1222/oj).
