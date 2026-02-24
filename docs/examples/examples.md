<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Examples

The examples below demonstrate PyPSA's capabilities for energy system modeling. They cover a broad range of topics, including electricity markets, linear optimal power flow, unit commitment, capacity expansion, grid modelling, and more.

## Operational Problems

<div class="grid cards" markdown>


-   :material-notebook:{ .lg .middle } **Electricity Market**

    Demonstrates basic electricity market modeling with with multiple bidding zones, renewables and storage.

    [:octicons-arrow-right-24: Go to example](simple-electricity-market-examples.ipynb)

-   :material-notebook:{ .lg .middle } **Demand and Supply Bids**

    Demonstrates market-clearing with supply and demand bids in single and two-zone configurations.

    [:octicons-arrow-right-24: Go to example](demand-supply-bids.ipynb)

-   :material-notebook:{ .lg .middle } **Unit Commitment**

    Models generator unit commitment with start-up and shut-down costs, ramping limits, minimum part loads, up and down times using binary variables.

    [:octicons-arrow-right-24: Go to example](unit-commitment.ipynb)

-   :material-notebook:{ .lg .middle } **Negative Prices in Linearized UC**

    Shows how negative electricity prices emerge from linearized unit commitment constraints due to the trade-off between cycling costs and operating at minimum load.

    [:octicons-arrow-right-24: Go to example](uc-prices.ipynb)

-   :material-notebook:{ .lg .middle } **Meshed AC-DC Networks**

    Builds a stylized 3-node AC network coupled via AC-DC converters to a 3-node DC network.

    [:octicons-arrow-right-24: Go to example](ac-dc-lopf.ipynb)

-   :material-notebook:{ .lg .middle } **SciGRID Network**

    Performs linear optimal power flow on a high-resolution German grid model to analyze power flows and nodal prices.

    [:octicons-arrow-right-24: Go to example](scigrid-lopf-then-pf.ipynb)

-   :material-notebook:{ .lg .middle } **Security-Constrained LOPF**

    Implements N-1 security constraints in linear optimal power flow models to ensure grid reliability under line outage events.

    [:octicons-arrow-right-24: Go to example](scigrid-sclopf.ipynb)

-   :material-notebook:{ .lg .middle } **Newton-Raphson Power Flow**

    Solves non-linear AC power flow equations using the Newton-Raphson method to inspect voltage magnitudes and angles.

    [:octicons-arrow-right-24: Go to example](minimal-example-pf.ipynb)

-   :material-notebook:{ .lg .middle } **Negative LMPs from Line Congestion**

    Explores how Kirchhoff's Voltage Law can lead to negative locational marginal prices when lines are congested.

    [:octicons-arrow-right-24: Go to example](negative-prices-kvl-baker.ipynb)

-   :material-notebook:{ .lg .middle } **Rolling-Horizon Optimization**

    Explores how rolling-horizon optimization can be used to account for imperfect forecast horizons in reality.

    [:octicons-arrow-right-24: Go to example](rolling-horizon.ipynb)

-   :material-notebook:{ .lg .middle } **Water Values**

    Explores how water values, the marginal values of stored energy, can improve seasonal storage operation in rolling-horizon optimization.

    [:octicons-arrow-right-24: Go to example](water-value.ipynb)

</div>

## Planning Problems


<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Single-Node Capacity Expansion**

    Models investment decisions for generation and storage in a single-node system in the style of [model.energy](https://model.energy).

    [:octicons-arrow-right-24: Go to example](capacity-expansion-planning-single-node.ipynb)

-   :material-notebook:{ .lg .middle } **Three-Node Capacity Expansion**

    Co-optimizes generation, storage and transmission investments in a stylized three-node network in Australia :flag_au:.

    [:octicons-arrow-right-24: Go to example](3-node-cem.ipynb)

-   :material-notebook:{ .lg .middle } **Pathway Planning**

    Optimizes investment decisions across multiple investment periods for a long-term transition pathway with perfect foresight.

    [:octicons-arrow-right-24: Go to example](multi-investment-optimisation.ipynb)

-   :material-notebook:{ .lg .middle } **Myopic Pathway Planning**

    Optimizes investment decisions across multiple investment periods for a
    long-term transition pathway with myopic foresight.

    [:octicons-arrow-right-24: Go to example](myopic-pathway.ipynb)


-   :material-notebook:{ .lg .middle } **Stochastic Optimization**

    Demonstrates investment planning under uncertainty with scenario-based
    two-stage stochastic optimization.

    [:octicons-arrow-right-24: Go to example](stochastic-optimization.ipynb)

-   :material-notebook:{ .lg .middle } **Modelling-to-Generate Alternatives**

    Explores near-optimal solution diversity by generating alternative system
    designs with similar costs.

    [:octicons-arrow-right-24: Go to example](mga.ipynb)

-   :material-notebook:{ .lg .middle } **Exploring Near-Optimal Spaces**

    Explores near-optimal space to understand flexibility in investment
    decisions while maintaining cost-effectiveness.

    [:octicons-arrow-right-24: Go to example](near-opt-space.ipynb)

-   :material-notebook:{ .lg .middle } **Modular Capacity Expansion**

    Models discrete capacity additions with integer constraints on investment
    decisions considering predefined unit sizes.

    [:octicons-arrow-right-24: Go to example](modular-expansion.ipynb)

-   :material-notebook:{ .lg .middle } **Committable and Extendable Components**

    Co-optimize capacity expansion and unit commitment using big-M linearization.
    Demonstrates continuous capacity decisions with start-up/shut-down costs,
    ramp limits, and minimum load constraints.

    [:octicons-arrow-right-24: Go to example](committable-extendable.ipynb)

-   :material-notebook:{ .lg .middle } **Modular and Committable Components**

    Model discrete capacity blocks with unit commitment where status represents
    the number of committed modules. Shows modular gas turbines, HVDC links,
    and multi-module operational dynamics.

    [:octicons-arrow-right-24: Go to example](modular-committable.ipynb)

</div>


## Special Problems

<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Redispatch**

    Sketches how redispatch can be modelled by separating market clearing and
    congestion management.

    [:octicons-arrow-right-24: Go to example](scigrid-redispatch.ipynb)

-   :material-notebook:{ .lg .middle } **Demand Elasticity**

    Demonstrates modelling of price-responsive electricity demands and how they
    affect price formation.

    [:octicons-arrow-right-24: Go to example](demand-elasticity.ipynb)

-   :material-notebook:{ .lg .middle } **Imperfect Competition**

    Models oligopolistic behavior in energy markets using Cournot-Nash equilibrium
    with the fictitious objective approach, avoiding KKT conditions.

    [:octicons-arrow-right-24: Go to example](imperfect-competition.ipynb)

-   :material-notebook:{ .lg .middle } **Screening Curves**

    Determines optimal generation capacity mix based on screening curves.

    [:octicons-arrow-right-24: Go to example](generation-investment-screening-curve.ipynb)

-   :material-notebook:{ .lg .middle } **Chained Hydro Reservoirs**

    Models cascaded hydropower systems with water flow constraints between
    reservoirs.

    [:octicons-arrow-right-24: Go to example](chained-hydro-reservoirs.ipynb)

-   :material-notebook:{ .lg .middle } **Transformers**

    Shows how transformers can be considered with varying tap ratios and phase
    shifts.

    [:octicons-arrow-right-24: Go to example](transformer-example.ipynb)


-   :material-notebook:{ .lg .middle } **Reserve Constraints**

    Implements operating reserve requirements in power system optimization.

    [:octicons-arrow-right-24: Go to example](reserve-power.ipynb)

-   :material-notebook:{ .lg .middle } **Link Delay**

    Demonstrates time-delayed energy transport through links, modeling pipeline or shipping delays with cyclic and non-cyclic boundary behavior.

    [:octicons-arrow-right-24: Go to example](link-delay.ipynb)

</div>

## Sector Coupling

<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Single-Node Sector-Coupling**

    Extends the 1-node capacity expansion example with hydrogen, heat and
    transport demand.

    [:octicons-arrow-right-24: Go to example](sector-coupling-single-node.ipynb)


-   :material-notebook:{ .lg .middle } **Islanded Methanol Production**

    Optimizes islanded renewable methanol production systems in Namibia
    :flag_na: or Argentina :flag_ar:.

    [:octicons-arrow-right-24: Go to example](islanded-methanol-production.ipynb)


-   :material-notebook:{ .lg .middle } **Electric Vehicles**

    Demonstrates how to model flexible electric vehicle charging and discharging.

    [:octicons-arrow-right-24: Go to example](battery-electric-vehicle-charging.ipynb)


-   :material-notebook:{ .lg .middle } **Backpressure CHPs**

    Models combined heat and power plants with fixed heat-to-power ratios.

    [:octicons-arrow-right-24: Go to example](chp-fixed-heat-power-ratio.ipynb)


-   :material-notebook:{ .lg .middle } **Extraction-Condensing CHPs**

    Models combined heat and power plants with variable heat-to-power ratios.

    [:octicons-arrow-right-24: Go to example](power-to-gas-boiler-chp.ipynb)


-   :material-notebook:{ .lg .middle } **Heat Pumps and Thermal Storage**

    Models sector coupling with heat pumps and thermal energy storage.

    [:octicons-arrow-right-24: Go to example](power-to-heat-water-tank.ipynb)


-   :material-notebook:{ .lg .middle } **Carbon Management**

    Models carbon flows between atmosphere, biomass, and synthetic fuels.

    [:octicons-arrow-right-24: Go to example](biomass-synthetic-fuels-carbon-management.ipynb)

</div>


## Complexity Management

<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Time Series Aggregation**

    Shows how model complexity can be reduced by aggregating snapshots.

    [:octicons-arrow-right-24: Go to example](time-series-aggregation.ipynb)

-   :material-notebook:{ .lg .middle } **Global Sensitivity Analysis**

    Combines PyPSA with SALib's Sobol indices to understand how technology cost uncertainties affect optimal system design and total costs.

    [:octicons-arrow-right-24: Go to example](gsa.ipynb)

-   :material-notebook:{ .lg .middle } **Storage Units as Links & Stores**

    Shows how storage units can be replaced by more fundamental links and stores.

    [:octicons-arrow-right-24: Go to example](replace-generator-storage-units-with-store.ipynb)

-   :material-notebook:{ .lg .middle } **Tracing Infeasibilities**

    Shows how to trace infeasibilities in the optimization problem using Irreducible Infeasible Subsets (IIS).

    [:octicons-arrow-right-24: Go to example](tracing-infeasibilities.ipynb)


</div>
