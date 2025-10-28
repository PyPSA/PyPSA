<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Features

PyPSA is a flexible framework for modelling and optimising modern energy
systems. It supports high spatial, temporal, and sectoral resolution, from
short-term dispatch to long-term planning. The following feature set outlines
its key features:

## Optimisation Functionalities

- **:material-power-plug-battery-outline: Economic Dispatch (ED):** Models
short-term market-based dispatch including unit commitment (either with integer
variables as MILP or in a relaxed approximation as LP), renewable availability,
short-duration and seasonal storage including hydro reservoirs with inflow and
spillage dynamics, elastic demands, load shedding and conversion between energy
carriers, using either perfect operational foresight or rolling horizon time
resolution.

- **:material-transmission-tower: Linear Optimal Power Flow (LOPF):** Extends economic dispatch to determine
the least-cost dispatch while respecting network constraints in meshed AC-DC
networks, using a linearised representation of power flow (KVL, KCL) with
optional loss approximations.

- **:material-security: Security-Constrained LOPF (SCLOPF):** Extends LOPF by accounting for line
outage contingencies to ensure system reliability under $N-1$ conditions.

- **:material-crane: Capacity Expansion Planning (CEP):** Supports least-cost
long-term system planning with investment decisions for generation, storage,
conversion, and transmission infrastructure. Handles both single and multiple
investment periods. Continuous and discrete investments are supported.

- **:material-road: Pathway Planning:** Supports co-optimisation of multiple investment periods to
plan energy system transitions over time with perfect planning foresight.

- **:material-rollerblade: Rolling-Horizon Optimisation:** Enables sequential optimisation of operation
with myopic foresight, allowing for dynamic information updates and breaking
down large problems into manageable time slices.

- **:material-crosshairs-question: Stochastic Optimisation:** Implements two-stage stochastic programming
framework with scenario-weighted uncertain inputs, with investments as
first-stage decisions and dispatch as recourse decisions.

- **:material-diversify: Modelling-to-Generate-Alternatives (MGA):** Explores near-optimal decision
spaces to provide insight into the range of feasible system configurations with
similar costs.

- **:fontawesome-solid-building-columns: Policy Constraints:** Built-in support
  for policy constraints such as CO~2~ emission limits and pricing, subsidies, resource
  limits, expansion limits, and growth limits. Extendable by custom constraints.

- **:material-screwdriver: Custom Constraints:** Users can impose own objectives, variables and
constraints, such as policy constraints or technical requirements, using
[Linopy](https://linopy.readthedocs.io/).

- **:material-dots-square: Solver Flexibility:** Supports a wide range of LP, MILP, and QP solvers from
open-source solutions (e.g. [HiGHS](https://highs.dev/),
[SCIP](https://scipopt.org)) to commercial products (e.g.
[Gurobi](https://www.gurobi.com/), [COPT](https://shanshu.ai/copt)).

## Use Cases and Grid Modelling

- **:material-heat-pump: Sector-Coupling:** Modelling integrated energy systems with multiple energy
  carriers (electricity, heat, hydrogen, etc.) and conversion between them.
  Flexible representation of technologies such as heat pumps, electrolysers,
  battery electric vehicles (BEVs), direct air capture (DAC), and synthetic
  fuels production.

- **:material-lightbulb: Diverse Applications:** Supports a wide range of energy system analyses for
  strategic decision support. Applications include techno-economic assessment of
  technologies, capacity expansion, transmission planning, market design,
  sector-coupling, integration of variable renewables such as wind and solar, flexibility needs and resource
  adequacy assessments, network congestion analysis, battery scheduling,
  electricity trading, planning of decarbonisation pathways, hydrogen
  infrastructure planning, electrolyser siting and operation, resilience to
  extreme weather, bidding zone configurations, and the design of islanded
  systems for remote renewable fuel production.

- **:material-transmission-tower: Standard Grid Components:** Includes standard types for lines and
  transformers from [pandapower](https://pandapower.org).

- **:simple-graphql: Static Power Flow Analysis:** Computes both full non-linear and linearised
  load flows for meshed AC and DC grids using Newton-Raphson method with
  optional distributed slack and shunt compensation.

## Architecture and Performance

- **:material-view-module: Modular Design:** Clean separation between data and modelling code enables
flexible scenario development.

- **:material-car-cruise-control: Resolution Control:** Offers flexible control over temporal, spatial, and
sectoral scope and detail.

- **:material-grid-large: Spatial Clustering:** Can reduce model size for large networks via spatial
  aggregation strategies.

- **:material-data-matrix: Data Backbone:** Uses [pandas](https://pandas.pydata.org/) for data handling
  and [linopy](https://linopy.readthedocs.io/) for optimisation and interfacing
  with solvers.

- **:material-run-fast: Performance:** Using [linopy](https://linopy.readthedocs.io/), the code is
designed to scale well with high resolution networks, minimising memory usage
and time spent outside the solver.

- **:fontawesome-solid-arrow-trend-up: Scalability:** Handles models ranging from small conceptual prototypes to
continent-scale high-resolution systems solved on high-performance compute
clusters.

## Analysis and Usability

- **:simple-shadow: Shadow Prices:** Outputs dual values such as nodal clearing prices (LMPs),
storage water values, scarcity and CO~2~ prices.

- **:simple-googlesheets: Statistics:** Built-in tools for summarising and visualising results, such
  as energy balances, capacities, costs, market values, curtailment, component
  revenues.
  
- **:octicons-paintbrush-16: Visualisations:** Built-in tools for plotting statistics, time series data
  and spatial distributions of line loadings and nodal dispatch decisions.
  
- **:material-notebook: Documentation:** Comprehensive user guide, API reference, and plenty of
examples are available.

- **:fontawesome-solid-people-group: Community Support:** Active community on
  [GitHub](https://github.com/pypsa/pypsa) and
  [Discord](https://discord.gg/AnuJBk23FU) for user support and development
  discussions.

- **:material-lock-open: MIT License:** Fully open-source and free for commercial and academic use.

## Illustrations

Interactive **network visualization** of SciGRID example network clustered by federal state, showing transmission capacities, flow directions, electricity supply (upper pie charts) and demand (lower pie charts), as well as average nodal prices (color scale):

<div style="width: 100%; height: 800px; overflow: hidden;">
    <iframe src="https://bxio.ng/assets/html/scigrid-interactive-map"
            width="100%" height="100%" frameborder="0" 
            style="border: 0px solid #ccc; transform: scale(1); transform-origin: 0 0;">
    </iframe>
</div>

Interactive area plot of **electricity balance time series** in a highly-renewable sector-coupled example network, showing temporal generation (positive values) and consumption (negative values) of different technologies:

<div style="width: 100%; height: 550px;">
    <iframe src="../../../assets/interactive/carbon_management-energy_balance-area_iplot-AC-bus_carrier.html" 
            width="100%" height="100%" frameborder="0" style="border: 0px solid #ccc;">
    </iframe>
</div>