<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Frequently Asked Questions

## Does PyPSA model more than just the power system? Can methane, hydrogen, carbon dioxide networks be included?

Yes, PyPSA can model any energy carrier or material flow. Typically this is done as a transport network with linear losses. See for example how gas networks are modelled in PyPSA-Eur in [Hofmann et al. (2025)](https://www.nature.com/articles/s41560-025-01752-6). On the demand side, PyPSA has been used to model the full energy system, including building heating (heat pumps, gas boilers, district heating), process heating in industry, process emissions in industry, electric vehicles with flexible charging, demands for transport fuels and industrial feedstocks.

## How is demand modelled?

Demand can be modelled with linear or convex-quadratic models as perfectly inelastic, perfectly inelastic up to a value of lost load, or with price elasticity (linear demand function modelled as quadratic program). Cross-price elasticity between different hours is also possible with custom modifications (see [Brown et al. (2025)](https://www.sciencedirect.com/science/article/pii/S014098832500307X)). Constant Elasticity of Substitution (CES) functions cannot be modelled out of the box.

## Can investments be modelled for different years over multiple decades?

Yes, see [Pathway Planning](../user-guide/optimization/pathway-planning.md) as
well as [Zeyen et al. (2023)](https://www.nature.com/articles/s41467-023-39397-2). Just expect some hit to solution speed, as the
model size grows with the number of investment periods.

## Can technological learning be modelled?

No, not yet out of the box. But it is possible to model technological learning with a piecewise linear approximation of the learning curve using SOS2 constraints, as shown in [Zeyen et al. (2023)](https://www.nature.com/articles/s41467-023-39397-2). This requires some customisation of the model.

## Can you do Generation Adequacy Studies with Monte Carlo unplanned outages?

No, this functionality is not offered directly in PyPSA, but it can be built in
an outer loop around PyPSA.

## Does PyPSA have a GUI?

There is currently no desktop application where you can build a model from scratch without programming in Python. The usual mode of interaction with PyPSA is via Python scripts or Jupyter Notebooks, where you enter code to build and inspect the model and plot inputs and outputs. There are also [online scenario generators](https://model.energy/scenarios/) for specific applications, where you can enter inputs and start simulations.

## How easy is it to add custom constraints to PyPSA?

PyPSA uses `linopy` in the background. Using its syntax, you can add custom
objectives, variables and constraints to the optimisation (see [Custom
Constraints](../user-guide/optimization/custom-constraints.md)).

## Can you model market clearing in PyPSA?

Yes, if you take care with demand side. The [`Load`][pypsa.components.Loads] component represents
perfectly inelastic demand. Use [`Generator`][pypsa.components.Generators] component with negative sign
attribute to represent elastic demand, and pair [`Load`][pypsa.components.Loads] components with [`Store`][pypsa.components.Stores]
and [`StorageUnit`][pypsa.components.StorageUnits] components for modelling shiftable loads.

## Does PyPSA do ancillary service co-optimization (e.g. frequency control)?

Not directly. It requires some customisation depending on what you want, but
there is an example in the documentation.

## Can you model intra-day as well as day-ahead markets?

Currently, this is not possible without customisation.

## Is storage capacity optimized endogenously?

Yes, see [Storage User Guide](../user-guide/optimization/storage.md).

## Does PyPSA offer stochastic optimisation?

Yes, a two-stage stochastic programming framework is available since Version `v1.0`
(see [Stochastic Optimisation](../user-guide/optimization/stochastic.md)).

## How does PyPSA model grid load flow physics?

PyPSA models the grid load flow physics with a standard linear DC power flow
approximation, using a cycle-based formulation for computational efficiency (see
[Power Flow](../user-guide/optimization/power-flow.md)). An optional
piecwise-linear loss approximation is also available (see [Neumann et al.
(2022)](https://www.sciencedirect.com/science/article/pii/S0306261922002938)).

## Can PyPSA model AC power flow?

Yes and no. PyPSA can only consider linear power flow constraints in
optimisation problems with `n.optimize()`. However, it is possible to use PyPSA
to compute AC power flow solutions after the optimisation has been solved, using
the `n.pf()` function. This is useful for post-processing and analysis of the
results, but it does not change the optimisation problem itself.

## How are N-1 and line outages handled?

An implementation of security constrained LOPF is offered (see
[Contingencies](../user-guide/optimization/contingencies.md)). A simplified
option would be to limit line loading to 70% of the thermal limit (see
[Gazafroudi et al. (2022)](https://doi.org/10.1016/j.ijepes.2021.107702)).

## Can transmission line projects be endogenously selected?

The general mode is continuous transmission expansion on existing lines. Changes
in impedance can be considered by an iterative approach with
`n.optimize.optimize_transmission_expansion_iteratively()`. This function also
allows for discretisation procedures in between iterations. PyPSA also supports
Limits](../user-guide/optimization/capacity-limits.md)). However, this option
does not yet consider changes in impedance endogenously.

<!-- ## How are capacity retirements and stranded assets handled?

Lisa knows. -->

## Is electric vehicle charging and V2G handled endogenously?

Yes, it can be modelled with a combination of loads and storage.

## How do I model a concentrating solar plant?

See, for instance, [Hampp (2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0292892).

## How do I model retrofitting of coal plants with CCS?

See, for instance, [Zhou et al. (2024)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ein2.12011).

## How long does it take a PyPSA model to run?

Depends on the size and complexity of the model, but typically from a few seconds to several hours.

## How much computing resource do I need to run a PyPSA model?

Depends on the size and complexity of the model, but typically a modern laptop
or desktop computer is sufficient for small to medium-sized models. For larger
models, a server with more RAM and CPU cores may be required.

## Do I need to buy a commercial solver?

No, but it helps for very large problems with high spatial and temporal
resolution, cross-sectoral scope, and multiple investment periods. PyPSA is
CPLEX are recommended for better performance or allow finding any solutions at
all.

## How long does it take to learn PyPSA?

This depends on your prior experience and the depth of knowledge you want to
achieve. If you are already familiar with Python and pandas, you can get started
with PyPSA in a few hours to a few days. If you are new to Python, it may take a
week or two to become comfortable with the basics. For more advanced usage, such
as customising the model or adding new components, it may take several weeks of
practice and learning.

## Can I get support for using PyPSA?

PyPSA has an active community of users and developers on Discord and elsewhere
who can help you with questions and issues (see
[Support](../user-guide/support.md)). Also, various consultancies offer paid
support, such as [OET](https://www.openenergytransition.org),
[d-fine](https://www.d-fine.com/en/),
[Energynautics](https://energynautics.com/en/), and
[CLIMACT](https://www.climact.com/).

## Can you guarantee PyPSA will be developed further in the future?

We cannot guarantee anything, but there is a lively community of developers,
several of whom have permanent contracts, and there is funding guaranteed until
2027 for a research software engineer.

## Is there an automatic conversion from PLEXOS to PyPSA?

Not yet. PyPSA relies on tabular data inputs, so if you can convert your data
into a pre-defined Excel or CSV format, you can read it into PyPSA.

## Is there an automatic conversion from MATPOWER to PyPSA?

No, but there is an importer from PYPOWER (the Python implementation of
MATPOWER). It is called `n.import_from_pypower_ppc()`. PyPSA relies on tabular
data inputs, so if you can convert your data into a pre-defined Excel or CSV
format, you can read it into PyPSA.

## Is there an automatic conversion from pandapower to PyPSA?

Yes, but not all components are supported yet (e.g. three-winding transformers).
The importer is called `n.import_from_pandapower_net()`. PyPSA relies on tabular
data inputs, so if you can convert your data into a pre-defined Excel or CSV
format, you can read it into PyPSA.

## Can I do nodal pricing / LMP calculations?

Yes. Dual values (shadow prices) of the nodal balance constraints can be used to infer nodal prices. These are available in `n.buses_t.marginal_price`.

## Can PyPSA model sub-hourly time resolutions?

Yes, PyPSA can handle sub-hourly resolution (e.g. 15-minute timesteps). Just
define them in `n.snapshots` and adjust the `n.snapshot_weightings` accordingly.

## Can I model unit commitment in PyPSA?

Yes, PyPSA can model ramping constraints, minimum up and down times, and other
unit commitment constraints for generators and links as a MILP and with an
LP-relaxation (see [Unit
Commitment](../user-guide/optimization/unit-commitment.md)).

## Can PyPSA model district heating networks?

Yes, at a high level of abstraction (e.g. via transport links with losses and
centralized heating technologies). Detailed hydraulic or temperature-dependent
physics are not represented.

## Can I model endogenous fuel prices in PyPSA?

Yes, the market clearing prices returned in `n.buses_t.marginal_price` also work
for non-electric buses, and depend on the supply and consumption options at that
bus (e.g. fossil gas, biogas upgrading, synthetic methane with different costs
and volume restrictions).
