<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Numerical Scaling

Solvers work best when the coefficients, bounds and right-hand sides of a problem sit in a narrow band around 1. Energy system models often mix units that break this, e.g. CO~2~ buses in tonnes next to electricity buses in MWh, global constraints with right-hand sides of `1e8` or objective coefficients spanning many orders of magnitude. Badly scaled models solve slower and trigger solver warnings such as Gurobi's *"Matrix range"* messages.

PyPSA lets you declare the **solver unit** of each bus, each global constraint and the objective. The factors only change the numbers handed to the solver. [linopy](https://linopy.readthedocs.io) transforms solutions, shadow prices and the objective back to original units, so a scaled and an unscaled run give the same result up to solver tolerance.

## Scaling factors

| Factor | Set via | Acts on |
|---|---|---|
| Bus column factor $s_b$ | `n.c.buses.static.scaling` | Dispatch, capacity and storage columns of all one-ports at bus $b$ and of all branches with `bus0` $= b$ (solver column holds $s_b \cdot x$), plus all rows of these components and the energy balance row of bus $b$ (`Bus-nodal_balance`) |
| Global constraint row factor | `n.c.global_constraints.static.scaling` | Coefficients and right-hand side of the `GlobalConstraint-{name}` row |
| Objective factor | `n.optimize(objective_scaling=...)` | All objective coefficients, plus the cost-valued `objective_constant` column and the columns and rows of [CVaR](stochastic.md) (`CVaR`, `CVaR-a`, `CVaR-theta`, `CVaR-excess-*`, `CVaR-def`) |

A bus factor `s = 1e-3` means the solver works in GW instead of MW at that bus. Since the factor applies to the columns of a component and to all of its internal rows, these rows are unchanged in the solver. A [Link](../components/links.md) coefficient in the energy balance of `bus1` reads $\eta \cdot s_{\text{bus1}} / s_{\text{bus0}}$ in the solver. With the same factor at both buses, the coefficients of the energy balance stay at their unscaled values.

Binary and integer columns (unit commitment status, modular capacities, maintenance), angles (transformer phase shifts), and the normalised weights and binaries of [piecewise](piecewise.md) formulations are never scaled. The output column of a piecewise curve (e.g. the piecewise cost or the converted flow of a Link) and its linking, chord and domain rows take the factor of the component, so breakpoints are expressed in solver units. The rows of the [Kirchhoff voltage law](power-flow.md) are scaled by `1e5` times the factor of their sub-network, which keeps their solver coefficients independent of the bus factor. Since these rows mix the flows of all branches of a sub-network, all buses of one AC or DC sub-network must share the same factor.

``` py
import pypsa

n = pypsa.examples.ac_dc_meshed()

# Carrier-wide factors via a mapping, e.g. GW for electricity
n.c.buses.static.scaling = n.c.buses.static.carrier.map({"AC": 1e-3, "DC": 1e-3})

# CO2 limit in kt instead of t
n.c.global_constraints.static.scaling = 1e-3

# Objective in million currency units
n.optimize(objective_scaling=1e-6)
```

Factors must be finite and positive; all default to `1`. Results such as `n.c.generators.static.p_nom_opt`, `n.c.buses.dynamic.marginal_price` or `n.c.global_constraints.static.mu` and the duals in `n.model` are reported in original units. LP files and `n.model.matrices` show solver units. Use `n.model.matrices` (`A`, `b`, `c`, `lb`, `ub`) to check the four ranges a solver judges.

Constraints added after [`n.optimize.create_model()`][pypsa.optimization.OptimizationAccessor.create_model], e.g. in `extra_functionality`, are not scaled by PyPSA. Set their factor with linopy, e.g. `n.model.add_constraints(..., scaling=1e-3)` or `n.model.constraints["my-constraint"].scaling = 1e-3`. [`optimize_security_constrained`][pypsa.optimization.OptimizationAccessor.optimize_security_constrained], [`optimize_mga`][pypsa.optimization.OptimizationAccessor.optimize_mga], [`optimize_mga_in_direction`][pypsa.optimization.OptimizationAccessor.optimize_mga_in_direction] and [`optimize_mga_in_multiple_directions`][pypsa.optimization.OptimizationAccessor.optimize_mga_in_multiple_directions] accept `objective_scaling` as well. In MGA, the budget constraint takes the factor, and the alternative objective keeps the factor of the cost objective although its units differ (e.g. MW for capacity weights). Choose the factor with both objectives in mind.

## Choosing factors

Three rules follow from how column, row and objective scaling act:

- **Only column scaling narrows a spread inside a row or inside the objective.** Row and objective factors change the unit of a whole row, so they only move its band.
- **Column scaling moves bounds and coefficients in opposite directions.** A column that is too small in both its bounds and its coefficients cannot be fixed in both places.
- **Column scaling never touches the right-hand side, and row scaling never touches the bounds.** After changing the unit of a bus, re-check the rows its columns appear in.

!!! tip "Scale units, not asset sizes"

    Scaling targets **mismatched units**, e.g. a tonne-based CO~2~ bus connected to MWh buses by links with efficiencies around `1e-3`, huge global constraint rows and the objective band. It does not help with large single assets: column-scaling the bus of a `1e8` MWh biomass store only moves the spread from its bounds into the coefficients of the links and rows it shares with other assets. A large spread of asset sizes at one bus is real and no choice of units hides it.

!!! note "Keep factors at 1 unless units genuinely mismatch"

    On a PyPSA-Eur sector network with 39 nodes, objective and global constraint scaling were neutral, and scaling the CO~2~ buses (tonnes) by `1e-3` made things worse: HiGHS failed to solve, Gurobi was unaffected. CO~2~ coefficients of `0.013` to `1.8` t/MWh are already of the same order as the energy flows, so the factor only pushed them away from 1. Keep all factors at the default `1` and scale a bus only for a genuine unit mismatch, e.g. kt next to MWh or link efficiencies around `1e-3` or `1e3`. Large right-hand sides from annual potentials, e.g. EU-wide biomass, are real magnitudes that no scaling removes.

Scaling also shifts what solver tolerances mean, since a solver checks feasibility and optimality on the scaled problem. Results of scaled and unscaled runs therefore agree only up to solver tolerance.
