<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->


For long-term planning problems where the network is optimised for different
time horizons, PyPSA offers functionality optimise the network across multiple
investment periods (e.g. 2030, 2040, 2050) simulataneously with perfect foresight.

## Literature

In the literature, two different methods can be distinguished for pathway
optimisation with perfect foresight, mainly differing in the way they handle
the investment costs of assets:[^1]

* In **Type I**, the complete overnight investment costs are applied in the investment period the asset is built.
* In **Type II**, the investment costs are annualised over the years in which an asset is active (i.e. build year plus lifetime).

PyPSA uses **Type II**, mainly because it allows a cleaner separation of the
discounting over different years and the end-of-horizon effects are smaller
compared to **Type I**. End-of-horizon effects occur when models undervalue long-lived assets and overbuild short-term ones because they ignore system needs beyond the final period.

## Implementation

Optimisation with multiple investment periods can be run with

``` py
n.optimize(multi_investment_periods=True)
```

!!! info "Overnight versus pathway optimisation"

    By default, there are no investment periods defined, and the network is
    optimised for a single investment period (overnight scenario).

For pathway optimisation, the `n.snapshots` have to be a `pandas.MultiIndex`,
with the first level as a subset of the investment periods (see [Investment
Periods](../design.md#investment-periods)). The investment periods are defined
in `n.investment_periods`, a `pandas.Index` of monotonically increasing integers
of years (e.g. `[2030, 2040, 2050]`).

Investment periods weightings are stored in `n.investment_period_weightings`, a
`pandas.DataFrame` indexed by `n.investment_periods` with two columns for
objective ($W^{\mathrm{o}}_a$, for social discounting of costs) and years ($W^{\mathrm{y}}_a$, for
emission budgets).

The general approach for modelling multiple investment periods is to add
components for each investment period $a$, in which their capacity should be
extendable (i.e. each generator $i$ with its capacity $G_i$ belongs to one investment period).
For example, optimising wind capacities from 2030 to 2050 in 10-year steps
requires a separate generator for each investment period (e.g. `wind-2030`,
`wind-2040`, `wind-2050`), each with the corresponding build year and lifetime.

This structure enables the specification of differing technological assumptions
per investment period, such as decreasing investment costs, improving
efficiencies, or better capacity factors from higher hub heights. Moreover, not
all technologies need to be available in each investment period. For instance,
to forbid new coal power plants after 2025 or to introduce new technologies like
small modular nuclear reactors (SMR) only from 2040 onwards.


## Objective

With multiple investment periods $a$, the objective weighs the cost of each
period with the period weighting $W^{\mathrm{o}}_a$. The investment cost of an
asset counts in every period in which the asset is active, by its `build_year`
and `lifetime`. Its capital weight is the sum of $W^{\mathrm{o}}_a$ over these
periods:

<!-- md:spec Generator_capex -->

The build cost uses the *annualised* investment costs (**Type II**).
Operational costs weigh each snapshot with $w^{\mathrm{o}}_t$ times the
weighting of its period:

<!-- md:spec Generator_opex -->

The other components follow the same pattern. See [Objective](objective.md) for
all terms of the total cost.

!!! warning

    Note that the `build_year` and `lifetime` attributes are only used to determined whether a component is active
    in a particular investment period. They are **not** used to annualise the costs to get the `capital_cost` attribute. This must be done by by the user, because there are use cases where
    the `capital_cost` is calculated from a loan period that is different from the technical lifetime of the component.

!!! note

    Extendable components with build years **before** the first investment period are considered as **non-extendable** components, i.e. taking whatever capacity is given for `{p,e,s}_nom`.

??? note "Symbols"

    <!-- md:spec-legend Generator_capex Generator_opex -->

    The investment periods are `n.investment_periods`, and $W^{\mathrm{o}}_a$ is `n.investment_period_weightings.objective`.

## Constraints

### Dispatch of Active Components

A component is active in investment period $a$ if `build_year` $\le a <$
`build_year` + `lifetime`. Dispatch variables exist only for snapshots in which
the component is active, so an inactive component does not dispatch:

<!-- md:spec Generator_p -->

In the example above, the `wind-2030` generator built in 2030 cannot contribute to electricity generation in 2025.

### Emission Budgets

Global constraints can be defined as a budget over all investment periods, e.g.
for CO~2~ emissions. The energy of each snapshot counts with the snapshot
weighting $w^{\mathrm{g}}_t$ times the years $W^{\mathrm{y}}_a$ of its period:

<!-- md:spec GlobalConstraint_upper GlobalConstraint_energy_weight -->

They can also be defined as a limit for each investment period, e.g. for CO~2~
emissions. Then the global constraint names an `investment_period`, and only
the snapshots of that period count (`GlobalConstraint_counts_snapshot`).
See [Primary Energy](../optimization/global-constraints.md#primary-energy) for the terms of
generators and storage.

??? note "Symbols"

    <!-- md:spec-legend GlobalConstraint_upper GlobalConstraint_energy_weight -->

    The dual variable $\mu$ of the constraint is stored in `n.global_constraints.mu`.

### Expansion Limits

The `{p,e,s}_nom_{max,min}` for each component now refers to the installable limit for
this build year. To ensure that the total capacity of all active components does
not exceed the technical potential in each region, we must add an additional
global constraint placing an expansion limit on the sum of carrier capacities at
each bus $n$ and investment period $a$ (`type="tech_capacity_expansion_limit"`).
The limit may even change with each investment period, e.g. due to changes in land use or
city development.

See [Expansion Limit](../optimization/global-constraints.md#expansion-limit) and
[Growth Limit](../optimization/global-constraints.md#growth-limit-per-carrier) for more
details on global constraints for expansion limits.

### Storage Cyclicity and Initial Energy Levels

The cyclicity constraints for the [`Store`][pypsa.components.Stores] and [`StorageUnit`][pypsa.components.StorageUnits] components and
initial energy levels require special attention in the context of multiple
investment periods.

By default, a storage component with a set initial state of charge (`e_initial`,
`state_of_charge_initial`) is not replenished at the beginning of each
investment period, but only once for the first snapshot of the first investment
period. To consider that the storage is filled to the initial state of charge
for each investment period, the `state_of_charge_initial_per_period` or
`e_initial_per_period` attribute must be set to `True`.

By default, the storage cyclicity constraints for a storage with cyclic
behaviour (`cyclic_state_of_charge`, `e_cyclic`) are applied across all investment
periods, i.e. the storage is cyclic across all investment periods. To consider that
the storage is cyclic for each investment period, the
`cyclic_state_of_charge_per_period` or `e_cyclic_per_period` attribute must be
set to `True`.

!!! note "Technological learning"

    Technological learning, i.e. the reduction of investment costs for technologies over investment periods as experience grows, is currently not yet implemented in PyPSA. However, a study by Zeyen et al. (2023)[^2]
    demonstrates how to implement this on top of PyPSA with a MILP formulation of a piecewise-linearised learning curve using [SOS2](https://en.wikipedia.org/wiki/Special_ordered_set) constraints.

## Examples


<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Pathway Planning**

    Optimizes investment decisions across multiple investment periods for a long-term transition pathway with perfect foresight.

    [:octicons-arrow-right-24: Go to example](../../examples/multi-investment-optimisation.ipynb)

-   :material-notebook:{ .lg .middle } **Myopic Pathway Planning**

    Optimizes investment decisions across multiple investment periods for a
    long-term transition pathway with myopic foresight.

    [:octicons-arrow-right-24: Go to example](../../examples/myopic-pathway.ipynb)

</div>

[^1]: T. Brown (2020), [Multi-Horizon Planning with Perfect Foresight](https://nworbmot.org/energy/multihorizon.pdf).

[^2]: E. Zeyen, M. Victoria, T. Brown (2023), [Endogenous learning for green hydrogen in a sector-coupled energy model for Europe](https://doi.org/10.1038/s41467-023-39397-2). Nature Communications, 14, 3743, doi:10.1038/s41467-023-39397-2.
