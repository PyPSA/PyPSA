<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Generator

The [`Generator`][pypsa.components.Generators] components attach to a single bus and can feed in power. They
convert energy from their carrier to the carrier of the bus to which they
attach. They can be used to represent dispatchable conventional power plants,
renewable generators with variable availability, supply of grid electricity or
biomass from an external source.

!!! note "When to use [`Load`][pypsa.components.Loads] instead?"

    To model withdrawal of power at a given price or elastic demand following a linear demand curve, use an active [`Load`][pypsa.components.Loads] with a positive `marginal_cost` (the willingness to pay) instead of a generator with inverted `sign`. For generators with `sign < 0`, `marginal_cost` currently applies to the dispatch variable regardless of `sign`; from PyPSA 2.0 on, it will apply to the power injected at the bus as for loads.

!!! note "When to use [`Link`][pypsa.components.Links] instead?"

    Use the [`Link`][pypsa.components.Links] component if you have the fuel of the generator represented by a [`Bus`][pypsa.components.Buses] and you want to model the conversion of that fuel to electricity, e.g. a gas-fired power plant with a gas bus.


{{ read_csv('../../../pypsa/data/component_attrs/generators.csv') }}

!!! note "Time-varying unit commitment costs"

    For committable generators, `start_up_cost` and `shut_down_cost` can be provided as time series, allowing costs to vary per snapshot. See the [Unit Commitment](../optimization/unit-commitment.md#start-up-and-shut-down) documentation for details.
