<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Storage Unit

The [`StorageUnit`][pypsa.components.StorageUnits] components connect to a single bus and enable inter-temporal energy shifting with coupled power and energy capacity modelling. This compoenent is suitable for modeling batteries, pumped hydro storage, and other storage technologies where power and energy capacities are coupled.

- Energy capacity is defined as a fixed ratio (`max_hours`) of power capacity: `e_nom = p_nom * max_hours` (MW × h = MWh)
- An `inflow` attribute (an exogenous parameter representing an energy input from external sources) and `spill` (energy overflow/spillage) variable are supported
- For storage units, if $p>0$ the storage unit is supplying active power to the bus (discharging) and if $q>0$ it is supplying reactive power.

!!! note "When to use [`Store`][pypsa.components.Stores] instead?"

    For independent optimization of power and energy capacities, use the [`Store`][pypsa.components.Stores] component at an auxiliary [`Bus`][pypsa.components.Buses] combined with separate [`Link`][pypsa.components.Links] components for charging and discharging. Add a [`Generator`][pypsa.components.Generators] or [`Load`][pypsa.components.Loads] at the auxiliary [`Bus`][pypsa.components.Buses] for modelling spillage and inflow from an external source. See also [this example](../../examples/replace-generator-storage-units-with-store.ipynb).

{{ read_csv('../../../pypsa/data/component_attrs/storage_units.csv') }}
