<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Storage Unit

The [`StorageUnit`][pypsa.components.StorageUnits] components connect to a single bus and enable inter-temporal energy shifting with coupled power and energy capacity modelling. This component is suitable for modeling batteries, pumped hydro storage, and other storage technologies where power and energy capacities are coupled.

- Energy capacity is defined as a fixed ratio (`max_hours`) of power capacity: `e_nom = p_nom * max_hours` (MW × h = MWh)
- An `inflow` attribute (an exogenous parameter representing an energy input from external sources) and `spill` (energy overflow/spillage) variable are supported
- For storage units, if $p>0$ the storage unit is supplying active power to the bus (discharging) and if $q>0$ it is supplying reactive power.

!!! note "[`Store`][pypsa.components.Stores] covers all storage unit functionality"

    The [`Store`][pypsa.components.Stores] component supports `max_hours`, `efficiency_store`, `efficiency_dispatch`, `inflow` and spillage as well. [`n.storage_units_to_stores()`][pypsa.Network.storage_units_to_stores] converts storage units into equivalent stores.

{{ read_csv('../../../pypsa/data/component_attrs/storage_units.csv') }}
