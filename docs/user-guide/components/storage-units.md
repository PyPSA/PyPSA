# Storage Unit

The [`StorageUnit`](/api/components/types/storage_units) components connect to a single bus and enable inter-temporal energy shifting with coupled power and energy capacity modelling. This compoenent is suitable for modeling batteries, pumped hydro storage, and other storage technologies where power and energy capacities are coupled.

- Energy capacity is defined as a fixed ratio (`max_hours`) of power capacity: `e_nom = p_nom * max_hours` (MW × h = MWh)
- An `inflow` attribute (an exogenous parameter representing an energy input from external sources) and `spill` (energy overflow/spillage) variable are supported
- For storage units, if $p>0$ the storage unit is supplying active power to the bus (discharging) and if $q>0$ it is supplying reactive power.

!!! note "When to use [`Store`](/api/components/types/stores) instead?"

    For independent optimization of power and energy capacities, use the [`Store`](/api/components/types/stores) component at an auxiliary [`Bus`](/api/components/types/buses) combined with separate [`Link`](/api/components/types/links) components for charging and discharging. Add a [`Generator`](/api/components/types/generators) or [`Load`](/api/components/types/loads) at the auxiliary [`Bus`](/api/components/types/buses) for modelling spillage and inflow from an external source. See also [this example](https://pypsa.readthedocs.io/en/latest/examples/replace-generator-storage-units-with-store.html).

{{ read_csv('../../../pypsa/data/component_attrs/storage_units.csv') }}
