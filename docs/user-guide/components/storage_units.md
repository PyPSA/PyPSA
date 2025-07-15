# Storage Unit

Storage units connect to a single bus and enable inter-temporal energy shifting with coupled power and energy capacity modeling. This compoenent is suitable for modeling batteries, pumped hydro storage, and other storage technologies where power and energy capacities are coupled.

- Energy capacity is defined as a fixed ratio (`max_hours`) of power capacity: `e_nom = p_nom * max_hours` (MW × h = MWh)
- Power capacity (`p_nom`) can be optimized in capacity expansion problems
- Supports `inflow` attribute (an exogenous parameter representing an energy input from external sources) and `spill` (energy overflow/spillage) variable, see attribute documentation below for details
- For storage units, if $p>0$ the storage unit is supplying active power to the bus (discharging) and if $q>0$ it is supplying reactive power (behaving like a capacitor).

## When to use store instead

For independent optimization of power and energy capacities, use a fundamental `Store` component combined with separate `Link` components for charging and discharging. See also [this example](https://pypsa.readthedocs.io/en/latest/examples/replace-generator-storage-units-with-store.html).

{{ read_csv('../../../pypsa/data/component_attrs/storage_units.csv') }}
