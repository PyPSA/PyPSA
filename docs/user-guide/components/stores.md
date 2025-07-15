# Store

The `Store` component connects to a single bus and provides fundamental energy storage functionality with independent storage energy and charge/discharge power capacity optimization.

To control charging and discharging power, separate `Link` components must be connected to the Store. This decoupled approach enables independent optimization of power and energy capacities, different efficiencies for charging and discharging processes, and asymmetric power ratings (different charging/discharging rate).

- `Store` component only "stores" energy without converting between energy carriers. It automatically inherits an energy carrier from the connected bus
- Energy capacity (`e_nom`) can be optimized independently of charge-/discharge power constraints
- `Store` component has no power capacity attribute `p_nom` (like `StorageUnit`), the charge/discharge power is controlled by a `Link` component connected to the Store
- `marginal_cost` attribute applies equally to both charging and discharging operations, representing the cost per unit of energy stored or released. This differs from `StorageUnit` components where marginal costs apply only to the marginal cost of production (discharging).
- The `marginal_cost` of the `Store` component can represent trading in external energy markets where the stored carrier can be bought or sold at market prices.
- For modeling technical marginal costs where both charging and discharging increase the objective function, use two separate Link components with distinct cost structures for charging and discharging processes.

## When to use storage unit instead

Use `StorageUnit` when power and energy capacities have a fixed relationship and you need integrated power dispatch control within a single component. This is simpler for e.g. a storage device where the power-to-energy ratio is predetermined by the manufacturer.

See [this example](https://pypsa.readthedocs.io/en/latest/examples/replace-generator-storage-units-with-store.html) for implementation details.

# TODO Table
