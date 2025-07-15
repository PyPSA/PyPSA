# Generator

The `Generator` components attach to a single bus and can feed in power. They
convert energy from their carrier to the carrier of the bus to which they
attach. They can be used to represent dispatchable conventional power plants,
renewable generators with variable availability, supply of grid electricity or
biomass from an external source. With inverted `sign`, they can also be used to
represent withdrawal of power at a given price or elastic demands following a
linear demand curve.

!!! note "When to use `Link` instead?"

    Use the `Link` component if you have the fuel of the generator represented by a `Bus` and you want to model the conversion of that fuel to electricity, e.g. a gas-fired power plant with a gas bus.


{{ read_csv('../../../pypsa/data/component_attrs/generators.csv') }} 
