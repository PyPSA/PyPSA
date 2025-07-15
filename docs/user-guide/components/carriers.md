# Carrier

The carrier describes energy carriers and defaults to `AC` for alternating current electricity networks. `DC` can be set for direct current electricity networks. It can also take arbitrary values for arbitrary energy carriers, e.g. `wind`, `heat`, `hydrogen` or `natural gas`.

Attributes relevant for global constraints can also be stored in this table, the canonical example being CO2 emissions of the carrier relevant for limits on CO2 emissions.

{{ read_csv('../../../pypsa/data/component_attrs/carriers.csv') }}
