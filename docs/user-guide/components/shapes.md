# Shapes

Shapes is of a `geopandas.GeoDataFrame` which can be used to store network-related geographical data (for plotting, calculating potentials, etc.). The dataframe has the columns geometry, component, idx and type. The columns component, idx and type do not require specific values, but give the user the possibility to store additional information about the shapes.

{{ read_csv('../../../pypsa/data/component_attrs/shapes.csv') }}
