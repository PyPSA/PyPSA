# Bus

The [`Bus`](/api/components/types/buses) is the fundamental node of the network, to which all other components
attach. It enforces energy conservation for all elements feeding in and out of
it in any snapshot (e.g. Kirchhoff's current law for electric buses). A [`Bus`](/api/components/types/buses)
can represent a grid connection point, but it can also be used for other,
non-electric energy carriers (e.g. hydrogen, heat, oil) or even non-energy
carriers (e.g. CO~2~ or steel) in different locations. 

{{ read_csv('../../../pypsa/data/component_attrs/buses.csv') }}