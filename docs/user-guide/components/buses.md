<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Bus

The [`Bus`][pypsa.components.Buses] is the fundamental node of the network, to which all other components
attach. It enforces energy conservation for all elements feeding in and out of
it in any snapshot (e.g. Kirchhoff's current law for electric buses). A [`Bus`][pypsa.components.Buses]
can represent a grid connection point, but it can also be used for other,
non-electric energy carriers (e.g. hydrogen, heat, oil) or even non-energy
carriers (e.g. CO~2~ or steel) in different locations. 

{{ read_csv('../../../pypsa/data/component_attrs/buses.csv') }}