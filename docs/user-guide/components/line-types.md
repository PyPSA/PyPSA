<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Line Types

The `LineType` components describe standard line types with per length values
for impedances. If for a [`Line`][pypsa.components.Lines] the attribute `type` is non-empty, it is mapped
to the `LineType`'s electrical parameters and multiplied with the `length`
attribute of the [`Line`][pypsa.components.Lines].

{{ read_csv('../../../pypsa/data/component_attrs/line_types.csv') }}

The following standard line types are available, which are based on [pandapower's standard types](https://pandapower.readthedocs.io/en/latest/std_types/basic.html), whose parameterisation is in turn based on [DIgSILENT PowerFactory](http://www.digsilent.de/index.php/products-powerfactory.html). Other sources include [JAO's Static Grid Model](https://www.jao.eu/static-grid-model).

{{ read_csv('../../../pypsa/data/standard_types/line_types.csv') }}
