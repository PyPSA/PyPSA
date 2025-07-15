# Line Types

Standard line types with per length values for impedances.

If for a line the attribute "type" is non-empty, then these values are multiplied with the line length to get the line's electrical parameters.

{{ read_csv('../../../pypsa/data/component_attrs/line_types.csv') }}

The following standard line types are available:

{{ read_csv('../../../pypsa/data/standard_types/line_types.csv') }}

The line type parameters in table above are based on [pandapower's standard types](https://pandapower.readthedocs.io/en/latest/std_types/basic.html), whose parameterisation is in turn loosely based on [DIgSILENT PowerFactory](http://www.digsilent.de/index.php/products-powerfactory.html). The parametrisation of lines is supplemented by additional sources such as [JAO's Static Grid Model](https://www.jao.eu/static-grid-model).
