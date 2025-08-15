# Transformer Types

The `TransformerType` components describe standard 2-winding transformer types.
If for a [`Transformer`](/api/components/types/transformers) the attribute `type` is non-empty, the electrical
parameters of the `TransformerType` are mapped to the [`Transformer`](/api/components/types/transformers).

{{ read_csv('../../../pypsa/data/component_attrs/transformer_types.csv') }}

The following standard transformer types are available, which are based on [pandapower's standard types](https://pandapower.readthedocs.io/en/latest/std_types/basic.html), whose parameterisation is in turn based on [DIgSILENT PowerFactory](http://www.digsilent.de/index.php/products-powerfactory.html).

{{ read_csv('../../../pypsa/data/standard_types/transformer_types.csv') }}
