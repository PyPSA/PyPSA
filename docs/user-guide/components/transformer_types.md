# Transformer Types

Standard 2-winding transformer types.

If for a transformer the attribute "type" is non-empty, then these values are used for the transformer's electrical parameters.

{{ read_csv('../../../pypsa/data/component_attrs/transformer_types.csv') }}

The following standard transformer types are available:

{{ read_csv('../../../pypsa/data/standard_types/transformer_types.csv') }}

The transformer type parameters in the table above are based on [pandapower's standard types](http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/std_types/basic.html), whose parameterisation is in turn loosely based on [DIgSILENT PowerFactory](http://www.digsilent.de/index.php/products-powerfactory.html).
