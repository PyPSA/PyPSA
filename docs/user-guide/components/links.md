# Link

The `Link` is a component for controllable directed flows between two buses `bus0` and `bus1` with arbitrary energy carriers. It can have an efficiency loss and a marginal cost; for this reason its default settings allow only for power flow in one direction, from `bus0` to `bus1` (i.e. `p_min_pu = 0`). To build a bidirectional lossless link, set `efficiency = 1`, `marginal_cost = 0` and `p_min_pu = -1`.

The `Link` component can be used for any element with a controllable power flow: a bidirectional point-to-point HVDC link, a unidirectional lossy HVDC link, a converter between an AC and a DC network, a heat pump, an electrolyser, or resistive heater from an AC/DC bus to a heat bus, etc.

#TODO TAbke

### Multilink

Links can also be defined with multiple outputs in fixed ratio to the power in the single input by defining new columns `bus2`, `bus3`, etc. in `n.links` along with associated columns for `efficiency2`, `efficiency3`, etc. The different outputs are then equal to the input multiplied by the corresponding efficiency; see [opf-links](#opf-links) for how these are used in the LOPF and the [example of a CHP with a fixed power-heat ratio](https://pypsa.readthedocs.io/en/latest/examples/chp-fixed-heat-power-ratio.html).

The columns `bus2`, `efficiency2`, `bus3`, `efficiency3`, etc. in `n.links` are automatically added to the component attributes. The values in these columns are not compulsory; if the link has no second output, simply leave it empty `n.links.at["my_link", "bus2"] = ""` or as NaN.

For links with multiple inputs in fixed ratio to one of the inputs, you can define the other inputs as outputs with a negative efficiency so that they withdraw energy or material from the bus if there is a positive flow in the link.

As an example, suppose a link representing a methanation process takes as inputs one unit of hydrogen and 0.5 units of carbon dioxide, and gives as outputs 0.8 units of methane and 0.2 units of heat. Then `bus0` connects to hydrogen, `bus1` connects to carbon dioxide with `efficiency=-0.5` (since 0.5 units of carbon dioxide is taken for each unit of hydrogen), `bus2` connects to methane with `efficiency2=0.8` and `bus3` to heat with `efficiency3=0.2`.

The example [Biomass, synthetic fuels and carbon management](https://pypsa.readthedocs.io/en/latest/examples/biomass-synthetic-fuels-carbon-management.html) provides many examples of modelling processes with multiple inputs and outputs using links.
