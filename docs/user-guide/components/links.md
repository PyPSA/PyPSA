<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Link

The [`Link`][pypsa.components.Links] components are used for controllable directed flows between two or
more buses with arbitrary energy carriers (`bus0`, `bus1`, `bus2`, etc.). For
instance, they can represent point-to-point HVDC links, unidirectional lossy
HVDC links, converters between AC and DC, net transfer capacities (NTCs) of HVAC
lines (neglecting Kirchhoff's voltage law), as well as any conversion between
carriers (e.g. electricity to hydrogen in electrolysis, electricity to heat in
heat pumps, or gas to electricity and heat in a combined heat and power (CHP)
plant).

- The [`Link`][pypsa.components.Links] component has one dispatch variable `p0` associated with the input
  from `bus0` (positive if withdrawing from `bus0`) and one or more outputs
  `p1`, `p2`, etc. associated with the output at `bus1`, `bus2`, etc. (negative
  if supplying to these buses). The  outputs at `bus1`, `bus2`, etc. are
  proportional to the input at `bus0` multiplied by the corresponding efficiency
  (e.g. `efficiency`, `efficiency2`, etc.). That means that the link can have
  multiple outputs in fixed ratio to the input.

- The columns `bus2`, `efficiency2`, `bus3`, `efficiency3`, etc. in `n.links`
  are automatically added to the component attributes.

- Any `marginal_cost` are related to the input `p0` at `bus0` (e.g. cost per
  unit of fuel consumed rather than cost per unit of electricity produced for a
  power plant).

- For links with multiple inputs in fixed ratio to one of the inputs, you can
  define the other inputs as outputs with a negative efficiency so that they
  withdraw from the corresponding bus if there is a positive flow for `p0`.

!!! note "When to use [`Line`][pypsa.components.Lines] instead?"

    Use the [`Line`][pypsa.components.Lines] component type for power lines for which their power flow is determined passively through Kirchhoff's voltage law.

!!! example "[`Link`][pypsa.components.Links] with bidirectional lossless flow"

    Because the [`Link`][pypsa.components.Links] component can have efficiency losses and marginal costs, the default settings allow only for flow in one direction (`p_min_pu=0`), from `bus0` to `bus1`. To build a bidirectional lossless link, set `efficiency = 1`, `marginal_cost = 0` and `p_min_pu = -1`.

!!! example "[`Link`][pypsa.components.Links] with a single input and multiple outputs"

    Suppose a link representing a combined heat and power (CHP) plant takes as input 1 unit of fuel and gives as outputs 0.3 units of electricity and 0.7 units of heat. Then `bus0` connects to the fuel, `bus1` connects to electricity with `efficiency=0.3` and `bus2` connects to heat with `efficiency2=0.7`. [This example](../../examples/chp-fixed-heat-power-ratio.ipynb) illustrates a CHP with a fixed power-heat ratio using links.

!!! example "[`Link`][pypsa.components.Links] with multiple inputs and a single output"

    Suppose a link representing a methanation process takes as inputs one unit of hydrogen and 0.5 units of carbon dioxide, and gives as outputs 0.8 units of methane and 0.2 units of heat. Then `bus0` connects to hydrogen, `bus1` connects to carbon dioxide with `efficiency=-0.5` (since 0.5 units of carbon dioxide is taken for each unit of hydrogen), `bus2` connects to methane with `efficiency2=0.8` and `bus3` to heat with `efficiency3=0.2`. [This example](../../examples/biomass-synthetic-fuels-carbon-management.ipynb) illustrates many modelling processes with multiple inputs and outputs using links.

{{ read_csv('../../../pypsa/data/component_attrs/links.csv', disable_numparse=True) }} 
