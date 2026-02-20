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

## Time-Delayed Energy Delivery

Links can model time-delayed energy transport via the `delay` and `cyclic_delay` attributes. This is useful for representing transport delays in pipelines, shipping and other transport modes, or any process where energy withdrawn at `bus0` arrives at output ports after a configurable time lag.

- **`delay`**: The delay in units of elapsed time, measured against cumulative `n.snapshot_weightings.generators`. Energy withdrawn from `bus0` at snapshot `t` arrives at `bus1` at the first snapshot where the cumulative weighting since `t` reaches at least `delay`. For example, with uniform hourly snapshots (weightings = 1), `delay=3` means a 3-hour delay. With 3-hourly snapshots (weightings = 3), the same `delay=3` shifts delivery by one snapshot.

- **`cyclic_delay`**: If `True` (default), energy wraps cyclically from the end of the optimization horizon back to the start — energy sent in the last snapshots arrives at the first snapshots. If `False`, energy is lost at the tail of the horizon and the first snapshots receive nothing from delayed links.

- **Multi-port delays**: Each output port can have its own delay. For additional ports (`bus2`, `bus3`, ...), use `delay2`, `delay3`, ... and `cyclic_delay2`, `cyclic_delay3`, ....

!!! info "Delays in multi investment periods"

    In multi-period optimization, delays are applied independently within each investment period. Energy does not cross period boundaries, so a delay in the last snapshots of one period does not carry over to the first snapshots of the next period.

!!! example "Adding a delayed link"

    ```python
    n.add(
        "Link",
        "H2 pipeline",
        bus0="production",
        bus1="demand",
        p_nom=500,
        efficiency=0.95,
        delay=3,           # 3 time-unit delivery delay
        cyclic_delay=True,  # energy wraps around at horizon boundary
    )
    ```

!!! tip

    For delays applied to chained hydro reservoirs, see the [Chained Hydro-Reservoirs example](../../examples/chained-hydro-reservoirs.ipynb). For non-cyclic behavior, multi-port delays, and non-uniform snapshot weightings, see the [Link Delay example](../../examples/link-delay.ipynb).

{{ read_csv('../../../pypsa/data/component_attrs/links.csv', disable_numparse=True) }}
