<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Line

The [`Line`][pypsa.components.Lines] components represent power transmission and distribution lines. They connect a `bus0` to a `bus1`. They can connect to buses with carrier "AC" or "DC". Power flow through lines is not directly controllable, but is determined passively by their impedances and the nodal power imbalances according to Kirchhoff's voltage law. To see how the impedances are used in the power flow, see the [line model](../power-flow.md#line-model).

!!! note "When to use [`Link`][pypsa.components.Links] instead?"

    - Use the [`Link`][pypsa.components.Links] for power lines with controllable power flow, such as point-to-point HVDC links.
    - Use the [`Link`][pypsa.components.Links] for any connection between buses with different carrier.


{{ read_csv('../../../pypsa/data/component_attrs/lines.csv', disable_numparse=True) }} 