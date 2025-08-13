# Line

The [`Line`](/api/components/types/lines) components represent power transmission and distribution lines. They connect a `bus0` to a `bus1`. They can connect to buses with carrier "AC" or "DC". Power flow through lines is not directly controllable, but is determined passively by their impedances and the nodal power imbalances according to Kirchhoff's voltage law. To see how the impedances are used in the power flow, see the [line model](/user-guide/power-flow/#line-model).

!!! note "When to use [`Link`](/api/components/types/links) instead?"

    - Use the [`Link`](/api/components/types/links) for power lines with controllable power flow, such as point-to-point HVDC links.
    - Use the [`Link`](/api/components/types/links) for any connection between buses with different carrier.

