# Line

The `Line` components represent power transmission and distribution lines. They connect a `bus0` to a `bus1`. They can connect to buses with carrier "AC" or "DC". Power flow through lines is not directly controllable, but is determined passively by their impedances and the nodal power imbalances according to Kirchhoff's voltage law. To see how the impedances are used in the power flow, see [line-model](#line-model).

!!! note "When to use `Link` instead?"

    - Use the `Link` for power lines with controllable power flow, such as point-to-point HVDC links.
    - Use the `Link` for any connection between buses with different carrier.


TODO Table
