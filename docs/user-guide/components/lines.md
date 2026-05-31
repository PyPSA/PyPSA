<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Line

The [`Line`][pypsa.components.Lines] components represent power transmission and distribution lines. They connect a `bus0` to a `bus1`. They can connect to buses with carrier "AC" or "DC". Power flow through lines is not directly controllable, but is determined passively by their impedances and the nodal power imbalances according to Kirchhoff's voltage law. To see how the impedances are used in the power flow, see the [line model](../power-flow.md#line-model).

Line loading limits are set with the nominal apparent-power rating `s_nom` and the per-unit time series limit `s_max_pu`. For a static 100 MW thermal rating, set `s_nom=100`; for dynamic line rating, provide a time series in `n.lines_t.s_max_pu`, such as `1.1` during cooler hours and `0.8` during warmer hours. A simple security margin for approximate $N-1$ operation can be represented by setting `s_max_pu=0.7`, reserving 30% of the line rating for contingencies.

!!! note "When to use [`Link`][pypsa.components.Links] instead?"

    - Use the [`Link`][pypsa.components.Links] for power lines with controllable power flow, such as point-to-point HVDC links.
    - Use the [`Link`][pypsa.components.Links] for any connection between buses with different carrier.


{{ read_csv('../../../pypsa/data/component_attrs/lines.csv', disable_numparse=True) }}
