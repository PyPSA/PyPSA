<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Carrier

The [`Carrier`][pypsa.components.Carriers] describes **energy carriers of buses** (e.g. "AC" for alternating current, "DC" for dicrect current, "hydrogen", or "heat") or **technologies** of other components (e.g. "wind", "gas turbine", "electrolyser", or "heat pump"). Besides descriptive names and colors for visualizations, attributes relevant for <!-- md:guide components/global-constraints.md --> can also be stored in this component class (e.g. CO$_2$ emissions of bus carriers relevant for emission limits).

{{ read_csv('../../../pypsa/data/component_attrs/carriers.csv') }}
