<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Transformer

The [`Transformer`][pypsa.components.Transformers] components represent 2-winding transformers that convert AC power from one voltage level to another. They connect a `bus0` (typically at higher voltage) to a `bus1` (typically at lower voltage). Power flow through transformers is not directly controllable, but is determined passively by their impedances and the nodal power imbalances. To see how the impedances are used in the power flow, see the [transformer model](../power-flow.md#transformer-model).

{{ read_csv('../../../pypsa/data/component_attrs/transformers.csv') }} 