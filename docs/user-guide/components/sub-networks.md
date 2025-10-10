<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Sub-Network

The [`SubNetwork`][pypsa.components.SubNetworks] components are network subsets formed by buses with the same
carrier that are connected by passive branches (i.e. [`Line`][pypsa.components.Lines] and [`Transformer`][pypsa.components.Transformers]).
Sub-networks with carrier "AC" correspond to synchronous areas, in which the
power flow is determined by the line and transformer impedances following
Kirchhoff's voltage law. Only "AC" and "DC" sub-networks can contain passive
branches. All other sub-networks must contain a single isolated bus.

!!! warning

    Sub-networks are not entered by the user but dynamically calculated by
    [`n.determine_network_topology()`][pypsa.Network.determine_network_topology].

{{ read_csv('../../../pypsa/data/component_attrs/sub_networks.csv') }}
