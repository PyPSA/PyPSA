<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Load

The [`Load`][pypsa.components.Loads] components attach to a single bus and represent a demand for the
[`Bus`][pypsa.components.Buses] carrier they are connected to. With inverted sign, they can also be used
to model an exogenous supply. For "AC" buses, they act as a PQ load. If $p>0$
the load is consuming active power from the bus and if $q>0$ it is consuming
reactive power (i.e. behaving like an inductor).

!!! note "When to use [`Generator`][pypsa.components.Generators] instead?"

    Use the [`Generator`][pypsa.components.Generators] component with a negative `sign` to model elastic demands following a linear demand curve or to represent a comnsumption at a given price.

{{ read_csv('../../../pypsa/data/component_attrs/loads.csv', disable_numparse=True) }} 