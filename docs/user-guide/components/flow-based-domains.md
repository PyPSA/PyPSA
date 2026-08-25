<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Flow-Based Domain

The [`FlowBasedDomain`][pypsa.components.FlowBasedDomains] components describe a flow-based market-coupling domain: a set of linear constraints that bound the *net positions* of the market zones (buses) by

$$\text{zonal\_ptdf} \cdot NP \le \text{RAM},$$

where each component (row) is one critical network element (CNEC). This replaces the internal transmission grid between zones with a compact market representation, as used in the Core region of the European day-ahead market.

- The net position $NP_z$ of a zone is its net export (`generation - load`). During optimization it is added as a variable directly inside the nodal balance, so no auxiliary buses or links are needed and the zonal prices remain the native duals of the nodal balance.
- A single zero-sum constraint $\sum_z NP_z = 0$ closes the copper-plate balance across zones.
- The zone net positions are the bus net injections and are read from `n.buses_t.p`; the per-CNEC shadow prices are assigned to the `mu_domain` output when optimizing with `assign_all_duals=True`.

## Zonal PTDF

Unlike the scalar attributes below, the zonal power transfer distribution factors form a matrix (CNEC × zone) and are stored in the dedicated frame `n.c.flow_based_domains.zonal_ptdf` (rows = CNECs, columns = zone buses). The name distinguishes it from the *nodal* PTDF computed per [sub-network](sub-networks.md) via `sub_network.calculate_PTDF()`. Pass it directly to [`n.add`][pypsa.Network.add] together with the remaining available margin `ram` (which may be static or time-varying):

```python
import pandas as pd
import pypsa

n = pypsa.Network()
n.add("Bus", ["DE", "FR", "BE"])

# one CNEC per row, one zone per column
zonal_ptdf = pd.DataFrame(
    {"DE": [0.4, -0.3], "FR": [-0.2, 0.5], "BE": [0.1, 0.2]},
    index=["cnec_1", "cnec_2"],
)
n.add("FlowBasedDomain", zonal_ptdf.index, zonal_ptdf=zonal_ptdf, ram=[1000.0, 800.0])
```

!!! note "Cross-zone links"

    The domain replaces the electrical exchange between zones, so the network must not contain electrical [`Link`][pypsa.components.Links] components that directly connect two zone buses. Non-electrical links (e.g. gas pipelines or electrolysers) with at least one non-zone end are ignored.

## Importing published domains

Published flow-based domains can be read directly. `from_eraa` parses an ERAA `FB-Domain-CORE` Excel workbook, selecting one target year and season (assumed time-invariant for now):

```python
n.c.flow_based_domains.from_eraa(
    "FB-Domain-CORE_simplified.xlsx", year="2030", season="winter1"
)
```

The zone labels in the file are used as bus names as-is; there is no fuzzy matching. Where the labels differ from your bus names, pass an explicit mapping, e.g. `buses={"DE00": "DE"}`. Zones that are not buses in the network raise an error rather than being dropped silently. Reading `.xlsx` requires the `excel` extra (`openpyxl`).

{{ read_csv('../../../pypsa/data/component_attrs/flow_based_domains.csv') }}
