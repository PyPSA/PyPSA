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

## Controllable link flows (AHC and EvFB)

A domain column may also name a [`Link`][pypsa.components.Links] instead of a zone bus. These are the controllable HVDC corridors of *advanced hybrid coupling* (AHC, a link from a zone to an external hub) and *evolved flow-based coupling* (EvFB, a link between two zones). Both enter the constraint identically as `zonal_ptdf · Link-p` terms, so there is no AHC/EvFB distinction and no auxiliary variables — the link's own `p_nom` is its capacity, and its flow delivers power through the ordinary nodal balance.

```python
n.add("Link", "ALEGrO", bus0="BE", bus1="DE", p_nom=1000)  # EvFB (two zones)
n.add("Link", "NorNed", bus0="NL", bus1="NO2", p_nom=700)   # AHC (zone to external NO2)
zonal_ptdf["ALEGrO"] = ...  # sensitivity to the link flow in its bus0 -> bus1 direction
zonal_ptdf["NorNed"] = ...
n.c.flow_based_domains.add(cnecs, zonal_ptdf=zonal_ptdf, ram=ram)
```

The link column's sign must follow the link's `bus0 → bus1` flow direction (the sign of `Link-p`); flip the column if your link orientation is opposite to the published convention.

!!! note "Cross-zone branches"

    The domain replaces the electrical grid *between* zones, so the network must not contain a [`Line`][pypsa.components.Lines], [`Transformer`][pypsa.components.Transformers] or [`Link`][pypsa.components.Links] directly connecting two zone buses — with one exception: a `Link` that is a declared domain column (an EvFB corridor) is kept, since its flow enters the constraint explicitly. Branches with at least one non-zone end (e.g. an external border or a gas pipeline) are ignored.

## Importing published domains

Published flow-based domains can be read directly. `from_eraa` parses an ERAA `FB-Domain-CORE` Excel workbook, selecting one target year and season (assumed time-invariant for now); `from_jao` parses a JAO `finalComputation` CSV for one market hour:

```python
n.c.flow_based_domains.from_eraa(
    "FB-Domain-CORE_simplified.xlsx", year="2030", season="winter1"
)
n.c.flow_based_domains.from_jao("finalComputation.csv")  # presolved rows by default
```

Both importers map the file's zone labels to bus names of the same name; there is no fuzzy matching. For JAO the `Ptdf_<hub>` columns are read and the `Ptdf_` prefix stripped to obtain the hub name. Where the labels differ from your bus names, pass an explicit mapping, e.g. `buses={"DE00": "DE"}`. Zones that are not buses in the network raise an error rather than being dropped silently. JAO `CneName` is not unique across directions and contingencies, so the numeric `Id` is used as the CNEC name by default (`name_col`). Reading `.xlsx` requires the `excel` extra (`openpyxl`).

To bring in the AHC/EvFB corridors as link terms, pass a `links` mapping. For ERAA the border columns are directed labels `"A-B"`, so the sign is aligned automatically to the link's `bus0 → bus1` orientation; unmapped corridors are dropped:

```python
n.c.flow_based_domains.from_eraa(
    "FB-Domain-CORE_simplified.xlsx", year="2030", season="winter1",
    links={"CH00-AT00": "AT_CH_dc", "BE00-DE00": "ALEGrO"},
)
```

For JAO the external hubs are undirected labels, so `links` only renames the column — set the link's orientation (or flip the column) to match the hub's net-position convention.

{{ read_csv('../../../pypsa/data/component_attrs/flow_based_domains.csv') }}
