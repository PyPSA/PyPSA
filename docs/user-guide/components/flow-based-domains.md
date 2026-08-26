<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Flow-Based Domain

The [`FlowBasedDomain`][pypsa.components.FlowBasedDomains] components describe a flow-based market-coupling domain: a set of linear constraints that bound the *net positions* of the market zones (buses) by

$$\text{zonal\_ptdf} \cdot NP \le \text{RAM},$$

where each component (row) is one critical network element (CNEC). This replaces the internal transmission grid between zones with a compact market representation, as used in the Core region of the European day-ahead market.

- The net position $NP_z$ of a zone is its net export (`generation - load`). During optimization it is added as a variable directly inside the nodal balance, so no auxiliary buses or links are needed and the zonal prices remain the native duals of the nodal balance.
- A single zero-sum constraint $\sum_z NP_z + \sum_v NP_v = 0$ closes the copper-plate balance across the zones and the AHC virtual hubs $v$ (see below); with no AHC borders it reduces to $\sum_z NP_z = 0$.
- The zone net positions are written to `n.buses_t.net_position` after optimizing. Without corridors these equal the bus net injections `n.buses_t.p`; when AHC/EvFB corridors touch a zone, `n.buses_t.p` is the *physical* injection (it includes the corridor flow) and diverges from the net position `generation - load` that the domain constrains, so read the net position from `n.buses_t.net_position`. The per-CNEC shadow prices are assigned to the `mu_domain` output when optimizing with `assign_all_duals=True`.

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

A domain column may also name a [`Link`][pypsa.components.Links] instead of a zone bus. These are the controllable HVDC corridors of *advanced hybrid coupling* (AHC, a link from a zone to an external hub) and *evolved flow-based coupling* (EvFB, a link between two zones). Each loads its CNECs through a `zonal_ptdf · Link-p` term on the existing interconnector (its `p_nom` is the capacity). To keep every zone's net position equal to `generation - load`, the corridor's contribution to its Core-side bus balance is cancelled — for EvFB (both ends zones) at both ends, for AHC (one end external) at the Core end only. An AHC border additionally enters the zero-sum balance as the net position $NP_v$ of its external virtual hub, so the imported power is priced by the external zone; EvFB, being internal and net-zero, stays out of the zero-sum. The AHC vs EvFB distinction is inferred from the link's endpoints — you do not declare it.

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

Published flow-based domains can be read directly. `from_eraa` parses an ERAA `FB-Domain-CORE` Excel workbook (one target year and season); `from_jao` parses a JAO `finalComputation` CSV (one market hour); `from_tso` parses a TSO `MS_FBMC` domain CSV (one typical situation). All are assumed time-invariant for now:

```python
n.c.flow_based_domains.from_eraa(
    "FB-Domain-CORE_simplified.xlsx", year="2030", season="winter1"
)
n.c.flow_based_domains.from_jao("finalComputation.csv")  # presolved rows by default
n.c.flow_based_domains.from_tso("MS_FBMC_Domain_TS1.csv")  # one file, self-typed
```

The TSO file carries a `!!OBJEKTTYP` header row that types every column, so `from_tso` reads just that one file: `RAM_MW` is the RAM, `FB_DOMAIN`/`FB_DOMAIN_AHC` columns are the zones, and `HGUE`/`HGUE_AHC` columns are HVDC converters (mapped via `links`, like JAO). It is Latin-1 by default and auto-detects the decimal separator (German `,` vs English `.`).

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
