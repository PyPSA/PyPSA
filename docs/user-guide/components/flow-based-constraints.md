<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Flow-Based Constraint

The [`FlowBasedConstraint`][pypsa.components.FlowBasedConstraints] components describe a flow-based market-coupling domain: a set of linear constraints that bound the net positions of the market zones (buses) by their zonal PTDF sensitivities, one row per critical network element (CNEC). This replaces the transmission grid *between* zones with a compact market representation, as used in the Core region of the European day-ahead market. For how the domain enters the optimisation, see [flow-based market coupling](../optimization/flow-based-constraints.md).

The net position of each zone is added as a variable inside the nodal balance, so the zonal prices remain the native duals of the nodal balance. After optimising, the net positions are written to `n.buses_t.net_position` and the per-CNEC shadow prices to `mu_domain` (with `assign_all_duals=True`).

## Zonal PTDF

Unlike the scalar attributes below, the zonal PTDF sensitivities form a matrix (CNEC × zone) and are stored in the dedicated frame `n.c.flow_based_constraints.zonal_ptdf` (rows = CNECs, columns = zone buses). The name distinguishes it from the *nodal* PTDF computed per [sub-network](../components/sub-networks.md). Pass it to [`n.add`][pypsa.Network.add] together with the remaining available margin `ram`:

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
n.add("FlowBasedConstraint", zonal_ptdf.index, zonal_ptdf=zonal_ptdf, ram=[1000.0, 800.0])
```

For a single CNEC, `zonal_ptdf` may also be a `pandas.Series` over zones. Both `ram` and the zonal PTDF may be time-varying: pass a `zonal_ptdf` with a `(snapshot, CNEC)` MultiIndex (zones as columns), and `c.zonal_ptdf.loc[snapshot]` selects one hour's matrix. A domain is static or time-varying as a whole, not a mix.

### Deriving a zonal PTDF

If you have a nodal, grid-resolved network, the zonal PTDF follows from its nodal PTDF and a *generation shift key* (GSK); see [the math](../optimization/flow-based-constraints.md#deriving-a-zonal-ptdf-from-a-nodal-grid). It is computed per [sub-network](../components/sub-networks.md) with [`calculate_zonal_PTDF`][pypsa.SubNetwork.calculate_zonal_PTDF], returning a labelled `branch × zone` frame:

```python
nodal.determine_network_topology()
sub = nodal.c.sub_networks.static.obj.iloc[0]
node_to_zone = nodal.buses["country"]  # any bus -> zone mapping (a pandas Series)

zonal_ptdf = sub.calculate_zonal_PTDF(node_to_zone, gsk="capacity")  # or "uniform"
```

The `gsk` argument is a scheme name or a ready bus × zone frame. Two builders are provided: [`gsk_uniform`][pypsa.SubNetwork.gsk_uniform] (equal weight per bus) and [`gsk_by_capacity`][pypsa.SubNetwork.gsk_by_capacity] (weight ∝ generator `p_nom`, optionally by `carrier`). Every sub-network bus must map to a zone. The `ram` is not derived — supply it yourself.

## Controllable link flows (AHC and EvFB)

A domain column may name a [`Link`][pypsa.components.Links] instead of a zone bus: the controllable HVDC corridors of *advanced hybrid coupling* (AHC, a link to an external hub) and *evolved flow-based coupling* (EvFB, a link between two zones). The link flow then loads its CNECs directly, and the AHC/EvFB distinction is inferred from the link's endpoints — you do not declare it. The column's sign must follow the link's `bus0 → bus1` direction.

```python
n.add("Link", "ALEGrO", bus0="BE", bus1="DE", p_nom=1000)  # EvFB (two zones)
n.add("Link", "NorNed", bus0="NL", bus1="NO2", p_nom=700)   # AHC (zone to external NO2)
zonal_ptdf["ALEGrO"] = ...  # sensitivity to the link flow in its bus0 -> bus1 direction
zonal_ptdf["NorNed"] = ...
n.c.flow_based_constraints.add(cnecs, zonal_ptdf=zonal_ptdf, ram=ram)
```

See [the optimisation page](../optimization/flow-based-constraints.md#controllable-link-flows-ahc-and-evfb) for how corridors are kept out of the zones' net positions.

!!! note "Cross-zone branches"

    The domain replaces the electrical grid *between* zones, so the network must not contain a [`Line`][pypsa.components.Lines], [`Transformer`][pypsa.components.Transformers] or [`Link`][pypsa.components.Links] directly connecting two zone buses — except a `Link` that is a declared domain column (an AHC/EvFB corridor). Branches with a non-zone end (an external border, a gas pipeline) are ignored.

## Importing published domains

Published domains can be read directly with `from_eraa` (ERAA `FB-Domain-CORE` workbook), `from_jao` (JAO `finalComputation` CSV) and `from_tso` (TSO `MS_FBMC` CSV). Each maps the file's zone labels to buses of the same name (pass `buses=` to remap) and accepts a `links=` mapping to include AHC/EvFB corridors. See the [example notebook](../../examples/flow-based-market-coupling.ipynb) for a worked import.

```python
n.c.flow_based_constraints.from_eraa("FB-Domain-CORE.xlsx", year="2030", season="winter1")
```

{{ read_csv('../../../pypsa/data/component_attrs/flow_based_constraints.csv', disable_numparse=True) }}
