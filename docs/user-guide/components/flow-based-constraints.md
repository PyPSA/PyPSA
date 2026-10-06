<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Flow-Based Constraint

The [`FlowBasedConstraint`][pypsa.components.FlowBasedConstraints] components describe a flow-based market-coupling domain: a set of linear constraints that bound the net positions of the market zones (buses) by their zonal PTDF sensitivities, one row per critical network element (CNEC). This replaces the grid constraints between zones with a compact market representation, as used in parts of the European day-ahead market. For how the domain enters the optimisation, see [flow-based market coupling](../optimization/flow-based-constraints.md).

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

For adding a single CNEC, `zonal_ptdf` may also be a `pandas.Series` over zones. Both `ram` and the zonal PTDF may be time-varying: pass a `zonal_ptdf` with a `(snapshot, CNEC)` MultiIndex (zones as columns).

### Deriving a zonal PTDF

If you have a nodal network, the zonal PTDF follows from its nodal PTDF and a *generation shift key* (GSK); see [the math](../optimization/flow-based-constraints.md#deriving-a-zonal-ptdf-from-a-nodal-grid). It is computed per [sub-network](../components/sub-networks.md) with [`calculate_zonal_PTDF`][pypsa.SubNetwork.calculate_zonal_PTDF], returning a labelled `branch × zone` frame:

```python
nodal.determine_network_topology()
sub = nodal.c.sub_networks.static.obj.iloc[0]
node_to_zone = nodal.buses["country"]  # any bus -> zone mapping (a pandas Series)

zonal_ptdf = sub.calculate_zonal_PTDF(node_to_zone, gsk="capacity")  # or "uniform"
```

The `gsk` argument is a scheme name or a ready bus × zone frame. Two builders are provided: [`gsk_uniform`][pypsa.SubNetwork.gsk_uniform] (equal weight per bus) and [`gsk_by_capacity`][pypsa.SubNetwork.gsk_by_capacity] (weight ∝ generator `p_nom`, optionally by `carrier`).

## Controllable link flows (AHC and EvFB)

A domain column may name a [`Link`][pypsa.components.Links] instead of a zone bus. This covers HVDC and other controllable corridors:

- **Advanced hybrid coupling (AHC):** a link from a bus outside the flow-based region into a zone.
- **Evolved flow-based (EvFB):** a link between two zones.

The link column holds the CNEC's sensitivity to the link flow in its `bus0 -> bus1` direction. An EvFB link has one column for both ends.

```python
n.add("Link", "BE-DE", bus0="BE", bus1="DE", p_nom=1000, p_min_pu=-1)  # EvFB
n.add("Link", "NO2-NL", bus0="NO2", bus1="NL", p_nom=700, p_min_pu=-1)  # AHC
zonal_ptdf["BE-DE"] = ...
zonal_ptdf["NO2-NL"] = ...
n.c.flow_based_constraints.add(cnecs, zonal_ptdf=zonal_ptdf, ram=ram)
```

See [the optimisation page](../optimization/flow-based-constraints.md#controllable-link-flows-ahc-and-evfb) for the definition of the link column.

!!! note "Cross-zone branches"

    The domain replaces the grid *between* zones. The network must not contain a [`Line`][pypsa.components.Lines], [`Transformer`][pypsa.components.Transformers] or [`Link`][pypsa.components.Links] between two zone buses, except a `Link` that is a domain column.
