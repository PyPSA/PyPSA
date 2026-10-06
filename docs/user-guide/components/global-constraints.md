<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Global Constraints

The [`GlobalConstraint`][pypsa.components.GlobalConstraints] components describe constraints in the optimisation
problem that apply to multiple components at once. Constraints of `type="flow_based"` form a [flow-based domain](#flow-based-domain).

{{ read_csv('../../../pypsa/data/component_attrs/global_constraints.csv') }}

## Flow-Based Domain

A flow-based market-coupling domain is a set of global constraints of `type="flow_based"`, one row per critical network element (CNEC), each with `sense="<="` and the remaining available margin (RAM) in `constant`. The concept and the constraints are explained on the [optimisation page](../optimization/global-constraints.md#flow-based-market-coupling); this section covers how to enter the data. See also the [example notebook](../../examples/flow-based-market-coupling.ipynb).

### Zonal PTDF

The zonal PTDF is a CNEC x zone matrix. Each zone column is stored as an ordinary attribute `ptdf_<zone>` of the rows, so a domain is added like any other global constraint, and `n.c.global_constraints.zonal_ptdf` assembles the matrix back:

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
ram = pd.Series([1000.0, 800.0], index=zonal_ptdf.index)
n.add(
    "GlobalConstraint",
    zonal_ptdf.index,
    type="flow_based",
    sense="<=",
    constant=ram,
    **zonal_ptdf.add_prefix("ptdf_"),
)
```

[`add_flow_based`][pypsa.components.GlobalConstraints.add_flow_based] does the same in one call, `n.c.global_constraints.add_flow_based(zonal_ptdf, ram)`, and also takes the time-varying layout: `zonal_ptdf` with a `(snapshot, CNEC)` MultiIndex (zones as columns) and `ram` as a snapshot x CNEC frame. Static and time-varying zone columns may be mixed; in the raw form a time-varying column is a snapshot x CNEC frame, e.g. `ptdf_DE=...`. A time-varying `constant` is only allowed for flow-based rows. With multiple investment periods, `investment_period` ties a row to one period; rows without it apply in all periods.

### Controllable link flows (AHC and EvFB)

A column may name a [`Link`][pypsa.components.Links] instead of a zone bus: an HVDC corridor into the region (AHC) or between two zones (EvFB). The column is the CNEC's sensitivity to the link flow (`bus0 -> bus1`) with the zone net positions held fixed; see [the optimisation page](../optimization/global-constraints.md#controllable-link-flows-ahc-and-evfb) for its definition and how it relates to published hub sensitivities.

```python
n.add("Link", "BE-DE", bus0="BE", bus1="DE", p_nom=1000, p_min_pu=-1)  # EvFB
n.add("Link", "NO2-NL", bus0="NO2", bus1="NL", p_nom=700, p_min_pu=-1)  # AHC
zonal_ptdf["BE-DE"] = ...
zonal_ptdf["NO2-NL"] = ...
n.c.global_constraints.add_flow_based(zonal_ptdf, ram)
```

!!! note "Cross-zone branches"

    The domain replaces the grid *between* zones. The network must not contain a [`Line`][pypsa.components.Lines], [`Transformer`][pypsa.components.Transformers] or [`Link`][pypsa.components.Links] between two zone buses, except a `Link` that is a domain column.
