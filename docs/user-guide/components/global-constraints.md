<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Global Constraints

The [`GlobalConstraint`][pypsa.components.GlobalConstraints] components describe constraints in the optimisation
problem that apply to multiple components at once. Constraints of `type="flow_based"` form a [flow-based domain](#flow-based-domain).

{{ read_csv('../../../pypsa/data/component_attrs/global_constraints.csv') }}

## Flow-Based Domain

A flow-based market-coupling domain is a set of global constraints of `type="flow_based"`, one row per critical network element (CNEC), each with `sense="<="` and the remaining available margin (RAM) in `ram` (`constant` is ignored for these rows). The concept and the constraints are explained on the [optimisation page](../optimization/global-constraints.md#flow-based-market-coupling); this section covers how to enter the data. See also the [example notebook](../../examples/fbmc-simple-example.ipynb).

### Zonal PTDF

The zonal PTDF is a CNEC x zone matrix. Each zone column is stored as an ordinary attribute `ptdf_<zone>` of the rows, so a domain is added like any other global constraint, and `n.c.global_constraints.zonal_ptdf` assembles the matrix back:

``` py
>>> n = pypsa.Network()
>>> n.add("Bus", ["DE", "FR", "BE"])
>>> zonal_ptdf = pd.DataFrame(  # one CNEC per row, one zone per column
...     {"DE": [0.4, -0.3], "FR": [-0.2, 0.5], "BE": [0.1, 0.2]},
...     index=["cnec_1", "cnec_2"],
... )
>>> ram = pd.Series([1000.0, 800.0], index=zonal_ptdf.index)
>>> n.add(
...     "GlobalConstraint",
...     zonal_ptdf.index,
...     type="flow_based",
...     sense="<=",
...     ram=ram,
...     **zonal_ptdf.add_prefix("ptdf_"),
... )
>>> n.c.global_constraints.zonal_ptdf
bus      DE   FR   BE
name
cnec_1  0.4 -0.2  0.1
cnec_2 -0.3  0.5  0.2
```

[`add_flow_based`][pypsa.components.GlobalConstraints.add_flow_based] does the same in one call, `n.c.global_constraints.add_flow_based(zonal_ptdf, ram)`, and also takes the time-varying layout: `zonal_ptdf` with a `(snapshot, CNEC)` MultiIndex (zones as columns) and `ram` as a snapshot x CNEC frame. Static and time-varying zone columns may be mixed; in the raw form a time-varying column is a snapshot x CNEC frame, e.g. `ptdf_DE=...`. `ram` may likewise be static or time-varying (`n.global_constraints_t.ram`). With multiple investment periods, `investment_period` ties a row to one period; rows without it apply in all periods.

A NaN in a time-varying `ram` drops that CNEC in that snapshot. This suits presolved domains, whose set of CNECs changes from hour to hour.

### Controllable link flows (AHC and EvFB)

A column may name a [`Link`][pypsa.components.Links] instead of a zone bus: an HVDC corridor into the region (AHC) or between two zones (EvFB). The column is the CNEC's sensitivity to the link flow (`bus0 -> bus1`) with the zone net positions held fixed; see [the optimisation page](../optimization/global-constraints.md#controllable-link-flows-ahc-and-evfb) for its definition and how it relates to published hub sensitivities.

``` py
>>> n.add("Bus", ["NO2", "NL"])
>>> n.add("Link", "BE-DE", bus0="BE", bus1="DE", p_nom=1000, p_min_pu=-1)  # EvFB
>>> n.add("Link", "NO2-NL", bus0="NO2", bus1="NL", p_nom=700, p_min_pu=-1)  # AHC
>>> zonal_ptdf["BE-DE"] = [0.3, -0.1]
>>> zonal_ptdf["NO2-NL"] = [0.05, 0.1]
>>> n.c.global_constraints.add_flow_based(zonal_ptdf, ram, overwrite=True)
>>> n.c.global_constraints.zonal_ptdf.columns.tolist()
['DE', 'FR', 'BE', 'BE-DE', 'NO2-NL']
```

!!! note "Cross-zone branches"

    The domain replaces the grid *between* zones. The network must not contain a [`Line`][pypsa.components.Lines], [`Transformer`][pypsa.components.Transformers] or [`Link`][pypsa.components.Links] between two zone buses, except a `Link` that is a domain column.

### Several domains side by side

Europe couples more than one flow-based domain, e.g. Core and Nordic. Name the domain of each row with `domain`, e.g. `add_flow_based(zonal_ptdf, ram, domain="Nordic")`. Each named domain gets its own zero-sum balance of net positions. Its zones are the zone columns its rows set (not NaN), so a zone must be set in the rows of exactly one domain. A `Link` between zones of two domains is a column of both. Name the domain of every `flow_based` row or of none; without names, all zones share one balance.

### Importing published domains

`flow_based_from_eraa` reads the Core and Nordic flow-based domains of ENTSO-E's European Resource Adequacy Assessment 2025,[^eraa] `flow_based_from_jao` the Core day-ahead domains of the JAO Publication Tool,[^core-pub] and `flow_based_from_tso` domains in the TSO `MS_FBMC` CSV format. See the [example notebook](../../examples/fbmc-simple-example.ipynb).

The network needs one bus per zone and one link per corridor, named as in the file:

| Format | Zone buses | EvFB link | AHC link |
|---|---|---|---|
| ERAA | `BE00`, `DE00`, ... | `BE00-DE00` | `DKE1-DE00` (from `DKE1` into `DE00`) |
| JAO | `BE`, `DE`, ... | `ALBE-ALDE` (between `BE` and `DE`) | `DE_SE4_Baltic` (from `SE4` into `DE`) |
| TSO | `BE`, `DE`, ... | `KONV_BE-DE1` | `KONV_AHC_DE-SE04` (HVDC into `DE`), `DKW` (AC exchange of `DKW`) |

Pass `buses=` or `links=` to use other names. A list in `links=` maps one corridor to several links, e.g. a pair of one-way links `{"DKW1-DE00": ["DKW1-DE00-DC", "DE00-DKW1-DC"]}`; each link gets the column in its own direction. The importer flips a corridor's sign if the link runs the other way, and drops corridors without a link (with a warning). It also converts each corridor to PyPSA's link column: it merges the two end columns of an EvFB link (JAO, TSO) and subtracts the zone PTDF of the link's ends from the published hub sensitivities (see [two equivalent net-position conventions](../optimization/global-constraints.md#controllable-link-flows-ahc-and-evfb)); ERAA's AHC columns are already published in this form and arrive unchanged. JAO's `CH` column is dropped, as Core publishes Swiss PTDFs for transparency only. For a multi-period network, `flow_based_from_eraa(..., investment_period=2030)` ties the domain to one period and appends the period to the CNEC names, so the domains of several target years can coexist. All importers take `domain=` to name the domain, e.g. `domain="Core"`. A corridor between two domains maps to one link in both, e.g. ERAA's `DKE1-DE00` (Core) and `DE00-DKE1` (Nordic) via `links={"DE00-DKE1": "DKE1-DE00"}` for Nordic. A Nordic domain is a RAM column (`season=0` to `6`), as assigned per hour in the workbook's `Domain Assignment` sheet.

Each zone bus carries its own generation and load. A bus outside the flow-based region, such as `DKE1`, needs its own supply or price. For an AC-coupled AHC zone such as `DKW`, the link carries the zone's AC exchange with the region. The zone's HVDC cables (e.g. to `NOS0`, or COBRA to `NL`) are ordinary links: they change that exchange through the zone's nodal balance, so their `DKW` ends need no column. For example, for ERAA:

=== "ERAA"

    ```python
    zones = ["AT00", "BE00", "CZ00", "DE00", "FR00", "HR00", "HU00",
             "ITN1", "NL00", "PL00", "RO00", "SI00", "SK00"]
    n = pypsa.Network()
    n.add("Bus", zones + ["DKE1"])
    n.add("Load", zones, bus=zones, p_set=demand)
    n.add("Generator", zones, bus=zones, p_nom=capacity, marginal_cost=cost)
    n.add("Generator", "DKE1", bus="DKE1", p_nom=5000, marginal_cost=20)
    n.add("Link", "BE00-DE00", bus0="BE00", bus1="DE00", p_nom=1000, p_min_pu=-1)
    n.add("Link", "DKE1-DE00", bus0="DKE1", bus1="DE00", p_nom=600, p_min_pu=-1)
    n.c.global_constraints.flow_based_from_eraa("FB-Domain-CORE_full.xlsx", year=2030, season="winter1")
    ```

=== "JAO"

    ```python
    zones = ["AT", "BE", "CZ", "DE", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK"]
    n = pypsa.Network()
    n.add("Bus", zones + ["SE4"])
    ...  # loads and generators as for ERAA
    n.add("Link", "ALBE-ALDE", bus0="BE", bus1="DE", p_nom=1000, p_min_pu=-1)
    n.add("Link", "DE_SE4_Baltic", bus0="SE4", bus1="DE", p_nom=600, p_min_pu=-1)
    n.c.global_constraints.flow_based_from_jao("finalComputation.csv")
    ```

=== "TSO"

    ```python
    zones = ["AT", "BE", "CZ", "DE", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK"]
    n = pypsa.Network()
    n.add("Bus", zones + ["SE04", "DKW"])
    ...  # loads and generators as for ERAA
    n.add("Link", "KONV_BE-DE1", bus0="BE", bus1="DE", p_nom=1000, p_min_pu=-1)
    n.add("Link", "KONV_AHC_DE-SE04", bus0="SE04", bus1="DE", p_nom=600, p_min_pu=-1)
    n.add("Link", "DKW-DE", bus0="DKW", bus1="DE", p_nom=2500, p_min_pu=-1)
    # the corridor "DKW" is named like its bus, so it needs a distinct link name
    n.c.global_constraints.flow_based_from_tso("MS_FBMC_Domain_TS1.csv", links={"DKW": "DKW-DE"})
    ```

[^eraa]: ENTSO-E (2025), [European Resource Adequacy Assessment 2025](https://www.entsoe.eu/eraa/2025/).

[^core-pub]: Core TSOs / JAO (2024), [Publication Tool for the Core day-ahead capacity calculation methodology, Publication Handbook v2.2](https://publicationtool.jao.eu/core/).
