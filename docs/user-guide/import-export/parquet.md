<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Parquet Store

<!-- md:badge-experimental -->

A parquet store is a directory of parquet files with a manifest, designed to
be self describing. The files plus one decode rule are enough to read the data
without PyPSA. Parquet is columnar and compressed, small and fast to scan, and
readable from many engines (pandas, polars, DuckDB, Arrow, R, Spark...), also
directly from cloud object storage. Install the optional dependencies via
`pip install pypsa[parquet]`.

To **export** a network run [`n.export_to_parquet()`][pypsa.network.io.NetworkIOMixin.export_to_parquet].
To **import** run [`n.import_from_parquet()`][pypsa.network.io.NetworkIOMixin.import_from_parquet]
or simply provide the path in the [`pypsa.Network`][] constructor.

``` py
n.export_to_parquet("foo/bar")
n_import = pypsa.Network("foo/bar")
```

## Store format

**Layout.** A store is a directory containing a manifest and parquet files:

```
my_network/
    dims/
        components/<Type>.parquet    members of a component type with static inputs
        snapshots.parquet            snapshot axis and weightings
        periods.parquet              period axis and weightings, only if multiperiod
        scenarios.parquet            scenario axis and weights, only if stochastic
    inputs/<attr>.parquet            one varying input per file
    outputs/<attr>.parquet           one result per file
    manifest.json                    see Manifest below
```

Solving only adds files under `outputs/`. Each `dims/components/<Type>.parquet`
is a subset of `c.static`: only the columns for attributes that cannot vary
across snapshots, scenarios or periods. Varying attributes live under `inputs/`
instead.

**Long schema.** Every file under `inputs/` and `outputs/` has the columns

```
component_type | name | snapshot | scenario | period | attribute | value
```

Each file holds exactly one attribute, so `value` carries that attribute's
dtype.

**Decode rule.** A row's `value` applies to every combination of its null
dimension columns, taken from the axis tables under `dims/`. Rows never
overlap. Anything not covered by a row, including an attribute with no file,
takes the attribute's `default` from the manifest. The writer mirrors the
in-memory representation, so a round-trip restores every value to the
container it came from.

**Manifest.** `manifest.json` holds the non-tabular network data and the
self-description of the store.

```json
{
    "format": "pypsa-parquet",
    "format_version": 1,
    "attributes": {"name": "my_network", ...},
    "meta": {...},
    "crs": "...",
    "attribute_catalog": {
        "Generator": {
            "p_max_pu": {
                "dims": ["name", "snapshot"],
                "default": 1.0
            },
            ...
        },
        ...
    }
}
```

The catalog is network independent, identical for every network with the
same custom attributes. Per attribute it records the allowed `dims` and
the `default` used wherever the files are silent. An attribute's actual
axes compose from its `dims` and the axis files under `dims/`. Reading
never depends on the catalog, just writing does, since it defines which
shapes are valid input.

!!! warning

    The catalog will be expanded in a future version to become the single
    source of truth for attribute metadata.
