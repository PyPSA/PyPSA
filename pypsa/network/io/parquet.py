# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Parquet store for networks.

A store is a self-describing directory of parquet files with a manifest:

    my_network/
    ├── dims/
    │   ├── components/<Type>.parquet  members of a type, static inputs only
    │   │                              (a subset of `c.static`)
    │   └── snapshots.parquet, ...     other dimensions with weightings
    ├── inputs/<attr>.parquet          one varying input, no file means default
    ├── outputs/<attr>.parquet         one result, defaults are not written
    └── manifest.json                  format version and metadata

The full store format is documented at
https://docs.pypsa.org/latest/user-guide/import-export/parquet/.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pyproj import CRS

from pypsa.common import _is_output, check_optional_dependency

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path

if TYPE_CHECKING:
    from pypsa import Network
    from pypsa.components.components import Components

logger = logging.getLogger(__name__)

_INSTALL_MSG = (
    "Missing optional dependencies to use parquet stores. Install them via "
    "`pip install pypsa[parquet]` or `conda install -c conda-forge pyarrow`."
)

PARQUET_FORMAT = "pypsa-parquet"
PARQUET_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
DIMS_DIR = "dims"
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
COMPONENTS_DIR = "components"
SCENARIOS_NAME = "scenarios.parquet"
SNAPSHOTS_NAME = "snapshots.parquet"
PERIODS_NAME = "periods.parquet"

# Reserved manifest key that tags infinities in JSON
_FLOAT_TAG = "__pypsa_float__"

# Recomputed by `calculate_dependent_values` on import, like the other exporters
# TODO: should be flagged in the component attribute definitions, not hardcoded
_DERIVED_STATIC_COLUMNS = ("g_pu", "b_pu")

# Leads the wide component tables
_DIM_COMPONENT_LEAD_COLS = ("component_type", "name", "scenario")

_LONG_COLUMNS = [
    "component_type",
    "name",
    "snapshot",
    "scenario",
    "period",
    "attribute",
    "value",
]


def _file_meta(kind: str, **extra: Any) -> dict:
    """File-level Arrow metadata stamped on every store file."""
    return {
        "format": PARQUET_FORMAT,
        "format_version": PARQUET_FORMAT_VERSION,
        "kind": kind,
        **extra,
    }


def _encode_meta(meta: dict) -> dict[bytes, bytes]:
    """Encode a dict to Arrow's bytes->bytes mapping, dropping `None` values."""
    return {k.encode(): str(v).encode() for k, v in meta.items() if v is not None}


def _write_table(
    df: pd.DataFrame,
    path: Path,
    *,
    file_meta: dict | None = None,
    sorting_columns: list[str] | None = None,
) -> None:
    """Write a DataFrame as a zstd parquet file."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pa.Table.from_pandas(df, preserve_index=False)
    # Empty columns would infer as Arrow `null`, giving stores different
    # schemas. Declare them as nullable strings instead.
    fields = [
        f.with_type(pa.string()) if pa.types.is_null(f.type) else f
        for f in table.schema
    ]
    meta = {**(table.schema.metadata or {}), **_encode_meta(file_meta or {})}
    table = table.cast(pa.schema(fields, metadata=meta))

    with Path(path).open("wb") as f:
        pq.write_table(
            table,
            f,
            compression="zstd",
            write_page_index=True,
            sorting_columns=(
                [
                    pq.SortingColumn(table.schema.get_field_index(c))
                    for c in sorting_columns
                ]
                if sorting_columns
                else None
            ),
        )


def _read_table(path: Path) -> pd.DataFrame:
    """Read a parquet file written by `_write_table`."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    with Path(path).open("rb") as f:
        table = pq.read_table(f)
    # Normalize temporal columns to the nanoseconds PyPSA uses
    return table.to_pandas(coerce_temporal_nanoseconds=True)


def _jsonable(obj: Any) -> Any:
    """Encode a manifest value as strict JSON."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            # Tagged to not mix it up with the string "inf"
            return {_FLOAT_TAG: "inf" if obj > 0 else "-inf"}
    return obj


def _write_manifest(path: Path, manifest: dict) -> None:
    Path(path).joinpath(MANIFEST_NAME).write_text(
        json.dumps(_jsonable(manifest), allow_nan=False, indent=2)
    )


_STORE_DIRS = (DIMS_DIR, INPUTS_DIR, OUTPUTS_DIR)


def _clear_store(path: Path) -> None:
    """Remove a previous store's files, manifest first so it is uncommitted."""
    (path / MANIFEST_NAME).unlink(missing_ok=True)
    for sub in _STORE_DIRS:
        directory = path / sub
        if directory.is_dir():
            for fn in directory.rglob("*.parquet"):
                fn.unlink()


def _check_target(path: Path) -> None:
    """Refuse to export anywhere but a new, empty or existing store directory."""
    path = Path(path)
    if not path.exists():
        return
    if path.is_dir():
        try:
            if _read_manifest(path) is not None:
                return
        except ValueError:
            pass  # unreadable manifest
        if not any(path.iterdir()):
            return
    msg = (
        f"{path} is not a parquet store and not empty. Refusing to export into "
        "it, pick a new or empty directory."
    )
    raise FileExistsError(msg)


def _safe_name(name: str, kind: str) -> str:
    """Return `name` if it is a single, non-traversing filename component."""
    # PureWindowsPath is stricter than PurePosixPath
    if not name or name == ".." or name != PureWindowsPath(name).name:
        msg = f"Cannot export {kind} {name}, not a valid file name."
        raise ValueError(msg)
    return name


def _read_manifest(path: Path) -> dict | None:
    """Read the commit marker, returning `None` if absent."""
    fn = Path(path).joinpath(MANIFEST_NAME)
    if not fn.exists():
        return None
    hint = f"Fix it or remove the file to read {path} as a CSV folder."
    try:
        manifest = json.loads(fn.read_text())
        fmt = manifest.get("format")
    except (OSError, ValueError, AttributeError) as err:
        msg = f"{fn} is not readable as a {PARQUET_FORMAT} manifest. {hint}"
        raise ValueError(msg) from err
    if fmt != PARQUET_FORMAT:
        msg = f"{fn} has format {fmt}, not {PARQUET_FORMAT}. {hint}"
        raise ValueError(msg)
    return manifest


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _write_store(path: str | Path, n: Network) -> None:
    """Write Network to a parquet store."""
    check_optional_dependency("pyarrow", _INSTALL_MSG)
    _check_snapshots_exportable(n)
    path = Path(path)
    _check_target(path)
    path.mkdir(parents=True, exist_ok=True)
    _clear_store(path)

    multiperiod = n.has_periods

    axes = [
        (SCENARIOS_NAME, n.scenario_weightings if n.has_scenarios else None),
        (SNAPSHOTS_NAME, n.snapshot_weightings),
        (PERIODS_NAME, n.investment_period_weightings if multiperiod else None),
    ]
    for name, table in axes:
        if table is not None:
            (path / DIMS_DIR).mkdir(parents=True, exist_ok=True)
            _write_table(
                table.reset_index(),
                path / DIMS_DIR / name,
                file_meta=_file_meta("axis"),
            )

    components_present: list[str] = []
    input_frames: list[pd.DataFrame] = []  # varying inputs -> inputs/
    output_frames: list[pd.DataFrame] = []  # every output -> outputs/
    for c in n.components:
        static = _build_static_frame(n, c)
        if static is None:
            continue
        components_present.append(c.name)

        # outputs -> outputs
        # varying inputs -> inputs
        # static inputs -> stay on the wide dim table
        defaults = c.defaults
        output_cols: list[str] = []
        static_cols: list[str] = []
        for col in static.columns:
            if col in _DIM_COMPONENT_LEAD_COLS:
                continue
            if _is_output(defaults, col):
                output_cols.append(col)
            elif not (col in defaults.index and bool(defaults.at[col, "varying"])):
                static_cols.append(col)  # varying inputs go to inputs/ instead

        comp_dir = path / DIMS_DIR / COMPONENTS_DIR
        comp_dir.mkdir(parents=True, exist_ok=True)
        _write_table(
            static[[*_DIM_COMPONENT_LEAD_COLS, *static_cols]],
            comp_dir / f"{_safe_name(c.name, 'component type')}.parquet",
            file_meta=_file_meta("components", component_type=c.name),
        )
        for frame in c._export_long_frames(output_cols):
            attr = frame["attribute"].iloc[0]
            if _is_output(defaults, attr):
                output_frames.append(frame)
            else:
                input_frames.append(frame)

    snapshot_dtype = _long_snapshot_dtype(n)
    n_periods = len(n.investment_periods) if multiperiod else 0
    n_scenarios = len(n.scenarios) if n.has_scenarios else 0
    _write_long_frames(
        input_frames,
        path / INPUTS_DIR,
        "inputs",
        snapshot_dtype,
        n_periods,
        n_scenarios,
    )
    _write_long_frames(output_frames, path / OUTPUTS_DIR, "outputs", snapshot_dtype)

    manifest = {
        "format": PARQUET_FORMAT,
        "format_version": PARQUET_FORMAT_VERSION,
        "kind": "network",
        "attributes": n._collect_network_attributes(),
        "meta": n.meta,
        "crs": n.crs.to_wkt() if n.crs is not None else None,
        "attribute_catalog": _attribute_catalog(n, components_present),
    }
    _write_manifest(path, manifest)


def _dejsonable(obj: Any) -> Any:
    """Reverse `_jsonable`'s infinity encoding."""
    if isinstance(obj, dict):
        # Decode only our own tag
        if set(obj) == {_FLOAT_TAG} and obj[_FLOAT_TAG] in ("inf", "-inf"):
            return float(obj[_FLOAT_TAG])
        return {k: _dejsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dejsonable(v) for v in obj]
    return obj


def _attribute_catalog(n: Network, component_classes: list[str]) -> dict:
    """Network-independent attribute descriptions per component type."""
    catalog: dict = {}
    for cls in component_classes:
        entries = {}
        for attr, row in n.components[cls].defaults.iterrows():
            # TODO: Currently just descriptive and not used. Needs new updated manifest as ground truth
            dims = ["name", "snapshot"] if row["varying"] else ["name"]
            entries[str(attr)] = {
                "dims": dims,
                "default": _jsonable(row["default"]),
            }
        catalog[cls] = entries
    return catalog


def _build_static_frame(n: Network, c: Components) -> pd.DataFrame | None:
    """Build a flat static table for one component, or `None` if it has no rows."""
    static = n._prepare_components_export(c, export_standard_types=False)
    if static.empty:
        return None
    static = static.drop(columns=list(_DERIVED_STATIC_COLUMNS), errors="ignore")

    # SubNetwork keeps live objects in `obj`, which cannot be serialized
    if c.name == "SubNetwork" and "obj" in static:
        static["obj"] = np.nan

    # Scenario is an index level only when stochastic, always a column here
    flat = static.reset_index()
    if "scenario" not in flat:
        flat.insert(0, "scenario", None)
    # string-with-nulls so deterministic and stochastic stores share one schema
    flat["scenario"] = flat["scenario"].astype("string")

    flat.insert(0, "component_type", c.name)
    return flat


def _long_snapshot_dtype(n: Network) -> str:
    """Pandas dtype for the long `snapshot` column (nullable, so all-null files unify)."""
    snapshots = n.snapshots
    if isinstance(snapshots, pd.MultiIndex):
        snapshots = snapshots.get_level_values("timestep")
    if pd.api.types.is_datetime64_any_dtype(snapshots):
        return "datetime64[ns]"
    if pd.api.types.is_timedelta64_dtype(snapshots):
        return "timedelta64[ns]"
    if pd.api.types.is_integer_dtype(snapshots):
        return "Int64"
    if pd.api.types.is_float_dtype(snapshots):
        return "float64"
    return "string"


def _check_snapshots_exportable(n: Network) -> None:
    """Refuse object snapshots (e.g. dates) that would import as silent NaN."""
    if _long_snapshot_dtype(n) != "string":
        return
    snapshots = n.snapshots
    if isinstance(snapshots, pd.MultiIndex):
        snapshots = snapshots.get_level_values("timestep")
    bad = next((s for s in snapshots if not isinstance(s, str)), None)
    if bad is not None:
        msg = f"Parquet cannot export {snapshots.dtype} snapshots like {bad!r}."
        raise ValueError(msg)


def _collapse_axis(
    frame: pd.DataFrame, col: str, n_axis: int, keys: list[str]
) -> pd.DataFrame:
    """Replace repeated values along `col` by one row with a null `col`.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {"name": ["g", "g", "h"], "scenario": ["low", "high", "low"], "value": [1.0, 1.0, 2.0]}
    ... )
    >>> _collapse_axis(df, "scenario", 2, ["name"])
      name scenario  value
    0    g     <NA>    1.0
    2    h      low    2.0

    """
    if n_axis < 2 or frame.empty:
        return frame
    grouped = frame.groupby(keys, dropna=False, sort=False)["value"]
    constant = (grouped.transform("size") == n_axis) & (
        grouped.transform("nunique", dropna=False) == 1
    )
    if not constant.any():
        return frame
    frame = frame.copy()
    frame.loc[constant, col] = pd.NA
    return frame[~constant | ~frame.duplicated(subset=keys)]


def _write_long_frames(
    frames: list[pd.DataFrame],
    directory: Path,
    role: str,
    snapshot_dtype: str,
    n_periods: int = 0,
    n_scenarios: int = 0,
) -> None:
    """Concat long frames per attribute, cast to the shared schema, sort and write."""
    if not frames:
        return
    by_attr: dict[str, list[pd.DataFrame]] = {}
    for frame in frames:
        by_attr.setdefault(frame["attribute"].iloc[0], []).append(frame)
    for attr in by_attr:
        _safe_name(attr, "attribute")

    directory.mkdir(parents=True, exist_ok=True)
    for attr, attr_frames in by_attr.items():
        cast = []
        for frame in attr_frames:
            # Unify dtypes before concat, empty scenario/period default to NaN
            frame = frame.astype({"scenario": "string", "period": "Int64"})

            # Solver outputs essentially never repeat across periods or
            # scenarios, so the collapse is not worth it
            if role == "inputs":
                frame = _collapse_axis(
                    frame, "period", n_periods, ["name", "scenario", "snapshot"]
                )
                frame = _collapse_axis(
                    frame, "scenario", n_scenarios, ["name", "snapshot", "period"]
                )
            frame["snapshot"] = (
                pd.to_datetime(frame["snapshot"])
                if snapshot_dtype == "datetime64[ns]"
                else frame["snapshot"].astype(snapshot_dtype)
            )
            cast.append(frame[_LONG_COLUMNS])
        data = pd.concat(cast, ignore_index=True)
        # One file per attribute, so `value` is holds the dtype
        if pd.api.types.is_numeric_dtype(
            data["value"]
        ) and not pd.api.types.is_bool_dtype(data["value"]):
            data["value"] = data["value"].astype("float64")
        data = data.sort_values(
            ["component_type", "snapshot", "scenario", "name"],
            kind="stable",
            ignore_index=True,
        )

        _write_table(
            data[_LONG_COLUMNS],
            directory / f"{_safe_name(attr, 'attribute')}.parquet",
            file_meta=_file_meta(role, attribute=attr),
            sorting_columns=["component_type", "snapshot", "scenario", "name"],
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def is_network_store(path: str | Path) -> bool:
    """Return `True` if `path` holds a store."""
    try:
        p = Path(path)
        if not p.is_dir():
            return False
    except (OSError, ValueError):
        return False
    return _read_manifest(p) is not None


def _read_store(path: str | Path, n: Network) -> None:
    """Read the store and load its dimensions, inputs and outputs into `n`."""
    check_optional_dependency("pyarrow", _INSTALL_MSG)
    path = Path(path)
    manifest = _read_manifest(path)
    if manifest is None:
        msg = f"No readable parquet manifest found at {path}."
        raise FileNotFoundError(msg)

    version = manifest.get("format_version")
    if not isinstance(version, int) or version > PARQUET_FORMAT_VERSION:
        msg = (
            f"Parquet store at {path} has format_version {version!r}; "
            f"this PyPSA version reads up to {PARQUET_FORMAT_VERSION}. "
            "Upgrade PyPSA to read it."
        )
        raise ValueError(msg)

    n._apply_network_attributes(_dejsonable(manifest.get("attributes", {}) or {}))
    n.meta = _dejsonable(manifest.get("meta") or {})
    crs = manifest.get("crs")
    if crs is not None:
        n._crs = CRS.from_wkt(crs)

    # Network shape is declared by which axis files exist under dims/
    dims = path / DIMS_DIR
    has_scenarios = (dims / SCENARIOS_NAME).exists()
    multiperiod = (dims / PERIODS_NAME).exists()

    n._apply_snapshots_import(_read_table(dims / SNAPSHOTS_NAME))

    if multiperiod:
        ip_df = _read_table(dims / PERIODS_NAME).set_index("period")
        if not ip_df.empty:
            n.periods = ip_df.index
            n._investment_periods_data = ip_df.reindex(n.investment_periods)

    # set directly to avoid broadcasting already scenario shaped data
    if has_scenarios:
        scen = _read_table(dims / SCENARIOS_NAME).set_index("scenario")
        scen.index = scen.index.astype(str)
        scen.index.name = "scenario"
        n._scenarios_data = scen[["weight"]]

    comp_dir = dims / COMPONENTS_DIR
    if comp_dir.is_dir():
        _apply_static(n, comp_dir, has_scenarios)

    for sub in (INPUTS_DIR, OUTPUTS_DIR):
        directory = path / sub
        if not directory.exists():
            continue
        for fn in sorted(directory.glob("*.parquet")):
            _apply_long_file(n, fn, has_scenarios, multiperiod)

    n._broadcast_standard_types()


def _apply_static(n: Network, comp_dir: Path, has_scenarios: bool) -> None:
    """Import the per type wide static tables."""
    present = sorted(p.stem for p in comp_dir.glob("*.parquet"))
    for cls in present:
        sub = _read_table(comp_dir / f"{cls}.parquet").drop(columns="component_type")
        if has_scenarios:
            sub = sub.set_index(["scenario", "name"])
        else:
            sub = sub.drop(columns="scenario").set_index("name")
        n._import_components_from_df(sub, cls)


def _apply_long_file(
    n: Network,
    fn: Path,
    has_scenarios: bool,
    multiperiod: bool,
) -> None:
    """Read one long file and route its rows to static and/or dynamic assignment."""
    attr = fn.stem
    df = _read_table(fn)
    if df.empty:
        return
    null_rows = df[df["snapshot"].isna()]
    ts_rows = df[df["snapshot"].notna()]
    if not null_rows.empty:
        _apply_static_long(n, null_rows, attr, has_scenarios)
    if not ts_rows.empty:
        _apply_dynamic(n, ts_rows, attr, has_scenarios, multiperiod)


def _expand_axis(df: pd.DataFrame, col: str, values: pd.Index) -> pd.DataFrame:
    """Broadcast rows with a null `col` across the whole axis.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {"name": ["g", "h"], "scenario": [pd.NA, "low"], "value": [1.0, 2.0]}
    ... )
    >>> _expand_axis(df, "scenario", pd.Index(["low", "high"]))
      name scenario  value
    0    h      low    2.0
    1    g      low    1.0
    2    g     high    1.0

    """
    null = df[col].isna()
    if not null.any():
        return df
    expanded = (
        df[null]
        .drop(columns=col)
        .merge(pd.Series(values, name=col).astype(df[col].dtype), how="cross")
    )
    return pd.concat([df[~null], expanded], ignore_index=True)


def _apply_dynamic(
    n: Network,
    data_df: pd.DataFrame,
    attr: str,
    has_scenarios: bool,
    multiperiod: bool,
) -> None:
    """Pivot one attribute's long table back to wide frames and feed the importer."""
    column_keys = ["scenario", "name"] if has_scenarios else "name"
    index_keys = ["period", "snapshot"] if multiperiod else "snapshot"

    if multiperiod:
        data_df = _expand_axis(data_df, "period", n.investment_periods)
    if has_scenarios:
        data_df = _expand_axis(data_df, "scenario", n.scenarios)

    for ctype, group in data_df.groupby("component_type", sort=False):
        if ctype not in n.all_components:
            continue
        wide = group.pivot(index=index_keys, columns=column_keys, values="value")
        # Restore static member order on the dynamic columns
        # TODO: handle this better when importing old networks
        static_index = n.components[ctype].static.index
        col_order = static_index[static_index.isin(wide.columns)]
        wide = wide.reindex(n.snapshots)[col_order]
        n._import_series_from_df(wide, ctype, attr)


def _apply_static_long(
    n: Network, df: pd.DataFrame, attr: str, has_scenarios: bool
) -> None:
    """Assign null snapshot rows in place as a static column, leaving other rows at their defaults."""
    index_keys = ["scenario", "name"] if has_scenarios else "name"
    if has_scenarios:
        df = _expand_axis(df, "scenario", n.scenarios)
    for ctype, group in df.groupby("component_type", sort=False):
        if ctype not in n.all_components:
            continue
        static = n.components[ctype].static
        if attr not in static.columns:
            continue
        values = group.set_index(index_keys)["value"]
        values = values[values.index.isin(static.index)]
        static.loc[values.index, attr] = values
