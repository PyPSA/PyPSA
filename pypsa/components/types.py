# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Components types module.

Loads default component types from package data CSVs at import time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pypsa.common import list_as_string
from pypsa.definitions.components import ComponentType

_DATA = Path(__file__).parent.parent / "data"
_COMPONENTS_PATH = _DATA / "components.csv"
_ATTRS_PATH = _DATA / "component_attrs"
_STANDARD_TYPES_PATH = _DATA / "standard_types"

_BOOL_MAP = {
    "True": True,
    "true": True,
    True: True,
    "False": False,
    "false": False,
    False: False,
}


def _process_defaults(defaults_df: pd.DataFrame) -> pd.DataFrame:
    """Process a defaults DataFrame from CSV.

    Derives ``static``/``varying`` flags from the ``type`` column, coerces
    ``dynamic``/``nullable`` to ``bool``, and casts ``default`` values to
    their declared dtypes.
    """
    df = defaults_df.copy()
    df["default"] = df["default"].astype(object)

    df["static"] = df["type"] != "series"
    df["varying"] = df["type"].isin({"series", "static or series"})
    df["dynamic"] = df["dynamic"].map(_BOOL_MAP).astype(bool)
    df["nullable"] = df["nullable"].map(_BOOL_MAP).astype(bool)

    bool_b = df["dtype"] == "bool"
    if bool_b.any():
        df.loc[bool_b, "default"] = df.loc[bool_b, "default"].isin({True, "True"})

    str_b = df["dtype"] == "str"
    df.loc[str_b, "default"] = df.loc[str_b, "default"].fillna("")
    for dtype_name, pytype in (("str", str), ("float", float), ("int", int)):
        mask = (df["dtype"] == dtype_name) & df["default"].notna()
        if mask.any():
            df.loc[mask, "default"] = df.loc[mask, "default"].astype(pytype)
    return df


def _load_standard_types(name: str) -> pd.DataFrame | None:
    path = _STANDARD_TYPES_PATH / f"{name}.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


component_types_df = pd.read_csv(_COMPONENTS_PATH, index_col=0)
default_components = component_types_df.index.to_list()

all_components: dict[str, ComponentType] = {}
for _name, _row in component_types_df.iterrows():
    _attrs_file = _ATTRS_PATH / f"{_row.list_name}.csv"
    if not _attrs_file.exists():
        msg = (
            f"Could not find {_attrs_file}. For each component, there must be "
            "a corresponding file for its attributes."
        )
        raise FileNotFoundError(msg)
    all_components[_row.list_name] = ComponentType(
        name=_name,
        list_name=_row.list_name,
        description=_row.description,
        category=_row.category if pd.notna(_row.category) else "",
        defaults=_process_defaults(pd.read_csv(_attrs_file, index_col=0)),
        standard_types=_load_standard_types(_row.list_name),
    )

all_standard_attrs_set = {
    attr for ct in all_components.values() for attr in ct.defaults.index
}

_NAME_TO_LIST_NAME = dict(
    zip(component_types_df.index, component_types_df.list_name, strict=True)
)


def get(name: str) -> ComponentType:
    """Get component type by name (PascalCase) or list_name.

    Examples
    --------
    >>> pypsa.components.types.get("Generator")
    'Generator' Component Type

    """
    key = _NAME_TO_LIST_NAME.get(name, name) if name not in all_components else name
    try:
        return all_components[key]
    except KeyError as e:
        msg = (
            f"Component type '{name}' not found. Available types: "
            f"{list_as_string(all_components)}."
        )
        raise ValueError(msg) from e
