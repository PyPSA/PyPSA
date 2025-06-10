"""Components types module.

Contains module wide component types. Default types are loaded from the package data.
Additional types can be added by the user.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pypsa.common import list_as_string
from pypsa.definitions.components import ComponentType
from pypsa.deprecations import COMPONENT_ALIAS_DICT

# TODO better path handeling, integrate custom components
_components_path = Path(__file__).parent.parent / "data" / "components.csv"
_attrs_path = Path(__file__).parent.parent / "data" / "component_attrs"
_standard_types_path = Path(__file__).parent.parent / "data" / "standard_types"

component_types_df = pd.read_csv(_components_path, index_col=0)
default_components = component_types_df.index.to_list()

all_components = {}


def add_component_type(
    name: str,
    list_name: str,
    description: str,
    category: str,
    defaults_df: pd.DataFrame,
    standard_types_df: pd.DataFrame | None = None,
) -> None:
    """Add component type to package wide component types library.

    The function is used to add the package default components but can also be used to
    add custom components, which then again can be used during the network creation.

    Parameters
    ----------
    name : str
        Name of the component type. Must be unique.
    list_name : str
        List name of the component type.
    description : str
        Description of the component type.
    category : str
        Category of the component type.
    defaults_df : pandas.DataFrame
        Default attributes of the component type. Pass as a DataFrame with the same
        structure as the default components in `/pypsa/data/default_components/`.
    standard_types_df : pandas.DataFrame, optional
        Standard types of the component type.


    Examples
    --------
    >>> import pandas as pd

    >>> defaults_data = {
    ...     "attribute": ["name", "attribute_a"],
    ...     "type": ["string", "float"],
    ...     "unit": ["n/a", "n/a"],
    ...     "default": ["n/a", 1],
    ...     "description": ["Unique name", "Some custom attribute"],
    ...     "status": ["Input (required)", "Input (optional)"]
    ... }
    >>> defaults_df = pd.DataFrame(defaults_data)
    >>> pypsa.components.types.add_component_type(
    ...     name="CustomComponent",
    ...     list_name="custom_components",
    ...     description="A custom component example",
    ...     category="custom",
    ...     defaults_df=defaults_df,
    ... )
    >>> # Check created component type
    >>> pypsa.components.types.get("custom_components")
    'CustomComponent' Component Type

    """
    if name in all_components:
        msg = f"Component type '{name}' already exists."
        raise ValueError(msg)

    # Format attributes
    defaults_df["default"] = defaults_df.default.astype(object)
    defaults_df["static"] = defaults_df["type"] != "series"
    defaults_df["varying"] = defaults_df["type"].isin({"series", "static or series"})
    defaults_df["typ"] = (
        defaults_df["type"]
        .map({"boolean": bool, "int": int, "string": str, "geometry": "geometry"})
        .fillna(float)
    )
    defaults_df["dtype"] = (
        defaults_df["type"]
        .map(
            {
                "boolean": np.dtype(bool),
                "int": np.dtype(int),
                "string": np.dtype("O"),
            }
        )
        .fillna(np.dtype(float))
    )

    bool_b = defaults_df.type == "boolean"
    if bool_b.any():
        defaults_df.loc[bool_b, "default"] = defaults_df.loc[bool_b, "default"].isin(
            {True, "True"}
        )

    str_b = defaults_df.typ.apply(lambda x: x is str)
    defaults_df.loc[str_b, "default"] = defaults_df.loc[str_b, "default"].fillna("")
    for typ in (str, float, int):
        typ_b = defaults_df.typ == typ
        defaults_df.loc[typ_b, "default"] = defaults_df.loc[typ_b, "default"].astype(
            typ
        )

    # Initialize Component
    all_components[list_name] = ComponentType(
        name=name,
        list_name=list_name,
        description=description,
        category=category,
        defaults=defaults_df,
        standard_types=standard_types_df,
    )


def _load_default_component_types(
    component_df: pd.DataFrame, attrs_path: Path, standard_types_path: Path
) -> None:
    """Load default component types from package data.

    Function is called during package import and should not be used otherwise.

    Parameters
    ----------
    component_df : pandas.DataFrame
        DataFrame which lists all default components. E.g. `/pypsa/data/components.csv`.
    attrs_path : pathlib.Path
        Path to the default attributes dir. E.g. `/pypsa/data/default_components/`.
    standard_types_path : pathlib.Path
        Path to the standard types dir. E.g. `/pypsa/data/standard_types/`.

    """
    for c_name, row in component_df.iterrows():
        # Read in defaults attributes
        attrs_file_path = attrs_path / f"{row.list_name}.csv"
        if not attrs_file_path.exists():
            msg = (
                f"Could not find {attrs_path}. For each component, there must be a "
                "corresponding file for its attributes."
            )
            raise FileNotFoundError(msg)
        attrs = pd.read_csv(attrs_file_path, index_col=0, na_values="n/a")

        # Read in standard types
        types_paths = standard_types_path / f"{row.list_name}.csv"
        if not types_paths.exists():
            standard_types = None
        else:
            standard_types = pd.read_csv(types_paths, index_col=0)

        add_component_type(
            name=c_name,
            list_name=row.list_name,
            description=row.description,
            category=row.category,
            defaults_df=attrs,
            standard_types_df=standard_types,
        )


def get(name: str) -> ComponentType:
    """Get component type instance from package wide component types library.

    The function is used to get the package default components but can also be used to
    get custom components. During network creation, the type instance is not needed but
    to pass a component type name as a string to the network constructor, a custom
    component must be added to the package wide component types library first.

    Parameters
    ----------
    name : str
        Name of the component type.

    Returns
    -------
    pypsa.components.types.ComponentType
        Component type instance.

    Examples
    --------
    >>> pypsa.components.types.get("Generator")
    'Generator' Component Type

    """
    if name in COMPONENT_ALIAS_DICT:
        name = COMPONENT_ALIAS_DICT[name]
    try:
        return all_components[name]
    except KeyError as e:
        msg = (
            f"Component type '{name}' not found. If you use a custom component, make "
            f"sure to have it added. Available types are: "
            f"{list_as_string(all_components)}."
        )
        raise ValueError(msg) from e


# Load default component types
_load_default_component_types(
    component_df=component_types_df,
    attrs_path=_attrs_path,
    standard_types_path=_standard_types_path,
)
