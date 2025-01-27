from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Annotated, Any

import pandas as pd
from deprecation import deprecated
from pandera import Column, DataFrameSchema, Index
from pydantic import BaseModel, ConfigDict, Field

from pypsa.common import format_str_dtype
from pypsa.definitions.structures import Dict

logger = logging.getLogger(__name__)


def to_list_name(name: str) -> str:
    """
    Convert a component type name to its list name.

    It also works for custom component types. List names should not be defined
    manually.

    Parameters
    ----------
    name : str
        Component type name.

    Returns
    -------
    str
        Component type list name.

    Examples
    --------
    >>> to_list_name("Generator")
    'generators'
    >>> to_list_name("StorageUnit")
    'storage_units'

    """
    # Singular to Plural
    if re.search("[sxz]$", name) or re.search("[^aeioudgkprt]h$", name):
        name = re.sub("$", "es", name)
    elif re.search("[aeiou]y$", name):
        name = re.sub("y$", "ies", name)
    else:
        name = name + "s"

    # CamelCase to snake_case
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)

    return name.lower()


def schema_input_static(data: dict) -> DataFrameSchema:
    schema = {}

    for name, row in data["defaults"][data["defaults"].status != "output"].iterrows():
        if name == "name":
            continue

        schema[name] = Column(
            row.type, default=format_str_dtype(row.default), nullable=row.nullable
        )

    return DataFrameSchema(
        schema,
        index=Index(str, name=data["name"]),
        add_missing_columns=True,
        # strict=True,
        coerce=True,
    )


def schema_output_static(data: dict) -> DataFrameSchema:
    return DataFrameSchema(
        {
            name: Column(
                row.type, default=format_str_dtype(row.default), nullable=row.nullable
            )
            for name, row in data["defaults"][
                (data["defaults"].status == "output")
                & (data["defaults"].dynamic == False)
            ]
            .drop(index=["name"], errors="ignore")
            .iterrows()
        },
        index=Index(str, name=data["name"]),
        add_missing_columns=True,
        # strict=True,
        coerce=True,
    )


def schemas_input_dynamic(data: dict) -> Dict:
    attrs = data["defaults"]
    attrs = attrs[attrs.dynamic & (attrs.status != "output")]

    return Dict(
        {
            name: DataFrameSchema(
                columns={
                    ".*": Column(
                        row.type,
                        default=format_str_dtype(row.default),
                        nullable=row.nullable,
                        regex=True,
                    )
                },
                # add_missing_columns=True,
                # strict=True,
                coerce=True,
            )
            for name, row in attrs.iterrows()
        }
    )


def schemas_output_dynamic(data: dict) -> Dict:
    attrs = data["defaults"]
    attrs = attrs[attrs.dynamic & (attrs.status == "output")]

    return Dict(
        {
            name: DataFrameSchema(
                columns={
                    ".*": Column(
                        row.type,
                        default=format_str_dtype(row.default),
                        nullable=row.nullable,
                        regex=True,
                    )
                },
                # add_missing_columns=True,
                # strict=True,
                coerce=True,
            )
            for name, row in attrs.iterrows()
        }
    )


class ComponentTypeEnum(Enum):
    # TODO Think about capitalization, since those are constants, but adds a third
    # representation
    SubNetwork = "SubNetwork"
    Bus = "Bus"
    Carrier = "Carrier"
    GlobalConstraint = "GlobalConstraint"
    Line = "Line"
    LineType = "LineType"
    Transformer = "Transformer"
    TransformerType = "TransformerType"
    Link = "Link"
    Load = "Load"
    Generator = "Generator"
    StorageUnit = "StorageUnit"
    Store = "Store"
    ShuntImpedance = "ShuntImpedance"
    Shape = "Shape"


class ComponentType(BaseModel):
    """
    Dataclass for network component type.

    Contains all information about a component type, such as its name and defaults
    attributes. Two different types are for example 'Generator' and 'Carrier'.

    Attributes
    ----------
    name : str
        Name of component type, e.g. 'Generator'.
    list_name : str
        Name of component type in list form, e.g. 'generators'.
    description : str
        Description of the component type.
    category : str
        Category of the component type, e.g. 'passive_branch'.
    defaults : pd.DataFrame
        Default values for the component type.
    standard_types : pd.DataFrame | None
        Standard types for the component type.

    """

    # Pydantic config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Immutable attributes
    name: Annotated[str, Field(frozen=True)]
    list_name: Annotated[
        str, Field(frozen=True, default_factory=lambda data: to_list_name(data["name"]))
    ]
    description: Annotated[
        str,
        Field(
            frozen=True,
            default_factory=lambda data: _get_component_data(
                data["name"], "description"
            ),
        ),
    ]
    category: Annotated[
        str,
        Field(
            frozen=True,
            default_factory=lambda data: _get_component_data(data["name"], "category"),
        ),
    ]
    defaults: Annotated[
        pd.DataFrame,
        Field(
            frozen=True,
            default_factory=lambda data: _get_component_data(data["name"], "defaults"),
        ),
    ]
    standard_types: Annotated[
        pd.DataFrame | None,
        Field(
            frozen=True,
            default_factory=lambda data: _get_component_data(
                data["name"], "standard_types"
            ),
        ),
    ]

    # Validation schemas
    schema_static: Annotated[
        DataFrameSchema, Field(default_factory=schema_input_static)
    ]
    schemas_input_dynamic: Annotated[Dict, Field(default_factory=schemas_input_dynamic)]
    schema_output_static: Annotated[
        DataFrameSchema, Field(default_factory=schema_output_static)
    ]
    schemas_output_dynamic: Annotated[
        Dict, Field(default_factory=schemas_output_dynamic)
    ]

    # Mutable attributes
    custom_attrs: Annotated[
        pd.DataFrame,
        Field(default_factory=lambda data: data["defaults"].head(0)),
    ]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ComponentType):
            return NotImplemented

        return (
            self.name == other.name
            and self.list_name == other.list_name
            and self.description == other.description
            and str(self.category) == str(other.category)
            and self.defaults.equals(other.defaults)
        )

    def __repr__(self) -> str:
        return f"'{self.name}' Component Type"

    # @property  # TODO: Caching
    # def schema_static(self) -> DataFrameSchema:
    #     """
    #     Pydantic schema for component type.

    #     Returns
    #     -------
    #     DataFrameSchema
    #         Pydantic schema for component type.

    #     """
    #     # TODO
    #     # if static_attrs.at[k, "type"] == "string":
    #     #     df[k] = df[k].replace({np.nan: ""})
    #     # if static_attrs.at[k, "type"] == "int":
    #     #     df[k] = df[k].fillna(0)
    #     #         geometry = df[k].replace({"": None, np.nan: None})
    #     # from shapely.geometry.base import BaseGeometry

    #     # if geometry.apply(lambda x: isinstance(x, BaseGeometry)).all():
    #     #     df[k] = gpd.GeoSeries(geometry)
    #     # else:
    #     #     df[k] = gpd.GeoSeries.from_wkt(geometry)

    @property  # TODO: Caching
    def static_attrs(self) -> set:
        return set(self.defaults[~self.defaults.dynamic].index)

    @property  # TODO: Caching
    def dynamic_attrs(self) -> set:
        return set(self.defaults[self.defaults.dynamic].index)

    @property  # TODO: Caching
    def required_attrs(self) -> set:
        return set(self.defaults[self.defaults.requirement == "required"].index)

    @property  # TODO: Caching
    def configurable_attrs(self) -> set:
        return set(self.defaults[self.defaults.requriement == "configurable"].index)

    @property  # TODO: Caching
    def optional_attrs(self) -> set:
        return set(self.defaults[self.defaults.requriement == "optional"].index)

    @property
    @deprecated(
        deprecated_in="0.32.0",
        details="Use the 'category' attribute instead.",
    )
    def type(self) -> str:
        return self.category

    @property
    @deprecated(
        deprecated_in="0.32.0",
        details="Use the 'defaults' attribute instead.",
    )
    def attrs(self) -> pd.DataFrame:
        return self.defaults


def _get_component_data(component: str, key: str) -> pd.DataFrame | str | None:
    try:
        cname = ComponentTypeEnum(component).value
    except ValueError:
        msg = (
            f"`{component}` is not a default component type. If you want to use a "
            "custom component type, just provide data for all attributes."
        )
        raise ValueError(msg)

    if key == "defaults":
        return pd.read_csv(f"pypsa/data/component_attrs/{cname}.csv", index_col=0)
    elif key == "standard_types":
        try:
            return pd.read_csv(f"pypsa/data/standard_types/{cname}.csv", index_col=0)
        except FileNotFoundError:
            return None
    elif key == "description":
        return str(
            pd.read_csv("pypsa/data/components.csv", index_col=0).loc[
                cname, "description"
            ]
        )
    elif key == "category":
        return str(
            pd.read_csv("pypsa/data/components.csv", index_col=0).loc[cname, "category"]
        )
    else:
        msg = f"`{key}` is not a valid key."
        raise ValueError(msg)
