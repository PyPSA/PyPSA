"""Generators components module."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

from pypsa.components.components import Components
from pypsa.components.types import get as get_component_type

if TYPE_CHECKING:
    from collections.abc import Callable


def create_docstring_parameters(component_name: str) -> str:
    """Create docstring parameters section for all attributes of component."""
    # Create the docstring parameters section
    docstring = "\n\nAdditional Parameters\n----------\n"
    ct = get_component_type(component_name)
    for attribute, row in ct.attrs.iterrows():
        attr_type = row["type"]
        description = row["description"]

        # Map the type to Python types
        if attr_type == "string":
            py_type = "str"
        elif attr_type == "float":
            py_type = "float"
        elif attr_type == "boolean":
            py_type = "bool"
        elif attr_type == "int":
            py_type = "int"
        elif "static or series" in attr_type:
            py_type = "float or pandas.Series"
        elif attr_type == "series":
            py_type = "pandas.Series"
        else:
            py_type = attr_type

        # Format the parameter line
        docstring += f"{attribute} : {py_type}\n"
        docstring += f"    {description}\n"

    return docstring


def patch_docstrings(cls: Components) -> Components:
    original_add = cls.add
    original_doc = Components.add.__doc__ or ""

    @wraps(original_add)
    def add(cls: Any, *args: Any, **kwargs: Any) -> Callable:
        return original_add(cls, *args, **kwargs)

    to_be_replaced = (
        "kwargs : Any\n"
        "    Component attributes, e.g. x=[0.1, 0.2], can be list, pandas.Series\n"
        "    of pandas.DataFrame for time-varying"
    )

    component_parameters_string = create_docstring_parameters(cls.__name__.lower())
    add.__doc__ = original_doc.replace(to_be_replaced, component_parameters_string)
    cls.add = add
    return cls
