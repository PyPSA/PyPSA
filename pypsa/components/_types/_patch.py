"""Generators components module."""

from __future__ import annotations

import re
from functools import wraps
from typing import Any, TypeVar

from pypsa.components.components import Components
from pypsa.components.types import get as get_component_type

_T = TypeVar("_T", bound=Components)


def create_docstring_parameters(component_name: str) -> str:
    """Create docstring parameters section for all attributes of component.

    Parameters
    ----------
    component_name : str
        The name of the component.

    Returns
    -------
    str
        The docstring parameters section.

    """
    # Create the docstring parameters section

    docstring = (
        "kwargs : Any\n"
        "    Component attributes to add. See Additinal Parameters for list of default"
        "    attributes but any attribute could be added.\n"
        "\n"
        "Other Parameters\n"
        "---------------------\n"
    )
    ct = get_component_type(component_name)
    for attribute, row in ct.defaults.iterrows():
        attr_type = row["type"]
        description = row["description"]

        # Skip name attribute since it is not a additional parameter
        if attribute == "name":
            continue

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
        else:
            py_type = attr_type
        if attr_type == "series":
            py_type = "pandas.Series"
            is_dynamic = True
        else:
            is_dynamic = False

        param_header = f"{attribute} : {py_type} or SeriesLike[{py_type}]"
        if is_dynamic:
            param_header += f" or ArrayLike[{py_type}]"
        default_arg = row["default"]
        if attr_type == "string":
            default_arg = f"'{default_arg}'"
        param_header += f", default={default_arg}\n"

        # Format the parameter line
        docstring += param_header
        docstring += f"    {description}\n"

    return docstring


def patch_add_docstring(cls: type[_T]) -> type[_T]:
    original_add = cls.add
    original_doc = Components.add.__doc__ or ""

    @wraps(original_add)
    def add(self: _T, *args: Any, **kwargs: Any) -> Any:
        return original_add(self, *args, **kwargs)

    to_be_replaced = (
        "kwargs : Any\n"
        "    Component attributes, e.g. x=[0.1, 0.2], can be list, pandas.Series\n"
        "    of pandas.DataFrame for time-varying\n"
    )

    def _camel_to_snake(name: str) -> str:
        return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()

    component_name = _camel_to_snake(cls.__name__)

    component_parameters_string = create_docstring_parameters(component_name)
    add.__doc__ = original_doc.replace(to_be_replaced, component_parameters_string)
    add.__doc__ = add.__doc__.replace(
        "Add new components.", f"Add new {component_name}."
    )
    cls.add = add  # type: ignore
    return cls
