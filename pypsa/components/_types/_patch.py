# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Generators components module."""

from __future__ import annotations

import re
from functools import wraps
from typing import Any, TypeVar

from pypsa.components.components import Components
from pypsa.components.types import get as get_component_type

_T = TypeVar("_T", bound=Components)


def _format_str_dtype(value: Any) -> Any:
    """Convert string representations from CSV cells to properly typed Python values.

    Handles booleans, numeric types, and special float values (inf, nan).
    Non-string values are returned as-is.
    """
    if not isinstance(value, str):
        return value
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    if value in ("inf", "Inf"):
        return float("inf")
    if value in ("-inf", "-Inf"):
        return float("-inf")
    if value in ("nan", "NaN", ""):
        return float("nan")
    try:
        int_val = int(value)
        if str(int_val) == value:
            return int_val
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


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
        "    Component attributes to add. See Other Parameters for list of default"
        "    attributes but any attribute could be added.\n"
        "\n"
        "Other Parameters\n"
        "---------------------\n"
    )
    ct = get_component_type(component_name)
    for attribute, row in ct.defaults.iterrows():
        description = row["description"]

        if attribute == "name":
            continue

        dtype = row["dtype"]
        default_arg = f"'{row['default']}'" if dtype == "str" else row["default"]

        if row["dynamic"]:
            param_header = (
                f"{attribute} : {dtype} or pandas.Series, default={default_arg}\n"
            )
        else:
            param_header = f"{attribute} : {dtype}, default={default_arg}\n"

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
