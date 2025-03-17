from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd
from deprecation import deprecated

from pypsa._options import options
from pypsa.deprecations import COMPONENT_ALIAS_DICT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentType:
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

    name: str
    list_name: str
    description: str
    category: str
    defaults: pd.DataFrame
    standard_types: pd.DataFrame | None = None

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


class ComponentsStore(dict):
    def __repr__(self) -> str:
        return "PyPSA Components Store\n======================\n- " + "\n- ".join(
            str(value) for value in self.values()
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(ComponentsStore, name):
            msg = f"'ComponentsStore' object attribute '{name}' can not be set."
            raise AttributeError(msg)
        self[name] = value

    def __getitem__(self, item: str | list | set) -> Any:
        """
        Index single and multiple items from the dictionary.

        Similar behavior to pandas.DataFrame.__getitem__.

        Examples
        --------
        >>> n.components
        PyPSA Components Store
        ======================
        - 0 'SubNetwork' Components
        - 9 'Bus' Components
        - 3 'Carrier' Components
        - 1 'GlobalConstraint' Components
        - 7 'Line' Components
        - 36 'LineType' Components
        - 0 'Transformer' Components
        - 14 'TransformerType' Components
        - 4 'Link' Components
        - 6 'Load' Components
        - 6 'Generator' Components
        - 0 'StorageUnit' Components
        - 0 'Store' Components
        - 0 'ShuntImpedance' Components
        - 0 'Shape' Components
        >>> n.components["generators"]
        PyPSA 'Generator' Components
        ----------------------------
        Attached to PyPSA Network 'AC-DC'
        Components: 6
        """
        if isinstance(item, (list | set)):
            return [self[key] for key in item]
        else:
            if item in COMPONENT_ALIAS_DICT:
                # TODO: Activate when changing logic
                # warnings.warn(
                #     f"Accessing components in n.components using capitalized singular "
                #     f"name is deprecated. Use lowercase list name instead: "
                #     f"'{COMPONENT_ALIAS_DICT[item]}' instead of '{item}'.",
                #     DeprecationWarning,
                #     stacklevel=2,
                # )
                return super().__getitem__(COMPONENT_ALIAS_DICT[item])
            return super().__getitem__(item)

    def __getattr__(self, item: str) -> Any:
        """
        Get attribute from the dictionary.

        Examples
        --------
        >>> n.components.generators
        PyPSA 'Generator' Components
        ----------------------------
        Attached to PyPSA Network 'AC-DC'
        Components: 6
        """
        try:
            return self[item]
        except KeyError:
            msg = f"Network has no components '{item}'"
            raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """
        Is invoked when del object.member is called.
        """
        del self[name]

    _re_pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

    def __dir__(self) -> list[str]:
        """
        Return a list of object attributes including dynamic ones from the dictionary keys.
        """
        dict_keys = [
            k for k in self.keys() if isinstance(k, str) and self._re_pattern.match(k)
        ]
        obj_attrs = list(dir(super()))
        return dict_keys + obj_attrs

    def __iter__(self) -> Any:
        """
        Value iterator over components in store.
        """
        if options.get_option("warnings.components_store_iter"):
            warnings.warn(
                "Iterating over `n.components` yields the values instead of keys from "
                "v0.33.0. This behavior might be breaking. Use `n.components.keys()` "
                "to iterate over the keys. To suppress this warning set "
                "`pypsa.options.warnings.components_store_iter = False`.",
                DeprecationWarning,
                stacklevel=2,
            )
        return iter(self.values())

    def __contains__(self, item: Any) -> bool:
        """
        Check if component is in store.
        """
        msg = (
            "Checking if a component is in `n.components` using the 'in' operator "
            "is deprecated. Use `item in n.components.keys()` to retain the old "
            "behavior. But with v0.33.0 custom components are deprecated and "
            "therefore keys in `n.components` never change. Check the release "
            "notes for more information."
        )
        raise DeprecationWarning(msg)
