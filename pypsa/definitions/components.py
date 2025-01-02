from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from deprecation import deprecated

from pypsa.deprecations import COMPONENT_ALIAS_DICT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentTypeInfo:
    name: str
    list_name: str
    description: str
    category: str
    defaults: pd.DataFrame
    standard_types: pd.DataFrame | None = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ComponentTypeInfo):
            return NotImplemented

        return (
            self.name == other.name
            and self.list_name == other.list_name
            and self.description == other.description
            and str(self.category) == str(other.category)
            and self.defaults.equals(other.defaults)
        )

    def __repr__(self) -> str:
        return self.name + " Component Type"

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
        return "PyPSA Components Store \n====================== \n- " + "\n- ".join(
            str(value) for value in self.values()
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(ComponentsStore, name):
            msg = "'ComponentsStore' object attribute " f"'{name}' can not be set."
            raise AttributeError(msg)
        self[name] = value

    def __getitem__(self, item: str | list | set) -> Any:
        """
        Index single and multiple items from the dictionary.

        Similar behavior to pandas.DataFrame.__getitem__.

        Examples
        --------
        >>> components = ComponentsStore()
        >>> components["generator"] = Generators()
        >>> components["stores"] = Stores()
        >>> components["generator"]
        'Generator class instance'
        >>> components[["generator", "stores"]]
        >>> components[["generator", "stores"]]
        ['Generator class instance', 'Stores class instance']
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
        >>> components = ComponentsStore()
        >>> components["generator"] = Generators()
        >>> components.generators
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
        return iter(self.values())
