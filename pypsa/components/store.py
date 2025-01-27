from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.constants import DEFAULT_TIMESTAMP
from pypsa.deprecations import COMPONENT_ALIAS_DICT

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ComponentsStore(dict):
    def __repr__(self) -> str:
        return "Components Store\n================\n- " + "\n- ".join(
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
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.components
        Components Store
        ================
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
        'Generator' Components
        ======================
        Attached to PyPSA Network 'AC-DC'
        Components: 6

        """
        if isinstance(item, (list | set)):
            return [self[key] for key in item]
        else:
            pass
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
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.components.generators
        'Generator' Components
        ======================
        Attached to PyPSA Network 'AC-DC'
        Components: 6

        """
        try:
            return self[item]
        except KeyError:
            msg = f"Network has no components '{item}'"
            raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Is invoked when del object.member is called."""
        del self[name]

    _re_pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

    def __dir__(self) -> list[str]:
        dict_keys = [
            k for k in self.keys() if isinstance(k, str) and self._re_pattern.match(k)
        ]
        obj_attrs = list(dir(super()))
        return dict_keys + obj_attrs

    def __iter__(self) -> Any:
        """Value iterator over components in store."""
        return iter(self.values())


# class StaticAttrsDataFrame(pd.DataFrame):
#     def __getattribute__(self, name):
#         logger.info(f"Accessing attribute: {name}")
#         return super().__getattribute__(name)

#     def __getitem__(self, key):
#         logger.info(f"Accessing item: {key}")
#         return super().__getitem__(key)

#     def __setitem__(self, key, value):
#         logger.info(f"Setting item: {key} = {value}")
#         return super().__setitem__(key, value)


class DynamicAttrsDict(dict):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize dict-like DynamicAttrsDict object.

        Behaves like a common Python dictionary but also sets hidden meta attributes.

        Examples
        --------
        Normal dictionary behavior:
        >>> some_dict = DynamicAttrsDict({'a': 1})
        >>> some_dict['a']
        1

        """
        self._default_snapshots = pd.Index([DEFAULT_TIMESTAMP], name="snapshot")
        # TODO snapshots can be moved to c once it handles snapshots without being
        # attached to a network
        super().__init__(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """Setattr is called when the syntax a.b = 2 is used to set a value."""
        if hasattr(DynamicAttrsDict, name):
            raise AttributeError(f"'Dict' object attribute '{name}' is read-only")
        self[name] = value

    def __getitem__(self, name: str) -> Any:
        """Getitem is called when the syntax a['b'] is used to get a value."""
        try:
            return super().__getitem__(name)
        except KeyError as e:
            if not name.startswith("_"):
                df = pd.DataFrame(
                    index=self._default_snapshots, columns=[], dtype=float
                )
                df.index.name = "snapshot"
                df.columns.name = self._pypsa_component.name

                # logger.warning(
                #     f"'{name}' for {self._pypsa_componentname} not dynamically set. Find attribute "
                #     f"in static DataFrame instead. Returning empty dynamic DataFrame "
                #     "instead."
                # )
                return df
            else:
                raise e

    def __getattr__(self, name: str) -> Any:
        """Getattr is called when the syntax a.b is used to get a value."""
        return self.__getitem__(name)

    def __delattr__(self, name: str) -> None:
        """Is invoked when del some_addict.b is called."""
        del self[name]

    def __iter__(self) -> Any:
        """Value iterator over components in store."""
        return iter(self.keys())

    def __repr__(self) -> str:
        """Return a string representation of the dict."""
        title = f"'{self._pypsa_component.name}' Dynamic Attributes Store"
        title += "\n" + "=" * len(title)
        return f"{title}\n" + "\n".join(
            f"{k}\n" + "-" * len(k) + f"\n{v}" for k, v in self.items()
        )

    def __dir__(self):  # type: ignore
        """Return a list of object attributes."""
        return list(self.keys()) + list(dir(super()))

    def __deepcopy__(self, memo):
        """Handle deep copying of the dictionary."""
        new_dict = DynamicAttrsDict()
        memo[id(self)] = new_dict

        # Copy the private attributes
        new_dict._pypsa_component = deepcopy(self._pypsa_component, memo)
        new_dict._default_snapshots = deepcopy(self._default_snapshots, memo)

        # Copy all other items
        for key, value in self.items():
            new_dict[key] = deepcopy(value, memo)

        return new_dict

    def keys(self) -> list[str]:  # type: ignore # TODO: Maybe switch typing
        """Return a list of keys in the dict."""
        return list(super().keys() - {"_pypsa_component", "_default_snapshots"})

    def values(self) -> list[Any]:  # type: ignore
        """Return a list of values in the dict."""
        return [v for k, v in self.items()]

    def items(self) -> list[tuple[str, Any]]:  # type: ignore
        """Return a list of items in the dict."""
        return [
            (k, v)
            for k, v in super().items()
            if k not in ["_pypsa_component", "_default_snapshots"]
        ]
