"""
Descriptors for component attributes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class Dict(dict):
    """
    Dict is a subclass of dict, which allows you to get AND SET items in the
    dict using the attribute syntax!

    Stripped down from addict https://github.com/mewwts/addict/ .
    """

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Setattr is called when the syntax a.b = 2 is used to set a value.
        """
        if hasattr(Dict, name):
            raise AttributeError("'Dict' object attribute " f"'{name}' is read-only")
        self[name] = value

    def __getattr__(self, item: str) -> Any:
        try:
            return self.__getitem__(item)
        except KeyError as e:
            raise AttributeError(e.args[0])

    def __delattr__(self, name: str) -> None:
        """
        Is invoked when del some_addict.b is called.
        """
        del self[name]

    _re_pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

    def __dir__(self) -> list[str]:
        """
        Return a list of object attributes.

        This includes key names of any dict entries, filtered to the
        subset of valid attribute names (e.g. alphanumeric strings
        beginning with a letter or underscore).  Also includes
        attributes of parent dict class.
        """
        dict_keys = []
        for k in self.keys():
            if isinstance(k, str):
                if m := self._re_pattern.match(k):
                    dict_keys.append(m.string)

        obj_attrs = list(dir(Dict))

        return dict_keys + obj_attrs


class ComponentsStore(dict):
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Setattr is called when the syntax a.b = 2 is used to set a value.
        """
        if hasattr(ComponentsStore, name):
            raise AttributeError("'Dict' object attribute " f"'{name}' is read-only")
        self[name] = value

    def __getattr__(self, item: str) -> Any:
        """
        Getattr allows attribute-style access and raises AttributeError if not found.
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'"
            )

    def __getitem__(self, item: str) -> Any:
        """
        Modified __getitem__ to handle single keys or lists of keys.
        """
        if isinstance(item, (list, set)):
            return [self[key] for key in item]
        else:
            return super().__getitem__(item)

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
        Make class iterable over the keys.
        """
        for key, value in super().items():
            if key != "Network":  # TODO Drop Network completly?
                yield value
