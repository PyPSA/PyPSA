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
