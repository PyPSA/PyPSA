"""Descriptors for component attributes."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class Dict(dict):
    """Subclass of dict, which allows to use attribute syntax to get and set items.

    Stripped down from addict https://github.com/mewwts/addict/ .
    """

    def __setattr__(self, name: str, value: Any) -> None:
        """Setattr is called when the syntax a.b = 2 is used to set a value."""
        if hasattr(Dict, name):
            msg = f"'Dict' object attribute '{name}' is read-only"
            raise AttributeError(msg)
        self[name] = value

    def __getattr__(self, item: str) -> Any:
        """Getattr is called when the syntax a.b is used to get a value."""
        try:
            return self.__getitem__(item)
        except KeyError as e:
            raise AttributeError(e.args[0]) from e

    def __delattr__(self, name: str) -> None:
        """Is invoked when del some_addict.b is called."""
        del self[name]

    _re_pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

    def __dir__(self) -> list[str]:
        """Return a list of object attributes.

        This includes key names of any dict entries, filtered to the
        subset of valid attribute names (e.g. alphanumeric strings
        beginning with a letter or underscore).  Also includes
        attributes of parent dict class.
        """
        dict_keys = [
            m.string
            for k in self.keys()
            if isinstance(k, str) and (m := self._re_pattern.match(k))
        ]

        obj_attrs = list(dir(Dict))

        return dict_keys + obj_attrs
