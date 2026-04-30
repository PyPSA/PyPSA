# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Legacy functionality which is kept for backwards compatibility."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pypsa.components.types import get as get_component_type

if TYPE_CHECKING:
    import pandas as pd

    from pypsa.definitions.components import ComponentType
    from pypsa.definitions.structures import Dict

# Legacy Component Class
# -----------------------------------


class Component:
    """Legacy component class.

    Allows to keep functionallity of previous dataclass/ named tuple and wraps
    around new structure.

    !!! warning
        This class is deprecated and should not be used anymore.
    """

    # ruff: noqa: D102
    def __new__(
        cls,
        name: str | None = None,
        ctype: ComponentType | None = None,
        n: Any | None = None,
        static: pd.DataFrame | None = None,
        dynamic: Dict | None = None,
    ) -> Any:
        warnings.warn(
            "Component() is deprecated as of 1.3.0 and will be removed in 2.0.0."
            " Components cannot be initialized directly and are always"
            " attached to a Network. Access via `n.components.<component>`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if (name and ctype is not None) or (not name and ctype is None):
            msg = "One out of 'name' or 'ctype' must be given."
            raise ValueError(msg)
        if n is None:
            msg = "Legacy `Component` requires a Network via `n=`."
            raise ValueError(msg)

        if name:
            ctype_ = get_component_type(name)
        else:
            ctype_ = ctype  # type: ignore

        instance = n.components[ctype_.list_name]
        if static is not None:
            instance.static = static
        if dynamic is not None:
            instance.dynamic = dynamic

        return instance
