# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Legacy functionality which is kept for backwards compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deprecation import deprecated

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
    @deprecated(
        deprecated_in="1.3.0",
        removed_in="2.0.0",
        details="Use `n.components.<component>` instead.",
    )
    def __new__(
        cls,
        name: str | None = None,
        ctype: ComponentType | None = None,
        n: Any | None = None,
        static: pd.DataFrame | None = None,
        dynamic: Dict | None = None,
    ) -> Any:
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
