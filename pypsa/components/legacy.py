"""Legacy functionality which is kept for backwards compatibility."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pypsa.common import UnexpectedError
from pypsa.components._types import (
    Buses,
    Carriers,
    Generators,
    GlobalConstraints,
    Lines,
    LineTypes,
    Links,
    Loads,
    Shapes,
    ShuntImpedances,
    StorageUnits,
    Stores,
    SubNetworks,
    Transformers,
    TransformerTypes,
)
from pypsa.components.types import get as get_component_type

if TYPE_CHECKING:
    import pandas as pd

    from pypsa.components.components import Components
    from pypsa.definitions.components import ComponentType
    from pypsa.definitions.structures import Dict

# Legacy Component Class
# -----------------------------------

_CLASS_MAPPING = {
    "Bus": Buses,
    "Carrier": Carriers,
    "Generator": Generators,
    "GlobalConstraint": GlobalConstraints,
    "Line": Lines,
    "LineType": LineTypes,
    "Link": Links,
    "Load": Loads,
    "Shape": Shapes,
    "ShuntImpedance": ShuntImpedances,
    "StorageUnit": StorageUnits,
    "Store": Stores,
    "SubNetwork": SubNetworks,
    "Transformer": Transformers,
    "TransformerType": TransformerTypes,
}


class Component:
    """Legacy component class.

    Allows to keep functionallity of previous dataclass/ named tuple and wraps
    around new structure.

    .. warning::
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
        list_name: str | None = None,
        attrs: pd.DataFrame | None = None,
        investment_periods: pd.Index | None = None,
        ind: None = None,
    ) -> Any:
        # Deprecation warnings
        if (name and ctype is not None) or (not name and ctype is None):
            msg = "One out of 'name' or 'ct' must be given."
            raise ValueError(msg)
        if list_name is not None or attrs is not None:
            warnings.warn(
                "Passing 'list_name' and 'attrs' is deprecated and they will be "
                "retrieved via the 'name' argument. Deprecated in version 0.31 and "
                "will be removed in version 1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if ind is not None:
            warnings.warn(
                "The 'ind' attribute is deprecated. Deprecated in version 0.31 and "
                "will be removed in version 1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if investment_periods is not None:
            msg = (
                "The 'investment_periods' attribute is deprecated. Pass 'n' instead."
                "Deprecated in version 0.31 and will be removed in version 1.0."
            )
            raise DeprecationWarning(msg)

        if name:
            ctype_ = get_component_type(name)
        else:
            ctype_ = ctype  # type: ignore

        component_class = _CLASS_MAPPING.get(ctype_.name, None)
        instance: Components
        if component_class is not None:
            instance = component_class(ctype=ctype_)
        else:
            msg = f"Component type '{ctype_.name}' not found."
            raise UnexpectedError(msg)

        if n is not None:
            instance.n = n
        if static is not None:
            instance.static = static
        if dynamic is not None:
            instance.dynamic = dynamic

        return instance
