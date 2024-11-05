"""
Components module.

Contains classes and logic relevant to specific component types in PyPSA.
Generic functionality is implemented in the abstract module.
"""

import logging
import warnings
from typing import Any

import pandas as pd

from pypsa.components.abstract import Components
from pypsa.components.types import ComponentTypeInfo, get_component_type
from pypsa.definitions.structures import Dict

logger = logging.getLogger(__name__)


class GenericComponents(Components):
    """
    Generic Components class

    This class is used for components that do not have a specific class implementation.
    All functionality specific to generic types only is implemented here. Functionality
    for all components is implemented in the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class Generators(Components):
    """
    Generators Components class

    This class is used for components that are generators. All functionality specific to
    generators is implemented here. Functionality for all components is implemented in
    the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


CLASS_MAPPING = {
    "Generator": Generators,
}


class Component:
    """
    Legacy Component Class

    Allows to keep functionallity of previous dataclass and named tuple and wraps
    around new structure.

    .. warning::
        This class is deprecated and should not be used anymore.
    """

    def __new__(
        cls,
        name: str | None = None,
        ct: ComponentTypeInfo | None = None,
        n: Any | None = None,
        static: pd.DataFrame | None = None,
        dynamic: Dict | None = None,
        list_name: str | None = None,
        attrs: pd.DataFrame | None = None,
        investment_periods: pd.Index | None = None,
        ind: None = None,
    ) -> Any:
        # Deprecation warnings
        if (name and ct is not None) or (not name and ct is None):
            msg = "One out of 'name' or 'ct' must be given."
            raise ValueError(msg)
        if list_name is not None or attrs is not None:
            warnings.warn(
                "Passing 'list_name' and 'attrs' is deprecated and they will be "
                "retrieved via the 'name' argument.",
                DeprecationWarning,
                stacklevel=2,
            )
        if ind is not None:
            warnings.warn(
                "The 'ind' attribute is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
        if investment_periods is not None:
            raise DeprecationWarning(
                "The 'investment_periods' attribute is deprecated. Pass 'n' instead."
            )

        if name:
            ct_ = get_component_type(name)
        else:
            ct_ = ct  # type: ignore

        component_class = CLASS_MAPPING.get(ct_.name, None)
        instance: Components
        if component_class is not None:
            instance = component_class(ct=ct_)
        else:
            instance = GenericComponents(ct=ct_)

        if n is not None:
            instance.n = n
        if static is not None:
            instance.static = static
        if dynamic is not None:
            instance.dynamic = dynamic

        return instance
