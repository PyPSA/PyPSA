"""
Components module.

Contains classes and logic relevant to specific component types in PyPSA.
Generic functionality is implemented in the abstract module.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from pypsa.components.abstract import Components
from pypsa.components.types import ComponentType
from pypsa.components.types import get as get_component_type
from pypsa.definitions.structures import Dict
from pypsa.geo import haversine_pts

logger = logging.getLogger(__name__)


class GenericComponents(Components):
    """
    Generic components class.

    This class is used for components that do not have a specific class implementation.
    All functionality specific to generic types only is implemented here. Functionality
    for all components is implemented in the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    See Also
    --------
    pypsa.components.abstract.Components : Base class for all components.
    pypsa.components.components.Generators : Generators components class.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize generic components class.

        See :class:`pypsa.components.abstract.Components` for more information.

        Parameters
        ----------
        args : Any
            Arguments of base class.
        kwargs : Any
            Keyword arguments of base class.

        Returns
        -------
        None

        """
        super().__init__(*args, **kwargs)


class Generators(Components):
    """
    Generators components class.

    This class is used for generator components. All functionality specific to
    generators is implemented here. Functionality for all components is implemented in
    the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    See Also
    --------
    pypsa.components.abstract.Components : Base class for all components.
    pypsa.components.components.GenericComponents : Generic components class.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize Generators class.

        See :class:`pypsa.components.abstract.Components` for more information.

        Parameters
        ----------
        args : Any
            Arguments of base class.
        kwargs : Any
            Keyword arguments of base class.

        Returns
        -------
        None

        """
        super().__init__(*args, **kwargs)


class Lines(Components):
    """
    Lines components class.

    This class is used for line components. All functionality specific to
    lines is implemented here. Functionality for all components is implemented in
    the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    See Also
    --------
    pypsa.components.abstract.Components : Base class for all components.
    pypsa.components.components.GenericComponents : Generic components class.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize Generators class.

        See :class:`pypsa.components.abstract.Components` for more information.

        Parameters
        ----------
        args : Any
            Arguments of base class.
        kwargs : Any
            Keyword arguments of base class.

        Returns
        -------
        None

        """
        super().__init__(*args, **kwargs)

    def calculate_line_length(self) -> pd.Series:
        """
        Get length of the lines in meters.

        Based on coordinates of attached buses. Buses must have 'x' and 'y' attributes,
        otherwise no line length can be calculated. By default the haversine formula is
        used to calculate the distance between two points.

        Returns
        -------
        pd.Series
            Length of the lines.

        See Also
        --------
        pypsa.geo.haversine : Function to calculate distance between two points.

        Examples
        --------
        >>> c = pypsa.examples.scigrid_de().c.lines
        >>> ds = c.calculate_line_length()
        >>> ds.head()
        0    34432.796096
        1    59701.666027
        2    32242.741010
        3    30559.154647
        4    21574.543367
        dtype: float64

        """
        return (
            pd.Series(
                haversine_pts(
                    a=np.array(
                        [
                            self.static.bus0.map(self.n_save.buses.x),
                            self.static.bus0.map(self.n_save.buses.y),
                        ]
                    ).T,
                    b=np.array(
                        [
                            self.static.bus1.map(self.n_save.buses.x),
                            self.static.bus1.map(self.n_save.buses.y),
                        ]
                    ).T,
                )
            )
            * 1_000
        )


CLASS_MAPPING = {
    "Generator": Generators,
    "Line": Lines,
}


class Component:
    """
    Legacy component class.

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
            ctype_ = get_component_type(name)
        else:
            ctype_ = ctype  # type: ignore

        component_class = CLASS_MAPPING.get(ctype_.name, None)
        instance: Components
        if component_class is not None:
            instance = component_class(ctype=ctype_)
        else:
            instance = GenericComponents(ctype=ctype_)

        if n is not None:
            instance.n = n
        if static is not None:
            instance.static = static
        if dynamic is not None:
            instance.dynamic = dynamic

        return instance
