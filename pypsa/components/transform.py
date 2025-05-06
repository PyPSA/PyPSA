"""
Components transform module.

Contains single helper class (_ComponentsTransform) which is used to inherit
to Components class. Should not be used directly.  Transform methods are methods which
modify and restructure data.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class _ComponentsTransform:
    """
    Helper class for components descriptors methods.

    Class only inherits to Components and should not be used directly.
    """

    static: pd.DataFrame
    dynamic: dict[str, pd.DataFrame]
    attached: Any
    n_save: Any
    name: Any

    def rename_component_names(self, **kwargs: str) -> None:
        """
        Rename component names.

        Rename components and also update all cross-references of the component in
        the network.

        Parameters
        ----------
        **kwargs
            Mapping of old names to new names.

        Returns
        -------
        None

        Examples
        --------
        Define some network
        >>> import pypsa
        >>> n = pypsa.Network()
        >>> n.add("Bus", ["bus1"])
        Index(['bus1'], dtype='object')
        >>> n.add("Generator", ["gen1"], bus="bus1")
        Index(['gen1'], dtype='object')
        >>> c = n.c.buses

        Now rename the bus

        >>> c.rename_component_names(bus1="bus2")

        Which updates the bus components

        >>> c.static.index
        Index(['bus2'], dtype='object', name='component')

        and all references in the network

        >>> n.generators.bus
        component
        gen1    bus2
        Name: bus, dtype: object

        """
        if not all(isinstance(v, str) for v in kwargs.values()):
            msg = "New names must be strings."
            raise ValueError(msg)

        # Rename component name definitions
        self.static = self.static.rename(index=kwargs)
        for k, v in self.dynamic.items():  # Modify in place
            self.dynamic[k] = v.rename(columns=kwargs)

        # Rename cross references in network (if attached to one)
        if self.attached:
            for component in self.n_save.components.values():
                col_name = self.name.lower()  # TODO: Generalize
                cols = [f"{col_name}{port}" for port in component.ports]
                if cols and not component.static.empty:
                    component.static[cols] = component.static[cols].replace(kwargs)
