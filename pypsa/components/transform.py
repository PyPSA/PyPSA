"""
Transform module of PyPSA components.

Contains all transform functions which can be used as methods of the
Components class. Transform are functions which modify data.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypsa import Components


def rename_component_names(c: Components, **kwargs: str) -> None:
    """
    Rename component names.

    Rename components and also update all cross-references of the component in
    the network.

    Parameters
    ----------
    c : pypsa.Components
        Components instance.
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
    Index(['bus2'], dtype='object', name='Bus')

    and all references in the network

    >>> n.generators.bus
    Generator
    gen1    bus2
    Name: bus, dtype: object

    """
    if not all(isinstance(v, str) for v in kwargs.values()):
        msg = "New names must be strings."
        raise ValueError(msg)

    # Rename component name definitions
    c.static = c.static.rename(index=kwargs)
    for k, v in c.dynamic.items():  # Modify in place
        c.dynamic[k] = v.rename(columns=kwargs)

    # Rename cross references in network (if attached to one)
    if c.attached:
        for component in c.n_save.components.values():
            col_name = c.name.lower()  # TODO: Generalize
            cols = [f"{col_name}{port}" for port in component.ports]
            if cols and not component.static.empty:
                component.static[cols] = component.static[cols].replace(kwargs)
