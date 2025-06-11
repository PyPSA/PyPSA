"""Definitions for network components."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deprecation import deprecated

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentType:
    """Dataclass for network component type.

    Contains all information about a component type, such as its name and defaults
    attributes. Two different types are for example 'Generator' and 'Carrier'.

    Attributes
    ----------
    name : str
        Name of component type, e.g. 'Generator'.
    list_name : str
        Name of component type in list form, e.g. 'generators'.
    description : str
        Description of the component type.
    category : str
        Category of the component type, e.g. 'passive_branch'.
    defaults : pd.DataFrame
        Default values for the component type.
    standard_types : pd.DataFrame | None
        Standard types for the component type.

    """

    name: str
    list_name: str
    description: str
    category: str
    defaults: pd.DataFrame
    standard_types: pd.DataFrame | None = None

    def __eq__(self, other: object) -> bool:
        """Check if two component types are equal.

        Parameters
        ----------
        other : Any
            The other object to compare to.

        Returns
        -------
        bool

        """
        if not isinstance(other, ComponentType):
            return NotImplemented
        return (
            self.name == other.name
            and self.list_name == other.list_name
            and self.description == other.description
            and str(self.category) == str(other.category)
            and self.defaults.equals(other.defaults)
        )

    def __repr__(self) -> str:
        """Representation of the component type.

        Returns
        -------
        str

        """
        # TODO make this actually for the REPL
        return f"'{self.name}' Component Type"

    @property
    @deprecated(
        deprecated_in="0.32.0",
        removed_in="1.0",
        details="Use the 'category' attribute instead.",
    )
    def type(self) -> str:
        """Getter for the 'type' attribute.

        Returns
        -------
        str

        """
        return self.category

    @property
    @deprecated(
        deprecated_in="0.32.0",
        removed_in="1.0",
        details="Use the 'defaults' attribute instead.",
    )
    def attrs(self) -> pd.DataFrame:
        """Getter for the 'attrs' attribute.

        Returns
        -------
        pd.DataFrame

        """
        return self.defaults
