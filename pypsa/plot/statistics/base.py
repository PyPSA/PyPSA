"""Abstract base class to generate any plots based on statistics functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypsa import Network


class PlotsGenerator(ABC):
    """
    Base plot generator class for statistics plots.

    This class provides a common interface for all plot generators which build up
    on statistics functions of :mod:`pypsa.statistics`. Defined methods need
    to be implemented by subclasses.
    """

    def __init__(self, n: Network) -> None:
        """
        Initialize plot generator.

        Parameters
        ----------
        n : pypsa.Network
            Network object.

        """
        self._n = n

    @abstractmethod
    def add_default_kwargs(
        self,
        *args: str | None,
        stats_kwargs: dict | None = None,  # make required
        method_name: str | None = None,  # make required
    ) -> dict[str, Any]:
        """Handle default statistics kwargs based on provided plot kwargs."""
        pass
