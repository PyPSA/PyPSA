"""Abstract base class to generate any plots based on statistics functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.plot.statistics.schema import schema

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
    def derive_statistic_parameters(
        self,
        *args: str | None,
        method_name: str = "",  # make required
    ) -> dict[str, Any]:
        """Handle default statistics kwargs based on provided plot kwargs."""
        pass

    def apply_parameter_schema(
        self, stats_name: str, plot_name: str, kwargs: dict
    ) -> dict:
        """
        Apply parameter schema to kwargs.

        The schema is used to for different statistics functions signatures based on
        plot type/ choosed statistics function. The schema is defined in
        :mod:`pypsa.plot.statistics.schema`.

        Parameters
        ----------
        stats_name : str
            Name of the statistics function.
        plot_name : str
            Name of the plot type.
        kwargs : dict
            Dictionary of keyword arguments to be filtered based on the schema.

        Returns
        -------
        dict
            Filtered dictionary of keyword arguments.

        """
        to_remove = []
        # Filter kwargs based on different statistics functions signatures
        for param, value in kwargs.items():
            if param not in schema[stats_name][plot_name]:
                continue
            if value is None:
                kwargs[param] = schema[stats_name][plot_name][param]["default"]
                if not schema[stats_name][plot_name][param]["allowed"]:
                    to_remove.append(param)
            else:
                if not schema[stats_name][plot_name][param]["allowed"]:
                    msg = f"Parameter {param} can not be used for {stats_name} {plot_name}."
                    raise ValueError(msg)

        for param in to_remove:
            kwargs.pop(param)

        return kwargs

    def get_carrier_colors(
        self, carriers: Sequence | None = None, nice_names: bool = True
    ) -> dict:
        """Get colors for carrier data with default gray colors."""
        if carriers is None:
            carriers = self._n.carriers.index
        colors = self._n.carriers.color[carriers]
        if nice_names:
            labels = self.get_carrier_labels(carriers=carriers, nice_names=nice_names)
            colors = colors.rename(labels)
        default_colors = {"-": "gray", None: "gray"}
        return {**default_colors, **colors}

    def get_carrier_labels(
        self, carriers: Sequence | None = None, nice_names: bool = True
    ) -> pd.Series:
        """Get mapping of carrier names to nice names if requested."""
        if carriers is None:
            carriers = self._n.carriers.index
        if nice_names:
            names = self._n.carriers.nice_name[carriers]
            return names.where(names != "", carriers)
        else:
            return pd.Series(carriers, index=carriers)
