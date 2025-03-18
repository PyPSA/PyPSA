"""Plotting accessor for PyPSA."""

import functools
from typing import TYPE_CHECKING, Any

from pypsa.plot.maps import plot

if TYPE_CHECKING:
    from pypsa import Network


class PlotAccessor:
    """
    Accessor for plotting statistics.

    The class inherits from StatisticsAccessor and provides the same statistic
    functions, but returns a StatisticsPlotter object instead of a DataFrame.
    """

    """Abstract accessor to calculate different statistical values."""

    def __init__(self, n: "Network") -> None:
        """Initialize the statistics accessor."""
        self.n = n  # TODO rename

    @functools.wraps(plot)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        """Legacy plot method."""
        return plot(self.n, *args, **kwargs)
