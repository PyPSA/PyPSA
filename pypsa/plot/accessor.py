"""Plotting accessor for PyPSA."""

import functools
from typing import TYPE_CHECKING, Any

from deprecation import deprecated

from pypsa.plot.maps import explore, iplot, plot

if TYPE_CHECKING:
    from pypsa import Network


class PlotAccessor:
    """Accessor for plotting statistics.

    The class inherits from StatisticsAccessor and provides the same statistic
    functions, but returns a StatisticPlotter object instead of a DataFrame.
    """

    """Abstract accessor to calculate different statistical values."""

    def __init__(self, n: "Network") -> None:
        """Initialize the statistics accessor."""
        self.n = n  # TODO rename

    @deprecated(
        deprecated_in="0.34",
        removed_in="1.0",
        details="Use `n.plot.map()` as a drop-in replacement instead.",
    )
    @functools.wraps(plot)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Legacy plot method."""
        return plot(self.n, *args, **kwargs)

    @functools.wraps(plot)
    def map(self, *args: Any, **kwargs: Any) -> Any:
        """Plot method."""
        return plot(self.n, *args, **kwargs)

    @functools.wraps(iplot)
    def iplot(self, *args: Any, **kwargs: Any) -> Any:
        """Interactive plot method."""
        return iplot(self.n, *args, **kwargs)

    @functools.wraps(explore)
    def explore(self, *args: Any, **kwargs: Any) -> Any:
        """Interactive map plot method."""
        return explore(self.n, *args, **kwargs)
