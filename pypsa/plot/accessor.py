# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Plotting accessor for PyPSA."""

import functools
from typing import TYPE_CHECKING, Any

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
        self._n = n

    @functools.wraps(plot)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for [pypsa.Network.plot.map][pypsa.plot.PlotAccessor.map]."""
        return plot(self._n, *args, **kwargs)

    @functools.wraps(plot)
    def map(self, *args: Any, **kwargs: Any) -> Any:
        """Plot method."""
        return plot(self._n, *args, **kwargs)

    @functools.wraps(iplot)
    def iplot(self, *args: Any, **kwargs: Any) -> Any:
        """Interactive plot method."""
        return iplot(self._n, *args, **kwargs)

    @functools.wraps(explore)
    def explore(self, *args: Any, **kwargs: Any) -> Any:
        """Interactive map plot method."""
        return explore(self._n, *args, **kwargs)
