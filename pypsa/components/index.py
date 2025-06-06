"""Components index module.

Contains single mixin class which is used to inherit to [pypsa.Components][] class.
Should not be used directly.

Index methods and properties are used to access the different index levels, based on
the attached parent network.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pypsa.components.abstract import _ComponentsABC

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class ComponentsIndexMixin(_ComponentsABC):
    """Mixin class for components index methods.

    Class only inherits to [pypsa.Components][] and should not be used directly.
    All attributes and methods can be used within any Components instance.

    """

    @property
    def component_names(self) -> pd.Series:
        """Get component names.

        Returns
        -------
        pd.Series
            Series with component names as index and values.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Generator", "g1")  # doctest: +ELLIPSIS
        Index(['g1'], dtype='object')
        >>> n.add("Generator", "g2")  # doctest: +ELLIPSIS
        Index(['g2'], dtype='object')
        >>> n.generators.index
        Index(['g1', 'g2'], dtype='object', name='Generator')

        """
        return self.static.index.get_level_values("component").unique()

    # Derived from attached Network

    @property
    def snapshots(self) -> pd.Index | pd.MultiIndex:
        """Snapshots of the network.

        See Also
        --------
        [pypsa.Network.snapshots][] :
            Snapshots of the network.

        """
        return self.n_save.snapshots

    @property
    def timesteps(self) -> pd.Index:
        """Time steps of the network.

        See Also
        --------
        [pypsa.Network.timesteps][] :
            Time steps of the network.

        """
        return self.n_save.timesteps

    @property
    def investment_periods(self) -> pd.Index:
        """Investment periods of the network.

        See Also
        --------
        [pypsa.Network.investment_periods][] :
            Investment periods of the network.

        """
        return self.n_save.investment_periods

    @property
    def has_investment_periods(self) -> bool:
        """Indicator whether network has investment periods.

        See Also
        --------
        [pypsa.Network.has_investment_periods][] :
            Indicator whether network has investment periods.

        """
        return self.n_save.has_investment_periods

    @property
    def periods(self) -> pd.Index:
        """Periods of the network.

        See Also
        --------
        [pypsa.Network.periods][] :
            Periods of the network.

        """
        return self.n_save.periods

    @property
    def has_periods(self) -> bool:
        """Indicator whether network has periods.

        See Also
        --------
        [pypsa.Network.has_periods][] :
            Indicator whether network has periods.

        """
        return self.n_save.has_periods
