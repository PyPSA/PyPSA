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
    def component_names(self) -> pd.Index:
        """Unique names of the components."""
        return self.static.index.get_level_values("component").unique()

    # Derived from attached Network

    @property
    def snapshots(self) -> pd.Index | pd.MultiIndex:
        """Snapshots of the network."""
        return self.n_save.snapshots

    @property
    def timesteps(self) -> pd.Index:
        """Time steps of the network."""
        return self.n_save.timesteps

    @property
    def investment_periods(self) -> pd.Index:
        """Investment periods of the network."""
        return self.n_save.investment_periods

    @property
    def has_investment_periods(self) -> bool:
        """Indicator whether network has investment persios."""
        return self.n_save.has_investment_periods

    @property
    def periods(self) -> pd.Index:
        """Periods of the network."""
        return self.n_save.periods

    @property
    def has_periods(self) -> bool:
        """Investment periods of the network."""
        return self.n_save.has_periods
