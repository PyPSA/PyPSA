# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Components index module.

Contains single mixin class which is used to inherit to [pypsa.Components][] class.
Should not be used directly.

Index methods and properties are used to access the different index levels, based on
the attached parent network.

"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from pypsa.components.abstract import _ComponentsABC

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


class ComponentsIndexMixin(_ComponentsABC):
    """Mixin class for components index methods.

    Class inherits to [pypsa.Components][]. All attributes and methods can be used
    within any Components instance.

    """

    @property
    def names(self) -> pd.Series:
        """Get component names.

        <!-- md:badge-version v1.0.0 -->

        The names are corresponding to [`c.static.index`][pypsa.Components.static] for
        non stochastic networks. For stochastic networks the index will be multi-indexed.

        Returns
        -------
        pd.Series
            Series with component names as index and values.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Generator", "g1")
        >>> n.add("Generator", "g2")
        >>> n.generators.index
        Index(['g1', 'g2'], dtype='object', name='name')

        """
        return self.static.index.get_level_values("name").drop_duplicates()

    @property
    def component_names(self) -> pd.Series:
        """Get component names.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"
            Use [`names`][pypsa.Components.names] instead.

        Returns
        -------
        pd.Series
            Series with component names as index and values.

        """
        warnings.warn(
            "c.component_names is deprecated, use c.names instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.names

    # Derived from attached Network

    @property
    def snapshots(self) -> pd.Index | pd.MultiIndex:
        """Snapshots of the network.

        <!-- md:badge-version v0.34.0 -->

        See Also
        --------
        [pypsa.Network.snapshots][]

        """
        return self.n_save.snapshots

    @property
    def timesteps(self) -> pd.Index:
        """Time steps of the network.

        <!-- md:badge-version v0.34.0 -->

        See Also
        --------
        [pypsa.Network.timesteps][]

        """
        return self.n_save.timesteps

    @property
    def investment_periods(self) -> pd.Index:
        """Investment periods of the network.

        <!-- md:badge-version v0.34.0 -->

        See Also
        --------
        [pypsa.Network.investment_periods][]

        """
        return self.n_save.investment_periods

    @property
    def has_investment_periods(self) -> bool:
        """Indicator whether network has investment periods.

        <!-- md:badge-version v0.34.0 -->

        See Also
        --------
        [pypsa.Network.has_investment_periods][]

        """
        return self.n_save.has_investment_periods

    @property
    def periods(self) -> pd.Index:
        """Periods of the network.

        <!-- md:badge-version v0.34.0 -->

        See Also
        --------
        [pypsa.Network.periods][]

        """
        return self.n_save.periods

    @property
    def has_periods(self) -> bool:
        """Investment periods of the network.

        <!-- md:badge-version v0.34.0 -->
        """
        return self.n_save.has_periods

    @property
    def scenarios(self) -> pd.Index:
        """Scenarios of networks.

        <!-- md:badge-version v1.0.0 -->
        """
        return self.n_save.scenarios

    @property
    def has_scenarios(self) -> bool:
        """Boolean indicating if the network has scenarios defined.

        <!-- md:badge-version v1.0.0 -->
        """
        return len(self.scenarios) > 0
