"""#TODO."""

from __future__ import annotations

import logging

import pandas as pd

from pypsa.components.abstract import _ComponentsABC

logger = logging.getLogger(__name__)


class _ComponentsIndex(_ComponentsABC):
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

    @property
    def scenarios(self) -> pd.Index:
        """Scenarios of networks."""
        return self.n_save.scenarios

    @property
    def has_scenarios(self) -> bool:
        """Boolean indicating if the network has scenarios defined."""
        return len(self.scenarios) > 0

    # Helpers

    @property
    def _static_index(self) -> pd.Index:
        """
        Static index of the network.

        Returns
        -------
        pd.Index
            Static index of the network.

        """
        if self.has_scenarios:
            return pd.MultiIndex.from_product(
                (self.scenarios.index, self.component_names),
                names=["scenario", "component"],
            )
        else:
            return self.component_names

    @property
    def _dynamic_index(self) -> pd.Index:
        """
        Dynamic index of the network.

        Returns
        -------
        pd.Index
            Dynamic index of the network.

        """
        if self.has_scenarios:
            return pd.MultiIndex.from_product(
                (self.scenarios.index, self.snapshots, self.component_names),
                names=["scenario", "snapshot", "component"],
            )
        else:
            return self.snapshots

    @property
    def _dynamic_columns(self) -> pd.Index:
        """
        Dynamic columns of the network.

        Returns
        -------
        pd.Index
            Dynamic columns of the network.

        """
        return self.component_names
