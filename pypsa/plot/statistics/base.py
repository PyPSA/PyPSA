# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Abstract base class to generate any plots based on statistics functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network


class PlotsGenerator(ABC):
    """Base plot generator class for statistics plots.

    This class provides a common interface for all plot generators which build up
    on statistics functions of [`pypsa.Network.statistics`][]. Defined methods need
    to be implemented by subclasses.
    """

    _n: Network

    def __init__(self, n: Network) -> None:
        """Initialize plot generator.

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

    def get_unique_carriers(self) -> pd.DataFrame:
        """Get unique carriers from the network.

        For stochastic networks with MultiIndex, returns deduplicated carriers
        across scenarios.

        Returns
        -------
        pd.DataFrame
            DataFrame with carrier data, deduplicated across scenarios if applicable.

        """
        carriers = self._n.c.carriers.static
        if isinstance(carriers.index, pd.MultiIndex):
            # For stochastic networks, deduplicate across scenarios
            for level in carriers.index.names:
                if level != "name":
                    carriers = carriers.droplevel(level)
            unique_carriers = carriers[~carriers.index.duplicated(keep="first")]
            return unique_carriers.sort_index()
        return carriers.sort_index()

    def _get_carrier_attribute(
        self,
        attribute: str,
        carriers: Sequence | None = None,
        apply_nice_names: bool = False,
    ) -> pd.Series:
        """Get attribute values for carriers.

        Helper method to extract carrier attributes with optional nice name mapping.

        Parameters
        ----------
        attribute : str
            Name of the attribute to extract (e.g., 'color', 'nice_name').
        carriers : Sequence or None
            Specific carriers to get attributes for. If None, uses all carriers.
        apply_nice_names : bool, default False
            Whether to map carrier names to nice names in the index.

        Returns
        -------
        pd.Series
            Series with carrier names (or nice names) as index and attribute values.

        """
        carriers_df = self.get_unique_carriers()
        if carriers is None:
            carriers = carriers_df.index

        values = carriers_df[attribute][carriers]

        if apply_nice_names and "nice_name" in carriers_df.columns:
            nice_names = carriers_df.nice_name[carriers]
            nice_names = nice_names.where(nice_names != "", carriers)
            values.index = nice_names

        # Remove any duplicate entries
        return values[~values.index.duplicated(keep="first")]

    def get_carrier_colors(
        self, carriers: Sequence | None = None, nice_names: bool = True
    ) -> dict:
        """Get colors for carrier data with default gray colors.

        Parameters
        ----------
        carriers : Sequence or None
            Specific carriers to get colors for. If None, uses all carriers.
        nice_names : bool, default True
            Whether to use nice names as keys in the returned dictionary.

        Returns
        -------
        dict
            Dictionary mapping carrier (or nice) names to color values.

        """
        colors = self._get_carrier_attribute(
            "color", carriers, apply_nice_names=nice_names
        )
        default_colors = {"-": "gray", None: "gray", "": "gray"}
        return {**default_colors, **colors.to_dict()}

    def get_carrier_labels(
        self, carriers: Sequence | None = None, nice_names: bool = True
    ) -> pd.Series:
        """Get mapping of carrier names to display labels.

        Parameters
        ----------
        carriers : Sequence or None
            Specific carriers to get labels for. If None, uses all carriers.
        nice_names : bool, default True
            Whether to use nice names as labels. If False, returns carrier names.

        Returns
        -------
        pd.Series
            Series mapping carrier names to display labels.

        """
        carriers_df = self.get_unique_carriers()
        if carriers is None:
            carriers = carriers_df.index

        if nice_names and "nice_name" in carriers_df.columns:
            names = carriers_df.nice_name[carriers]
            return names.where(names != "", carriers)

        return pd.Series(carriers, index=carriers)
