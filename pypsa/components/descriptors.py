"""
Components descriptor module.

Contains single helper class (__ComponentsDescriptors) which is used to inherit
to Components class. Should not be used directly. Descriptor functions only describe
data and do not modify it.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pypsa.common import as_index
from pypsa.components.abstract import _ComponentsABC

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypsa import Components
logger = logging.getLogger(__name__)


def get_active_assets(c: Components, *args: Any, **kwargs: Any) -> Any:
    """
    Deprecated function to get active assets. Use `c.get_active_assets`.

    Examples
    --------
    >>> import pytest
    >>> with pytest.warns(DeprecationWarning):
    ...     get_active_assets(c)
    Generator
    Manchester Wind    True
    Manchester Gas     True
    Norway Wind        True
    Norway Gas         True
    Frankfurt Wind     True
    Frankfurt Gas      True
    Name: active, dtype: bool

    """
    warnings.warn(
        (
            "pypsa.components.descriptors.get_active_assets is deprecated. "
            "Use c.get_active_assets instead."
            "Deprecated in version 0.35 and will be removed in version 1.0."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return c.get_active_assets(*args, **kwargs)


class _ComponentsDescriptors(_ComponentsABC):
    """
    Helper class for components descriptors methods.

    Class only inherits to Components and should not be used directly.
    """

    def get_active_assets(
        self,
        investment_period: int | str | Sequence | None = None,
    ) -> pd.Series:
        """
        Get active components mask of componen type in investment period(s).

        A component is considered active when:

        - it's active attribute is True
        - it's build year + lifetime is smaller than the investment period (if given)

        Parameters
        ----------
        investment_period : int, str, Sequence
            Investment period(s) to check for active within build year and lifetime. If
            none only the active attribute is considered and build year and lifetime are
            ignored. If multiple periods are given the mask is True if component is
            active in any of the given periods.

        Returns
        -------
        pd.Series
            Boolean mask for active components

        """
        if investment_period is None:
            return self.static.active
        if not {"build_year", "lifetime"}.issubset(self.static):
            return self.static.active

        # Logical OR of active assets in all investment periods and
        # logical AND with active attribute
        active = {}
        for period in np.atleast_1d(investment_period):
            if period not in self.n_save.investment_periods:
                raise ValueError("Investment period not in `n.investment_periods`")
            active[period] = self.static.eval(
                "build_year <= @period < build_year + lifetime"
            )
        return pd.DataFrame(active).any(axis=1) & self.static.active

    def get_activity_mask(
        self,
        sns: Sequence | None = None,
        index: pd.Index | None = None,
    ) -> pd.DataFrame:
        """
        Get active components mask indexed by snapshots.

        Gets the boolean mask for active components, indexed by snapshots and
        components instead of just components.

        Parameters
        ----------
        c : pypsa.Components
            Components instance.
        sns : pandas.Index, default None
            Set of snapshots for the mask. If None (default) all snapshots are returned.
        index : pd.Index, default None
            Subset of the component elements. If None (default) all components are returned.

        Returns
        -------
        pd.DataFrame
            Boolean mask for active components indexed by snapshots.

        """
        sns_ = as_index(self.n_save, sns, "snapshots")

        if self.has_investment_periods:
            active_assets_per_period = {
                period: self.get_active_assets(investment_period=period)
                for period in self.investment_periods
            }
            mask = (
                pd.concat(active_assets_per_period, axis=1)
                .T.reindex(self.snapshots, level=0)
                .loc[sns_]
            )
        else:
            active_assets = self.get_active_assets()
            mask = pd.DataFrame(
                np.tile(active_assets, (len(sns_), 1)),
                index=sns_,
                columns=active_assets.index,
            )

        if index is not None:
            mask = mask.reindex(columns=index)

        mask.index.name = "snapshot"
        if isinstance(mask.index, pd.MultiIndex):
            mask.index.names = ["period", "timestep"]

        return mask

    # TODO: remove as soon as deprecated renaming is removed
    def get_extendable_i(self, rename_index: bool = True) -> pd.Index:
        """Get the index of extendable elements of this component."""
        idx = self.extendables
        return idx.rename(idx.name) if rename_index else idx

    # TODO: remove as soon as deprecated renaming is removed
    def get_non_extendable_i(self, rename_index: bool = True) -> pd.Index:
        """Get the index of non-extendable elements of this component."""
        idx = self.fixed
        return idx.rename(idx.name) if rename_index else idx

    # TODO: remove as soon as deprecated renaming is removed
    def get_committable_i(self, rename_index: bool = True) -> pd.Index:
        """Get the index of committable elements of this component."""
        idx = self.committables
        return idx.rename(idx.name) if rename_index else idx
