"""Components descriptor module.

Contains single mixin class which is used to inherit to [pypsa.Components][] class.
Should not be used directly.

Descriptor functions only describe data and do not modify it.

"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pypsa.common import as_index
from pypsa.components.abstract import _ComponentsABC

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Components
logger = logging.getLogger(__name__)


def get_active_assets(c: Components, *args: Any, **kwargs: Any) -> Any:
    """Get active assets. Use `c.get_active_assets` instead.

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


class ComponentsDescriptorsMixin(_ComponentsABC):
    """Mixin class for components descriptors methods.

    Class only inherits to [pypsa.Components][] and should not be used directly.
    All attributes and methods can be used within any Components instance.

    """

    @property
    def _operational_attrs(self) -> dict[str, str]:
        """Get operational attributes of component for optimization.

        Provides a dictionary of attribute patterns used in optimization constraints,
        based on the component type. This makes constraint formulation more modular
        by avoiding hardcoded attribute names.

        Returns
        -------
        dict[str, str]
            Dictionary of operational attribute names

        """
        # TODO: refactor component attrs store

        base = {
            "Generator": "p",
            "Line": "s",
            "Link": "p",
            "Load": "p",
            "StorageUnit": "p",
            "Store": "e",
            "Transformer": "s",
        }[self.name]

        return {
            "base": base,
            "nom": f"{base}_nom",
            "nom_extendable": f"{base}_nom_extendable",
            "nom_min": f"{base}_nom_min",
            "nom_max": f"{base}_nom_max",
            "nom_set": f"{base}_nom_set",
            "min_pu": f"{base}_min_pu",
            "max_pu": f"{base}_max_pu",
            "set": f"{base}_set",
        }

    def get_active_assets(
        self,
        investment_period: int | str | Sequence | None = None,
    ) -> pd.Series:
        """Get active components mask of componen type in investment period(s).

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

        Examples
        --------
        Without investment periods

        >>> n = pypsa.Network()
        >>> n.add("Generator", "g1", active=False)
        Index(['g1'], dtype='object')
        >>> n.add("Generator", "g2", active=True)
        Index(['g2'], dtype='object')
        >>> n.components.generators.get_active_assets()
        Generator
        g1    False
        g2     True
        Name: active, dtype: bool

        With investment periods
        >>> n = pypsa.Network()
        >>> n.snapshots = pd.MultiIndex.from_product([[2020, 2021, 2022], ["1", "2", "3"]])
        >>> n.add("Generator", "g1", build_year=2020, lifetime=1)
        Index(['g1'], dtype='object')
        >>> n.add("Generator", "g2", active=False)
        Index(['g2'], dtype='object')
        >>> n.components.generators.get_active_assets()
        Generator
        g1     True
        g2    False
        Name: active, dtype: bool

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
                msg = "Investment period not in `n.investment_periods`"
                raise ValueError(msg)
            active[period] = self.static.eval(
                "build_year <= @period < build_year + lifetime"
            )
        return pd.DataFrame(active).any(axis=1) & self.static.active

    def get_activity_mask(
        self,
        sns: Sequence | None = None,
        index: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Get active components mask indexed by snapshots.

        Wrapper around the
        `:py:meth:`pypsa.descriptors.components.Componenet.get_active_assets` method.
        Get's the boolean mask for active components, but indexed by snapshots and
        components instead of just components.

        Parameters
        ----------
        n : pypsa.Network
            Network instance
        c : string
            Component name
        sns : pandas.Index, default None
            Set of snapshots for the mask. If None (default) all snapshots are returned.
        index : pd.Index, default None
            Subset of the component elements. If None (default) all components are returned.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.snapshots = pd.MultiIndex.from_product([[2020, 2021, 2022], ["1", "2", "3"]])
        >>> n.add("Generator", "g1", build_year=2020, lifetime=1)  # doctest: +ELLIPSIS
        Index(['g1'], dtype='object')
        >>> n.add("Generator", "g2", active=False)  # doctest: +ELLIPSIS
        Index(['g2'], dtype='object')
        >>> n.components.generators.get_activity_mask()  # doctest: +ELLIPSIS
        Generator          g1     g2
        period timestep
        2020   1         True  False
               2         True  False
               3         True  False
        2021   1         True  False
               2         True  False
               3         True  False
        2022   1         True  False
               2         True  False
               3         True  False

        """
        sns_ = as_index(self.n_save, sns, "snapshots")

        if getattr(self.n_save, "_multi_invest", False):
            active_assets_per_period = {
                period: self.get_active_assets(period)
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

        return mask

    @property
    def extendables(self) -> pd.Index:
        """Get the index of extendable components of this component type.

        Returns
        -------
        pd.Index
            Index of extendable components.

        """
        extendable_col = self._operational_attrs["nom_extendable"]
        if extendable_col not in self.static.columns:
            return self.static.iloc[:0].index

        idx = self.static.loc[self.static[extendable_col]].index

        return idx.rename(f"{self.name}-ext")

    @property
    def fixed(self) -> pd.Index:
        """Get the index of non-extendable components of this component type.

        Returns
        -------
        pd.Index
            Index of non-extendable components.

        """
        extendable_col = self._operational_attrs["nom_extendable"]
        if extendable_col not in self.static.columns:
            return self.static.iloc[:0].index

        idx = self.static.loc[~self.static[extendable_col]].index
        return idx.rename(f"{self.name}-fix")

    @property
    def committables(self) -> pd.Index:
        """Get the index of committable components of this component type.

        Returns
        -------
        pd.Index
            Index of committable components.

        """
        if "committable" not in self.static:
            return self.static.iloc[:0].index

        idx = self.static.loc[self.static["committable"]].index
        return idx.rename(f"{self.name}-com")
