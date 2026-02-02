# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Components descriptor module.

Contains single mixin class which is used to inherit to [pypsa.Components][] class.
Should not be used directly.

Descriptor functions only describe data and do not modify it.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pypsa.common import as_index
from pypsa.components.abstract import _ComponentsABC

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class ComponentsDescriptorsMixin(_ComponentsABC):
    """Mixin class for components descriptors methods.

    Class inherits to [pypsa.Components][]. All attributes and methods can be used
    within any Components instance.

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
        # TODO if we expose this, this needs further refinment and checks, specially
        # because of edge case StorageUnit with multiple operational variables

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
            "nom_mod": f"{base}_nom_mod",
            "min_pu": f"{base}_min_pu",
            "max_pu": f"{base}_max_pu",
            "set": f"{base}_set",
        }

    def get_active_assets(
        self,
        investment_period: int | str | Sequence | None = None,
    ) -> pd.Series:
        """Get active components mask of component type in investment period(s).

        <!-- md:badge-version v0.33.0 -->

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
        >>> n.add("Generator", "g2", active=True)
        >>> n.components.generators.get_active_assets()
        name
        g1    False
        g2     True
        Name: active, dtype: bool

        With investment periods
        >>> n = pypsa.Network()
        >>> n.snapshots = pd.MultiIndex.from_product([[2020, 2021, 2022], ["1", "2", "3"]])
        >>> n.add("Generator", "g1", build_year=2020, lifetime=1)
        >>> n.add("Generator", "g2", active=False)
        >>> n.components.generators.get_active_assets()
        name
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

    @property
    def active_assets(self) -> pd.Series:
        """Get list of active assets.

        <!-- md:badge-version v1.0.0 -->

        See corresponding [pypsa.Components.inactive_assets][] for details.

        Returns
        -------
        pd.Series
            List of inactive assets

        See Also
        --------
        [pypsa.Components.inactive_assets][]

        """
        active_assets = self.get_active_assets()
        return active_assets[active_assets].index.get_level_values("name").unique()

    @property
    def inactive_assets(self) -> pd.Series:
        """Get list of inactive assets.

        <!-- md:badge-version v1.0.0 -->

        An asset is considered inactive when one of the following conditions is met:
        - `active` is set to False across all dimensions (investment periods, scenarios)
        - `build_year` + `lifetime` never satisfies the condition for any investment period

        Inactive assets are not considered in the optimization and are excluded from
        the model entirely.

        Returns
        -------
        pd.Series
            List of inactive assets

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.snapshots = pd.MultiIndex.from_product([[2020, 2021, 2022], ["1", "2", "3"]])
        >>> n.add("Generator", "g1", build_year=2020, lifetime=1)
        >>> n.add("Generator", "g2", active=False)

        List all components
        >>> n.generators.index
        Index(['g1', 'g2'], dtype='object', name='name')

        List of inactive components
        'g1' will be considered as active because it is active in at least one investment period
        >>> n.components.generators.inactive_assets
        Index(['g2'], dtype='object', name='name')

        List of active components
        >>> n.components.generators.active_assets
        Index(['g1'], dtype='object', name='name')

        `c.active_assets` and `c.inactive_assets` are mutually exclusive

        See Also
        --------
        [pypsa.Components.get_active_assets][]

        """
        active_assets = self.get_active_assets()
        return active_assets[~active_assets].index.get_level_values("name").unique()

    def filter_by_active_assets(
        self,
        data: pd.DataFrame | pd.Index,
        investment_period: int | float | Sequence | None = None,  # noqa: PYI041
    ) -> pd.DataFrame | pd.Index:
        """Filter DataFrame or Index to only include active assets.

        Parameters
        ----------
        data : pd.DataFrame | pd.Index
            DataFrame or Index to filter. Must have a "name" level in its index.
        investment_period : int | float | Sequence, optional
            If provided, additionally filter by assets active in this
            specific investment period(s). If a sequence is given, assets
            active in any of the periods are included. NaN values are
            treated as None (no period filtering).

        Returns
        -------
        pd.DataFrame | pd.Index
            Filtered DataFrame or Index with only active assets.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", "bus")
        >>> n.add("Generator", "g1", bus="bus")
        >>> n.add("Generator", "g2", bus="bus", active=False)
        >>> df = n.generators[['p_nom', 'bus']]
        >>> n.components.generators.filter_by_active_assets(df)
           p_nom  bus
        name
        g1     0.0  bus

        """
        # Normalize investment_period: NaN -> None, float -> int
        if investment_period is not None and isinstance(investment_period, float):
            if np.isnan(investment_period):
                investment_period = None
            else:
                investment_period = int(investment_period)

        if isinstance(data, pd.Index):
            if investment_period is not None:
                active = self.get_active_assets(investment_period)
                return data[active.reindex(data, fill_value=False)]
            return data.intersection(self.active_assets)

        names = data.index.get_level_values("name").unique()
        if investment_period is not None:
            active = self.get_active_assets(investment_period)
            names = names[active.reindex(names, fill_value=False)]
        else:
            names = names.intersection(self.active_assets)

        return data[data.index.get_level_values("name").isin(names)]

    def get_activity_mask(
        self,
        sns: Sequence | None = None,
        index: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Get active components mask indexed by snapshots.

        <!-- md:badge-version v0.33.0 -->

        Gets the boolean mask for active components, indexed by snapshots and
        components instead of just components.

        Parameters
        ----------
        sns : pandas.Index, default None
            Set of snapshots for the mask. If None (default) all snapshots are returned.
        index : pd.Index, default None
            Subset of the component elements. If None (default) all components are returned.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.snapshots = pd.MultiIndex.from_product([[2020, 2021, 2022], ["1", "2", "3"]])
        >>> n.add("Generator", "g1", build_year=2020, lifetime=1)
        >>> n.add("Generator", "g2", active=False)
        >>> n.components.generators.get_activity_mask()  # doctest: +ELLIPSIS
        name                g1     g2
        period timestep
        2020   1          True  False
               2          True  False
               3          True  False
        2021   1         False  False
               2         False  False
               3         False  False
        2022   1         False  False
               2         False  False
               3         False  False

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
