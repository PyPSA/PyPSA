"""Network descriptors module.

Contains single mixin class which is used to inherit to [pypsa.Networks] class.
Should not be used directly.

Descriptor functions only describe data and do not modify it.

"""

from __future__ import annotations

import logging
from itertools import repeat
from typing import TYPE_CHECKING

import pandas as pd

from pypsa.common import as_index, deprecated_in_next_major
from pypsa.network.abstract import _NetworkABC

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class NetworkDescriptorsMixin(_NetworkABC):
    """Mixin class for network descriptor methods.

    Class only inherits to [pypsa.Network][] and should not be used directly.
    All attributes and methods can be used within any Network instance.
    """

    @deprecated_in_next_major(
        details="Use `n.components[c].extendables` instead.",
    )
    def get_extendable_i(self, c: str) -> pd.Index:
        """Getter function.

        Get the index of extendable elements of a given component.

        Deprecated: Use n.components[c].get_extendable_i() instead.
        """
        return self.components[c].extendables

    @deprecated_in_next_major(details="Use `n.components[c].fixed` instead.")
    def get_non_extendable_i(self, c: str) -> pd.Index:
        """Getter function.

        Get the index of non-extendable elements of a given component.

        Deprecated: Use n.components[c].self.fixed instead.
        """
        return self.components[c].fixed

    @deprecated_in_next_major(details="Use `n.components[c].committables` instead.")
    def get_committable_i(self, c: str) -> pd.Index:
        """Getter function.

        Get the index of commitable elements of a given component.

        Deprecated: Use n.components[c].get_committable_i() instead.
        """
        return self.components[c].committables

    @deprecated_in_next_major(
        details="Use `n.components[c].get_active_assets` instead."
    )
    def get_active_assets(
        self,
        c: str,
        investment_period: int | str | Sequence | None = None,
    ) -> pd.Series:
        """Get active components mask of component type in investment period(s).

        See the :py:meth:`pypsa.descriptors.components.Component.get_active_assets`.

        Parameters
        ----------
        c : string
            Component name
        investment_period : int, str, Sequence
            Investment period(s) to check

        Returns
        -------
        pd.Series
            Boolean mask for active components

        """
        return self.components[c].get_active_assets(investment_period=investment_period)

    def get_switchable_as_dense(
        self,
        component: str,
        attr: str,
        snapshots: Sequence | None = None,
        inds: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Return a Dataframe for a time-varying component attribute .

        Values for all non-time-varying components are filled in with the default
        values for the attribute.

        Parameters
        ----------
        component : string
            Component object name, e.g. 'Generator' or 'Link'
        attr : string
            Attribute name
        snapshots : pandas.Index
            Restrict to these snapshots rather than n.snapshots.
        inds : pandas.Index
            Restrict to these components rather than n.components.index

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> n.get_switchable_as_dense('Generator', 'p_max_pu', n.snapshots[:2]) # doctest: +SKIP
        Generator            Manchester Wind  Manchester Gas  Norway Wind  Norway Gas  Frankfurt Wind  Frankfurt Gas
        snapshot
        2015-01-01 00:00:00         0.930020             1.0     0.974583         1.0        0.559078            1.0
        2015-01-01 01:00:00         0.485748             1.0     0.481290         1.0        0.752910            1.0

        """
        sns = as_index(self, snapshots, "snapshots")

        static = self.static(component)[attr]
        empty = pd.DataFrame(index=sns)
        dynamic = self.dynamic(component).get(attr, empty).loc[sns]

        index = static.index
        if inds is not None:
            index = index.intersection(inds)

        diff = index.difference(dynamic.columns)
        static_to_dynamic = pd.DataFrame({**static[diff]}, index=sns)
        res = pd.concat([dynamic, static_to_dynamic], axis=1, names=sns.names)[index]
        res.index.name = sns.name
        res.columns.name = component
        return res

    def get_switchable_as_iter(
        self,
        component: str,
        attr: str,
        snapshots: Sequence,
        inds: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Return an iterator over snapshots for a time-varying component attribute.

        Values for all non-time-varying components are filled in with the default
        values for the attribute.

        Parameters
        ----------
        component : string
            Component object name, e.g. 'Generator' or 'Link'
        attr : string
            Attribute name
        snapshots : pandas.Index
            Restrict to these snapshots rather than n.snapshots.
        inds : pandas.Index
            Restrict to these items rather than all of n.{generators, ..}.index

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> gen = n.get_switchable_as_iter('Generator', 'p_max_pu', n.snapshots[:2])
        >>> next(gen)  # doctest: +ELLIPSIS
        Generator
        Manchester Wind    0.930020
        Manchester Gas     1.000000
        Norway Wind        0.974583
        Norway Gas         1.000000
        Frankfurt Wind     0.559078
        Frankfurt Gas      1.000000
        dtype: float64

        """
        static = self.static(component)
        dynamic = self.dynamic(component)

        index = static.index
        varying_i = dynamic[attr].columns
        fixed_i = static.index.difference(varying_i)

        if inds is not None:
            inds = pd.Index(inds)
            index = inds.intersection(index)
            varying_i = inds.intersection(varying_i)
            fixed_i = inds.intersection(fixed_i)

        # Short-circuit only fixed
        if len(varying_i) == 0:
            return repeat(static.loc[fixed_i, attr], len(snapshots))

        def is_same_indices(i1: pd.Index, i2: pd.Index) -> bool:
            return len(i1) == len(i2) and (i1 == i2).all()

        if is_same_indices(fixed_i.append(varying_i), index):

            def reindex_maybe(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
                return s

        else:

            def reindex_maybe(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
                return s.reindex(index)

        return (
            reindex_maybe(
                pd.concat(
                    [static.loc[fixed_i, attr], dynamic[attr].loc[sn, varying_i]],
                    axis=0,
                )
            )
            for sn in snapshots
        )

    def bus_carrier_unit(self, bus_carrier: str | Sequence[str] | None) -> str:
        """Determine the unit associated with a specific bus carrier in the network.

        Parameters
        ----------
        bus_carrier : str | Sequence[str] | None
            The carrier type of the bus to query.

        Returns
        -------
        str:
            The unit associated with the specified bus carrier. If no bus carrier is
            provided, returns `"carrier dependent"`.

        Raises
        ------
        ValueError:
            If the specified bus carrier is not found in the network or if multiple
            units are found for the specified bus carrier.

        """
        if bus_carrier is None:
            return "carrier dependent"

        if isinstance(bus_carrier, str):
            bus_carrier = [bus_carrier]

        not_included = set(bus_carrier) - set(self.c.buses.static.carrier.unique())
        if not_included:
            msg = f"Bus carriers {not_included} not in network"
            raise ValueError(msg)
        unit = self.c.buses.static[
            self.c.buses.static.carrier.isin(bus_carrier)
        ].unit.unique()
        if len(unit) > 1:
            logger.warning("Multiple units found for carrier %s: %s", bus_carrier, unit)
            return "carrier dependent"
        return unit.item()
