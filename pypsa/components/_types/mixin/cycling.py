# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Cycle life mixin for storage components."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pypsa.components.components import Components
from pypsa.constants import HOURS_PER_YEAR

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class _Cycling(Components):
    """Shared cycle life functionality of storage units and stores.

    See Also
    --------
    [pypsa.Components][]

    """

    @abstractmethod
    def _throughput_flow(self) -> pd.DataFrame:
        """Get the energy passing through the storage level per snapshot in MW."""

    @abstractmethod
    def _energy_capacity(self, nom_attr: str) -> pd.Series:
        """Get the nominal energy capacity in MWh from a nominal power or energy attribute."""

    def get_cycles(self) -> pd.Series:
        """Get the equivalent full cycles performed over the snapshots.

        One cycle is twice the nominal energy capacity of energy passing through
        the storage level, weighted by `snapshot_weightings.stores`. Throughput
        before the horizon (`throughput_initial`) is not counted.

        Returns
        -------
        pd.Series
            Equivalent full cycles per asset. NaN where the optimised capacity is
            zero, e.g. before the optimisation or for unbuilt extendable assets.

        Examples
        --------
        >>> n.c.storage_units.get_cycles()  # doctest: +SKIP

        """
        weights = self.n_save.snapshot_weightings.stores
        throughput = self._throughput_flow().mul(weights, axis=0).sum()
        nom_attr = self._operational_attrs["nom"]
        return throughput / (2 * self._energy_capacity(f"{nom_attr}_opt"))

    def set_cycle_life(
        self,
        cycles_life: float,
        soh_end: float = 0.8,
        names: Sequence | None = None,
    ) -> None:
        """Set `degradation_per_cycle` and `cycles_max` from a cycle life.

        Sets `degradation_per_cycle = (1 - soh_end) / cycles_life` and
        `cycles_max = cycles_life * horizon_years / lifetime` plus the cycles
        already in `throughput_initial`, the share of the cycle life that fits
        into the snapshots without shortening the technical `lifetime`. The
        horizon is measured with the `stores` snapshot weighting over all
        snapshots of the network, so the budget covers every rolling-horizon
        window together. Assets with an infinite `lifetime` keep an infinite
        `cycles_max`. This assumes `lifetime` is at least the horizon; a shorter
        lifetime over-budgets a single row, which should instead be modelled as
        separate rebuilds.

        Parameters
        ----------
        cycles_life : float
            Equivalent full cycles until the state of health reaches `soh_end`.
        soh_end : float, default 0.8
            State of health at the end of the cycle life, as a fraction of the
            nominal energy capacity.
        names : Sequence | None, default None
            Assets to set; all assets by default.

        Examples
        --------
        >>> n.c.storage_units.set_cycle_life(6000, soh_end=0.8)  # doctest: +SKIP

        """
        selected: pd.Index = self.names if names is None else pd.Index(names)
        unknown = selected.difference(self.names)
        if not unknown.empty:
            msg = f"Cannot set cycle life of unknown {self.list_name}: {list(unknown)}"
            raise ValueError(msg)

        rows = self.static.index.get_level_values("name").isin(selected)
        lifetime = self.static.loc[rows, "lifetime"]
        horizon_years = self.n_save.snapshot_weightings.stores.sum() / HOURS_PER_YEAR
        throughput_initial = self.static.loc[rows, "throughput_initial"]
        capacity = self._energy_capacity(self._operational_attrs["nom"]).loc[rows]
        prior_cycles = (throughput_initial / (2 * capacity)).where(
            throughput_initial > 0, 0
        )
        cycles_max = prior_cycles + cycles_life * horizon_years / lifetime

        self.static.loc[rows, "degradation_per_cycle"] = (1 - soh_end) / cycles_life
        self.static.loc[rows, "cycles_max"] = cycles_max.where(
            lifetime < np.inf, np.inf
        )

        infinite = lifetime.index[lifetime == np.inf]
        if not infinite.empty:
            logger.warning(
                "%s %s: lifetime is infinite, cycles_max stays infinite. "
                "Set it directly for a cycle budget.",
                self.name,
                infinite.tolist(),
            )
