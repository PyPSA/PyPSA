# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Storage units components module."""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components
from pypsa.constants import HOURS_PER_YEAR

logger = logging.getLogger(__name__)


@patch_add_docstring
class StorageUnits(Components):
    """StorageUnits components class.

    This class is used for storage unit components. All functionality specific to
    storage units is implemented here. Functionality for all components is implemented
    in the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.storage_units
    Empty 'StorageUnit' Components

    """

    _operational_variables = ["p_dispatch", "p_store", "state_of_charge"]

    def get_bounds_pu(
        self,
        attr: str = "p_store",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for storage units.

        <!-- md:badge-version v1.0.0 -->

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "p", "p_store", "state_of_charge"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = f"Bounds can only be retrieved for operational attributes. For storage_units those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        max_pu = self.da.p_max_pu

        if attr == "p_store":
            max_pu = -self.da.p_min_pu
            min_pu = xr.zeros_like(max_pu)
        elif attr == "state_of_charge":
            max_pu = self.da.max_hours
            min_pu = xr.zeros_like(max_pu)
        else:
            max_pu = self.da.p_max_pu
            min_pu = xr.zeros_like(max_pu)

        return min_pu, max_pu

    def get_cycles(self) -> pd.Series:
        """Get the equivalent full cycles performed over the snapshots.

        The energy throughput of the state of charge from the solved dispatch,
        i.e. the energy charged into it after `efficiency_store` plus the
        energy discharged from it before `efficiency_dispatch`, weighted by
        `snapshot_weightings.stores`, divided by `2 * max_hours * p_nom_opt`.
        Throughput before the horizon (`throughput_initial`) is not counted.

        Returns
        -------
        pd.Series
            Equivalent full cycles per storage unit. NaN where `p_nom_opt` is
            zero, e.g. before the optimisation or for extendable units that
            were not built.

        Examples
        --------
        >>> n.c.storage_units.get_cycles()  # doctest: +SKIP

        """
        n = self.n_save
        weights = n.snapshot_weightings.stores
        eff_store = n.get_switchable_as_dense(self.name, "efficiency_store")
        eff_dispatch = n.get_switchable_as_dense(self.name, "efficiency_dispatch")
        flow = self.dynamic.p_store * eff_store + self.dynamic.p_dispatch / eff_dispatch
        throughput = flow.mul(weights, axis=0).sum()
        return throughput / (2 * self.static.max_hours * self.static.p_nom_opt)

    def set_cycle_life(
        self,
        cycles_life: float,
        soh_end: float = 0.8,
        names: Sequence | None = None,
    ) -> None:
        """Set `degradation_per_cycle` and `cycles_max` from a cycle life.

        Sets `degradation_per_cycle = (1 - soh_end) / cycles_life` and
        `cycles_max = cycles_life * horizon_years / lifetime`, the share of the
        cycle life that fits into the optimised snapshots without shortening
        the technical `lifetime`. The horizon is measured with the `stores`
        snapshot weighting, which the cycle budget sums over (`n.nyears` uses
        the `objective` weighting instead); with investment periods it covers
        the whole horizon. Storage units with an infinite `lifetime` get an
        infinite `cycles_max`; set it directly for those.

        Parameters
        ----------
        cycles_life : float
            Equivalent full cycles until the state of health reaches `soh_end`.
        soh_end : float, default 0.8
            State of health at the end of the cycle life, as a fraction of
            the nominal energy capacity.
        names : Sequence | None, default None
            Storage units to set; all storage units by default.

        Examples
        --------
        >>> n.c.storage_units.set_cycle_life(6000, soh_end=0.8)  # doctest: +SKIP

        """
        index = self.static.index
        if names is not None:
            unknown = pd.Index(names).difference(self.names)
            if len(unknown) > 0:
                msg = f"Cannot set cycle life of unknown storage units: {list(unknown)}"
                raise ValueError(msg)
            index = index[index.get_level_values("name").isin(names)]
        lifetime = self.static.loc[index, "lifetime"]
        horizon_years = self.n_save.snapshot_weightings.stores.sum() / HOURS_PER_YEAR

        self.static.loc[index, "degradation_per_cycle"] = (1 - soh_end) / cycles_life
        self.static.loc[index, "cycles_max"] = (
            cycles_life * horizon_years / lifetime
        ).where(np.isfinite(lifetime), np.inf)

        if not np.isfinite(lifetime).all():
            logger.warning(
                "StorageUnits %s: lifetime is infinite, cycles_max is set to "
                "infinity. Set it directly for a cycle budget.",
                lifetime.index[~np.isfinite(lifetime)].tolist(),
            )

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str | Sequence[str] = "",
        overwrite: bool = False,
        return_names: bool | None = None,
        **kwargs: Any,
    ) -> pd.Index | None:
        """Wrap Components.add() and docstring is patched via decorator."""
        return super().add(
            name=name,
            suffix=suffix,
            overwrite=overwrite,
            return_names=return_names,
            **kwargs,
        )
