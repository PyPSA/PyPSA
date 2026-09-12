# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Stores components module."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components
from pypsa.constants import HOURS_PER_YEAR

if TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr

logger = logging.getLogger(__name__)


@patch_add_docstring
class Stores(Components):
    """Stores components class.

    This class is used for store components. All functionality specific to
    stores is implemented here. Functionality for all components is implemented in
    the abstract base class.

    Examples
    --------
    >>> n.components.stores
    Empty 'Store' Components

    See Also
    --------
    [pypsa.Components][]

    """

    _operational_variables = ["e"]

    def get_bounds_pu(
        self,
        attr: str = "e",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for stores.

        <!-- md:badge-version v1.0.0 -->

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "e"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = f"Bounds can only be retrieved for operational attributes. For stores those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        return self.da.e_min_pu, self.da.e_max_pu

    def get_cycles(self) -> pd.Series:
        """Get the equivalent full cycles performed over the snapshots.

        The energy throughput of the energy level from the solved dispatch,
        i.e. `|p|` weighted by `snapshot_weightings.stores`, divided by
        `2 * e_nom_opt`. Throughput before the horizon (`throughput_initial`)
        is not counted.

        Returns
        -------
        pd.Series
            Equivalent full cycles per store. NaN before the optimisation and
            for extendable stores that were not built.

        Examples
        --------
        >>> n.c.stores.get_cycles()  # doctest: +SKIP

        """
        weights = self.n_save.snapshot_weightings.stores
        throughput = self.dynamic.p.abs().mul(weights, axis=0).sum()
        return throughput / (2 * self.static.e_nom_opt)

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
        the whole horizon. Stores with an infinite `lifetime` get an infinite
        `cycles_max`; set it directly for those.

        Parameters
        ----------
        cycles_life : float
            Equivalent full cycles until the state of health reaches `soh_end`.
        soh_end : float, default 0.8
            State of health at the end of the cycle life, as a fraction of
            the nominal energy capacity.
        names : Sequence | None, default None
            Stores to set; all stores by default.

        Examples
        --------
        >>> n.c.stores.set_cycle_life(6000, soh_end=0.8)  # doctest: +SKIP

        """
        index = self.static.index
        if names is not None:
            unknown = pd.Index(names).difference(self.names)
            if len(unknown) > 0:
                msg = f"Cannot set cycle life of unknown stores: {list(unknown)}"
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
                "Stores %s: lifetime is infinite, cycles_max is set to infinity. "
                "Set it directly for a cycle budget.",
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
