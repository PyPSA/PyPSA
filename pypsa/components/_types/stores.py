# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Stores components module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


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

    _operational_variables = ["e", "p", "p_store"]

    def get_bounds_pu(
        self,
        attr: str = "e",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for stores.

        <!-- md:badge-version v1.0.0 -->

        Power bounds are given per unit of `e_nom`, i.e. `p_max_pu / max_hours`
        for the discharge `p + p_store` and `-p_min_pu / max_hours` for the
        charge `p_store`. They are infinite where `max_hours` is infinite. For
        stores without a [split dispatch][pypsa.components.Stores.split_dispatch],
        `p_store` is not defined, its bounds are infinite and the charging
        bound applies to `p`.

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "e", "p", "p_store"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = f"Bounds can only be retrieved for operational attributes. For stores those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        if attr == "e":
            return self.da.e_min_pu, self.da.e_max_pu

        finite = np.isfinite(self.da.max_hours)
        store_pu = (-self.da.p_min_pu / self.da.max_hours).where(finite, np.inf)
        dispatch_pu = (self.da.p_max_pu / self.da.max_hours).where(finite, np.inf)
        split = self.split_dispatch()

        if attr == "p":
            return -store_pu.where(~split, 0), dispatch_pu

        lower = xr.zeros_like(store_pu).where(split, -np.inf)
        upper = store_pu.where(split, np.inf)
        return lower, upper

    def split_dispatch(self) -> xr.DataArray:
        """Get which stores need separate charging and discharging variables.

        The net dispatch `p` is sufficient unless the efficiencies differ from
        one or a cost applies to one direction only. Adding the charging
        variable `p_store` only where needed avoids degenerate variable pairs
        and speeds up solving.

        Returns
        -------
        xr.DataArray
            Boolean per store.

        """
        split = (
            (self.da.efficiency_store != 1)
            | (self.da.efficiency_dispatch != 1)
            | (self.da.marginal_cost_dispatch != 0)
            | (self.da.marginal_cost_store != 0)
        )
        return split.any("snapshot") if "snapshot" in split.dims else split

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
