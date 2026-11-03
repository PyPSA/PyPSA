# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Loads components module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import xarray as xr


@patch_add_docstring
class Loads(Components):
    """Loads components class.

    This class is used for load components. All functionality specific to
    loads is implemented here. Functionality for all components is implemented in
    the abstract base class.

    Loads are active components which mirror generators with the opposite sign.
    A load whose `p_set` is given for all optimized snapshots (static values are
    broadcast) and which uses no feature requiring a decision variable enters
    the optimization as a constant, as passive loads always did. All other
    loads receive a consumption variable `Load-p`. Linear marginal costs are
    applied to the power injected at the bus (`sign * p`), so a positive
    `marginal_cost` is the price a consumer pays for 1 MWh.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.loads
    'Load' Components
    -----------------
    Attached to PyPSA Network 'AC-DC-Meshed'
    Components: 6

    """

    _operational_variables = ["p"]
    _constant_fixed_operation = True
    _cost_follows_sign = True

    def get_bounds_pu(
        self,
        attr: str = "p",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for loads.

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "p"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = f"Bounds can only be retrieved for operational attributes. For loads those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        return self.da.p_min_pu, self.da.p_max_pu

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
