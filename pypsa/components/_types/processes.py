# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Processes components module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components._types.multiports import Multiport
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import xarray as xr


@patch_add_docstring
class Processes(Multiport, Components):
    """Processes components class.

    This class is used for process components. All functionality specific to
    processes is implemented here. Functionality for all components is implemented in
    the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.processes
    Empty 'Process' Components

    """

    _operational_variables = ["p"]

    @property
    def _coefficient_attr(self) -> str:
        return "rate"

    def get_bounds_pu(
        self,
        attr: str = "p",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for processes.

        <!-- md:badge-version v1.0.0 -->

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
            msg = f"Bounds can only be retrieved for operational attributes. For processes those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        return self.da.p_min_pu, self.da.p_max_pu

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str = "",
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
