# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Loads components module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import pandas as pd
    import xarray as xr


@patch_add_docstring
class Loads(Components):
    """Loads components class.

    This class is used for load components. All functionality specific to
    loads is implemented here. Functionality for all components is implemented in
    the abstract base class.

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

    @property
    def dispatchable(self) -> pd.Index:
        """Loads dispatched as variables, i.e. with any NaN ``p_set`` entry.

        These receive a ``Load-p`` dispatch variable, while the remaining
        [passive][pypsa.components.Loads.passive] loads enter the nodal balance as
        a constant.
        """
        names = self.names
        if names.empty:
            return names

        # A load is dispatchable if its static p_set is NaN, or (where a p_set
        # time series is given) any snapshot value is NaN.
        dynamic = self.dynamic["p_set"]
        has_nan = self.static["p_set"].isnull()
        has_nan.loc[dynamic.columns] = dynamic.isnull().any()
        has_nan = has_nan.groupby(level="name").any()
        return names[has_nan.reindex(names, fill_value=False)]

    @property
    def passive(self) -> pd.Index:
        """Passive loads (fully set ``p_set``) that enter the balance as a constant.

        The complement of [dispatchable][pypsa.components.Loads.dispatchable].
        """
        return self.names.difference(self.dispatchable)

    @property
    def extendables(self) -> pd.Index:
        """Loads are never extendable."""
        return self.static.iloc[:0].index

    @property
    def fixed(self) -> pd.Index:
        """Non-extendable loads with a dispatch variable (all dispatchable loads)."""
        return self.dispatchable

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
