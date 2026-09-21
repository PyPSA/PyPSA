# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Global constraints components module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

from pypsa.components._types._flow_based_io import FlowBasedImportersMixin
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Column prefix under which a flow-based row stores its zonal PTDF, one column per zone.
PTDF_PREFIX = "ptdf_"


@patch_add_docstring
class GlobalConstraints(FlowBasedImportersMixin, Components):
    """Global constraints components class.

    This class is used for global constraint components. All functionality specific to
    global constraints is implemented here. Functionality for all components is implemented in
    the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.global_constraints
    'GlobalConstraint' Components
    -----------------------------
    Attached to PyPSA Network 'AC-DC-Meshed'
    Components: 1

    """

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

    def add_flow_based(
        self,
        zonal_ptdf: pd.DataFrame,
        ram: pd.Series | pd.DataFrame,
        *,
        investment_period: int | None = None,
        overwrite: bool = False,
    ) -> pd.Index | None:
        """Add a flow-based domain, one row ``zonal_ptdf . NP <= ram`` per CNEC.

        Each zone column is stored as the attribute ``ptdf_<zone>``, so this equals
        ``n.add("GlobalConstraint", cnecs, type="flow_based", sense="<=", constant=ram,
        **zonal_ptdf.add_prefix("ptdf_"))`` and also handles the time-varying layout.

        Parameters
        ----------
        zonal_ptdf : pandas.DataFrame
            Zonal PTDF, ``CNEC x zone``; columns are zone buses or corridor links. A
            ``(snapshot, CNEC)`` MultiIndex gives a time-varying domain.
        ram : pandas.Series or pandas.DataFrame
            Remaining available margin per CNEC, or ``snapshot x CNEC`` if time-varying.
        investment_period : int, optional
            Apply the rows only in this investment period (default: all).
        overwrite : bool, default False
            Replace rows that already exist.

        """
        df = zonal_ptdf.astype(float)
        names = df.index.get_level_values(-1).unique()
        if isinstance(df.index, pd.MultiIndex):
            cols = {z: df[z].unstack(-1).reindex(columns=names) for z in df}
        else:
            cols = {z: df[z] for z in df}
        return self.add(
            names,
            type="flow_based",
            sense="<=",
            investment_period=investment_period,
            constant=ram,
            overwrite=overwrite,
            **{PTDF_PREFIX + z: v for z, v in cols.items()},
        )

    @property
    def zonal_ptdf(self) -> pd.DataFrame:
        """Zonal PTDF of the flow-based rows, assembled from the ``ptdf_<zone>`` columns.

        For a static domain the index is the CNEC; if any zone column is time-varying it
        is a ``(snapshot, CNEC)`` MultiIndex, so ``c.zonal_ptdf.loc[sns]`` selects one
        snapshot. The xarray view is ``c.da.zonal_ptdf``.
        """
        da = self.da.zonal_ptdf
        if "snapshot" in da.dims:
            return (
                da.stack(row=("snapshot", "name")).transpose("row", "bus").to_pandas()
            )
        return da.to_pandas()

    def _as_xarray(self, attr: str) -> xr.DataArray:
        """Expose ``zonal_ptdf`` as a (name, bus) or (snapshot, name, bus) DataArray.

        Static and time-varying zone columns broadcast against each other; a zone missing
        for a row has sensitivity zero.
        """
        if attr != "zonal_ptdf":
            return super()._as_xarray(attr)
        rows = self.static.index[self.static["type"] == "flow_based"]
        attrs = [*self.static.columns, *self.dynamic]
        cols = [a for a in attrs if a.startswith(PTDF_PREFIX)]
        if not cols:
            return xr.DataArray(
                pd.DataFrame(index=rows, columns=pd.Index([], name="bus"))
            )
        arrays = {c.removeprefix(PTDF_PREFIX): self.da[c].sel(name=rows) for c in cols}
        da = xr.Dataset(arrays).to_dataarray("bus").fillna(0.0)
        da.name = "zonal_ptdf"
        return da.transpose(..., "name", "bus")
