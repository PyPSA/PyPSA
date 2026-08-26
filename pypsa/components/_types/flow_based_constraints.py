# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Flow-based constraint components module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

from pypsa.components._types._flow_based_io import FlowBasedImportersMixin
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network
    from pypsa.components.types import ComponentType


@patch_add_docstring
class FlowBasedConstraints(FlowBasedImportersMixin, Components):
    """Flow-based market-coupling constraint components.

    A non-physical component holding a flow-based domain: linear constraints on the net
    positions of the market zones (buses), ``zonal_ptdf . NP <= RAM``, one entity per
    critical network element (CNEC). The zonal PTDF sensitivities form a matrix and are
    stored in the dedicated frame ``c.zonal_ptdf`` (rows = CNECs, columns = zone buses),
    analogous to ``c.piecewise``. Pass it to ``add`` as ``zonal_ptdf``; a time-varying
    domain passes a ``(snapshot, cnec)`` MultiIndex frame instead.

    See Also
    --------
    [pypsa.Components][]

    """

    frame_attrs: tuple[str, ...] = ("zonal_ptdf",)

    def __init__(
        self,
        ctype: ComponentType,
        n: Network | None = None,
        names: str | int | Sequence[int | str] | None = None,
        suffix: str = "",
    ) -> None:
        """Initialise the component and its (empty) zonal PTDF frame."""
        super().__init__(ctype=ctype, n=n, names=names, suffix=suffix)
        self._zonal_ptdf = pd.DataFrame()

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

    @property
    def zonal_ptdf(self) -> pd.DataFrame:
        """Zonal PTDF sensitivities (one column per zone bus).

        The stored frame itself, so in-place edits write through. For a static domain the
        index is the CNEC; for a time-varying one it is a ``(snapshot, CNEC)`` MultiIndex,
        so ``c.zonal_ptdf.loc[sns]`` selects one snapshot. The internal xarray view is
        ``c.da.zonal_ptdf``.
        """
        return self._zonal_ptdf

    def _set_frame(self, attr: str, value: Any, names: pd.Index) -> None:
        """Store the ``zonal_ptdf`` matrix (a Series, a cnec x zone frame, or a MultiIndex one).

        Rows already present are overwritten; missing zone entries default to zero. A domain
        is static or time-varying as a whole; mixing the two raises.
        """
        if attr != "zonal_ptdf":
            super()._set_frame(attr, value, names)
            return
        if isinstance(value, pd.DataFrame) and isinstance(value.index, pd.MultiIndex):
            df = self._time_varying_frame(value)
        elif isinstance(value, pd.Series):
            df = value.to_frame(names[0]).T.rename_axis(index="name", columns="bus")
        else:
            df = (
                pd.DataFrame(value)
                .reindex(names)
                .rename_axis(index="name", columns="bus")
            )
        df = df.astype(float)
        existing = self._zonal_ptdf
        if not existing.empty and existing.index.nlevels != df.index.nlevels:
            msg = "Cannot mix static and time-varying zonal PTDF rows in one domain."
            raise ValueError(msg)
        keep = existing.drop(index=df.index, errors="ignore")
        self._zonal_ptdf = pd.concat([keep, df]).fillna(0.0)

    def _time_varying_frame(self, value: pd.DataFrame) -> pd.DataFrame:
        """Validate and label a time-varying ``(snapshot, cnec) x zone`` frame."""
        if not value.index.get_level_values(0).isin(self.n_save.snapshots).all():
            msg = (
                "Time-varying zonal_ptdf must be indexed by (snapshot, CNEC); its outer "
                "index level must be network snapshots."
            )
            raise ValueError(msg)
        return value.rename_axis(index=["snapshot", "name"], columns="bus")

    def _as_xarray(self, attr: str) -> xr.DataArray:
        """Expose ``zonal_ptdf`` as a (name, bus) or (snapshot, name, bus) DataArray."""
        if attr != "zonal_ptdf":
            return super()._as_xarray(attr)
        z = self._zonal_ptdf
        if isinstance(z.index, pd.MultiIndex):
            da = z.stack(future_stack=True).to_xarray()
        else:
            da = xr.DataArray(z.rename_axis(index="name", columns="bus"))
        da.name = "zonal_ptdf"
        return da
