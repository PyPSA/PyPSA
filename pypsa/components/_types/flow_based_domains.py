# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Flow-based domain components module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa.components.types import ComponentType


@patch_add_docstring
class FlowBasedDomains(Components):
    """Flow-based domain components class.

    A non-physical component holding a flow-based market-coupling domain: a set of
    linear constraints on the net positions of the market zones (buses), of the form
    ``PTDF . NP <= RAM``. Each entity is one critical network element (CNEC).

    Unlike the scalar attributes (``ram``, ``mu``, ...), the zonal PTDF sensitivities
    form a matrix (cnec x zone) and are stored in a dedicated frame ``c.ptdf`` (rows =
    CNECs, columns = zone buses), analogous to ``c.piecewise``. Pass it directly to
    ``add`` via the ``ptdf`` argument; read it back as a pandas DataFrame from ``c.ptdf``
    or as an xarray DataArray from ``c.da.ptdf``.

    See Also
    --------
    [pypsa.Components][]

    """

    frame_attrs: tuple[str, ...] = ("ptdf",)

    def __init__(
        self,
        ctype: ComponentType,
        n: Any = None,
        names: Any = None,
        suffix: str = "",
    ) -> None:
        """Initialise the component and its (empty) PTDF frame."""
        super().__init__(ctype=ctype, n=n, names=names, suffix=suffix)
        self._ptdf = pd.DataFrame()

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
    def ptdf(self) -> pd.DataFrame:
        """Zonal PTDF sensitivities of the domain.

        Returns
        -------
        pandas.DataFrame
            Power transfer distribution factors with one row per CNEC (the component
            index) and one column per zone bus. This is the stored frame itself, so
            in-place edits write through; assign a new frame or use ``add`` to replace
            it. The xarray view used internally by the optimisation is ``c.da.ptdf``.

        """
        return self._ptdf

    def _set_frame(self, attr: str, value: Any, names: pd.Index) -> None:
        """Store a matrix-valued attribute (currently only ``ptdf``).

        ``value`` is a Series over zones (single CNEC) or a DataFrame (cnec x zone).
        Rows already present are overwritten; missing zone entries default to zero.
        """
        if attr != "ptdf":
            super()._set_frame(attr, value, names)
            return
        if isinstance(value, pd.Series):
            df = value.to_frame(names[0]).T
        else:
            df = pd.DataFrame(value).reindex(names)
        df = df.rename_axis(index="name", columns="bus").astype(float)
        keep = self._ptdf.drop(index=df.index, errors="ignore")
        self._ptdf = pd.concat([keep, df]).fillna(0.0)

    def _as_xarray(self, attr: str) -> xr.DataArray:
        """Expose ``ptdf`` as a (name, bus) DataArray; defer to the base otherwise."""
        if attr == "ptdf":
            da = xr.DataArray(self._ptdf.rename_axis(index="name", columns="bus"))
            da.name = "ptdf"
            return da
        return super()._as_xarray(attr)
