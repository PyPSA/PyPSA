# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Array module of the PyPSA network.

Exposes network-level (i.e. non-component) time series as xarray DataArrays on the
labels of the live optimization model, mirroring the component accessor
[pypsa.Components.da][].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import xarray as xr

from pypsa.network.abstract import _NetworkABC

if TYPE_CHECKING:
    import pandas as pd

    from pypsa.optimization.window import SnapshotWindow


class _FrameArrayAccessor:
    """Columns of a snapshot-indexed network frame as model-aligned DataArrays."""

    __slots__ = ("_frame", "_window")

    def __init__(self, frame: pd.DataFrame, window: SnapshotWindow | None) -> None:
        self._frame = frame
        self._window = window

    def _get_array(self, attr: str) -> xr.DataArray:
        res = xr.DataArray(self._frame[attr])
        return res if self._window is None else self._window.flatten(res)

    def __getattr__(self, attr: str) -> xr.DataArray:
        """Access a column as a DataArray via dot notation."""
        return self._get_array(attr)

    def __getitem__(self, attr: str) -> xr.DataArray:
        """Access a column as a DataArray via bracket notation."""
        return self._get_array(attr)

    def __iter__(self) -> NoReturn:
        """Raise a clear error, as XarrayAccessor objects are not iterable."""
        msg = "XarrayAccessor objects are not iterable."
        raise TypeError(msg)

    def __dir__(self) -> list[str]:
        """List available columns for tab-completion."""
        return sorted(self._frame.columns)


class _NetworkXarrayAccessor:
    """Accessor providing xarray access to the network's own time series."""

    __slots__ = ("_n",)

    def __init__(self, n: _NetworkABC) -> None:
        self._n = n

    @property
    def snapshot_weightings(self) -> _FrameArrayAccessor:
        """Snapshot weightings on the model's snapshot labels."""
        return _FrameArrayAccessor(
            self._n.snapshot_weightings, self._n._snapshot_window
        )

    def __repr__(self) -> str:
        """Get representation of the xarray accessor."""
        return "Network XarrayAccessor"


class NetworkArrayMixin(_NetworkABC):
    """Helper class for network array methods.

    Class inherits to [pypsa.Network][]. All attributes and methods can be used
    within any Network instance.
    """

    @property
    def da(self) -> _NetworkXarrayAccessor:
        """Xarray accessor for the network's own time series.

        While an optimization model is live, the arrays carry the model's snapshot
        labels, which differ from `n.snapshots` for a multi-period model built under
        linopy's v1 semantics (see [pypsa.optimization.window][]).

        Examples
        --------
        >>> n.da.snapshot_weightings.objective
        <xarray.DataArray 'objective' (snapshot: 10)> Size: 80B
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        Coordinates:
          * snapshot  (snapshot) datetime64[ns] 80B 2015-01-01 ... 2015-01-01T09:00:00

        """
        return _NetworkXarrayAccessor(self)
