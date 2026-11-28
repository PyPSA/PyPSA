# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Snapshot window of a single optimization model build.

linopy's v1 convention forbids a `MultiIndex` dimension coordinate, so
multi-period models are built over a flat `snapshot` dim labelled by the
`(period, timestep)` tuples of `n.snapshots`, with the levels attached as
auxiliary coordinates. `SnapshotWindow` carries both labellings of the build's
snapshots and every operation bridging them. Without investment periods the two
labellings coincide and all operations degenerate to identity, so call sites
never branch on the representation.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from pypsa import Network

SNAPSHOT_LEVELS = ("period", "timestep")


def _snapshot_is_multiindex(obj: Any) -> bool:
    """Whether `obj` carries a `snapshot` MultiIndex."""
    return isinstance(obj.indexes.get("snapshot"), pd.MultiIndex)


def _level_coords(mi: pd.MultiIndex) -> dict[str, tuple[str, Any]]:
    """Auxiliary `snapshot` coords holding the level values of `mi`."""
    return {
        name: ("snapshot", mi.get_level_values(name).to_numpy()) for name in mi.names
    }


def tuple_snapshot_index(mi: pd.MultiIndex) -> pd.Index:
    """Flat index holding the MultiIndex's `(period, timestep)` tuples."""
    return mi.to_flat_index().rename("snapshot")


def apply_period_weighting(weight: xr.DataArray, weighting: pd.Series) -> xr.DataArray:
    """Multiply `weight` by the investment-period weighting of its snapshots."""
    period_of = weight.coords["period"].to_numpy()
    return weight * weighting.reindex(period_of).to_numpy()


def snapshot_array(values: Any, index: pd.Index) -> xr.DataArray:
    """Array over a `snapshot` dim labelled by `index`.

    Names the dim explicitly. Inferring it from the pandas object is
    unreliable: the snapshots MultiIndex only carries its "snapshot" label as
    an ad-hoc scalar name, which pandas 3 drops on any derived index.
    """
    return xr.DataArray(values, coords=[index], dims="snapshot")


class SnapshotWindow:
    """Snapshots of one model build, in the network's and in linopy's labelling.

    Created once per model build and kept on the network for as long as that
    model lives. Internal. User code sees the model's snapshot labels only
    through the `sns` argument of `extra_functionality` and the `c.da`
    accessors.

    Parameters
    ----------
    n : pypsa.Network
        Network the model is built for.
    network_index : pandas.Index
        Build snapshots as the network indexes them (a `pandas.MultiIndex` for
        multi-period networks).
    model_index : pandas.Index
        Labels of the model's `snapshot` dimension, position-aligned with
        `network_index`. Equal to `network_index` unless the build has
        investment periods.

    """

    def __init__(
        self, n: Network, network_index: pd.Index, model_index: pd.Index
    ) -> None:
        """Initialize the window from both snapshot labellings."""
        self._n = n
        self.network_index = network_index
        self.model_index = model_index

    @classmethod
    def build(cls, n: Network, sns: pd.Index) -> SnapshotWindow:
        """Create the window for a model built over `sns`."""
        if isinstance(sns, pd.MultiIndex):
            return cls(n, sns, tuple_snapshot_index(sns))
        return cls(n, sns, sns)

    def subset(self, sns: pd.Index) -> SnapshotWindow:
        """Window restricted to `sns`, given in the model's labelling."""
        if sns.equals(self.model_index):
            return self
        positions = self.model_index.get_indexer(sns)
        if (positions < 0).any():
            outside = sns[positions < 0].tolist()
            msg = f"Snapshots outside the model's build window: {outside}"
            raise KeyError(msg)
        return SnapshotWindow(
            self._n,
            self.network_index.take(positions),
            self.model_index.take(positions),
        )

    @property
    def has_periods(self) -> bool:
        """Whether the window is resolved by investment period."""
        return isinstance(self.network_index, pd.MultiIndex)

    @property
    def periods(self) -> pd.Index:
        """Investment periods covered by the window."""
        return self.network_index.unique("period")

    @property
    def period_of(self) -> pd.Index:
        """Investment period of each snapshot of the window."""
        return self.network_index.get_level_values("period")

    def _attach_aux(self, obj: xr.DataArray) -> xr.DataArray:
        """Attach the `period`/`timestep` aux coords to an array over the window's flat labels.

        A no-op without investment periods.
        """
        if not self.has_periods:
            return obj
        return obj.assign_coords(_level_coords(self.network_index))

    def drop_aux(self, obj: Any) -> Any:
        """Drop the flat-snapshot `period`/`timestep` aux coords."""
        return obj.drop_vars(list(SNAPSHOT_LEVELS), errors="ignore")

    def flatten(self, obj: xr.DataArray) -> xr.DataArray:
        """Put an `n.snapshots`-indexed array on the model's flat labels.

        Relabels without restricting to the window: rolling-horizon code reads
        history from before the window. A no-op without investment periods.
        """
        if not self.has_periods or not _snapshot_is_multiindex(obj):
            return obj
        return obj.reset_index("snapshot", drop=True).assign_coords(self._flat_coords)

    @functools.cached_property
    def _flat_coords(self) -> dict[str, Any]:
        """Flat `snapshot` labels and aux coords of all of `n.snapshots`."""
        snapshots = self._n.snapshots
        labels = tuple_snapshot_index(snapshots).values
        return {"snapshot": labels, **_level_coords(snapshots)}

    def snapshot_weightings(self, col: str) -> xr.DataArray:
        """Snapshot-weighting column of the window's snapshots, read live."""
        series = self._n.snapshot_weightings[col].loc[self.network_index]
        return self._attach_aux(snapshot_array(series.to_numpy(), self.model_index))

    @functools.cached_property
    def _network_coords(self) -> xr.Coordinates:
        """Window snapshots as a ready-made MultiIndex coordinate set."""
        return xr.Coordinates.from_pandas_multiindex(self.network_index, "snapshot")

    def recompose(self, obj: xr.DataArray) -> xr.DataArray:
        """Put a model-labelled array back on the network's snapshots.

        Inverse of `flatten`, for results handed back to the pandas side. A
        no-op unless both level coords are present.
        """
        if "snapshot" not in obj.dims or _snapshot_is_multiindex(obj):
            return obj
        if not all(lvl in obj.coords for lvl in SNAPSHOT_LEVELS):
            return obj
        if obj.indexes["snapshot"].equals(self.model_index):
            obj = self.drop_aux(obj).drop_vars("snapshot")
            return obj.assign_coords(self._network_coords)
        return obj.set_index(snapshot=list(SNAPSHOT_LEVELS))
