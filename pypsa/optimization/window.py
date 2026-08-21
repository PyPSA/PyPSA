# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Snapshot window of a single optimization model build.

linopy's v1 convention forbids a `MultiIndex` dimension coordinate, so
multi-period models are built over a flat `snapshot` dim labelled by the
`(period, timestep)` tuples of `n.snapshots`, with the levels attached as
auxiliary coordinates. `SnapshotWindow` carries both labellings of the build's
snapshots and every operation bridging them. Outside the flat path the two
labellings coincide and all operations degenerate to identity, so call sites
never branch on the representation.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from linopy import merge

from pypsa._linopy_compat import use_flat_snapshot_index

if TYPE_CHECKING:
    from collections.abc import Iterator

    from linopy import LinearExpression, Variable

    from pypsa import Network

logger = logging.getLogger(__name__)

SNAPSHOT_LEVELS = ("period", "timestep")


@functools.cache
def _notify_multiindex_snapshot_kept() -> None:
    """One-shot notice that the multi-period `snapshot` stays a MultiIndex.

    Under legacy linopy the MultiIndex is kept first-class (unchanged `n.model`),
    which linopy flags per variable/constraint as deprecated; suppress that spam
    and inform once. The flat tuple representation can be opted into explicitly.
    """
    logger.info(
        "Building the multi-period model over a MultiIndex `snapshot`. Opt into "
        "the flat representation via "
        "`pypsa.options.optimization.model_snapshot_index = 'flat'`."
    )


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
        `network_index`. Equal to `network_index` unless the build is flat.

    """

    def __init__(
        self, n: Network, network_index: pd.Index, model_index: pd.Index
    ) -> None:
        """Initialize the window from both snapshot labellings."""
        self._n = n
        self.network_index = network_index
        self.model_index = model_index

    @classmethod
    def build(cls, n: Network, sns: pd.Index, representation: str) -> SnapshotWindow:
        """Create the window for a model built over `sns`."""
        if use_flat_snapshot_index(sns, representation):
            return cls(n, sns, tuple_snapshot_index(sns))
        if isinstance(sns, pd.MultiIndex):
            _notify_multiindex_snapshot_kept()
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
    def is_flat(self) -> bool:
        """Whether the model's `snapshot` dim carries flat tuple labels."""
        return self.has_periods and not isinstance(self.model_index, pd.MultiIndex)

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

    @property
    def start(self) -> Any:
        """First snapshot of the window, in the network's labelling."""
        return self.network_index[0]

    def iter_periods(self) -> Iterator[tuple[Any, pd.Index]]:
        """Yield `(period, snapshots)` pairs for per-period constraint building.

        Yields a single `(None, model_index)` pair unless the model was built
        with `multi_investment_periods=True`.
        """
        if not self._n._multi_invest:
            yield None, self.model_index
            return
        period_of = self.period_of
        for period in self.periods:
            yield period, self.model_index[period_of == period]

    def period_start_mask(self) -> xr.DataArray:
        """Mark the first snapshot of each investment period within the window."""
        is_start = np.zeros(len(self.model_index), dtype=bool)
        is_start[0] = True
        if self.has_periods:
            periods = self.period_of.to_numpy()
            is_start[1:] = periods[1:] != periods[:-1]
        return snapshot_array(is_start, self.model_index)

    def take(self, v: Variable, positions: Any) -> Variable:
        """Positionally select snapshots from `v`, keeping the window's labels.

        For shifts and rolls: the result carries the window's own snapshot
        coordinates, staying aligned with the un-shifted variable.
        """
        if not v.indexes["snapshot"].equals(self.model_index):
            v = v.sel(snapshot=self.model_index)
        keep = {c: v.coords[c] for c in ("snapshot", *SNAPSHOT_LEVELS) if c in v.coords}
        return v.isel(snapshot=positions).assign_coords(keep)

    def roll_within_periods(self, v: Variable) -> Variable:
        """Cyclically roll `v` by one snapshot within each investment period."""
        n_sns = len(self.model_index)
        starts = np.flatnonzero(self.period_start_mask().to_numpy())
        positions = np.arange(n_sns) - 1
        positions[starts] = np.append(starts[1:], n_sns) - 1
        return self.take(v, positions)

    def merge(self, exprs: list[LinearExpression]) -> LinearExpression:
        """Outer-merge expressions on `snapshot`, preserving the flat aux coords.

        The aux coords must be dropped before the strict outer merge — differing
        periods on collided labels read as a conflict — and re-derived afterwards.
        """
        merged = merge([self.drop_aux(e) for e in exprs], dim="snapshot", join="outer")
        return self._attach_aux(merged)

    def _attach_aux(self, obj: Any) -> Any:
        """Re-derive the aux coords of a flat `snapshot` dim from its tuple labels.

        A no-op outside a multi-period flat build.
        """
        if not self.has_periods or "snapshot" not in obj.dims:
            return obj
        if _snapshot_is_multiindex(obj):
            return obj
        idx = obj.indexes["snapshot"]
        if idx.equals(self.model_index):
            return obj.assign_coords(_level_coords(self.network_index))
        mi = pd.MultiIndex.from_tuples(list(idx), names=SNAPSHOT_LEVELS)
        return obj.assign_coords(_level_coords(mi))

    def drop_aux(self, obj: Any) -> Any:
        """Drop the flat-snapshot `period`/`timestep` aux coords.

        Needed where the snapshot labels are rewritten (a positional shift, a
        merge over collided labels): the coords still describe the source
        snapshots and contradict their own dimension, which v1 rejects. A no-op
        while `snapshot` is a live MultiIndex.
        """
        if "snapshot" in obj.dims and _snapshot_is_multiindex(obj):
            return obj
        return obj.drop_vars(list(SNAPSHOT_LEVELS), errors="ignore")

    def flatten(self, obj: xr.DataArray) -> xr.DataArray:
        """Put an `n.snapshots`-indexed array on the model's flat labels.

        Relabels without restricting to the window, mirroring the non-flat path
        where arrays keep all of `n.snapshots`: rolling-horizon code reads
        history from before the window. A no-op unless the build is flat.
        """
        if not self.is_flat or not _snapshot_is_multiindex(obj):
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
