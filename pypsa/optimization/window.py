# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Snapshot window of a single optimization model build.

A multi-period model may be built over a flat ``snapshot`` dimension whose labels
are the ``(period, timestep)`` tuples of ``n.snapshots`` (linopy's v1 convention,
see [pypsa._linopy_compat][]). [SnapshotWindow][pypsa.optimization.window.SnapshotWindow]
carries both labellings of the build's snapshots and every operation that bridges
them. Outside that path the two labellings are the same index and all operations
degenerate to identity, so call sites never branch on the representation.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr
from linopy import merge
from numpy import roll, zeros

from pypsa._linopy_compat import (
    SNAPSHOT_LEVELS,
    attach_snapshot_aux,
    drop_snapshot_aux,
    flatten_snapshot_dim,
    resolve_snapshot_representation,
    tuple_snapshot_index,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from linopy import LinearExpression, Variable

    from pypsa import Network

logger = logging.getLogger(__name__)


@functools.cache
def _notify_multiindex_snapshot_kept() -> None:
    """One-shot notice that the multi-period ``snapshot`` stays a MultiIndex.

    Under legacy linopy the MultiIndex is kept first-class (unchanged ``n.model``),
    which linopy flags per variable/constraint as deprecated; suppress that spam
    and inform once. The flat tuple representation can be opted into explicitly.
    """
    logger.info(
        "Building the multi-period model over a MultiIndex `snapshot`. The flat "
        "`snapshot` dim with `period`/`timestep` auxiliary coordinates (linopy's "
        "v1 convention, `linopy.options['semantics'] = 'v1'`) is enabled via "
        "`pypsa.options.optimization.snapshot_representation = 'flat'`; this "
        "becomes PyPSA's default in 2.0."
    )


class SnapshotWindow:
    """Snapshots of one model build, in the network's and in linopy's labelling.

    Created once per
    [create_model][pypsa.optimization.OptimizationAccessor.create_model] and reachable as
    ``n.optimize.window`` for as long as that model lives. Use it in
    ``extra_functionality`` to move pandas data onto the model's ``snapshot``
    dimension and back.

    Parameters
    ----------
    n : pypsa.Network
        Network the model is built for.
    network_index : pandas.Index
        Build snapshots as the network indexes them (a ``pandas.MultiIndex`` for
        multi-period networks).
    model_index : pandas.Index
        Labels of the model's ``snapshot`` dimension, position-aligned with
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
        """Create the window for a model built over `sns`.

        Parameters
        ----------
        n : pypsa.Network
            Network the model is built for.
        sns : pandas.Index
            Snapshots selected for the build, in the network's labelling.
        representation : str
            Value of ``pypsa.options.optimization.snapshot_representation``.

        Returns
        -------
        SnapshotWindow

        """
        if resolve_snapshot_representation(sns, representation) == "flat":
            return cls(n, sns, tuple_snapshot_index(sns))
        if isinstance(sns, pd.MultiIndex):
            _notify_multiindex_snapshot_kept()
        return cls(n, sns, sns)

    def subset(self, sns: pd.Index) -> SnapshotWindow:
        """Window restricted to `sns`, given in the model's labelling.

        Parameters
        ----------
        sns : pandas.Index
            Subset of ``model_index``.

        Returns
        -------
        SnapshotWindow

        """
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
        """Whether the model's ``snapshot`` dim carries flat tuple labels."""
        return isinstance(self.network_index, pd.MultiIndex) and not isinstance(
            self.model_index, pd.MultiIndex
        )

    @property
    def has_periods(self) -> bool:
        """Whether the window is resolved by investment period."""
        return (
            isinstance(self.network_index, pd.MultiIndex)
            and "period" in self.network_index.names
        )

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

    def on_network(self, obj: pd.DataFrame | pd.Series) -> Any:
        """Restrict a network-indexed pandas object to the window's snapshots.

        Parameters
        ----------
        obj : pandas.DataFrame or pandas.Series
            Object indexed by ``n.snapshots``.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            `obj` restricted to the window, keeping the network's labels.

        """
        return obj.loc[self.network_index]

    def on_model(self, obj: pd.DataFrame | pd.Series) -> Any:
        """Put a snapshot-indexed pandas object on the model's snapshot labels.

        The index name is carried over, as linopy derives the dimension name from it.

        Parameters
        ----------
        obj : pandas.DataFrame or pandas.Series
            Object indexed by ``n.snapshots``.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            `obj` restricted to the window and relabelled for the model.

        """
        labels = self.model_index.copy()
        labels.name = obj.index.name
        return self.on_network(obj).set_axis(labels)

    def iter_periods(self) -> Iterator[tuple[Any, pd.Index]]:
        """Yield ``(period, snapshots)`` pairs for per-period constraint building.

        Yields a single ``(None, model_index)`` pair unless the model was built
        with ``multi_investment_periods=True``.

        Yields
        ------
        tuple of (int or None, pandas.Index)

        """
        if not self._n._multi_invest:
            yield None, self.model_index
            return
        period_of = self.period_of
        for period in self.periods:
            yield period, self.model_index[period_of == period]

    def period_start_mask(self) -> xr.DataArray:
        """Mark the first snapshot of each investment period within the window.

        Returns
        -------
        xarray.DataArray
            Boolean array over the model's ``snapshot`` dimension.

        """
        is_start = zeros(len(self.model_index), dtype=bool)
        is_start[0] = True
        if self.has_periods:
            periods = self.period_of.to_numpy()
            is_start[1:] = periods[1:] != periods[:-1]
        return xr.DataArray(is_start, coords=[self.model_index])

    def roll_within_periods(self, v: Variable) -> Variable:
        """Cyclically roll `v` by one snapshot within each investment period.

        Rolls positionally within each period and restores the original snapshot
        coordinates, so the result stays aligned with the un-rolled variable.

        Parameters
        ----------
        v : linopy.Variable
            Variable over the model's ``snapshot`` dimension.

        Returns
        -------
        linopy.Variable

        """
        positions = pd.Series(range(len(self.model_index)), index=self.model_index)
        rolled_at = positions.groupby(self.period_of.to_numpy()).transform(
            lambda s: roll(s, 1)
        )
        rolled = v.isel(snapshot=rolled_at.to_numpy())
        keep = {c: v.coords[c] for c in ("snapshot", *SNAPSHOT_LEVELS) if c in v.coords}
        return rolled.assign_coords(keep)

    def merge(self, exprs: list[LinearExpression]) -> LinearExpression:
        """Outer-merge expressions on ``snapshot``, preserving the flat aux coords.

        The aux coords must be dropped before the strict outer merge — differing
        periods on collided labels read as a conflict — and re-derived afterwards.

        Parameters
        ----------
        exprs : list of linopy.LinearExpression
            Expressions to merge.

        Returns
        -------
        linopy.LinearExpression

        """
        merged = merge([self.drop_aux(e) for e in exprs], dim="snapshot", join="outer")
        return self._attach_aux(merged)

    def _attach_aux(self, obj: Any) -> Any:
        """Re-derive the ``period``/``timestep`` aux coords from the tuple labels."""
        return attach_snapshot_aux(obj, self.network_index)

    def drop_aux(self, obj: Any) -> Any:
        """Drop the flat-snapshot ``period``/``timestep`` aux coords.

        They linger on reduced dimensions and break strict concat/merge against
        operands that never carried them.

        Parameters
        ----------
        obj : linopy.Variable, linopy.LinearExpression or xarray.DataArray
            Object to strip.

        Returns
        -------
        Same type as `obj`

        """
        return drop_snapshot_aux(obj)

    def flatten(self, obj: xr.DataArray) -> xr.DataArray:
        """Select the window from an xarray object and put it on the model labels.

        Parameters
        ----------
        obj : xarray.DataArray
            Array indexed by ``n.snapshots`` along ``snapshot``.

        Returns
        -------
        xarray.DataArray
            Unchanged unless the build is flat.

        """
        if not self.is_flat:
            return obj
        if "snapshot" in obj.dims:
            obj = obj.sel(snapshot=self.network_index)
        return flatten_snapshot_dim(obj)
