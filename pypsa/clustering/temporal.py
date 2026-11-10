# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Functions for temporal clustering of networks.

This module provides methods to reduce the temporal resolution of PyPSA networks
while preserving the total modeled hours through snapshot weighting adjustments,
so that the total number of hours modelled is kept invariant.

**Invariant**: ``sum(weights) == total_modeled_hours`` (e.g., 8760 for one year)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib.metadata import version
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from deprecation import deprecated

if TYPE_CHECKING:
    from collections.abc import Collection

    from pypsa import Network


def _warn_if_solved(n: Network) -> None:
    """Warn if applying temporal clustering to a solved network."""
    has_results = any(
        not c.dynamic.get("p", pd.DataFrame()).empty
        for c in n.components
        if "p" in c.dynamic
    )
    if has_results:
        warnings.warn(
            "Applying temporal clustering to a solved network may result in "
            "inconsistent storage state of charge and dispatch values.",
            UserWarning,
            stacklevel=3,
        )


def _check_no_scenarios(n: Network) -> None:
    """Raise error if network has scenarios (stochastic)."""
    if n.has_scenarios:
        msg = "Temporal clustering does not yet support stochastic networks."
        raise NotImplementedError(msg)


HOURS_PER_YEAR = 8760

TEMPORAL_AGGREGATION_DEFAULTS: dict[str, str] = {
    "default": "mean",
    "e_max_pu": "min",
    "e_min_pu": "max",
}


def _warn_solver_deprecated(solver: str | None) -> None:
    """Warn that `solver` has no effect, if the caller passed it."""
    if solver is not None:
        warnings.warn(
            "`solver` is deprecated as an argument to `segment` and "
            "`get_segment_map`; it has no effect with tsam >= 3. "
            "Deprecated in version 1.5.0. Will be removed in version 2.0.0.",
            DeprecationWarning,
            stacklevel=3,
        )


def _enforce_pu_bounds(n: Network) -> None:
    """Clip aggregated `*_min_pu` series so they never exceed `*_max_pu`.

    Aggregation can lift a lower bound above its paired upper bound, either from
    floating-point error (segmentation normalises per column) or from min/max
    aggregation rules (`e_min_pu` uses `max`, `e_max_pu` uses `min`).
    """
    for c in n.components:
        for min_attr in [a for a in c.dynamic if a.endswith("_min_pu")]:
            max_attr = min_attr.replace("_min_pu", "_max_pu")
            if max_attr not in c.dynamic:
                continue
            min_df = c.dynamic[min_attr]
            max_df = c.dynamic[max_attr]
            common = min_df.columns.intersection(max_df.columns)
            if common.empty:
                continue
            c.dynamic[min_attr].loc[:, common] = min_df[common].clip(
                upper=max_df[common]
            )


@dataclass
class TemporalClustering:
    """Result of temporal clustering.

    Attributes
    ----------
    n : Network
        The clustered network with reduced temporal resolution.
    snapshot_map : pd.Series
        Mapping from original snapshots to aggregated snapshots.
        Index: original snapshots, Values: aggregated snapshots.

    """

    n: Network
    snapshot_map: pd.Series

    def __post_init__(self) -> None:
        """Sanitize time clustered data."""
        _enforce_pu_bounds(self.n)


def _get_aggregation_rule(
    attr: str,
    aggregation_rules: dict[str, str] | None = None,
) -> str:
    """Get aggregation rule for a component attribute.

    Parameters
    ----------
    attr : str
        Attribute name (e.g., "p_max_pu", "e_min_pu").
    aggregation_rules : dict, optional
        Custom aggregation rules overriding defaults.

    Returns
    -------
    str
        Aggregation function name ("mean", "min", "max", "sum").

    """
    if aggregation_rules and attr in aggregation_rules:
        return aggregation_rules[attr]
    return TEMPORAL_AGGREGATION_DEFAULTS.get(attr, "mean")


def _row_groups(n: Network, by_year: bool) -> list[np.ndarray]:
    """Return the row positions of each investment period of `n`.

    Without investment periods, return the rows of each year if `by_year`, and
    all rows as one group otherwise.
    """
    if n.has_periods:
        keys = n.snapshots.get_level_values("period")
    elif by_year:
        keys = pd.DatetimeIndex(n.snapshots).year
    else:
        keys = pd.Index(np.zeros(len(n.snapshots), dtype=int))
    return list(pd.Series(keys).groupby(keys).indices.values())


def _periods(snapshots: pd.Index) -> np.ndarray:
    """Return the investment period of each snapshot, or 0 without investment periods."""
    if isinstance(snapshots, pd.MultiIndex):
        return snapshots.get_level_values("period").to_numpy()
    return np.zeros(len(snapshots), dtype=int)


def _timesteps(snapshots: pd.Index) -> pd.DatetimeIndex:
    """Return the timestep of each snapshot, without its investment period."""
    if isinstance(snapshots, pd.MultiIndex):
        return pd.DatetimeIndex(snapshots.get_level_values("timestep"))
    return pd.DatetimeIndex(snapshots)


def _snapshot_labels(
    periods: np.ndarray, timesteps: pd.DatetimeIndex, has_periods: bool
) -> pd.Index:
    """Return snapshot labels from investment periods and timesteps."""
    if has_periods:
        return pd.MultiIndex.from_arrays(
            [periods, timesteps], names=["period", "timestep"]
        )
    return timesteps


def _resample_labels(timesteps: pd.DatetimeIndex, offset: str) -> np.ndarray:
    """Return the label that `resample(offset)` gives the bin of each timestep."""
    labels = timesteps.to_numpy().copy()
    bins = pd.Series(0, index=timesteps).resample(offset).indices
    for label, positions in bins.items():
        labels[positions] = label
    return labels


def _skip_leap_days(periods: np.ndarray, labels: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Replace each Feb 29 label by the first Mar 1 label of its period and year."""
    groups = pd.MultiIndex.from_arrays([periods, labels.year])
    on_march_first = (labels.month == 3) & (labels.day == 1)
    leap = (labels.month == 2) & (labels.day == 29)
    first_march = (
        pd.Series(labels[on_march_first], index=groups[on_march_first])
        .groupby(level=[0, 1])
        .min()
    )
    targets = first_march.reindex(groups[leap])
    if targets.isna().any():
        msg = "drop_leap_day=True requires a snapshot on March 1 after each February 29"
        raise ValueError(msg)
    moved = labels.to_numpy().copy()
    moved[leap] = targets.to_numpy()
    return pd.DatetimeIndex(moved)


def _drop_leap_days(m: Network, snapshot_map: pd.Series) -> pd.Series:
    """Drop the Feb 29 snapshots of `m` and add their weightings to Mar 1.

    Return `snapshot_map` with the snapshots of the dropped ones mapped to Mar 1.
    """
    labels = m.snapshots
    periods = _periods(labels)
    timesteps = _timesteps(labels)
    moved = _skip_leap_days(periods, timesteps)
    leap = np.flatnonzero(moved != timesteps)
    targets = labels.get_indexer(
        _snapshot_labels(periods[leap], moved[leap], m.has_periods)
    )
    weightings = m.snapshot_weightings.copy()
    for target in np.unique(targets):
        weightings.iloc[target] += weightings.iloc[leap[targets == target]].sum()
    kept = labels.delete(leap)
    m.set_snapshots(kept)
    m.snapshot_weightings = weightings.loc[kept]
    lookup = np.arange(len(labels))
    lookup[leap] = targets
    positions = lookup[labels.get_indexer(pd.Index(snapshot_map.to_numpy()))]
    values = labels.to_flat_index()[positions].to_numpy()
    return pd.Series(values, index=snapshot_map.index)


def _apply_snapshot_map(
    n: Network,
    snapshot_map: pd.Series | pd.DataFrame,
    aggregation_rules: dict[str, str] | None,
    *,
    select: bool = False,
) -> Network:
    """Return a copy of `n` aggregated by `snapshot_map`.

    With `select`, each aggregated snapshot takes the values of its first
    snapshot, missing ones included, instead of applying the aggregation rules.
    """
    _check_no_scenarios(n)

    if isinstance(snapshot_map, pd.DataFrame):
        snapshot_map_series = snapshot_map.iloc[:, 0]
    else:
        snapshot_map_series = snapshot_map

    if not snapshot_map_series.index.equals(n.snapshots):
        msg = "snapshot_map index must match network snapshots"
        raise ValueError(msg)

    aggregated_snapshots = snapshot_map_series.unique()
    aggregated_snapshots = pd.Index(sorted(aggregated_snapshots))

    weightings = n.snapshot_weightings.groupby(snapshot_map_series).sum()
    weightings = weightings.reindex(aggregated_snapshots)

    m = n.copy()
    m.set_snapshots(aggregated_snapshots)
    m.snapshot_weightings = weightings

    for c in n.components:
        if c.static.empty:
            continue
        for attr, df in c.dynamic.items():
            if df.empty:
                continue
            grouped = df.groupby(snapshot_map_series)
            aggregated = (
                grouped.first(skipna=False)
                if select
                else grouped.agg(_get_aggregation_rule(attr, aggregation_rules))
            )
            aggregated = aggregated.reindex(aggregated_snapshots)
            m._import_series_from_df(aggregated, c.name, attr, overwrite=True)

    _enforce_pu_bounds(m)
    return m


def _segmentation_matrix(n: Network, exclude_attrs: Collection[str]) -> np.ndarray:
    """Stack the time series of `n` into one matrix with a row per snapshot.

    NaN values are set to 0.
    """
    frames = [
        df.to_numpy(dtype=float)
        for c in n.components
        if not c.static.empty
        for attr, df in c.dynamic.items()
        if not df.empty and attr not in exclude_attrs
    ]
    if not frames:
        msg = "No time-varying data found for segmentation"
        raise ValueError(msg)
    matrix = np.hstack(frames)
    matrix[np.isnan(matrix)] = 0.0
    return matrix


def _distinct_varying_columns(matrix: np.ndarray, drop_duplicates: bool) -> np.ndarray:
    """Keep the columns of `matrix` that vary, and optionally only the first of identical ones.

    Constant columns do not move tsam's segments, so they are dropped unless no
    column varies.
    """
    varies = np.ptp(matrix, axis=0) > 0
    varying = matrix[:, varies] if varies.any() else matrix
    if not drop_duplicates:
        return varying
    duplicated = pd.Index([column.tobytes() for column in varying.T]).duplicated()
    return varying[:, ~duplicated]


def _segment_starts(matrix: np.ndarray, num_segments: int) -> np.ndarray:
    """Return, for each row of `matrix`, the row that starts its tsam segment."""
    import tsam  # noqa: PLC0415

    segmented = tsam.aggregate(
        pd.DataFrame(matrix),
        n_clusters=1,
        period_duration=len(matrix),
        temporal_resolution=1.0,
        segments=tsam.SegmentConfig(n_segments=num_segments),
    ).cluster_representatives
    durations = segmented.index.get_level_values("Segment Duration").to_numpy(int)
    return np.repeat(np.cumsum(durations) - durations, durations)


class TemporalClusteringMixin:
    """Mixin for temporal clustering methods.

    Class inherits to [`pypsa.clustering.TemporalClusteringAccessor`][]. All methods
    available via `n.cluster.temporal`.
    """

    _n: Network

    def resample(
        self,
        offset: str,
        *,
        drop_leap_day: bool = False,
        aggregation_rules: dict[str, str] | None = None,
    ) -> Network:
        """Resample network to coarser temporal resolution.

        Builds the map with `get_resample_map` and applies it as `from_snapshot_map`
        does.

        Parameters
        ----------
        offset : str
            Pandas offset string (e.g., "3h", "6h", "24h").
        drop_leap_day : bool, default False
            Transfer Feb 29 weights to March 1.
        aggregation_rules : dict, optional
            Override default aggregation per attribute.

        Returns
        -------
        Network
            The resampled network.

        Note
        ----
        This is not an inplace operation. Returns a new network.

        Examples
        --------
        >>> m = n.cluster.temporal.resample("3h")  # doctest: +SKIP

        """
        n = self._n
        _warn_if_solved(n)
        snapshot_map = self.get_resample_map(offset)
        m = _apply_snapshot_map(n, snapshot_map, aggregation_rules)
        if drop_leap_day:
            _drop_leap_days(m, snapshot_map)
        return m

    @deprecated(
        deprecated_in="1.5.0",
        removed_in="2.0.0",
        details="Use `get_resample_map()` and `from_snapshot_map()` instead.",
    )
    def get_resample_result(
        self,
        offset: str,
        *,
        drop_leap_day: bool = False,
        aggregation_rules: dict[str, str] | None = None,
    ) -> TemporalClustering:
        """Get the resampled network and its snapshot map.

        !!! warning "Deprecated in <!-- md:badge-version v1.5.0 -->"

            Use `get_resample_map()` and `from_snapshot_map()` instead.

        """
        return resample(
            self._n,
            offset,
            drop_leap_day=drop_leap_day,
            aggregation_rules=aggregation_rules,
        )

    def get_resample_map(
        self,
        offset: str,
        *,
        drop_leap_day: bool = False,
    ) -> pd.Series:
        """Map each snapshot to the label of its resampling bin.

        Bins are formed within each year, or within each investment period, and
        carry the labels that `pandas.DataFrame.resample` gives them.

        Parameters
        ----------
        offset : str
            Pandas offset string (e.g., "3h", "6h", "24h").
        drop_leap_day : bool, default False
            Map the snapshots of Feb 29 bins to the first Mar 1 bin.

        Returns
        -------
        pd.Series
            Index: original snapshots, Values: label of each snapshot's bin.

        """
        n = self._n
        _check_no_scenarios(n)

        if not isinstance(n.snapshots, pd.DatetimeIndex) and not n.has_periods:
            msg = "resample() requires snapshots to be a DatetimeIndex"
            raise TypeError(msg)

        periods = _periods(n.snapshots)
        timesteps = _timesteps(n.snapshots)
        labels = timesteps.to_numpy().copy()
        for rows in _row_groups(n, by_year=True):
            labels[rows] = _resample_labels(timesteps[rows], offset)
        labels = pd.DatetimeIndex(labels)
        if drop_leap_day:
            labels = _skip_leap_days(periods, labels)
        values = _snapshot_labels(periods, labels, n.has_periods)
        return pd.Series(values.to_flat_index(), index=n.snapshots)

    def downsample(
        self,
        stride: int,
    ) -> Network:
        """Select every Nth snapshot as representative.

        Each selected snapshot keeps its values and takes the summed weightings
        of its block, which preserves the total modeled hours.

        Parameters
        ----------
        stride : int
            Select every stride-th snapshot.

        Returns
        -------
        Network
            The downsampled network.

        Examples
        --------
        >>> m = n.cluster.temporal.downsample(4)  # doctest: +SKIP

        """
        n = self._n
        _warn_if_solved(n)
        return _apply_snapshot_map(
            n, self.get_downsample_map(stride), None, select=True
        )

    @deprecated(
        deprecated_in="1.5.0",
        removed_in="2.0.0",
        details="Use `get_downsample_map()` and `from_snapshot_map()` instead.",
    )
    def get_downsample_result(
        self,
        stride: int,
    ) -> TemporalClustering:
        """Get the downsampled network and its snapshot map.

        !!! warning "Deprecated in <!-- md:badge-version v1.5.0 -->"

            Use `get_downsample_map()` and `from_snapshot_map()` instead.

        """
        return downsample(self._n, stride)

    def get_downsample_map(self, stride: int) -> pd.Series:
        """Map each snapshot to every stride-th snapshot before it.

        The blocks of `stride` snapshots start again in each investment period.

        Parameters
        ----------
        stride : int
            Select every stride-th snapshot.

        Returns
        -------
        pd.Series
            Index: original snapshots, Values: first snapshot of each block.

        """
        n = self._n
        _check_no_scenarios(n)

        if stride < 1:
            msg = f"stride must be >= 1, got {stride}"
            raise ValueError(msg)

        representatives = np.arange(len(n.snapshots))
        for rows in _row_groups(n, by_year=False):
            representatives[rows] = rows[np.arange(len(rows)) // stride * stride]
        return pd.Series(
            n.snapshots[representatives].to_flat_index(), index=n.snapshots
        )

    def segment(
        self,
        num_segments: int,
        *,
        solver: str | None = None,
        exclude_attrs: list[str] | None = None,
        aggregation_rules: dict[str, str] | None = None,
        drop_duplicates: bool = False,
    ) -> Network:
        """Cluster time series into variable-duration segments using TSAM.

        Builds the map with `get_segment_map` and applies it as `from_snapshot_map`
        does.

        Parameters
        ----------
        num_segments : int
            Target number of segments, in each investment period.
        solver : str, optional
            Deprecated. Has no effect.
        exclude_attrs : list, optional
            Attributes to exclude from clustering (default: ["e_min_pu"]).
        aggregation_rules : dict, optional
            Override default aggregation per attribute.
        drop_duplicates : bool, default False
            Count identical time series once when placing the segments.

        Returns
        -------
        Network
            The segmented network.

        Note
        ----
        Requires `tsam>=3`: pip install tsam

        """
        n = self._n
        _warn_if_solved(n)
        _warn_solver_deprecated(solver)
        snapshot_map = self.get_segment_map(
            num_segments, exclude_attrs=exclude_attrs, drop_duplicates=drop_duplicates
        )
        return _apply_snapshot_map(n, snapshot_map, aggregation_rules)

    @deprecated(
        deprecated_in="1.5.0",
        removed_in="2.0.0",
        details="Use `get_segment_map()` and `from_snapshot_map()` instead.",
    )
    def get_segment_result(
        self,
        num_segments: int,
        *,
        solver: str | None = None,
        exclude_attrs: list[str] | None = None,
        aggregation_rules: dict[str, str] | None = None,
        drop_duplicates: bool = False,
    ) -> TemporalClustering:
        """Get the segmented network and its snapshot map.

        !!! warning "Deprecated in <!-- md:badge-version v1.5.0 -->"

            Use `get_segment_map()` and `from_snapshot_map()` instead.

        """
        return segment(
            self._n,
            num_segments,
            solver=solver,
            exclude_attrs=exclude_attrs,
            aggregation_rules=aggregation_rules,
            drop_duplicates=drop_duplicates,
        )

    def get_segment_map(
        self,
        num_segments: int,
        *,
        solver: str | None = None,
        exclude_attrs: list[str] | None = None,
        drop_duplicates: bool = False,
    ) -> pd.Series:
        """Map each snapshot to the first snapshot of its TSAM segment.

        Segments are placed by the time series of the network, separately in each
        investment period, so no segment crosses a period.

        Parameters
        ----------
        num_segments : int
            Number of segments, in each investment period.
        solver : str, optional
            Deprecated. Has no effect.
        exclude_attrs : list, optional
            Attributes to exclude from clustering (default: ["e_min_pu"]).
        drop_duplicates : bool, default False
            Count identical time series once, within each investment period.

        Returns
        -------
        pd.Series
            Index: original snapshots, Values: first snapshot of each segment.

        """
        n = self._n
        _check_no_scenarios(n)
        _warn_solver_deprecated(solver)

        if num_segments < 1:
            msg = f"num_segments must be >= 1, got {num_segments}"
            raise ValueError(msg)

        rows_per_period = _row_groups(n, by_year=False)
        shortest = min(len(rows) for rows in rows_per_period)
        if num_segments > shortest:
            scope = " per investment period" if n.has_periods else ""
            msg = f"num_segments ({num_segments}) cannot exceed number of snapshots{scope} ({shortest})"
            raise ValueError(msg)

        if find_spec("tsam") is None:
            msg = (
                "Optional dependency 'tsam' not found. "
                "Install via 'pip install tsam' or 'conda install -c conda-forge tsam'"
            )
            raise ModuleNotFoundError(msg)
        if int(version("tsam").split(".")[0]) < 3:
            msg = (
                f"tsam >= 3 is required for segmentation, found {version('tsam')}. "
                "Upgrade via 'pip install -U tsam' or 'conda install -c conda-forge \"tsam>=3\"'"
            )
            raise ImportError(msg)

        matrix = _segmentation_matrix(
            n, ["e_min_pu"] if exclude_attrs is None else exclude_attrs
        )
        starts = np.arange(len(n.snapshots))
        for rows in rows_per_period:
            distinct = _distinct_varying_columns(matrix[rows], drop_duplicates)
            starts[rows] = rows[_segment_starts(distinct, num_segments)]
        return pd.Series(n.snapshots[starts].to_flat_index(), index=n.snapshots)

    def from_snapshot_map(
        self,
        snapshot_map: pd.Series | pd.DataFrame,
        *,
        aggregation_rules: dict[str, str] | None = None,
    ) -> Network:
        """Apply pre-computed temporal aggregation mapping.

        Parameters
        ----------
        snapshot_map : pd.Series or pd.DataFrame
            Mapping from original snapshots to aggregated snapshots.
            If DataFrame, its first column is used.
        aggregation_rules : dict, optional
            Override default aggregation per attribute.

        Returns
        -------
        Network
            The aggregated network.

        """
        n = self._n
        _warn_if_solved(n)
        return _apply_snapshot_map(n, snapshot_map, aggregation_rules)

    @deprecated(
        deprecated_in="1.5.0",
        removed_in="2.0.0",
        details="Use `from_snapshot_map()` instead.",
    )
    def get_from_snapshot_map_result(
        self,
        snapshot_map: pd.Series | pd.DataFrame,
        *,
        aggregation_rules: dict[str, str] | None = None,
    ) -> TemporalClustering:
        """Get the aggregated network and its snapshot map.

        !!! warning "Deprecated in <!-- md:badge-version v1.5.0 -->"

            Use `from_snapshot_map()` instead.

        """
        return from_snapshot_map(
            self._n, snapshot_map, aggregation_rules=aggregation_rules
        )


# Backward-compatible module-level functions


def resample(
    n: Network,
    offset: str,
    *,
    drop_leap_day: bool = False,
    aggregation_rules: dict[str, str] | None = None,
) -> TemporalClustering:
    """Resample network to coarser temporal resolution.

    Parameters
    ----------
    n : Network
        Network with original resolution.
    offset : str
        Pandas offset string (e.g., "3h", "6h", "24h").
    drop_leap_day : bool, default False
        Transfer Feb 29 weights to March 1.
    aggregation_rules : dict, optional
        Override default aggregation per attribute.

    Returns
    -------
    TemporalClustering
        Result with clustered network and snapshot mapping.

    """
    _warn_if_solved(n)
    obj = TemporalClusteringMixin()
    obj._n = n
    snapshot_map = obj.get_resample_map(offset)
    m = _apply_snapshot_map(n, snapshot_map, aggregation_rules)
    if drop_leap_day:
        snapshot_map = _drop_leap_days(m, snapshot_map)
    return TemporalClustering(m, snapshot_map)


def downsample(
    n: Network,
    stride: int,
) -> TemporalClustering:
    """Select every Nth snapshot as representative.

    Parameters
    ----------
    n : Network
        Network with original resolution.
    stride : int
        Select every stride-th snapshot.

    Returns
    -------
    TemporalClustering
        Result with downsampled network.

    """
    _warn_if_solved(n)
    obj = TemporalClusteringMixin()
    obj._n = n
    snapshot_map = obj.get_downsample_map(stride)
    m = _apply_snapshot_map(n, snapshot_map, None, select=True)
    return TemporalClustering(m, snapshot_map)


def segment(
    n: Network,
    num_segments: int,
    *,
    solver: str | None = None,
    exclude_attrs: list[str] | None = None,
    aggregation_rules: dict[str, str] | None = None,
    drop_duplicates: bool = False,
) -> TemporalClustering:
    """Cluster time series into variable-duration segments using TSAM.

    Parameters
    ----------
    n : Network
        Network with original resolution.
    num_segments : int
        Target number of segments, in each investment period.
    solver : str, optional
        Deprecated. Has no effect.
    exclude_attrs : list, optional
        Attributes to exclude from clustering (default: ["e_min_pu"]).
    aggregation_rules : dict, optional
        Override default aggregation per attribute.
    drop_duplicates : bool, default False
        Count identical time series once when placing the segments.

    Returns
    -------
    TemporalClustering
        Result with segmented network.

    """
    _warn_if_solved(n)
    _warn_solver_deprecated(solver)
    obj = TemporalClusteringMixin()
    obj._n = n
    snapshot_map = obj.get_segment_map(
        num_segments, exclude_attrs=exclude_attrs, drop_duplicates=drop_duplicates
    )
    m = _apply_snapshot_map(n, snapshot_map, aggregation_rules)
    return TemporalClustering(m, snapshot_map)


def from_snapshot_map(
    n: Network,
    snapshot_map: pd.Series | pd.DataFrame,
    *,
    aggregation_rules: dict[str, str] | None = None,
) -> TemporalClustering:
    """Apply pre-computed temporal aggregation mapping.

    Parameters
    ----------
    n : Network
        Network with original resolution.
    snapshot_map : pd.Series or pd.DataFrame
        Mapping from original snapshots to aggregated snapshots.
    aggregation_rules : dict, optional
        Override default aggregation per attribute.

    Returns
    -------
    TemporalClustering
        Result with aggregated network.

    """
    _warn_if_solved(n)
    m = _apply_snapshot_map(n, snapshot_map, aggregation_rules)
    snapshot_series = (
        snapshot_map.iloc[:, 0]
        if isinstance(snapshot_map, pd.DataFrame)
        else snapshot_map
    )
    return TemporalClustering(m, snapshot_series)
