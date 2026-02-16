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
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
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


def _handle_leap_day(weightings: pd.DataFrame) -> pd.DataFrame:
    """Transfer leap day weights to March 1 rather than dropping hours.

    Parameters
    ----------
    weightings : pd.DataFrame
        Snapshot weightings with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Weightings with leap days removed and weights transferred.

    """
    if not isinstance(weightings.index, pd.DatetimeIndex):
        return weightings

    leap_days = weightings.index[
        (weightings.index.month == 2) & (weightings.index.day == 29)
    ]
    if leap_days.empty:
        return weightings

    weightings = weightings.copy()
    for year in leap_days.year.unique():
        march_first_candidates = weightings.index[
            (weightings.index.year == year)
            & (weightings.index.month == 3)
            & (weightings.index.day == 1)
        ]
        if len(march_first_candidates) > 0:
            march_first = march_first_candidates[0]
            leap_mask = (
                (weightings.index.year == year)
                & (weightings.index.month == 2)
                & (weightings.index.day == 29)
            )
            weightings.loc[march_first] += weightings.loc[leap_mask].sum()

    return weightings.drop(leap_days).sort_index()


def _build_resample_map(
    original_snapshots: pd.Index,
    resampled_snapshots: pd.Index,
    offset: str,
) -> pd.Series:
    """Build mapping from original to resampled snapshots.

    Parameters
    ----------
    original_snapshots : pd.Index
        Original snapshot index.
    resampled_snapshots : pd.Index
        Resampled snapshot index.
    offset : str
        Pandas offset string used for resampling.

    Returns
    -------
    pd.Series
        Mapping from original to resampled snapshots.

    """
    if isinstance(original_snapshots, pd.DatetimeIndex):
        resampled_dt = pd.DatetimeIndex(resampled_snapshots)
        idx = np.searchsorted(resampled_dt, original_snapshots, side="right") - 1
        idx = np.clip(idx, 0, len(resampled_dt) - 1)
        mapped_index = pd.DatetimeIndex(resampled_dt[idx])
        return pd.Series(mapped_index, index=original_snapshots)
    return pd.Series(resampled_snapshots[0], index=original_snapshots)


def _resample_with_periods(
    n: Network,
    offset: str,
    *,
    drop_leap_day: bool = False,
    aggregation_rules: dict[str, str] | None = None,
) -> TemporalClustering:
    """Resample network with investment periods, clustering within each period."""
    m = n.copy()
    original_snapshots = n.snapshots

    resampled_weightings_list = []
    for period in n.periods:
        period_mask = n.snapshots.get_level_values("period") == period
        sws_period = n.snapshot_weightings[period_mask]
        timesteps = sws_period.index.get_level_values("timestep")
        sws_flat = pd.DataFrame(
            sws_period.values,
            index=pd.DatetimeIndex(timesteps),
            columns=sws_period.columns,
        )
        sws_resampled = sws_flat.resample(offset).sum()
        sws_resampled = sws_resampled.query("objective != 0")
        if drop_leap_day:
            sws_resampled = _handle_leap_day(sws_resampled)
        sws_resampled.index = pd.MultiIndex.from_product(
            [[period], sws_resampled.index], names=["period", "timestep"]
        )
        sws_resampled.index.name = "snapshot"
        resampled_weightings_list.append(sws_resampled)

    snapshot_weightings = pd.concat(resampled_weightings_list)
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.components:
        if c.static.empty:
            continue
        for attr, df in c.dynamic.items():
            if df.empty:
                continue
            agg = _get_aggregation_rule(attr, aggregation_rules)
            resampled_list = []
            for period in n.periods:
                period_mask = df.index.get_level_values("period") == period
                df_period = df[period_mask]
                timesteps = df_period.index.get_level_values("timestep")
                df_flat = pd.DataFrame(
                    df_period.values,
                    index=pd.DatetimeIndex(timesteps),
                    columns=df_period.columns,
                )
                resampled = df_flat.resample(offset).agg(agg)
                resampled = resampled.query(
                    "index in @snapshot_weightings.index.get_level_values('timestep').unique()"
                )
                resampled.index = pd.MultiIndex.from_product(
                    [[period], resampled.index], names=["period", "timestep"]
                )
                resampled.index.name = "snapshot"
                resampled_list.append(resampled)
            resampled_all = pd.concat(resampled_list)
            m._import_series_from_df(resampled_all, c.name, attr, overwrite=True)

    snapshot_map = pd.Series(index=original_snapshots, dtype=object)
    for period in n.periods:
        orig_mask = original_snapshots.get_level_values("period") == period
        new_mask = m.snapshots.get_level_values("period") == period
        orig_timesteps = original_snapshots[orig_mask].get_level_values("timestep")
        new_timesteps = m.snapshots[new_mask].get_level_values("timestep")
        period_map = _build_resample_map(
            pd.DatetimeIndex(orig_timesteps),
            pd.DatetimeIndex(new_timesteps),
            offset,
        )
        for orig, new in zip(
            original_snapshots[orig_mask],
            [(period, ts) for ts in period_map.values],
            strict=False,
        ):
            snapshot_map[orig] = new

    return TemporalClustering(m, snapshot_map)


def _downsample_with_periods(n: Network, stride: int) -> TemporalClustering:
    """Downsample network with investment periods, selecting within each period."""
    m = n.copy()
    original_snapshots = n.snapshots

    downsampled_weightings_list = []
    selected_snapshots_list = []

    for period in n.periods:
        period_mask = n.snapshots.get_level_values("period") == period
        period_snapshots = n.snapshots[period_mask]
        period_weightings = n.snapshot_weightings[period_mask]

        # Select every Nth snapshot within this period
        selected_idx = range(0, len(period_snapshots), stride)
        selected = period_snapshots[list(selected_idx)]
        selected_snapshots_list.append(selected)

        # Scale weightings, handle remainder for last snapshot
        remainder = len(period_snapshots) % stride
        new_weights = period_weightings.loc[selected].copy()
        new_weights *= stride
        if remainder != 0:
            new_weights.iloc[-1] = new_weights.iloc[-1] / stride * remainder
        downsampled_weightings_list.append(new_weights)

    all_selected = pd.MultiIndex.from_tuples(
        [s for idx in selected_snapshots_list for s in idx],
        names=["period", "timestep"],
    )
    snapshot_weightings = pd.concat(downsampled_weightings_list)

    m.set_snapshots(all_selected)
    m.snapshot_weightings = snapshot_weightings

    for c in n.components:
        if c.static.empty:
            continue
        for attr, df in c.dynamic.items():
            if df.empty:
                continue
            m._import_series_from_df(df.loc[all_selected], c.name, attr, overwrite=True)

    # Build snapshot map using vectorized assignment per period
    snapshot_map = pd.Series(index=original_snapshots, dtype=object)
    for period in n.periods:
        period_mask = original_snapshots.get_level_values("period") == period
        period_snapshots = original_snapshots[period_mask]
        new_mask = all_selected.get_level_values("period") == period
        period_selected = all_selected[new_mask]

        # Vectorized: assign each original snapshot to its representative
        num_period = len(period_snapshots)
        indices = np.arange(num_period) // stride
        indices = np.clip(indices, 0, len(period_selected) - 1)
        snapshot_map.loc[period_snapshots] = [period_selected[i] for i in indices]

    return TemporalClustering(m, snapshot_map)


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
        return self.get_resample_result(
            offset,
            drop_leap_day=drop_leap_day,
            aggregation_rules=aggregation_rules,
        ).n

    def get_resample_result(
        self,
        offset: str,
        *,
        drop_leap_day: bool = False,
        aggregation_rules: dict[str, str] | None = None,
    ) -> TemporalClustering:
        """Get full TemporalClustering result from resample.

        Returns the full result including both the clustered network and the
        snapshot mapping. Use this when you need the snapshot_map for
        disaggregation or debugging.

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
        TemporalClustering
            Result with clustered network and snapshot mapping.

        """
        n = self._n
        _warn_if_solved(n)
        _check_no_scenarios(n)

        if not isinstance(n.snapshots, pd.DatetimeIndex) and not n.has_periods:
            msg = "resample() requires snapshots to be a DatetimeIndex"
            raise TypeError(msg)

        if n.has_periods:
            return _resample_with_periods(
                n,
                offset,
                drop_leap_day=drop_leap_day,
                aggregation_rules=aggregation_rules,
            )

        m = n.copy()

        original_snapshots = n.snapshots

        # Year-wise resampling handles non-contiguous years gracefully
        years = pd.DatetimeIndex(n.snapshots).year.unique()
        snapshot_weightings_list = []
        for year in years:
            year_mask = n.snapshots.year == year
            sws_year = n.snapshot_weightings[year_mask]
            sws_year = sws_year.resample(offset).sum()
            snapshot_weightings_list.append(sws_year)
        snapshot_weightings = pd.concat(snapshot_weightings_list)

        # The resampling produces a contiguous date range. In case the original
        # index was not contiguous, all rows with zero weight must be dropped
        # (corresponding to time steps not included in the original snapshots).
        snapshot_weightings = snapshot_weightings.query("objective != 0")

        if drop_leap_day:
            snapshot_weightings = _handle_leap_day(snapshot_weightings)

        m.set_snapshots(snapshot_weightings.index)
        m.snapshot_weightings = snapshot_weightings

        for c in n.components:
            if c.static.empty:
                continue
            for attr, df in c.dynamic.items():
                if df.empty:
                    continue
                agg = _get_aggregation_rule(attr, aggregation_rules)
                resampled_list = []
                for year in years:
                    year_mask = n.snapshots.year == year
                    df_year = df[year_mask]
                    resampled_year = df_year.resample(offset).agg(agg)
                    resampled_list.append(resampled_year)
                resampled = pd.concat(resampled_list)
                resampled = resampled[resampled.index.isin(m.snapshots)]
                m._import_series_from_df(resampled, c.name, attr, overwrite=True)

        snapshot_map = _build_resample_map(original_snapshots, m.snapshots, offset)

        return TemporalClustering(m, snapshot_map)

    def downsample(
        self,
        stride: int,
    ) -> Network:
        """Select every Nth snapshot as representative.

        Weightings are scaled by stride to preserve total modeled hours.

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
        return self.get_downsample_result(stride).n

    def get_downsample_result(
        self,
        stride: int,
    ) -> TemporalClustering:
        """Get full TemporalClustering result from downsample.

        Returns the full result including both the clustered network and the
        snapshot mapping.

        Parameters
        ----------
        stride : int
            Select every stride-th snapshot.

        Returns
        -------
        TemporalClustering
            Result with downsampled network and snapshot mapping.

        """
        n = self._n
        _warn_if_solved(n)
        _check_no_scenarios(n)

        if stride < 1:
            msg = f"stride must be >= 1, got {stride}"
            raise ValueError(msg)

        if n.has_periods:
            return _downsample_with_periods(n, stride)

        m = n.copy()
        original_snapshots = n.snapshots

        selected = n.snapshots[::stride]
        m.set_snapshots(selected)

        num_orig = len(original_snapshots)
        remainder = num_orig % stride

        new_weightings = n.snapshot_weightings.loc[selected].copy()
        new_weightings *= stride
        if remainder != 0:
            new_weightings.iloc[-1] = new_weightings.iloc[-1] / stride * remainder
        m.snapshot_weightings = new_weightings

        for c in n.components:
            if c.static.empty:
                continue
            for attr, df in c.dynamic.items():
                if df.empty:
                    continue
                m._import_series_from_df(df.loc[selected], c.name, attr, overwrite=True)

        snapshot_map = pd.Series(index=original_snapshots, dtype=object)
        for i, sel in enumerate(selected):
            start = i * stride
            end = min((i + 1) * stride, num_orig)
            snapshot_map.iloc[start:end] = sel

        return TemporalClustering(m, snapshot_map)

    def segment(
        self,
        num_segments: int,
        *,
        solver: str = "highs",
        exclude_attrs: list[str] | None = None,
        aggregation_rules: dict[str, str] | None = None,
    ) -> Network:
        """Cluster time series into variable-duration segments using TSAM.

        Parameters
        ----------
        num_segments : int
            Target number of segments.
        solver : str, default "highs"
            MIP solver for time clustering.
        exclude_attrs : list, optional
            Attributes to exclude from clustering (default: ["e_min_pu"]).
        aggregation_rules : dict, optional
            Override default aggregation per attribute.

        Returns
        -------
        Network
            The segmented network.

        Note
        ----
        Requires `tsam` package: pip install tsam

        """
        return self.get_segment_result(
            num_segments,
            solver=solver,
            exclude_attrs=exclude_attrs,
            aggregation_rules=aggregation_rules,
        ).n

    def get_segment_result(
        self,
        num_segments: int,
        *,
        solver: str = "highs",
        exclude_attrs: list[str] | None = None,
        aggregation_rules: dict[str, str] | None = None,
    ) -> TemporalClustering:
        """Get full TemporalClustering result from segment.

        Returns the full result including both the clustered network and the
        snapshot mapping.

        Parameters
        ----------
        num_segments : int
            Target number of segments.
        solver : str, default "highs"
            MIP solver for time clustering.
        exclude_attrs : list, optional
            Attributes to exclude from clustering (default: ["e_min_pu"]).
        aggregation_rules : dict, optional
            Override default aggregation per attribute.

        Returns
        -------
        TemporalClustering
            Result with segmented network and snapshot mapping.

        """
        n = self._n
        _warn_if_solved(n)
        _check_no_scenarios(n)

        if num_segments < 1:
            msg = f"num_segments must be >= 1, got {num_segments}"
            raise ValueError(msg)

        if num_segments > len(n.snapshots):
            msg = f"num_segments ({num_segments}) cannot exceed number of snapshots ({len(n.snapshots)})"
            raise ValueError(msg)

        if find_spec("tsam") is None:
            msg = (
                "Optional dependency 'tsam' not found. "
                "Install via 'pip install tsam' or 'conda install -c conda-forge tsam'"
            )
            raise ModuleNotFoundError(msg)

        import tsam.timeseriesaggregation as tsam_module  # noqa: PLC0415

        if exclude_attrs is None:
            exclude_attrs = ["e_min_pu"]

        if n.has_periods:
            msg = "segment() does not yet support networks with investment periods"
            raise NotImplementedError(msg)

        dfs = []
        col_attrs = []
        for c in n.components:
            if c.static.empty:
                continue
            for attr, df in c.dynamic.items():
                if not df.empty and attr not in exclude_attrs:
                    for col in df.columns:
                        dfs.append(df[[col]])
                        col_attrs.append((c.name, attr, col))

        if not dfs:
            msg = "No time-varying data found for segmentation"
            raise ValueError(msg)

        combined = pd.concat(dfs, axis=1)
        combined.columns = range(len(combined.columns))

        normalization_factors = combined.max().replace(0, 1)
        df_normalized = combined.div(normalization_factors)
        df_normalized = df_normalized.fillna(0)

        agg = tsam_module.TimeSeriesAggregation(
            df_normalized,
            hoursPerPeriod=len(df_normalized),
            noTypicalPeriods=1,
            noSegments=num_segments,
            segmentation=True,
            solver=solver,
        )
        segmented = agg.createTypicalPeriods()

        weightings_raw = segmented.index.get_level_values("Segment Duration")
        offsets = np.insert(np.cumsum(weightings_raw.values[:-1]), 0, 0).astype(int)
        original_snapshots = n.snapshots

        if isinstance(original_snapshots, pd.DatetimeIndex):
            new_snapshots = original_snapshots[offsets]
        else:
            new_snapshots = pd.Index(offsets)

        m = n.copy()
        m.set_snapshots(new_snapshots)

        new_weightings = pd.DataFrame(
            dict.fromkeys(n.snapshot_weightings.columns, weightings_raw.values),
            index=new_snapshots,
        )
        m.snapshot_weightings = new_weightings

        segmented_values = segmented.values * normalization_factors.values
        segmented_df = pd.DataFrame(
            segmented_values,
            index=new_snapshots,
            columns=range(len(col_attrs)),
        )

        for i, (comp, attr, col_name) in enumerate(col_attrs):
            data_col = segmented_df[[i]]
            data_col.columns = [col_name]
            if attr in m.c[comp].dynamic and not m.c[comp].dynamic[attr].empty:
                m.c[comp].dynamic[attr][col_name] = data_col[col_name]
            else:
                m._import_series_from_df(data_col, comp, attr, overwrite=True)

        segment_indices = (
            np.searchsorted(offsets, np.arange(len(original_snapshots)), side="right")
            - 1
        )
        snapshot_map = pd.Series(
            np.asarray(new_snapshots)[segment_indices], index=original_snapshots
        )

        return TemporalClustering(m, snapshot_map)

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
            If DataFrame, must have 'snapshot' index and columns for weightings.
        aggregation_rules : dict, optional
            Override default aggregation per attribute.

        Returns
        -------
        Network
            The aggregated network.

        """
        return self.get_from_snapshot_map_result(
            snapshot_map, aggregation_rules=aggregation_rules
        ).n

    def get_from_snapshot_map_result(
        self,
        snapshot_map: pd.Series | pd.DataFrame,
        *,
        aggregation_rules: dict[str, str] | None = None,
    ) -> TemporalClustering:
        """Get full TemporalClustering result from from_snapshot_map.

        Returns the full result including both the clustered network and the
        snapshot mapping.

        Parameters
        ----------
        snapshot_map : pd.Series or pd.DataFrame
            Mapping from original snapshots to aggregated snapshots.
            If DataFrame, must have 'snapshot' index and columns for weightings.
        aggregation_rules : dict, optional
            Override default aggregation per attribute.

        Returns
        -------
        TemporalClustering
            Result with aggregated network and snapshot mapping.

        """
        n = self._n
        _warn_if_solved(n)
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
                agg = _get_aggregation_rule(attr, aggregation_rules)
                aggregated = df.groupby(snapshot_map_series).agg(agg)
                aggregated = aggregated.reindex(aggregated_snapshots)
                m._import_series_from_df(aggregated, c.name, attr, overwrite=True)

        return TemporalClustering(m, snapshot_map_series)


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
    obj = TemporalClusteringMixin()
    obj._n = n
    return obj.get_resample_result(
        offset, drop_leap_day=drop_leap_day, aggregation_rules=aggregation_rules
    )


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
    obj = TemporalClusteringMixin()
    obj._n = n
    return obj.get_downsample_result(stride)


def segment(
    n: Network,
    num_segments: int,
    *,
    solver: str = "highs",
    exclude_attrs: list[str] | None = None,
    aggregation_rules: dict[str, str] | None = None,
) -> TemporalClustering:
    """Cluster time series into variable-duration segments using TSAM.

    Parameters
    ----------
    n : Network
        Network with original resolution.
    num_segments : int
        Target number of segments.
    solver : str, default "highs"
        MIP solver for time clustering.
    exclude_attrs : list, optional
        Attributes to exclude from clustering (default: ["e_min_pu"]).
    aggregation_rules : dict, optional
        Override default aggregation per attribute.

    Returns
    -------
    TemporalClustering
        Result with segmented network.

    """
    obj = TemporalClusteringMixin()
    obj._n = n
    return obj.get_segment_result(
        num_segments,
        solver=solver,
        exclude_attrs=exclude_attrs,
        aggregation_rules=aggregation_rules,
    )


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
    obj = TemporalClusteringMixin()
    obj._n = n
    return obj.get_from_snapshot_map_result(
        snapshot_map, aggregation_rules=aggregation_rules
    )
