# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Thermal rating helpers for transmission lines.

Currently provides:

- :func:`apply_seasonal_line_ratings`: broadcast a per-line (summer, winter)
  rating table onto ``n.lines.s_nom`` and ``n.lines_t.s_max_pu``.

The module is intentionally policy-light: it implements the per-snapshot
broadcast mechanism only. Source-of-truth selection (which lines to include,
whether grid upgrades replace the rating, hemisphere-specific summer months)
is left to the caller.

See Also
--------
PyPSA GH #1693 for the discussion that motivated this helper.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pypsa.networks import Network


_DEFAULT_SUMMER_MONTHS_NH: tuple[int, ...] = (4, 5, 6, 7, 8, 9)


def apply_seasonal_line_ratings(
    n: Network,
    ratings: pd.DataFrame,
    summer_months: Sequence[int] = _DEFAULT_SUMMER_MONTHS_NH,
    *,
    compose: bool = True,
) -> None:
    """Apply seasonal ``s_max_pu`` derating to lines based on snapshot month.

    For each line in ``ratings``, set ``n.lines.at[line, 's_nom']`` to the
    winter envelope (``max(summer, winter)``) and broadcast a seasonal factor
    of ``min(summer, winter) / max(summer, winter)`` onto summer snapshots,
    via ``n.lines_t.s_max_pu``. Non-summer snapshots receive a factor of 1.0.

    Parameters
    ----------
    n : pypsa.Network
        Network with a ``DatetimeIndex`` on ``n.snapshots``.
    ratings : pandas.DataFrame
        Index: line names (must be a subset of ``n.lines.index``).
        Columns: ``'summer'`` and ``'winter'`` (case-sensitive), per-line MVA
        ratings. Rows where the two values are equal still set ``s_nom`` to
        that value but skip the per-snapshot broadcast.
    summer_months : Sequence[int], default ``(4, 5, 6, 7, 8, 9)``
        Snapshot months treated as summer. The default is northern-hemisphere;
        for southern-hemisphere use, pass e.g. ``(10, 11, 12, 1, 2, 3)``.
    compose : bool, default True
        If ``True``, multiply the seasonal factor into any pre-existing
        ``n.lines_t.s_max_pu`` entry (preserves N-1 margins). If ``False``,
        overwrite ``n.lines_t.s_max_pu`` with the seasonal factor alone.

    Convention
    ----------
    Summer is the period with the *lower* rating (cooling-limited). This
    matches IEEE 738 / IEC 61597 conductor thermal models and TenneT, RTE,
    National Grid, CAISO published rating periods.

    Raises
    ------
    TypeError
        If ``n.snapshots`` is not a ``DatetimeIndex``.
    KeyError
        If ``ratings`` references a line not in ``n.lines.index``, or if
        ``ratings`` is missing the ``summer`` / ``winter`` columns.
    ValueError
        If any rating is non-positive or ``summer_months`` is empty.

    Examples
    --------
    >>> import pandas as pd, pypsa
    >>> from pypsa.utils import apply_seasonal_line_ratings
    >>> n = pypsa.Network()
    >>> n.set_snapshots(pd.date_range('2025-01-01', periods=8760, freq='h'))
    >>> n.add('Bus', ['a', 'b'])
    >>> n.add('Line', 'a-b', bus0='a', bus1='b', x=0.1, s_nom=1000)
    >>> ratings = pd.DataFrame(
    ...     {'summer': [800], 'winter': [1000]}, index=['a-b']
    ... )
    >>> apply_seasonal_line_ratings(n, ratings)
    >>> float(n.lines.at['a-b', 's_nom'])
    1000.0
    >>> n.lines_t.s_max_pu['a-b'].iloc[3000]  # mid-summer hour
    0.8

    """
    if not isinstance(n.snapshots, pd.DatetimeIndex):
        msg = (
            "apply_seasonal_line_ratings requires n.snapshots to be a "
            f"DatetimeIndex; got {type(n.snapshots).__name__}."
        )
        raise TypeError(msg)

    missing_cols = {"summer", "winter"} - set(ratings.columns)
    if missing_cols:
        msg = f"`ratings` DataFrame is missing required columns: {sorted(missing_cols)}"
        raise KeyError(msg)

    missing_lines = ratings.index.difference(n.lines.index)
    if len(missing_lines) > 0:
        msg = (
            f"`ratings` references {len(missing_lines)} line(s) not in "
            f"n.lines.index: {list(missing_lines[:5])}"
            + ("..." if len(missing_lines) > 5 else "")
        )
        raise KeyError(msg)

    if not len(summer_months):
        msg = "`summer_months` must be non-empty."
        raise ValueError(msg)

    summer = pd.to_numeric(ratings["summer"], errors="raise")
    winter = pd.to_numeric(ratings["winter"], errors="raise")
    if (summer <= 0).any() or (winter <= 0).any():
        msg = "All summer and winter ratings must be strictly positive."
        raise ValueError(msg)

    envelope = np.maximum(summer, winter)
    factor = np.minimum(summer, winter) / envelope

    summer_mask = n.snapshots.month.isin(summer_months)
    # Per-snapshot factor as a (T, N_lines) DataFrame.
    seasonal = pd.DataFrame(
        np.where(summer_mask[:, None], factor.to_numpy()[None, :], 1.0),
        index=n.snapshots,
        columns=ratings.index,
    )

    # Set s_nom to the winter envelope on every affected line.
    n.lines.loc[ratings.index, "s_nom"] = envelope.to_numpy()

    # Broadcast s_max_pu. Compose multiplicatively with any pre-existing
    # per-snapshot entry so callers retain N-1 margins.
    if compose:
        existing = n.lines_t.s_max_pu.reindex(index=n.snapshots, columns=ratings.index)
        static = n.lines.loc[ratings.index, "s_max_pu"].to_numpy()
        existing = existing.fillna(pd.Series(static, index=ratings.index))
        n.lines_t.s_max_pu[ratings.index] = (existing * seasonal).to_numpy()
    else:
        n.lines_t.s_max_pu[ratings.index] = seasonal.to_numpy()
