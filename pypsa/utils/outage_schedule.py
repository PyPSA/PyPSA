# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Exogenous outage-schedule helper.

Writes user-supplied outage windows onto a built PyPSA network by setting
time-varying ``p_max_pu`` (and ``p_min_pu`` for asymmetric Links). The
helper is policy-light: the caller decides which windows apply; PyPSA
broadcasts them onto the snapshot index and composes multiplicatively with
any existing dynamic availability.

This is **deterministic** scheduling of known outage windows (planned
maintenance records, ENTSO-E unavailability reports, post-mortem
reconstruction for validation runs). It is intentionally orthogonal to
PR #1576 ("Maintenance optimization"), which co-optimises maintenance
windows as MILP variables.

Schema for the ``outages`` DataFrame
------------------------------------
Required columns:

- ``asset_id``: index entry in ``n.generators`` or ``n.links``
- ``component``: ``'Generator'`` or ``'Link'``
- ``start``: ``pd.Timestamp`` (tz-aware or tz-naive UTC)
- ``end``: ``pd.Timestamp`` (same tz convention as ``start``)
- ``p_max_pu``: float in ``[0, 1]``, fractional availability inside the
  window

Overlapping rows on the same asset collapse by ``min(p_max_pu)`` per
snapshot (worst outage wins). The factor composes multiplicatively with
any existing ``p_max_pu_t`` profile, so a generator at CF 0.30 during a
0.50 derate becomes 0.15.

See Also
--------
PyPSA GH #1576 for the maintenance-optimization PR (orthogonal).

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pypsa.networks import Network

logger = logging.getLogger(__name__)

OnUnknown = Literal["raise", "warn", "ignore"]

_REQUIRED_COLUMNS = ("asset_id", "component", "start", "end", "p_max_pu")
_SUPPORTED_COMPONENTS = ("Generator", "Link")


def _build_factor_series(
    rows: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
) -> pd.Series:
    """Collapse outage rows for one asset into a per-snapshot factor series.

    Overlapping rows take the minimum factor (worst outage wins). Outage
    bounds may be tz-aware or tz-naive; they are normalized to the
    snapshots' tz-awareness before comparison so the mask never silently
    evaluates all-False.

    Parameters
    ----------
    rows : pd.DataFrame
        Outage rows with columns ``start``, ``end``, ``p_max_pu``.
    snapshots : pd.DatetimeIndex
        Network snapshots.

    Returns
    -------
    pd.Series
        Factor series indexed by ``snapshots``, dtype ``float``,
        initialised at ``1.0``.

    """
    factor = pd.Series(1.0, index=snapshots, dtype=float)
    tz_naive_snapshots = snapshots.tz is None
    for _, row in rows.iterrows():
        start = row["start"]
        end = row["end"]
        start_tz = getattr(start, "tz", None)
        if tz_naive_snapshots and start_tz is not None:
            start = start.tz_convert("UTC").tz_localize(None)
            end = end.tz_convert("UTC").tz_localize(None)
        elif not tz_naive_snapshots and start_tz is None:
            start = pd.Timestamp(start, tz="UTC")
            end = pd.Timestamp(end, tz="UTC")
        mask = (snapshots >= start) & (snapshots <= end)
        if not mask.any():
            continue
        p = float(row["p_max_pu"])
        if not (0.0 <= p <= 1.0):
            asset = row.get("asset_id", "?")
            msg = f"p_max_pu must be in [0, 1]; got {p} for asset_id={asset}"
            raise ValueError(msg)
        factor.loc[mask] = np.minimum(factor.loc[mask].values, p)
    return factor


def build_factor_series_per_asset(
    outages: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    component: str | None = None,
) -> dict[str, pd.Series]:
    """Return a per-asset factor series for every asset that has an outage.

    Parameters
    ----------
    outages : pd.DataFrame
        Outage rows; must contain ``asset_id``, ``start``, ``end``,
        ``p_max_pu``, and (if ``component`` is ``None``) ``component``.
    snapshots : pd.DatetimeIndex
        Network snapshots.
    component : {'Generator', 'Link', None}
        If given, filter ``outages`` to that component before collapsing.

    Returns
    -------
    dict[str, pd.Series]
        Only assets whose factor dips below ``1.0`` anywhere are returned.

    """
    if outages.empty:
        return {}
    df = outages
    if component is not None and "component" in df.columns:
        df = df[df["component"] == component]
    if df.empty:
        return {}
    out: dict[str, pd.Series] = {}
    for asset_id, rows in df.groupby("asset_id"):
        factor = _build_factor_series(rows, snapshots)
        if (factor < 1.0).any():
            out[str(asset_id)] = factor
    return out


def _apply_link_factor(n: Network, link_id: str, factor: pd.Series) -> None:
    """Write an outage factor to a Link, respecting asymmetric capacity.

    The same factor applies in both directions: forward goes into
    ``links_t.p_max_pu``; reverse is encoded as ``links_t.p_min_pu`` and
    clamped so reverse availability is never more permissive than the
    static ``links.p_min_pu``.
    """
    static_p_min_pu = float(n.links.at[link_id, "p_min_pu"])

    pmax = n.links_t.p_max_pu
    if link_id in pmax.columns:
        existing = pmax[link_id].reindex(factor.index).fillna(1.0)
        pmax[link_id] = np.minimum(existing.values, factor.values)
    else:
        pmax[link_id] = factor.values

    reverse_cap = -factor
    reverse_series = reverse_cap.clip(lower=static_p_min_pu)

    pmin = n.links_t.p_min_pu
    if link_id in pmin.columns:
        existing = pmin[link_id].reindex(factor.index).fillna(static_p_min_pu)
        pmin[link_id] = np.maximum(existing.values, reverse_series.values)
    else:
        pmin[link_id] = reverse_series.values


def _apply_generator_factor(n: Network, gen_id: str, factor: pd.Series) -> None:
    """Multiply outage factor into ``generators_t.p_max_pu``.

    Composes with any existing time-varying availability so a renewable
    generator's capacity-factor profile is preserved underneath the
    derate.
    """
    pmax = n.generators_t.p_max_pu
    if gen_id in pmax.columns:
        existing = pmax[gen_id].reindex(factor.index).fillna(1.0)
        pmax[gen_id] = (existing.values * factor.values).clip(0.0, 1.0)
    else:
        pmax[gen_id] = factor.values


def apply_outage_schedule(
    n: Network,
    outages: pd.DataFrame,
    on_unknown: OnUnknown = "warn",
) -> dict[str, int]:
    """Write exogenous outage windows onto a built network.

    Generators receive ``n.generators_t.p_max_pu *= factor``; Links
    receive ``n.links_t.p_max_pu = min(existing, factor)`` plus a clamped
    ``n.links_t.p_min_pu`` for the reverse direction. Overlapping rows on
    the same asset collapse by ``min(p_max_pu)``.

    Parameters
    ----------
    n : pypsa.Network
        Network with snapshots already set.
    outages : pd.DataFrame
        Outage rows; required columns: ``asset_id``, ``component``,
        ``start``, ``end``, ``p_max_pu``. ``component`` must be
        ``'Generator'`` or ``'Link'``.
    on_unknown : {'raise', 'warn', 'ignore'}
        Behaviour when an ``asset_id`` is not in the network's component
        index. Default ``'warn'``.

    Returns
    -------
    dict[str, int]
        ``{'Generator': n_modified, 'Link': n_modified}``.

    Raises
    ------
    ValueError
        If ``outages`` is missing required columns, ``component`` is
        unsupported, ``p_max_pu`` is outside ``[0, 1]``, or
        ``n.snapshots`` is empty.
    KeyError
        If ``on_unknown == 'raise'`` and any ``asset_id`` is unknown.

    Examples
    --------
    >>> import pandas as pd
    >>> import pypsa
    >>> n = pypsa.Network()
    >>> n.set_snapshots(pd.date_range("2025-01-01", periods=24, freq="h"))
    >>> _ = n.add("Bus", "b")
    >>> _ = n.add("Generator", "g", bus="b", p_nom=100)
    >>> outages = pd.DataFrame({
    ...     "asset_id": ["g"],
    ...     "component": ["Generator"],
    ...     "start": [pd.Timestamp("2025-01-01 06:00")],
    ...     "end": [pd.Timestamp("2025-01-01 12:00")],
    ...     "p_max_pu": [0.0],
    ... })
    >>> _ = pypsa.utils.apply_outage_schedule(n, outages)
    >>> bool(n.generators_t.p_max_pu["g"].loc["2025-01-01 09:00"] == 0.0)
    True

    """
    counts: dict[str, int] = {"Generator": 0, "Link": 0}
    if not isinstance(n.snapshots, pd.DatetimeIndex):
        msg = "n.snapshots must be a DatetimeIndex; call n.set_snapshots(...) first"
        raise TypeError(msg)

    if outages.empty:
        return counts

    missing = set(_REQUIRED_COLUMNS) - set(outages.columns)
    if missing:
        msg = f"outages is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    bad = ~outages["component"].isin(_SUPPORTED_COMPONENTS)
    if bad.any():
        unsupported = sorted(outages.loc[bad, "component"].unique().tolist())
        msg = (
            f"unsupported component(s) in outages: {unsupported}; "
            f"expected one of {list(_SUPPORTED_COMPONENTS)}"
        )
        raise ValueError(msg)

    snapshots = n.snapshots
    unknown: list[str] = []

    for component, group in outages.groupby("component"):
        if component == "Generator":
            index = n.generators.index
            apply_fn = _apply_generator_factor
        else:  # 'Link'
            index = n.links.index
            apply_fn = _apply_link_factor
        id_to_factor = build_factor_series_per_asset(group, snapshots)
        for asset_id, factor in id_to_factor.items():
            if asset_id not in index:
                unknown.append(f"{component}:{asset_id}")
                continue
            apply_fn(n, asset_id, factor)
            counts[component] += 1

    if unknown:
        preview = unknown[:5]
        ellipsis = "..." if len(unknown) > 5 else ""
        msg = (
            f"{len(unknown)} outage asset_id(s) not in network: "
            f"{preview}{ellipsis}"
        )
        if on_unknown == "raise":
            raise KeyError(msg)
        if on_unknown == "warn":
            logger.warning(msg)
    return counts
