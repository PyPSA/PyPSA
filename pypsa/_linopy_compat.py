# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""linopy v1 semantics compatibility for the multi-period ``snapshot`` dimension.

linopy's v1 arithmetic convention forbids a ``pd.MultiIndex`` on a variable's
dimension coordinate. Under v1, PyPSA therefore builds multi-period models over a
flat ``snapshot`` dim whose labels are the ``(period, timestep)`` tuples, carrying
``period``/``timestep`` as auxiliary coordinates. The tuple labels are the source
of truth: the MultiIndex and the aux coords are both derived from them, so any two
arrays over the same snapshots align by label and the round-trip back is a plain
``pd.MultiIndex.from_tuples``. Legacy linopy — and any release without the
``semantics`` option — keeps the MultiIndex first-class, leaving ``n.model``
unchanged. This shim is temporary; it goes away once v1 is linopy's default.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from linopy import options as linopy_options

SNAPSHOT_LEVELS = ("period", "timestep")


def linopy_uses_v1() -> bool:
    """Whether linopy's active arithmetic semantics is the v1 convention."""
    try:
        return linopy_options["semantics"] == "v1"
    except KeyError:
        return False


def _snapshot_is_multiindex(obj: Any) -> bool:
    """Whether ``obj`` carries a live ``snapshot`` MultiIndex (non-flattened build)."""
    return isinstance(obj.indexes.get("snapshot"), pd.MultiIndex)


def _level_coords(mi: pd.MultiIndex) -> dict[str, tuple[str, Any]]:
    """``{level: ('snapshot', values)}`` aux coords from a ``(period, timestep)`` index."""
    return {
        name: ("snapshot", mi.get_level_values(name).to_numpy()) for name in mi.names
    }


def tuple_snapshot_index(mi: pd.MultiIndex) -> pd.Index:
    """Flat object index of the MultiIndex's ``(period, timestep)`` tuple labels."""
    return pd.Index(list(mi), tupleize_cols=False, name="snapshot")


def flatten_snapshot_dim(obj: Any) -> Any:
    """Relabel a MultiIndex ``snapshot`` with its ``(period, timestep)`` tuples + aux coords.

    A no-op when ``snapshot`` is not a MultiIndex. Inverse of
    :func:`recompose_snapshot_dim`.
    """
    idx = obj.indexes.get("snapshot")
    if not isinstance(idx, pd.MultiIndex):
        return obj
    tuples = tuple_snapshot_index(idx).values
    obj = obj.reset_index("snapshot", drop=True)
    return obj.assign_coords(snapshot=tuples, **_level_coords(idx))


def recompose_snapshot_dim(obj: Any) -> Any:
    """Rebuild a MultiIndex ``snapshot`` from a flat dim + level aux coords.

    Inverse of :func:`flatten_snapshot_dim`. A no-op unless both level coords are present.
    """
    if "snapshot" not in obj.dims or _snapshot_is_multiindex(obj):
        return obj
    if not all(lvl in obj.coords for lvl in SNAPSHOT_LEVELS):
        return obj
    return obj.set_index(snapshot=list(SNAPSHOT_LEVELS))


def attach_snapshot_aux(obj: Any, window: pd.Index) -> Any:
    """Re-derive flat-snapshot aux coords from the ``(period, timestep)`` tuple labels.

    ``window`` guards the transform: a no-op outside a multi-period (flat) build.
    Inverse companion of :func:`drop_snapshot_aux`.
    """
    if not isinstance(window, pd.MultiIndex) or "snapshot" not in obj.dims:
        return obj
    if _snapshot_is_multiindex(obj):
        return obj
    mi = pd.MultiIndex.from_tuples(list(obj.indexes["snapshot"]), names=SNAPSHOT_LEVELS)
    return obj.assign_coords(_level_coords(mi))


def drop_snapshot_aux(obj: Any) -> Any:
    """Drop the flat-snapshot ``period``/``timestep`` aux coords.

    They linger on the ``_term`` dim after a reduction and break strict concat/merge
    against operands that never carried them. A no-op while ``snapshot`` is a live
    MultiIndex (the levels are the index itself, not droppable aux coords).
    """
    if "snapshot" in obj.dims and _snapshot_is_multiindex(obj):
        return obj
    return obj.drop_vars(list(SNAPSHOT_LEVELS), errors="ignore")
