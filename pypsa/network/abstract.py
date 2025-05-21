"""
Abstract network module.

Only defines a base class for all Network helper classes which inherit to
`Network` class.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from pypsa.components.store import ComponentsStore


class _NetworkABC(ABC):
    snapshots: pd.Index | pd.MultiIndex
    _snapshot_weightings: pd.DataFrame
    _investment_period_weightings: pd.DataFrame
    all_components: set[str]
    static: pd.DataFrame
    dynamic: Callable
    components: ComponentsStore
    c: ComponentsStore
    _import_series_from_df: Callable
