"""
Abstract network module.

Only defines a base class for all Network helper classes which inherit to
`Network` class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from pypsa.components.store import ComponentsStore


class _NetworkABC(ABC):
    snapshots: pd.Index | pd.MultiIndex
    _snapshot_weightings: pd.DataFrame
    _investment_period_weightings: pd.DataFrame
    static: pd.DataFrame
    dynamic: Callable
    _import_series_from_df: Callable
    add: Callable

    @property
    @abstractmethod
    def all_components(self) -> set[str]:
        """Read only placeholder."""
        ...

    @property
    @abstractmethod
    def components(self) -> ComponentsStore:
        """Read only placeholder."""
        ...

    @property
    @abstractmethod
    def c(self) -> ComponentsStore:
        """Read only placeholder."""
        ...
