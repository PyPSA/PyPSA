"""Abstract network module.

Only defines a base class for all Network mixin classes which inherit to
`Network` class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import pandas as pd

    from pypsa.components.store import ComponentsStore


class _NetworkABC(ABC):
    snapshots: pd.Index | pd.MultiIndex
    snapshot_weightings: pd.DataFrame
    _snapshot_weightings: pd.DataFrame
    _investment_period_weightings: pd.DataFrame
    static: pd.DataFrame
    dynamic: Callable
    _import_series_from_df: Callable
    add: Callable
    crs: Any
    investment_period_weightings: pd.DataFrame
    standard_type_components: pd.DataFrame
    srid: Any
    set_snapshots: Callable
    investment_periods: pd.Index
    remove: Callable
    iterate_components: Callable
    copy: Callable
    _import_components_from_df: Callable

    sub_networks: pd.DataFrame
    buses: pd.DataFrame
    generators: pd.DataFrame
    lines: pd.DataFrame
    links: pd.DataFrame
    transformers: pd.DataFrame
    stores: pd.DataFrame
    shunt_impedances: pd.DataFrame

    passive_branches: pd.DataFrame

    @property
    @abstractmethod
    def passive_branch_components(self) -> set[str]:
        """Read only placeholder."""
        ...

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
