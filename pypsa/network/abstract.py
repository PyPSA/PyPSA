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
    # Indexing
    snapshots: pd.Index | pd.MultiIndex
    snapshot_weightings: pd.DataFrame
    _snapshots_data: pd.DataFrame
    investment_periods: pd.Index
    investment_period_weightings: pd.DataFrame
    _investment_periods_data: pd.DataFrame
    scenarios: pd.Series
    scenario_weightings: pd.DataFrame
    _scenarios_data: pd.DataFrame
    _risk_preference: dict[str, float] | None
    static: pd.DataFrame
    dynamic: Callable
    _import_series_from_df: Callable
    add: Callable
    crs: Any
    standard_type_components: pd.DataFrame
    srid: Any
    set_snapshots: Callable
    remove: Callable
    iterate_components: Callable
    copy: Callable
    _import_components_from_df: Callable

    sub_networks: pd.DataFrame
    buses: pd.DataFrame
    generators: pd.DataFrame
    lines: pd.DataFrame
    line_types: pd.DataFrame
    links: pd.DataFrame
    transformers: pd.DataFrame
    stores: pd.DataFrame
    shunt_impedances: pd.DataFrame
    carriers: pd.DataFrame
    get_switchable_as_dense: Callable
    get_committable_i: Callable
    get_extendable_i: Callable
    shapes: pd.DataFrame
    component_attrs: pd.DataFrame
    global_constraints: pd.DataFrame
    calculate_dependent_values: Callable

    passive_branches: pd.DataFrame

    @property
    @abstractmethod
    def has_scenarios(self) -> bool:
        """Read only placeholder."""
        ...

    @property
    @abstractmethod
    def has_periods(self) -> bool:
        """Read only placeholder."""
        ...

    @property
    @abstractmethod
    def passive_branch_components(self) -> set[str]:
        """Read only placeholder."""
        ...

    @property
    @abstractmethod
    def one_port_components(self) -> set[str]:
        """Read only placeholder."""
        ...

    @property
    @abstractmethod
    def branch_components(self) -> set[str]:
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
