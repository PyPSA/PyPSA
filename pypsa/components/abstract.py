"""Abstract components module.

Only defines a base class for all Components helper classes which inherit to
`Components` class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from pypsa import Network
    from pypsa.components.types import ComponentType
    from pypsa.definitions.structures import Dict


class _ComponentsABC(ABC):
    ctype: ComponentType
    n: Network | None
    static: pd.DataFrame
    dynamic: Dict

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        pass

    @property
    @abstractmethod
    def n_save(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def snapshots(self) -> pd.Index:
        pass

    @property
    @abstractmethod
    def has_investment_periods(self) -> bool:
        pass

    @property
    @abstractmethod
    def investment_periods(self) -> pd.Index:
        pass

    @abstractmethod
    def get_active_assets(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        pass
