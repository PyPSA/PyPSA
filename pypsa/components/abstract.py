"""
Abstract components module.

Contains classes and properties relevant to all component types in PyPSA. Also imports
logic from other modules:
- components.types
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from pypsa import Network
    from pypsa.components.types import ComponentType
# logger = logging.getLogger(__name__)

# if TYPE_CHECKING:
#     from pypsa import Network


class _ComponentsABC(ABC):
    ctype: ComponentType
    n: Network | None
    static: pd.DataFrame
    dynamic: dict

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

    @property
    @abstractmethod
    def operational_attrs(self) -> dict:
        pass

    @property
    @abstractmethod
    def extendables(self) -> pd.Index:
        pass

    @property
    @abstractmethod
    def committables(self) -> pd.Index:
        pass

    @property
    @abstractmethod
    def fixed(self) -> pd.Index:
        pass

    @property
    def base_attr(self) -> str:
        msg = f"base_attr does not exist for {self.name} components"
        raise ValueError(msg)

    @abstractmethod
    def get_activity_mask(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        pass

    @abstractmethod
    def _as_dynamic(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_active_assets(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        pass

    def get_bounds_pu(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get per unit bounds for components.

        Parameters
        ----------
        args : Any
            Arguments for the method
        kwargs : Any
            Keyword arguments for the method

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple of (min_pu, max_pu) DataFrames.

        """
        msg = f"get_bounds_pu does not exist for {self.name} components"
        raise ValueError(msg)
