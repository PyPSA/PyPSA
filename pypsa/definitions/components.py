from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pypsa.definitions.structures import Dict


@dataclass
class Component:
    """
    Container class of energy system related assets, such as generators or
    transmission lines.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    Parameters
    ----------
    name : str
        The singular name of the component (e.g., 'Generator').
    list_name : str
        The plural name used for lists of components (e.g., 'generators').
    attrs : Dict[str, Any]
        A dictionary of attributes and their metadata.
    static : pd.DataFrame
        A DataFrame containing data for each component instance.
    dynamic : Dict[str, pd.DataFrame]
        A dictionary of time series data (panel data) for the component.
    ind : pd.Index
        An index of component identifiers.
    """

    name: str
    list_name: str
    attrs: pd.DataFrame
    investment_periods: pd.Index  # TODO: Needs better general approach
    static: pd.DataFrame
    dynamic: Dict
    ind: None  # deprecated

    # raise a deprecation warning if ind attribute is not None
    def __post_init__(self) -> None:
        if self.ind is not None:
            raise DeprecationWarning(
                "The 'ind' attribute is deprecated and will be removed in future versions."
            )

    def __repr__(self) -> str:
        return (
            f"Component(name={self.name!r}, list_name={self.list_name!r}, "
            f"attrs=Keys({list(self.attrs.keys())}), static=DataFrame(shape={self.static.shape}), "
            f"dynamic=Keys({list(self.dynamic.keys())}))"
        )

    # @deprecated(
    #     deprecated_in="0.32",
    #     removed_in="1.0",
    #     details="Use `c.static` instead.",
    # )
    @property
    def df(self) -> pd.DataFrame:
        return self.static

    # @deprecated(
    #     deprecated_in="0.32",
    #     removed_in="1.0",
    #     details="Use `c.dynamic` instead.",
    # )
    @property
    def pnl(self) -> Dict:
        return self.dynamic

    def get_active_assets(
        self,
        investment_period: int | str | Sequence | None = None,
    ) -> pd.Series:
        """
        Get active components mask of componen type in investment period(s).

        A component is considered active when:
        - it's active attribute is True
        - it's build year + lifetime is smaller than the investment period (if given)

        Parameters
        ----------
        n : pypsa.Network
            Network instance
        c : str
            Component name
        investment_period : int, str, Sequence
            Investment period(s) to check for active within build year and lifetime. If none
            only the active attribute is considered and build year and lifetime are ignored.
            If multiple periods are given the mask is True if the component is active in any
            of the given periods.

        Returns
        -------
        pd.Series
            Boolean mask for active components
        """
        if investment_period is None:
            return self.static.active
        if not {"build_year", "lifetime"}.issubset(self.static):
            return self.static.active

        # Logical OR of active assets in all investment periods and
        # logical AND with active attribute
        active = {}
        for period in np.atleast_1d(investment_period):
            if period not in self.investment_periods:
                raise ValueError("Investment period not in `n.investment_periods`")
            active[period] = self.static.eval(
                "build_year <= @period < build_year + lifetime"
            )
        return pd.DataFrame(active).any(axis=1) & self.static.active
