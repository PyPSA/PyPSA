"""
Abstract components module.

Contains classes and logic relevant to all component types in PyPSA.
"""

from __future__ import annotations

import logging
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from pypsa.definitions.components import ComponentTypeInfo
from pypsa.definitions.structures import Dict
from pypsa.utils import equals

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypsa import Network

# TODO attachment todos
# - crs
# - snapshots, investment_periods


@dataclass
class ComponentsData:
    ct: ComponentTypeInfo
    n: Network | None
    static: pd.DataFrame
    dynamic: dict


class Components(ComponentsData, ABC):
    """
    Abstract base class for Container of energy system related assets, such as
    generators or transmission lines. Use the specific subclasses for concrete or
    a generic component type.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.
    """

    def __init__(
        self,
        ct: ComponentTypeInfo,
        n: Network | None = None,
        names: str | int | Sequence[int | str] | None = None,
        suffix: str = "",
    ) -> None:
        if names is not None:
            msg = "Adding components during initialisation is not yet supported."
            raise NotImplementedError(msg)
        if n is not None:
            msg = (
                "Attaching components to Network during initialisation is not yet "
                "supported."
            )
            raise NotImplementedError(msg)
        static, dynamic = self._get_data_containers(ct)
        super().__init__(ct, n=None, static=static, dynamic=dynamic)

    def __repr__(self) -> str:
        num_components = len(self.static)
        if not num_components:
            return f"Empty PyPSA {self.ct.name} Components\n"
        text = f"PyPSA '{self.ct.name}' Components"
        text += "\n" + "-" * len(text) + "\n"

        # Add attachment status
        if self.attached:
            network_name = f"'{self.n_save.name}'" if self.n_save.name else ""
            text += f"Attached to PyPSA Network {network_name}\n"

        text += f"Components: {len(self.static)}"

        return text

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.__dict__:
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in Component")

    def __eq__(self, other: Any) -> bool:
        return (
            equals(self.ct, other.ct)
            and equals(self.static, other.static)
            and equals(self.dynamic, other.dynamic)
        )

    @staticmethod
    def _get_data_containers(ct: ComponentTypeInfo) -> tuple[pd.DataFrame, Dict]:
        static_dtypes = ct.defaults.loc[ct.defaults.static, "dtype"].drop(["name"])
        if ct.name == "Shape":
            # TODO: Needs general default crs
            crs = CRS.from_epsg(
                "4326"
            )  # if n is None else n.crs #TODO attach mechanism
            static = gpd.GeoDataFrame(
                {k: gpd.GeoSeries(dtype=d) for k, d in static_dtypes.items()},
                columns=static_dtypes.index,
                crs=crs,
            )
        else:
            static = pd.DataFrame(
                {k: pd.Series(dtype=d) for k, d in static_dtypes.items()},
                columns=static_dtypes.index,
            )
        static.index.name = ct.name

        # # it's currently hard to imagine non-float series,
        # but this could be generalised
        dynamic = Dict()
        # TODO: Needs general default
        snapshots = pd.Index(
            ["now"]
        )  # if n is None else n.snapshots #TODO attach mechanism
        for k in ct.defaults.index[ct.defaults.varying]:
            df = pd.DataFrame(index=snapshots, columns=[], dtype=float)
            df.index.name = "snapshot"
            df.columns.name = ct.name
            dynamic[k] = df

        return static, dynamic

    @property
    def standard_types(self) -> pd.DataFrame | None:
        return self.ct.standard_types

    @property
    def name(self) -> str:
        return self.ct.name

    @property
    def list_name(self) -> str:
        return self.ct.list_name

    @property
    def description(self) -> str:
        return self.ct.description

    @property
    def category(self) -> str:
        return self.ct.category

    @property
    def type(self) -> str:
        return self.ct.category

    @property
    def attrs(self) -> pd.DataFrame:
        return self.ct.defaults

    @property
    def defaults(self) -> pd.DataFrame:
        return self.ct.defaults

    def get(self, attribute_name: str, default: Any = None) -> Any:
        return getattr(self, attribute_name, default)

    @property
    def attached(self) -> bool:
        return self.n is not None

    @property
    def n_save(self) -> Any:
        """A save property to access the network (component must be attached)."""
        if not self.attached:
            raise ValueError("Component must be attached to a Network.")
        return self.n

    @property
    def df(self) -> pd.DataFrame:
        return self.static

    @property
    def pnl(self) -> dict:
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
            if period not in self.n_save.investment_periods:
                raise ValueError("Investment period not in `n.investment_periods`")
            active[period] = self.static.eval(
                "build_year <= @period < build_year + lifetime"
            )
        return pd.DataFrame(active).any(axis=1) & self.static.active


class SubNetworkComponents:
    def __init__(self, wrapped_data: Components, wrapper_func: Callable) -> None:
        self._wrapped_data = wrapped_data
        self._wrapper_func = wrapper_func

    def __getattr__(self, item: str) -> Any:
        # Determine which attributes to monitor
        # Delegate attribute access to the wrapped data object

        return self._wrapper_func(item, self._wrapped_data)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in {"_wrapped_data", "_wrapper_func"}:
            super().__setattr__(key, value)
        else:
            raise AttributeError("SubNetworkComponents is read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SubNetworkComponents is read-only")
