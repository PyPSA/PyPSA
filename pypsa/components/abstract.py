"""
Abstract components module.

Contains classes and logic relevant to all component types in PyPSA.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray
from pyproj import CRS

from pypsa.constants import DEFAULT_EPSG, DEFAULT_TIMESTAMP
from pypsa.definitions.components import ComponentType
from pypsa.definitions.structures import Dict
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import nominal_attrs
from pypsa.utils import equals

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypsa import Network

# TODO attachment todos
# - crs
# - snapshots, investment_periods


@dataclass
class ComponentsData:
    """
    Dataclass for Components.

    Dataclass to store all data of a Components object and used to separate data from
    logic.

    Attributes
    ----------
    ctype : ComponentType
        Component type information containing all default values and attributes.
    n : Network | None
        Network object to which the component might be attached.
    static : pd.DataFrame
        Static data of components.
    dynamic : dict
        Dynamic data of components.

    """

    ctype: ComponentType
    n: Network | None
    static: pd.DataFrame
    dynamic: dict


class Components(ComponentsData, ABC):
    """
    Components base class.

    Abstract base class for Container of energy system related assets, such as
    generators or transmission lines. Use the specific subclasses for concrete or
    a generic component type.
    All data is stored in dataclass :class:`pypsa.components.abstract.ComponentsData`.
    Components inherits from it, adds logic and methods, but does not store any data
    itself.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    """

    def __init__(
        self,
        ctype: ComponentType,
        n: Network | None = None,
        names: str | int | Sequence[int | str] | None = None,
        suffix: str = "",
    ) -> None:
        """
        Initialize Components object.

        Parameters
        ----------
        ctype : ComponentType
            Component type information.
        n : Network, optional
            Network object to attach to, by default None.
        names : str, int, Sequence[int | str], optional
            Names of components to attach to, by default None.
        suffix : str, optional
            Suffix to add to component names, by default "".

        """
        if names is not None:
            msg = "Adding components during initialisation is not yet supported."
            raise NotImplementedError(msg)
        if n is not None:
            msg = (
                "Attaching components to Network during initialisation is not yet "
                "supported."
            )
            raise NotImplementedError(msg)
        static, dynamic = self._get_data_containers(ctype)
        super().__init__(ctype, n=None, static=static, dynamic=dynamic)

    def __repr__(self) -> str:
        """
        Get string representation of component.

        Returns
        -------
        str
            String representation of component.

        Examples
        --------
        >>> import pypsa
        >>> c = pypsa.examples.ac_dc_meshed().components.generators
        >>> c
        PyPSA 'Generator' Components
        ----------------------------
        Attached to PyPSA Network 'AC-DC'
        Components: 6

        """
        num_components = len(self.static)
        if not num_components:
            return f"Empty PyPSA {self.ctype.name} Components\n"
        text = f"PyPSA '{self.ctype.name}' Components"
        text += "\n" + "-" * len(text) + "\n"

        # Add attachment status
        if self.attached:
            network_name = f"'{self.n_save.name}'" if self.n_save.name else ""
            text += f"Attached to PyPSA Network {network_name}\n"

        text += f"Components: {len(self.static)}"

        return text

    def __str__(self) -> str:
        """
        Get string representation of component.

        Returns
        -------
        str
            String representation of component.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.components.generators
        PyPSA 'Generator' Components
        ----------------------------
        Attached to PyPSA Network 'AC-DC'
        Components: 6

        """
        num_components = len(self.static)
        text = f"{num_components} '{self.ctype.name}' Components"
        return text

    def __getitem__(self, key: str) -> Any:
        """
        Get attribute of component.

        Parameters
        ----------
        key : str
            Attribute name to get.

        Returns
        -------
        Any
            Attribute value of component.

        """
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set attribute of component.

        Parameters
        ----------
        key : str
            Attribute name to set.
        value : Any
            Attribute value to set.

        Raises
        ------
        KeyError
            If the attribute is not found in component.

        """
        if key in self.__dict__:
            setattr(self, key, value)
        else:
            # TODO: Is this to strict?
            raise KeyError(f"'{key}' not found in Component")

    def __eq__(self, other: Any) -> bool:
        """
        Check if two Components are equal.

        Does not check the attached Network, but only component specific data. Therefore
        two components can be equal even if they are attached to different networks.

        Parameters
        ----------
        other : Any
            Other object to compare with.

        Returns
        -------
        bool
            True if components are equal, otherwise False.

        """
        return (
            equals(self.ctype, other.ctype)
            and equals(self.static, other.static)
            and equals(self.dynamic, other.dynamic)
        )

    @staticmethod
    def _get_data_containers(ct: ComponentType) -> tuple[pd.DataFrame, Dict]:
        static_dtypes = ct.defaults.loc[ct.defaults.static, "dtype"].drop(["name"])
        if ct.name == "Shape":
            crs = CRS.from_epsg(
                DEFAULT_EPSG
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
        snapshots = pd.Index(
            [DEFAULT_TIMESTAMP]
        )  # if n is None else n.snapshots #TODO attach mechanism
        for k in ct.defaults.index[ct.defaults.varying]:
            df = pd.DataFrame(index=snapshots, columns=[], dtype=float)
            df.index.name = "snapshot"
            df.columns.name = ct.name
            dynamic[k] = df

        return static, dynamic

    @property
    def standard_types(self) -> pd.DataFrame | None:
        """
        Get standard types of component.

        It is an alias for the `standard_types` attribute of the underlying
        :class:`pypsa.definitions.ComponentType`.

        Returns
        -------
        pd.DataFrame
            DataFrame with standard types of component.

        """
        return self.ctype.standard_types

    @property
    def name(self) -> str:
        """
        Name of component type.

        Returns
        -------
        str
            Name of component.

        See Also
        --------
        pypsa.definitions.ComponentType :
            This property directly references the same property in the
            associated underlying class.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.components.generators.name
        'Generator'

        """
        return self.ctype.name

    @property
    def list_name(self) -> str:
        """
        List name of component type.

        Returns
        -------
        str
            List name of component.

        See Also
        --------
        pypsa.definitions.ComponentType :
            This property directly references the same property in the
            associated underlying class.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.components.generators.list_name
        'generators'

        """
        return self.ctype.list_name

    @property
    def description(self) -> str:
        """
        Description of component.

        Returns
        -------
        str
            Description of component.

        See Also
        --------
        pypsa.definitions.ComponentType :
            This property directly references the same property in the
            associated underlying class.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.components.generators.description
        'Power generator.'

        """
        return self.ctype.description

    @property
    def category(self) -> str:
        """
        Category of component.

        Returns
        -------
        str
            Category of component.

        See Also
        --------
        pypsa.definitions.ComponentType :
            This property directly references the same property in the
            associated underlying class.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.components.generators.category
        'controllable_one_port'

        """
        return self.ctype.category

    @property
    def type(self) -> str:
        """
        Get category of component.

        .. note ::
            While not actively deprecated yet, :meth:`category` is the preferred method
            to access component type.

        Returns
        -------
        str
            Category of component.

        See Also
        --------
        pypsa.definitions.ComponentType :
            This property directly references the same property in the
            associated underlying class.

        """
        return self.ctype.category

    @property
    def attrs(self) -> pd.DataFrame:
        """
        Default values of corresponding component type.

        .. note::
            While not actively deprecated yet, :meth:`defaults` is the preferred method
            to access component attributes.

        Returns
        -------
        pd.DataFrame
            DataFrame with component attribute names as index and the information
            like type, unit, default value and description as columns.

        See Also
        --------
        pypsa.definitions.ComponentType :
            This property directly references the same property in the
            associated underlying class.


        """
        return self.ctype.defaults

    @property
    def defaults(self) -> pd.DataFrame:
        """
        Default values of corresponding component type.

        .. note::
            While not actively deprecated yet, :meth:`defaults` is the preferred method
            to access component attributes.

        Returns
        -------
        pd.DataFrame
            DataFrame with component attribute names as index and the information
            like type, unit, default value and description as columns.

        See Also
        --------
        pypsa.definitions.ComponentType :
            This property directly references the same property in the
            associated underlying class.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.components.generators.defaults.head() # doctest: +SKIP
                     type unit default                                        description            status  static  varying              typ    dtype
        attribute
        name       string  NaN                                                Unique name  Input (required)    True    False    <class 'str'>   object
        bus        string  NaN                 name of bus to which generator is attached  Input (required)    True    False    <class 'str'>   object
        control    string  NaN      PQ  P,Q,V control strategy for PF, must be "PQ", "...  Input (optional)    True    False    <class 'str'>   object
        type       string  NaN          Placeholder for generator type. Not yet implem...  Input (optional)    True    False    <class 'str'>   object
        p_nom       float   MW     0.0          Nominal power for limits in optimization.  Input (optional)    True    False  <class 'float'>  float64

        """
        return self.ctype.defaults

    @property
    def component_names(self) -> pd.Index:
        return self.static.index.get_level_values(self.ctype.name).unique()

    @property
    def snapshots(self) -> pd.Index:
        return self.n_save.snapshots

    @property
    def timesteps(self) -> pd.Index:
        return self.n_save.timesteps

    @property
    def investment_periods(self) -> pd.Index:
        return self.n_save.investment_periods

    @property
    def has_investment_periods(self) -> bool:
        return self.n_save.has_investment_periods

    @property
    def scenarios(self) -> pd.Index:
        return self.n_save.scenarios

    @property
    def has_scenarios(self) -> bool:
        return self.n_save.has_scenarios

    def get(self, attribute_name: str, default: Any = None) -> Any:
        """
        Get attribute of component.

        Just an alias for built-in getattr and allows for default values.

        Parameters
        ----------
        attribute_name : str
            Name of the attribute to get.
        default : Any, optional
            Default value to return if attribute is not found.

        Returns
        -------
        Any
            Value of the attribute if found, otherwise the default value.

        """
        return getattr(self, attribute_name, default)

    @property
    def attached(self) -> bool:
        """
        Check if component is attached to a Network.

        Some functionality of the component is only available when attached to a
        Network.

        Returns
        -------
        bool
            True if component is attached to a Network, otherwise False.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.components.generators.attached
        True

        """
        return self.n is not None

    @property
    def n_save(self) -> Any:
        """A save property to access the network (component must be attached)."""
        if not self.attached:
            raise AttributeError("Component must be attached to a Network.")
        return self.n

    @property
    def df(self) -> pd.DataFrame:
        """
        Get static data of all components as pandas DataFrame.

        .. note::
            While not actively deprecated yet, :meth:`static` is the preferred method
            to access static components data.

        Returns
        -------
        pd.DataFrame
            DataFrame with components as index and attributes as columns.

        """
        return self.static

    @property
    def pnl(self) -> dict:
        """
        Get dynamic data of all components as a dictionary of pandas DataFrames.

        .. note::
            While not actively deprecated yet, :meth:`dynamic` is the preferred method
            to access dynamic components data.

        Returns
        -------
        dict
            Dictionary of dynamic components. Keys are the attribute and each value is
            a pandas DataFrame with snapshots as index and the component names as
            columns.

        """
        return self.dynamic

    @property
    def ds(self) -> xarray.Dataset:
        ds = self.static.to_xarray().assign_coords(
            snapshot=("snapshot", self.snapshots)
        )

        if not self.has_investment_periods:
            # Only needed for single index, when dimension name is can not be different
            # from the coordinate name (e.g. "snapshot" == "timestep")
            ds = ds.assign_coords(timestep=("snapshot", self.timesteps))

        for k, v in self.dynamic.items():
            assert v.index.name == "snapshot"
            # Empty dfs must be handled separately, since stack will remove the index
            if v.empty:
                da = xarray.DataArray(np.nan, coords=ds.coords, dims=ds.dims)
            else:
                da = v.stack().to_xarray().reindex({self.name: self.component_names})
            if k in self.static:
                da = da.fillna(self.static[k].to_xarray())
            ds[k] = da
        return ds

    @property
    def nominal_attr(self) -> pd.Index:
        # TODO: MOve to components.py
        nominal_attrs = {
            "Generator": "p_nom",
            "Line": "s_nom",
            "Transformer": "s_nom",
            "Link": "p_nom",
            "Store": "e_nom",
            "StorageUnit": "p_nom",
        }
        if self.ctype.name not in nominal_attrs:
            raise ValueError(f"Nominal attribute not defined for {self.ctype.name}")
        return nominal_attrs[self.ctype.name]

    def get_extendable_i(self) -> pd.Index:
        """
        Get the index of extendable elements of a given component.
        """
        idx = self.static[lambda ds: ds[self.nominal_attr + "_extendable"]].index
        return idx.rename(f"{self.name}-ext")

    def get_non_extendable_i(self) -> pd.Index:
        """
        Get the index of non-extendable elements of a given component.
        """
        idx = self.static[lambda ds: ~ds[self.nominal_attr + "_extendable"]].index
        return idx.rename(f"{self.name}-non-ext")

    def copy(self) -> Components:
        return copy.deepcopy(self)

    @overload
    def get_active_assets(
        self,
        investment_period: int | str | Sequence | None = None,
        as_xarray: Literal[True] = True,
    ) -> xarray.DataArray: ...

    @overload
    def get_active_assets(
        self,
        investment_period: int | str | Sequence | None = None,
        as_xarray: Literal[False] = False,
    ) -> pd.Series: ...

    def get_active_assets(
        self,
        investment_period: int | str | Sequence | None = None,
        as_xarray: bool = False,
    ) -> pd.Series | xarray.DataArray:
        """
        Get active components mask of component type in investment period(s).

        A component is considered active when:
        - its active attribute is True
        - if an investment_period is provided and if build_year and lifetime exist in self.static,
        the component is active in that period if:
            build_year <= period < build_year + lifetime.
        If multiple periods are given, the asset is active if it is active in any period.
        If investment_period is None, only the active attribute is considered.

        Parameters
        ----------
        investment_period : int, str, Sequence, or None, optional
            Investment period(s) to check for active status. If provided, build_year and lifetime
            are used to determine activity. A ValueError is raised if any period is not in
            self.n_save.investment_periods.
        as_xarray : bool, default False
            If True, the result is returned as an xarray.DataArray; otherwise as a pandas.Series.

        Returns
        -------
        pd.Series or xarray.DataArray
            Boolean mask for active components.
        """
        # If no investment period is provided or build_year/lifetime information is missing,
        # fall back to the basic active mask.
        if investment_period is None or not {"build_year", "lifetime"}.issubset(
            self.static.columns
        ):
            active_result = (
                self.static.active
                if hasattr(self.static, "active")
                else pd.Series(True, index=self.static.index)
            )
        else:
            raise NotImplementedError("Investment period not yet implemented.")
            # active = {}
            # # Ensure investment_period is iterable
            # for period in np.atleast_1d(investment_period):
            #     if period not in self.n_save.investment_periods:
            #         raise ValueError("Investment period not in `n.investment_periods`")
            #     active[period] = self.static.eval(
            #         "build_year <= @period < build_year + lifetime"
            #     )
            # # Logical OR across all periods, then AND with the base active attribute
            # active_result = pd.DataFrame(active).any(axis=1) & self.static.active

        if as_xarray:
            return (
                active_result
                if isinstance(active_result, xarray.DataArray)
                else xarray.DataArray(active_result)
            )
        else:
            return (
                active_result
                if isinstance(active_result, pd.Series)
                else active_result.to_series()
            )

    @overload
    def get_bounds_pu(
        self,
        c: str,
        sns: Sequence,
        index: pd.Index | None = None,
        attr: str | None = None,
        as_xarray: Literal[True] = True,
    ) -> tuple[xarray.DataArray, xarray.DataArray]: ...

    @overload
    def get_bounds_pu(
        self,
        c: str,
        sns: Sequence,
        index: pd.Index | None = None,
        attr: str | None = None,
        as_xarray: Literal[False] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    def get_bounds_pu(
        self,
        c: str,
        sns: Sequence,
        index: pd.Index | None = None,
        attr: str | None = None,
        as_xarray: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Getter function to retrieve the per unit bounds of a given compoent for
        given snapshots and possible subset of elements (e.g. non-extendables).
        Depending on the attr you can further specify the bounds of the variable
        you are looking at, e.g. p_store for storage units.

        Parameters
        ----------
        c : string
            Component name, e.g. "Generator", "Line".
        sns : pandas.Index/pandas.DateTimeIndex
            set of snapshots for the bounds
        index : pd.Index, default None
            Subset of the component elements. If None (default) bounds of all
            elements are returned.
        attr : string, default None
            attribute name for the bounds, e.g. "p", "s", "p_store"
        as_xarray : bool, default False
            If True, return as xarray.DataArray

        Returns
        -------
        min_pu, max_pu : tuple(pd.DataFrame, pd.DataFrame) or tuple(xr.DataArray, xr.DataArray) -> Minimum and maximum per unit bounds
        """
        min_pu_str = nominal_attrs[c].replace("nom", "min_pu")
        max_pu_str = nominal_attrs[c].replace("nom", "max_pu")

        max_pu = get_as_dense(self.n, c, max_pu_str, sns)

        if c in self.n.passive_branch_components:
            min_pu = -max_pu
        elif c == "StorageUnit":
            min_pu = pd.DataFrame(0, index=max_pu.index, columns=max_pu.columns)
            if attr == "p_store":
                max_pu = -get_as_dense(self.n, c, min_pu_str, sns, inds=index)
            if attr == "state_of_charge":
                max_pu = get_as_dense(self.n, c, "max_hours", sns, inds=index)
        else:
            min_pu = get_as_dense(self.n, c, min_pu_str, sns, inds=index)

        if index is not None:
            min_pu = min_pu.reindex(columns=index)
            max_pu = max_pu.reindex(columns=index)

        if as_xarray:
            min_pu = xarray.DataArray(min_pu)
            max_pu = xarray.DataArray(max_pu)

        return min_pu, max_pu


class SubNetworkComponents:
    """
    Wrapper class to allow for custom attribute handling of components.

    SubNetworkComponents are read-only and delegate attribute access to it's wrapped
    Components object of the PyPSA Network. This allows for custom attribute handling
    and getter functions to be implemented, e.g. to filter sub-network specific
    components from the main network components.

    Also See
    --------
    pypsa.components.abstract.Components : Base class for all PyPSA components in the
    network.
    """

    def __init__(self, wrapped_data: Components, wrapped_get: Callable) -> None:
        """
        Initialize SubNetworkComponents.

        Parameters
        ----------
        wrapped_data : Components
            Components object to wrap around.
        wrapped_get : Callable
            Custom getter function to delegate attribute access to the wrapped data
            object and allow for custom attribute handling.

        Returns
        -------
        None

        """
        self._wrapped_data = wrapped_data
        self._wrapper_func = wrapped_get

    def __getattr__(self, item: str) -> Any:
        """
        Delegate attribute access to the wrapped data object.

        Parameters
        ----------
        item : str
            Attribute name to access.

        Returns
        -------
        Any
            Attribute value of the wrapped data object.

        """
        return self._wrapper_func(item, self._wrapped_data)

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Prevent setting of attributes.

        Parameters
        ----------
        key : str
            Attribute name to set.
        value : Any
            Attribute value to set.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If attribute setting is attempted.

        """
        if key in {"_wrapped_data", "_wrapper_func"}:
            super().__setattr__(key, value)
        else:
            raise AttributeError("SubNetworkComponents is read-only")

    def __delattr__(self, name: str) -> None:
        """
        Prevent deletion of attributes.

        Parameters
        ----------
        name : str
            Attribute name to delete.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If attribute deletion is attempted.

        """
        raise AttributeError("SubNetworkComponents is read-only")
