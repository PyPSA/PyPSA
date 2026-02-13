# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Components module.

Contains classes and properties relevant to all component types in PyPSA. Also imports
logic from other modules:
- components.types

Contains classes and logic relevant to specific component types in PyPSA.
Generic functionality is implemented in the abstract module.
ponents module.
Components module.

Contains classes and logic relevant to specific component types in PyPSA.
Generic functionality is implemented in the abstract module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
import xarray
from pyproj import CRS

from pypsa.common import deprecated_in_next_major, equals
from pypsa.components.array import ComponentsArrayMixin
from pypsa.components.descriptors import ComponentsDescriptorsMixin
from pypsa.components.index import ComponentsIndexMixin
from pypsa.components.transform import ComponentsTransformMixin
from pypsa.constants import DEFAULT_EPSG, DEFAULT_TIMESTAMP, RE_PORTS
from pypsa.costs import annuity, periodized_cost
from pypsa.definitions.structures import Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pypsa import Network
    from pypsa.definitions.components import ComponentType

# TODO attachment todos
# - crs
# - snapshots, investment_periods


@dataclass
class ComponentsData:
    """Dataclass for Components.

    <!-- md:guide components.md -->

    This class is used to store all data of a Components object. Other classes inherit
    from this class to implement logic and methods, but do not store any data next
    to the data in here.

    All attributes can therefore also be accessed directly from
    any [`Components`][pypsa.Components] object (which defines all
    attributes and properties which are available for all component types) as well as
    in specific type classes as [`Generators`][pypsa.components.Generators] (which
    define logic and methods specific to the component type).

    Attributes
    ----------
    ctype : ComponentType
        Component type information containing all default values and attributes. #TODO
    n : Network | None
        Network to which the component might be attached.
    static : pd.DataFrame
        Static data of components as a pandas DataFrame. Columns are the attributes
        and the index is the component name.
    dynamic : dict
        Dynamic (time-varying) data of components as a dict-like object of pandas
        DataFrames. Keys of the dict are the attribute names and each value is a pandas
        DataFrame with snapshots as index and the component names as columns.

    """

    ctype: ComponentType
    n: Network | None
    static: pd.DataFrame
    """
    Dataframe with static data for all components of this type.

    Returns
    -------
    pandas.DataFrame
        Static data of the component.

    Examples
    --------
    >>> c.static
    """
    dynamic: Dict
    """
    Dataframe with dynamic data for all components of this type.

    Returns
    -------
    pandas.DataFrame
        Dynamic data of the component.

    Examples
    --------
    >>> c.dynamic
    """


class Components(
    ComponentsData,
    ComponentsDescriptorsMixin,
    ComponentsTransformMixin,
    ComponentsIndexMixin,
    ComponentsArrayMixin,
):
    """Components base class.

    <!-- md:badge-version v0.33.0 --> | <!-- md:guide components.md -->

    Base class for container of energy system related assets, such as
    generators or transmission lines. Use the specific subclasses for concrete or
    a generic component type.
    All data is stored in the dataclass [ComponentsData][pypsa.components.components.ComponentsData].
    Components inherits from it, adds logic and methods, but does not store any data
    itself.

    """

    def __init__(
        self,
        ctype: ComponentType,
        n: Network | None = None,
        names: str | int | Sequence[int | str] | None = None,
        suffix: str = "",
    ) -> None:
        """Initialize Components object.

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
        ComponentsData.__init__(self, ctype, n=None, static=static, dynamic=dynamic)
        ComponentsArrayMixin.__init__(self)

    def __str__(self) -> str:
        """Get string representation of component.

        <!-- md:badge-version v0.33.0 -->

        Examples
        --------
        >>> str(n.components.generators)
        "'Generator' Components"

        """
        return f"'{self.ctype.name}' Components"

    def __repr__(self) -> str:
        """Get representation of component.

        <!-- md:badge-version v0.33.0 -->

        Examples
        --------
        >>> c = n.components.generators
        >>> c
        'Generator' Components
        ----------------------
        Attached to PyPSA Network 'AC-DC-Meshed'
        Components: 6

        """
        num_components = len(self.static)
        if not num_components:
            return f"Empty {self}"
        text = f"{self}\n" + "-" * len(str(self)) + "\n"

        # Add attachment status
        if self.attached:
            text += f"Attached to {str(self.n)}\n"

        text += f"Components: {len(self.static)}"

        return text

    def __getitem__(self, key: str) -> Any:
        """Get attribute of component.

        <!-- md:badge-version v0.33.0 -->

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
        """Set attribute of component.

        <!-- md:badge-version v0.33.0 -->

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
            msg = f"'{key}' not found in Component"
            raise KeyError(msg)

    def __eq__(self, other: object) -> bool:
        """Check if two Components are equal.

        <!-- md:badge-version v0.33.0 -->

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

        See Also
        --------
        [pypsa.Components.equals][]

        """
        return self.equals(other)

    def __len__(self) -> int:
        """Get the number of components.

        <!-- md:badge-version v1.0.0 -->

        Returns
        -------
        int
            Number of components.

        Examples
        --------
        >>> len(n.components.generators)
        6

        Which is the same as:
        >>> n.components.generators.static.shape[0]
        6

        """
        return len(self.static)

    def equals(self, other: Any, log_mode: str = "silent") -> bool:
        """Check if two Components are equal.

        <!-- md:badge-version v0.33.0 -->

        Does not check the attached Network, but only component specific data. Therefore
        two components can be equal even if they are attached to different networks.

        Parameters
        ----------
        other : Any
            The other network to compare with.
        log_mode: str, default="silent"
            Controls how differences are reported:
            - 'silent': No logging, just returns True/False
            - 'verbose': Prints differences but doesn't raise errors
            - 'strict': Raises ValueError on first difference

        Raises
        ------
        ValueError
            If log_mode is 'strict' and components are not equal.

        Returns
        -------
        bool
            True if components are equal, otherwise False.

        Examples
        --------
        >>> n1 = pypsa.Network()
        >>> n2 = pypsa.Network()
        >>> n1.add("Bus", "bus1")
        >>> n2.add("Bus", "bus1")
        >>> n1.buses.equals(n2.buses)
        True

        """
        return (
            equals(self.ctype, other.ctype, log_mode=log_mode, path="c.ctype")
            and equals(self.static, other.static, log_mode=log_mode, path="c.static")
            and equals(self.dynamic, other.dynamic, log_mode=log_mode, path="c.dynamic")
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
        static.index.name = "name"

        # # it's currently hard to imagine non-float series,
        # but this could be generalised
        dynamic = Dict()
        snapshots = pd.Index(
            [DEFAULT_TIMESTAMP]
        )  # if n is None else n.snapshots #TODO attach mechanism
        for k in ct.defaults.index[ct.defaults.varying]:
            df = pd.DataFrame(index=snapshots, columns=[], dtype=float)
            df.index.name = "snapshot"
            df.columns.name = "name"
            dynamic[k] = df

        return static, dynamic

    @property
    def standard_types(self) -> pd.DataFrame | None:
        """Get standard types of component.

        <!-- md:badge-version v0.33.0 -->

        Returns
        -------
        pd.DataFrame
            DataFrame with standard types of component.

        Examples
        --------
        >>> n.components.transformers.standard_types

        """
        return self.ctype.standard_types

    @property
    def name(self) -> str:
        """Name of component type.

        <!-- md:badge-version v0.33.0 -->

        Returns
        -------
        str
            Name of component.

        Examples
        --------
        >>> n.components.generators.name
        'Generator'

        """
        return self.ctype.name

    @property
    def list_name(self) -> str:
        """List name of component type.

        <!-- md:badge-version v0.33.0 -->

        Returns
        -------
        str
            List name of component.

        Examples
        --------
        >>> n.components.generators.list_name
        'generators'

        """
        return self.ctype.list_name

    @property
    def description(self) -> str:
        """Description of component.

        <!-- md:badge-version v0.33.0 -->

        Returns
        -------
        str
            Description of component.

        Examples
        --------
        >>> n.components.generators.description
        'Power generator for the bus carrier it attaches to.'

        """
        return self.ctype.description

    @property
    def category(self) -> str:
        """Category of component.

        <!-- md:badge-version v0.33.0 -->

        Returns
        -------
        str
            Category of component.

        Examples
        --------
        >>> n.components.generators.category
        'controllable_one_port'

        """
        return self.ctype.category

    @property
    def type(self) -> str:
        """Get category of component.

        <!-- md:badge-version v0.33.0 -->

        !!! note
            While not actively deprecated yet, [`category`][pypsa.Components.category] is the preferred method
            to access component type.

        Returns
        -------
        str
            Category of component.

        """
        return self.ctype.category

    @property
    @deprecated_in_next_major(details="Use `c.defaults` instead.")
    def attrs(self) -> pd.DataFrame:
        """Default values of corresponding component type.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use [`c.defaults`][pypsa.Components.defaults] instead.

        Returns
        -------
        pd.DataFrame
            DataFrame with component attribute names as index and the information
            like type, unit, default value and description as columns.

        """
        return self.ctype.defaults

    @property
    def defaults(self) -> pd.DataFrame:
        """Default values of corresponding component type.

        <!-- md:badge-version v0.33.0 -->

        Returns
        -------
        pd.DataFrame
            DataFrame with component attribute names as index and the information
            like type, unit, default value and description as columns.

        Examples
        --------
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
    def empty(self) -> bool:
        """Check if component is empty.

        <!-- md:badge-version v1.0.0 -->

        Returns
        -------
        bool
            True if component is empty, otherwise False.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add('Generator', 'g1')
        >>> n.components.generators.empty
        False

        >>> n.components.buses.empty
        True

        """
        return self.static.empty

    @property
    def attached(self) -> bool:
        """Check if component is attached to a Network.

        <!-- md:badge-version v0.33.0 -->

        Some functionality of the component is only available when attached to a
        Network.

        Returns
        -------
        bool
            True if component is attached to a Network, otherwise False.

        Examples
        --------
        >>> n.components.generators.attached
        True

        """
        return self.n is not None

    @property
    def n_save(self) -> Any:
        """A save property to access the network (component must be attached).

        <!-- md:badge-version v0.33.0 -->

        Returns
        -------
        Network
            Network to which the component is attached.

        Raises
        ------
        AttributeError
            If component is not attached to a Network.

        """
        if not self.attached:
            msg = "Component must be attached to a Network."
            raise AttributeError(msg)
        return self.n

    @property
    @deprecated_in_next_major(details="Use `c.static` instead.")
    def df(self) -> pd.DataFrame:
        """Get static data of all components as pandas DataFrame.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"
            Use [`c.static`][pypsa.Components.static] instead.

        Returns
        -------
        pd.DataFrame
            DataFrame with components as index and attributes as columns.

        """
        return self.static

    @property
    @deprecated_in_next_major(details="Use `c.dynamic` instead.")
    def pnl(self) -> dict:
        """Get dynamic data of all components as a dictionary of pandas DataFrames.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"
            Use [`c.dynamic`][pypsa.Components.dynamic] instead.

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
        """Create a xarray data array view of the component.

        <!-- md:badge-version v1.0.0 -->

        !!! note

            Note that this will create a full copy of the component data. For large networks
            this may be a bottleneck. Use the [pypsa.Components.da][] accessor instead to
            access a specific attribute of the component.

        Returns
        -------
        xarray.Dataset
            Dataset with component attributes as variables and snapshots as coordinates.

        See Also
        --------
        [pypsa.Components.da][]

        Examples
        --------
        >>> c = n.components.generators
        >>> c.ds  # doctest: +ELLIPSIS
        <xarray.Dataset> Size: ...
        Dimensions:                  (name: 6, snapshot: 10)
        Coordinates:
          * name                     (name) object ... 'Manchester Wind' ... 'Frankfu...
          * snapshot                 (snapshot) datetime64[ns] ... 2015-01-01 ... 201...
        Data variables: (12/43)
            bus                      (name) object ... 'Manchester' ... 'Frankfurt'
            control                  (name) object ... 'Slack' 'PQ' ... 'Slack' 'PQ'
            type                     (name) object ... '' '' '' '' '' ''
            p_nom                    (name) float64 ... 80.0 5e+04 100.0 ... 110.0 8e+04
            p_nom_mod                (name) float64 ... 0.0 0.0 0.0 0.0 0.0 0.0
            p_nom_extendable         (name) bool ... True True True True True True
            ...

        """
        data = {}

        for attr in self.static.columns:
            data[attr] = self._as_xarray(attr)

        for attr, df in self.dynamic.items():
            if not df.empty:
                data[attr] = self._as_xarray(attr)

        return xarray.Dataset(data)

    @property
    def units(self) -> pd.Series:
        """Get units of all attributes of components.

        <!-- md:badge-version v0.34.0 -->

        Returns
        -------
        pd.Series
            Series with attribute names as index and units as values.

        Examples
        --------
        >>> c = n.components.generators
        >>> c.units.head() # doctest: +SKIP
                        unit
        attribute
        p_nom            MW
        p_nom_mod        MW
        p_nom_min        MW
        p_nom_max        MW
        p_min_pu   per unit

        """
        return self.defaults.unit[self.defaults.unit.notnull()].to_frame()

    @property
    def ports(self) -> list:
        """Get ports of all components.

        <!-- md:badge-version v0.34.0 -->

        Returns
        -------
        pd.Series
            Series with attribute names as index and port names as values.

        Examples
        --------
        >>> c = n.components.lines
        >>> c.ports
        ['0', '1']

        See Also
        --------
        [pypsa.components.Links.additional_ports][]

        """
        return [
            match.group(1) for col in self.static if (match := RE_PORTS.search(col))
        ]

    @property
    def unique_carriers(self) -> set[str]:
        """Get all unique carrier values for this component.

        <!-- md:badge-version v1.1.0 -->

        Returns
        -------
        set of str
            Set of all unique carrier names found in this component.

        Examples
        --------
        >>> sorted(n.c.generators.unique_carriers)
        ['gas', 'wind']

        >>> sorted(n.c.buses.unique_carriers)
        ['AC', 'DC']

        See Also
        --------
        [pypsa.components.Carriers.add_missing_carriers][]

        """
        if self.static.empty or "carrier" not in self.static.columns:
            return set()

        # Get carriers and filter out empty strings and NaN
        c_carriers = self.static["carrier"].dropna()
        c_carriers = c_carriers[c_carriers != ""]
        return set(c_carriers.unique())

    @property
    def extendables(self) -> pd.Index:
        """Get the index of extendable elements of this component.

        <!-- md:badge-version v1.0.0 -->

        Returns
        -------
        pd.Index
            Single-level index of extendable elements.

        """
        extendable_col = self._operational_attrs["nom_extendable"]
        if extendable_col not in self.static.columns:
            return self.static.iloc[:0].index

        idx = self.static.loc[self.static[extendable_col]].index

        # Remove scenario dimension, since they cannot vary across scenarios
        if self.has_scenarios:
            idx = idx.get_level_values("name").drop_duplicates()

        return idx

    @property
    def fixed(self) -> pd.Index:
        """Get the index of non-extendable elements of this component.

        <!-- md:badge-version v1.0.0 -->

        Returns
        -------
        pd.Index
            Single-level index of non-extendable elements.

        """
        extendable_col = self._operational_attrs["nom_extendable"]
        if extendable_col not in self.static.columns:
            return self.static.iloc[:0].index

        idx = self.static.loc[~self.static[extendable_col]].index

        # Remove scenario dimension, since they cannot vary across scenarios
        if self.has_scenarios:
            idx = idx.get_level_values("name").drop_duplicates()

        return idx

    @property
    def committables(self) -> pd.Index:
        """Get the index of committable elements of this component.

        <!-- md:badge-version v1.0.0 -->

        Returns
        -------
        pd.Index
            Single-level index of committable elements.

        """
        if "committable" not in self.static:
            return self.static.iloc[:0].index

        idx = self.static.loc[self.static["committable"]].index

        # Remove scenario dimension, since they cannot vary across scenarios
        if self.has_scenarios:
            idx = idx.get_level_values("name").drop_duplicates()

        return idx

    @property
    def periodized_cost(self) -> xarray.DataArray:
        """Calculate periodized cost from component attributes as xarray DataArray.

        <!-- md:badge-version v1.1.0 -->

        See Also
        --------
        `pypsa.costs.periodized_cost`

        """
        static = self.static
        cost = periodized_cost(
            capital_cost=static["capital_cost"],
            overnight_cost=static["overnight_cost"],
            discount_rate=static["discount_rate"],
            lifetime=static["lifetime"],
            fom_cost=static.get("fom_cost", 0),
            nyears=self.nyears,
        )
        da = xarray.DataArray(cost)
        if self.has_scenarios:
            da = da.unstack().reindex(name=self.names, scenario=self.scenarios)
        return da

    @property
    def capital_cost(self) -> pd.Series:
        """Calculate annuitized investment cost per unit of capacity (no fom).

        <!-- md:badge-version v1.1.0 -->

        See Also
        --------
        `pypsa.costs.periodized_cost`

        """
        static = self.static
        return periodized_cost(
            capital_cost=static["capital_cost"],
            overnight_cost=static["overnight_cost"],
            discount_rate=static["discount_rate"],
            lifetime=static["lifetime"],
            fom_cost=None,
            nyears=self.nyears,
        )

    @property
    def nyears(self) -> float | pd.Series:
        """Return the modeled time horizon in years.

        <!-- md:badge-version v1.1.0 -->

        See Also
        --------
        `pypsa.Network.nyears`

        """
        return self.n_save.nyears

    @property
    def annuity(self) -> pd.Series:
        """Calculate annuity factor for all components.

        <!-- md:badge-version v1.1.0 -->

        Returns the annuity factor based on `discount_rate` and `lifetime`.
        If `discount_rate` is NaN (no `overnight_cost` provided), returns 1.0.

        Returns
        -------
        pd.Series
            Annuity factor for each component.

        Examples
        --------
        >>> n.c.generators.annuity  # doctest: +SKIP
        name
        gen1    0.085...
        gen2    1.0
        dtype: float64

        See Also
        --------
        `pypsa.costs.annuity_factor`

        """
        static = self.static
        discount_rate = static["discount_rate"]
        lifetime = static["lifetime"]
        return annuity(discount_rate, lifetime)

    @property
    def overnight_cost(self) -> pd.Series:
        """Calculate overnight cost from component attributes.

        <!-- md:badge-version v1.1.0 -->

        If overnight_cost column is provided (not NaN), returns it directly.
        Otherwise, converts annualized capital_cost back to overnight cost using
        the formula: overnight_cost = capital_cost / (annuity_factor × nyears).

        Note: When nyears == 1, capital_cost represents the annualized cost per year,
        so overnight_cost = capital_cost / annuity_factor.

        Returns
        -------
        pd.Series
            Overnight (upfront) investment cost per unit of capacity.

        Examples
        --------
        >>> n.c.generators.overnight_cost  # doctest: +SKIP
        name
        gen1    1000.0   # overnight_cost used directly
        gen2    1166.0   # 100 / annuity(0.07, 25) - back-calculated from capital_cost
        dtype: float64

        See Also
        --------
        `capital_cost` : Annuitized investment cost for the modeled horizon.
        `annuity` : Annuity factor for each component.

        """
        static = self.static
        overnight = static["overnight_cost"]
        capital = static["capital_cost"]
        has_overnight = overnight.notna()

        needs_back_calc = ~has_overnight & (capital != 0)
        discount_rate = static["discount_rate"]
        lifetime = static["lifetime"]
        missing_params = needs_back_calc & (discount_rate.isna() | lifetime.isna())

        if missing_params.any():
            bad = static.index[missing_params].tolist()
            msg = (
                f"Cannot back-calculate overnight_cost for {bad}: "
                "both 'discount_rate' and 'lifetime' must be provided "
                "when 'overnight_cost' is not set."
            )
            raise ValueError(msg)

        ann_factor = self.annuity
        nyears = self.nyears
        nyears_scalar = nyears.mean() if isinstance(nyears, pd.Series) else nyears
        back_calculated = capital / (ann_factor * nyears_scalar)

        return overnight.where(has_overnight, back_calculated)


class SubNetworkComponents:
    """Wrapper class to allow for custom attribute handling of components.

    SubNetworkComponents are read-only and delegate attribute access to it's wrapped
    Components object of the PyPSA Network. This allows for custom attribute handling
    and getter functions to be implemented, e.g. to filter sub-network specific
    components from the main network components.

    Also See
    --------
    pypsa.Components : Base class for all PyPSA components in the
    network.
    """

    def __init__(self, wrapped_data: Components, wrapped_get: Callable) -> None:
        """Initialize SubNetworkComponents.

        Parameters
        ----------
        wrapped_data : Components
            Components object to wrap around.
        wrapped_get : Callable
            Custom getter function to delegate attribute access to the wrapped data
            object and allow for custom attribute handling.

        """
        self._wrapped_data = wrapped_data
        self._wrapper_func = wrapped_get

    def __getattr__(self, item: str) -> Any:
        """Delegate attribute access to the wrapped data object.

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
        """Prevent setting of attributes.

        Parameters
        ----------
        key : str
            Attribute name to set.
        value : Any
            Attribute value to set.

        Raises
        ------
        AttributeError
            If attribute setting is attempted.

        """
        if key in {"_wrapped_data", "_wrapper_func"}:
            super().__setattr__(key, value)
        else:
            msg = "SubNetworkComponents is read-only"
            raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Prevent deletion of attributes.

        Parameters
        ----------
        name : str
            Attribute name to delete.

        Raises
        ------
        AttributeError
            If attribute deletion is attempted.

        """
        msg = "SubNetworkComponents is read-only"
        raise AttributeError(msg)

    def __str__(self) -> str:
        """Get string representation of sub-network components.

        Examples
        --------
        >>> str(sub_network.components.generators)
        "'Generator' SubNetworkComponents"

        """
        return f"'{self.ctype.name}' SubNetworkComponents"

    def __repr__(self) -> str:
        """Get representation of sub-network components.

        Examples
        --------
        >>> sub_network.components.generators
        'Generator' SubNetworkComponents
        --------------------------------
        Attached to Sub-Network of PyPSA Network 'AC-DC-Meshed'
        Components: 6

        """
        num_components = len(self._wrapped_data.static)
        if not num_components:
            return f"Empty {self}"
        text = f"{self}\n" + "-" * len(str(self)) + "\n"

        # Add attachment status
        if self.attached:
            text += f"Attached to Sub-Network of {str(self.n)}\n"

        text += f"Components: {len(self._wrapped_data.static)}"

        return text
