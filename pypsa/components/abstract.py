"""
Abstract components module.

Contains classes and properties relevant to all component types in PyPSA. Also imports
logic from other modules:
- components.types
"""

from __future__ import annotations

import logging
from abc import ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from pypsa.common import equals
from pypsa.components.descriptors import get_active_assets
from pypsa.constants import DEFAULT_EPSG, DEFAULT_TIMESTAMP
from pypsa.definitions.components import ComponentType
from pypsa.definitions.structures import Dict

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

    # Methods
    # -------

    # from pypsa.components.descriptors
    get_active_assets = get_active_assets

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

    def __str__(self) -> str:
        """
        Get string representation of component.

        Returns
        -------
        str
            String representation of component.

        Examples
        --------
        >>> str(n.components.generators)
        "PyPSA 'Generator' Components"

        """
        return f"PyPSA '{self.ctype.name}' Components"

    def __repr__(self) -> str:
        """
        Get representation of component.

        Returns
        -------
        str
            Representation of component.

        Examples
        --------
        >>> c = n.components.generators
        >>> c
        PyPSA 'Generator' Components
        ----------------------------
        Attached to PyPSA Network 'AC-DC'
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
    def units(self) -> pd.Series:
        """
        Get units of all attributes of components.

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
        """
        Get ports of all components.

        Returns
        -------
        pd.Series
            Series with attribute names as index and port names as values.

        Examples
        --------
        >>> c = n.components.lines
        >>> c.ports
        ['0', '1']

        """
        return [str(col)[3:] for col in self.static if str(col).startswith("bus")]

    @property
    def nominal_attr(self) -> str:
        """
        Get nominal attribute of component.

        Returns
        -------
        str
            Name of the nominal attribute of the component.

        Examples
        --------
        >>> c = n.components.generators
        >>> c.nominal_attr
        'p_nom'

        """
        # TODO: move to Component Specific class
        nominal_attr = {
            "Generator": "p_nom",
            "Line": "s_nom",
            "Transformer": "s_nom",
            "Link": "p_nom",
            "Store": "e_nom",
            "StorageUnit": "p_nom",
        }
        try:
            return nominal_attr[self.ctype.name]
        except KeyError:
            msg = f"Component type '{self.ctype.name}' has no nominal attribute."
            raise AttributeError(msg)

    # TODO move
    def rename_component_names(self, **kwargs: str) -> None:
        """
        Rename component names.

        Rename components and also update all cross-references of the component in
        the network.

        Parameters
        ----------
        **kwargs
            Mapping of old names to new names.

        Returns
        -------
        None

        Examples
        --------
        Define some network

        >>> n = pypsa.Network()
        >>> n.add("Bus", ["bus1"])
        Index(['bus1'], dtype='object')
        >>> n.add("Generator", ["gen1"], bus="bus1")
        Index(['gen1'], dtype='object')
        >>> c = n.c.buses

        Now rename the bus

        >>> c.rename_component_names(bus1="bus2")

        Which updates the bus components

        >>> c.static.index
        Index(['bus2'], dtype='object', name='Bus')

        and all references in the network

        >>> n.generators.bus
        Generator
        gen1    bus2
        Name: bus, dtype: object

        """
        if not all(isinstance(v, str) for v in kwargs.values()):
            msg = "New names must be strings."
            raise ValueError(msg)

        # Rename component name definitions
        self.static = self.static.rename(index=kwargs)
        for k, v in self.dynamic.items():  # Modify in place
            self.dynamic[k] = v.rename(columns=kwargs)

        # Rename cross references in network (if attached to one)
        if self.attached:
            for c in self.n_save.components.values():
                col_name = self.name.lower()  # TODO: Generalize
                cols = [f"{col_name}{port}" for port in c.ports]
                if cols and not c.static.empty:
                    c.static[cols] = c.static[cols].replace(kwargs)


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
