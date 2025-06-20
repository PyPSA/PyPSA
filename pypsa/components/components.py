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
from pyproj import CRS

from pypsa.common import equals
from pypsa.components.descriptors import ComponentsDescriptorsMixin
from pypsa.components.index import ComponentsIndexMixin
from pypsa.components.transform import ComponentsTransformMixin
from pypsa.constants import DEFAULT_EPSG, DEFAULT_TIMESTAMP, RE_PORTS
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

    This class is used to store all data of a Components object. Other classes inherit
    from this class to implement logic and methods, but do not store any data next
    to the data in here.

    All attributes can therefore also be accessed directly from
    any [`Components`][pypsa.components.Components] object (which defines all
    attributes and properties which are available for all component types) as well as
    in specific type classes as [`Generators`][pypsa.components.Generators] (which
    define logic and methods specific to the component type).

    User Guide
    ----------
    Check out the corresponding user guide: [:material-bookshelf: Components](/user-guide/components)

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
    dynamic: Dict


class Components(
    ComponentsData,
    ComponentsDescriptorsMixin,
    ComponentsTransformMixin,
    ComponentsIndexMixin,
):
    """Components base class.

    Base class for container of energy system related assets, such as
    generators or transmission lines. Use the specific subclasses for concrete or
    a generic component type.
    All data is stored in the dataclass [pypsa.components.components.ComponentsData][].
    Components inherits from it, adds logic and methods, but does not store any data
    itself.

    !!! warning
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
        super().__init__(ctype, n=None, static=static, dynamic=dynamic)

    def __str__(self) -> str:
        """Get string representation of component.

        Returns
        -------
        str
            String representation of component.

        Examples
        --------
        >>> str(n.components.generators)
        "'Generator' Components"

        """
        return f"'{self.ctype.name}' Components"

    def __repr__(self) -> str:
        """Get representation of component.

        Returns
        -------
        str
            Representation of component.

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
            msg = f"'{key}' not found in Component"
            raise KeyError(msg)

    def __eq__(self, other: object) -> bool:
        """Check if two Components are equal.

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
        [pypsa.Components.equals][] :
            Check for equality of two networks.

        """
        return self.equals(other)

    def equals(self, other: Any, log_mode: str = "silent") -> bool:
        """Check if two Components are equal.

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
        Index(['bus1'], dtype='object')
        >>> n2.add("Bus", "bus1")
        Index(['bus1'], dtype='object')
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
        """Get standard types of component.

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

        Returns
        -------
        str
            Description of component.

        Examples
        --------
        >>> n.components.generators.description
        'Power generator.'

        """
        return self.ctype.description

    @property
    def category(self) -> str:
        """Category of component.

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

        .. note ::
            While not actively deprecated yet, :meth:`category` is the preferred method
            to access component type.

        Returns
        -------
        str
            Category of component.

        """
        return self.ctype.category

    @property
    def attrs(self) -> pd.DataFrame:
        """Default values of corresponding component type.

        .. note::
            While not actively deprecated yet, :meth:`defaults` is the preferred method
            to access component attributes.

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

        .. note::
            While not actively deprecated yet, :meth:`defaults` is the preferred method
            to access component attributes.

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

        Returns
        -------
        bool
            True if component is empty, otherwise False.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add('Generator', 'g1')  # doctest: +ELLIPSIS
        Index(['g1'], dtype='object')
        >>> n.components.generators.empty
        False

        >>> n.components.buses.empty
        True

        """
        return self.static.empty

    def get(self, attribute_name: str, default: Any = None) -> Any:
        """Get attribute of component.

        Just an alias for built-in getattr and allows for default values.
        #TODO change to handle data access instead

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
        """Check if component is attached to a Network.

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
    def df(self) -> pd.DataFrame:
        """Get static data of all components as pandas DataFrame.

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
        """Get dynamic data of all components as a dictionary of pandas DataFrames.

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
        """Get units of all attributes of components.

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
        [pypsa.components.Links.additional_ports][] :
            Additional ports of components.

        """
        return [
            match.group(1) for col in self.static if (match := RE_PORTS.search(col))
        ]


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

        Returns
        -------
        str
            String representation of sub-network components.

        Examples
        --------
        >>> str(sub_network.components.generators)
        "'Generator' SubNetworkComponents"

        """
        return f"'{self.ctype.name}' SubNetworkComponents"

    def __repr__(self) -> str:
        """Get representation of sub-network components.

        Returns
        -------
        str
            Representation of sub-network components.

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
