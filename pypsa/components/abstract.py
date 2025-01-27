"""
Abstract components module.

Contains classes and properties relevant to all component types in PyPSA. Also imports
logic from other modules:
- components.types
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Any

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pyproj import CRS

from pypsa.common import equals
from pypsa.components.addremove import add
from pypsa.components.descriptors import get_active_assets
from pypsa.components.store import DynamicAttrsDict
from pypsa.components.types import ComponentType
from pypsa.components.validation import validate
from pypsa.constants import DEFAULT_EPSG, DEFAULT_TIMESTAMP

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypsa import Network

# TODO attachment todos
# - crs
# - snapshots, investment_periods


def _init_static_container(c: ComponentType) -> pd.DataFrame | gpd.GeoDataFrame:
    if c.name == "Shape":
        df = gpd.GeoDataFrame(columns=["geometry"], crs=CRS.from_epsg(DEFAULT_EPSG))
    else:
        df = pd.DataFrame()
    df.index.name = c.name
    return c.schema_static(df)


def _init_dynamic_container(c: ComponentType) -> DynamicAttrsDict:
    store = DynamicAttrsDict()
    store._pypsa_component = c
    store._default_snapshots = pd.Index(pd.Index([DEFAULT_TIMESTAMP]), name="snapshots")
    return store


class ComponentsData(BaseModel):
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

    # Pydantic config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Immutable attributes
    ctype: Annotated[ComponentType, Field(frozen=True)]

    # Mutable attributes
    n: Annotated[Any, Field(default=None)]
    static: Annotated[
        pd.DataFrame | gpd.GeoDataFrame,
        Field(default_factory=lambda data: _init_static_container(data["ctype"])),
    ]
    dynamic: Annotated[
        DynamicAttrsDict,
        Field(default_factory=lambda data: _init_dynamic_container(data["ctype"])),
    ]


class Components(ComponentsData):
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

    # from pypsa.components.addremove
    add = add

    # from pypsa.components.validation
    validate = validate  # type: ignore # Overwrites deprecated BaseModel method

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
            Suffix to add to component names, by default no suffix is added.

        """
        if names is not None:
            msg = "Adding components during initialisation is not yet supported."
            raise NotImplementedError(msg)
        # TODOS
        # if n is not None:
        #     msg = (
        #         "Attaching components to Network during initialisation is not yet "
        #         "supported."
        #     )
        #     raise NotImplementedError(msg)
        super().__init__(ctype=ctype, n=n)

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
        'Generator' Components
        ======================
        Attached to PyPSA Network 'AC-DC'
        Components: 6

        """
        num_components = len(self.static)
        if not num_components:
            return f"Empty {self.ctype.name} Components\n"
        text = f"'{self.ctype.name}' Components"
        text += "\n" + "=" * len(text) + "\n"

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
        'Generator' Components
        ======================
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

    @property
    def snapshots(self) -> pd.Index:
        """
        Get snapshots of the component.

        Returns
        -------
        pd.Index
            Snapshots of the component.

        """
        return self.n_save.snapshots

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
