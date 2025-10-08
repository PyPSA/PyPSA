"""Network components module.

Contains single mixin class which is used to inherit to [pypsa.Networks] class.
Should not be used directly.

Adds all properties and access methods to the Components of a network. `n.components`
is already defined during the Network initialization and here just the access properties
are set.

"""
# ruff: noqa: D102

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

from pypsa._options import options
from pypsa.common import deprecated_in_next_major
from pypsa.components.legacy import Component
from pypsa.components.store import ComponentsStore
from pypsa.components.types import (
    component_types_df,
)
from pypsa.components.types import (
    get as get_component_type,
)
from pypsa.definitions.structures import Dict
from pypsa.network.abstract import _NetworkABC

if TYPE_CHECKING:
    import pandas as pd
logger = logging.getLogger(__name__)

# TODO Change to UserWarning when they are all resolved and raised
# TODO Change types back to class with release. With legacy API type hints
# will be unsupported.

_STATIC_SETTER_WARNING = (
    "You are overwriting the network components with a new object. This is "
    "not supported, since it may lead to unexpected behavior. See #TODO. "
    "for more information."
)

_DYNAMIC_GETTER_WARNING = (
    "With PyPSA 1.0, the API for how to access components data has changed. "
    "See #TODO for more information. Use `n.{0}.dynamic` as a "
    "drop-in replacement for `n.{0}_t`."
)

_DYNAMIC_SETTER_WARNING = (
    "With PyPSA 1.0, the API for how to access components data has changed. "
    "See #TODO for more information. `n.{0}_t` is deprecated and "
    "cannot be set."
)


class NetworkComponentsMixin(_NetworkABC):
    """Mixin class for network components methods.

    Class only inherits to [pypsa.Network][] and should not be used directly.
    All attributes and methods can be used within any Network instance.
    """

    def __init__(self) -> None:
        """Initialize the NetworkComponentsMixin.

        The class should not be used directly and initialzed outside of PyPSA.
        """
        self._components: ComponentsStore | None = None

    def _read_in_default_standard_types(self) -> None:
        """Read in the default standard types from the data folder."""
        for std_type in self.standard_type_components:
            self.add(
                std_type,
                self.components[std_type].ctype.standard_types.index,
                **self.components[std_type].ctype.standard_types,
            )

    @property
    def components(self) -> ComponentsStore:
        """Network components store.

        Access all components of the network via `n.components.<component>`.

        Examples
        --------
        >>> n.components # doctest: +ELLIPSIS
        PyPSA Components Store
        ======================
        - 9 'Bus' Components
        - 6 'Carrier' Components
        ...

        Access a single component:
        >>> n.components.generators
        'Generator' Components
        ----------------------
        Attached to PyPSA Network 'AC-DC-Meshed'
        Components: 6

        Which is the same reference when accessing the component directly:
        >>> n.generators
                                        bus control  ... weight    p_nom_opt
        name                                 ...
        Manchester Wind  Manchester   Slack  ...    1.0  4090.809778
        Manchester Gas   Manchester      PQ  ...    1.0    -0.000000
        Norway Wind          Norway      PQ  ...    1.0  1533.599858
        Norway Gas           Norway      PQ  ...    1.0    -0.000000
        Frankfurt Wind    Frankfurt   Slack  ...    1.0  1667.724420
        Frankfurt Gas     Frankfurt      PQ  ...    1.0   982.034483
        <BLANKLINE>
        [6 rows x 37 columns]
        >>> n.generators is n.components.generators.static
        True

        Returns
        -------
        ComponentsStore

        """
        if self._components is None:
            components = component_types_df.index.to_list()

            self._components = ComponentsStore()
            for c_name in components:
                ctype = get_component_type(c_name)

                self._components[ctype.list_name] = Component(ctype=ctype, n=self)
        return self._components

    @property
    def c(self) -> ComponentsStore:
        """Network components store.

        Access all components of the network via `n.c.<component>`. Alias for
        [`n.components`][pypsa.Network.components].

        Returns
        -------
        ComponentsStore

        """
        return self.components

    @property
    def sub_networks(self) -> Any:
        return (
            self.c.sub_networks.static
            if not options.api.new_components_api
            else self.c.sub_networks
        )

    @sub_networks.setter
    def sub_networks(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.sub_networks.static = value

    @property
    def buses(self) -> Any:
        return (
            self.c.buses.static if not options.api.new_components_api else self.c.buses
        )

    @buses.setter
    def buses(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.buses.static = value

    @property
    def carriers(self) -> Any:
        return (
            self.c.carriers.static
            if not options.api.new_components_api
            else self.c.carriers
        )

    @carriers.setter
    def carriers(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.carriers.static = value

    @property
    def global_constraints(self) -> Any:
        return (
            self.c.global_constraints.static
            if not options.api.new_components_api
            else self.c.global_constraints
        )

    @global_constraints.setter
    def global_constraints(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.global_constraints.static = value

    @property
    def lines(self) -> Any:
        return (
            self.c.lines.static if not options.api.new_components_api else self.c.lines
        )

    @lines.setter
    def lines(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.lines.static = value

    @property
    def line_types(self) -> Any:
        return (
            self.c.line_types.static
            if not options.api.new_components_api
            else self.c.line_types
        )

    @line_types.setter
    def line_types(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.line_types.static = value

    @property
    def transformers(self) -> Any:
        return (
            self.c.transformers.static
            if not options.api.new_components_api
            else self.c.transformers
        )

    @transformers.setter
    def transformers(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.transformers.static = value

    @property
    def transformer_types(self) -> Any:
        return (
            self.c.transformer_types.static
            if not options.api.new_components_api
            else self.c.transformer_types
        )

    @transformer_types.setter
    def transformer_types(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.transformer_types.static = value

    @property
    def links(self) -> Any:
        return (
            self.c.links.static if not options.api.new_components_api else self.c.links
        )

    @links.setter
    def links(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.links.static = value

    @property
    def loads(self) -> Any:
        return (
            self.c.loads.static if not options.api.new_components_api else self.c.loads
        )

    @loads.setter
    def loads(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.loads.static = value

    @property
    def generators(self) -> Any:
        return (
            self.c.generators.static
            if not options.api.new_components_api
            else self.c.generators
        )

    @generators.setter
    def generators(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.generators.static = value

    @property
    def storage_units(self) -> Any:
        return (
            self.c.storage_units.static
            if not options.api.new_components_api
            else self.c.storage_units
        )

    @storage_units.setter
    def storage_units(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.storage_units.static = value

    @property
    def stores(self) -> Any:
        return (
            self.c.stores.static
            if not options.api.new_components_api
            else self.c.stores
        )

    @stores.setter
    def stores(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.stores.static = value

    @property
    def shunt_impedances(self) -> Any:
        return (
            self.c.shunt_impedances.static
            if not options.api.new_components_api
            else self.c.shunt_impedances
        )

    @shunt_impedances.setter
    def shunt_impedances(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.shunt_impedances.static = value

    @property
    def shapes(self) -> Any:
        return (
            self.c.shapes.static
            if not options.api.new_components_api
            else self.c.shapes
        )

    @shapes.setter
    def shapes(self, value: pd.DataFrame) -> None:
        if options.api.new_components_api:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.shapes.static = value

    @property
    def sub_networks_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("sub_networks"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.sub_networks.dynamic

    @sub_networks_t.setter
    def sub_networks_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("sub_networks"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.sub_networks.dynamic = value

    @property
    def buses_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("buses"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.buses.dynamic

    @buses_t.setter
    def buses_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("buses"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.buses.dynamic = value

    @property
    def carriers_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("carriers"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.carriers.dynamic

    @carriers_t.setter
    def carriers_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("carriers"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.carriers.dynamic = value

    @property
    def global_constraints_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("global_constraints"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.global_constraints.dynamic

    @global_constraints_t.setter
    def global_constraints_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("global_constraints"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.global_constraints.dynamic = value

    @property
    def lines_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("lines"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.lines.dynamic

    @lines_t.setter
    def lines_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("lines"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.lines.dynamic = value

    @property
    def line_types_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("line_types"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.line_types.dynamic

    @line_types_t.setter
    def line_types_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("line_types"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.line_types.dynamic = value

    @property
    def transformers_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("transformers"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.transformers.dynamic

    @transformers_t.setter
    def transformers_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("transformers"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.transformers.dynamic = value

    @property
    def transformer_types_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("transformer_types"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.transformer_types.dynamic

    @transformer_types_t.setter
    def transformer_types_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("transformer_types"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.transformer_types.dynamic = value

    @property
    def links_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("links"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.links.dynamic

    @links_t.setter
    def links_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("links"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.links.dynamic = value

    @property
    def loads_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("loads"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.loads.dynamic

    @loads_t.setter
    def loads_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("loads"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.loads.dynamic = value

    @property
    def generators_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("generators"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.generators.dynamic

    @generators_t.setter
    def generators_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("generators"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.generators.dynamic = value

    @property
    def storage_units_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("storage_units"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.storage_units.dynamic

    @storage_units_t.setter
    def storage_units_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("storage_units"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.storage_units.dynamic = value

    @property
    def stores_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("stores"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.stores.dynamic

    @stores_t.setter
    def stores_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("stores"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.stores.dynamic = value

    @property
    def shunt_impedances_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("shunt_impedances"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.shunt_impedances.dynamic

    @shunt_impedances_t.setter
    def shunt_impedances_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("shunt_impedances"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.shunt_impedances.dynamic = value

    @property
    def shapes_t(self) -> Dict:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_GETTER_WARNING.format("shapes"),
                DeprecationWarning,
                stacklevel=2,
            )
        return self.c.shapes.dynamic

    @shapes_t.setter
    def shapes_t(self, value: Dict) -> None:
        if options.api.new_components_api:
            warnings.warn(
                _DYNAMIC_SETTER_WARNING.format("shapes"),
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.shapes.dynamic = value

    @property
    def controllable_branch_components(self) -> set[str]:
        """Controllable branch components of the network as set of strings.

        Examples
        --------
        >>> sorted(n.controllable_branch_components)
        ['Link']

        """
        return {"Link"}

    @property
    def controllable_one_port_components(self) -> set[str]:
        """Controllable one port components of the network as set of strings.

        Examples
        --------
        >>> sorted(n.controllable_one_port_components)
        ['Generator', 'Load', 'StorageUnit', 'Store']

        """
        return {"StorageUnit", "Store", "Generator", "Load"}

    @property
    def passive_branch_components(self) -> set[str]:
        """Passive branch components of the network as set of strings.

        Examples
        --------
        >>> sorted(n.passive_branch_components)
        ['Line', 'Transformer']

        """
        return {"Transformer", "Line"}

    @property
    def passive_one_port_components(self) -> set[str]:
        """Passive one port components of the network as set of strings.

        Examples
        --------
        >>> sorted(n.passive_one_port_components)
        ['ShuntImpedance']

        """
        return {"ShuntImpedance"}

    @property
    def standard_type_components(self) -> set[str]:
        """Standard type components of the network as set of strings.

        Examples
        --------
        >>> sorted(n.standard_type_components)
        ['LineType', 'TransformerType']

        """
        return {"LineType", "TransformerType"}

    @property
    def one_port_components(self) -> set[str]:
        """One port components of the network as set of strings.

        Examples
        --------
        >>> sorted(n.one_port_components)
        ['Generator', 'Load', 'ShuntImpedance', 'StorageUnit', 'Store']

        """
        return self.passive_one_port_components | self.controllable_one_port_components

    @property
    def branch_components(self) -> set[str]:
        """Branch components of the network as set of strings.

        Examples
        --------
        >>> sorted(n.branch_components)
        ['Line', 'Link', 'Transformer']

        """
        return self.passive_branch_components | self.controllable_branch_components

    @property
    def all_components(self) -> set[str]:
        """All components of the network as set of strings.

        Examples
        --------
        >>> sorted(n.all_components)
        ['Bus', 'Carrier', 'Generator', 'GlobalConstraint', 'Line', 'LineType', 'Link', 'Load', 'Shape', 'ShuntImpedance', 'StorageUnit', 'Store', 'SubNetwork', 'Transformer', 'TransformerType']

        """
        return {
            "Carrier",
            "Line",
            "Transformer",
            "Shape",
            "Generator",
            "StorageUnit",
            "Store",
            "ShuntImpedance",
            "Link",
            "GlobalConstraint",
            "SubNetwork",
            "TransformerType",
            "LineType",
            "Bus",
            "Load",
        }

    @property
    def _index_names(self) -> list[str]:
        """Compatibility property for NetworkCollection object.

        Returns
        -------
        list of str
            Empty list, since the Network class does not have any index names.

        """
        return []

    @property
    @deprecated_in_next_major(
        details="Use `self.components.<component>.defaults` instead.",
    )
    def component_attrs(self) -> pd.DataFrame:
        """Component attributes.

        Deprecation
        -----------
        Deprecated in [:material-tag-outline: v1.0](/release-notes/#v1.0.0) and will be
        removed in v2.0:

        Use the [Components Class][pypsa.Components] to access components attributes.
        As a drop in replacement you can use either
        [`n.components[<component>].defaults`][pypsa.Components.defaults] or
        `n.components.<component>.defaults`. You can also use the alias [
        `n.c`][pypsa.Network.c] for [`n.components`][pypsa.Network.components].

        Parameters
        ----------
        component_name : string

        Returns
        -------
        pandas.DataFrame
            Component attributes informations.

        """
        return Dict({value.name: value.defaults for value in self.components})

    @deprecated_in_next_major(
        details="Use `self.components[<component>].static` instead."
    )
    def df(self, component_name: str) -> pd.DataFrame:
        """Alias for [`n.static`][pypsa.Network.static].

        Deprecation
        -----------
        Deprecated in [:material-tag-outline: v1.0](/release-notes/#v1.0.0) and will be
        removed in v2.0:

        Use the [Components Class][pypsa.Components] to access components attributes.
        As a drop in replacement you can use either
        [`n.components[<component>].static`][pypsa.Components.static] or
        `n.components.<component>.static`. You can also use the alias [
        `n.c`][pypsa.Network.c] for [`n.components`][pypsa.Network.components].

        Parameters
        ----------
        component_name : string
            Name of the component.

        Returns
        -------
        pandas.DataFrame
            Static data of the component.

        """
        return self.components[component_name].static

    @deprecated_in_next_major(
        details="Use `self.components.<component>.static` instead."
    )
    def static(self, component_name: str) -> pd.DataFrame:
        """Return the DataFrame of static components for component_name.

        Deprecation
        -----------
        Deprecated in [:material-tag-outline: v1.0](/release-notes/#v1.0.0) and will be
        removed in v2.0:

        Use the [Components Class][pypsa.Components] to access components attributes.
        As a drop in replacement you can use either
        [`n.components[<component>].static`][pypsa.Components.static] or
        `n.components.<component>.static`. You can also use the alias [
        `n.c`][pypsa.Network.c] for [`n.components`][pypsa.Network.components].

        Parameters
        ----------
        component_name : string
            Name of the component.

        Returns
        -------
        pandas.DataFrame
            Static data of the component.

        """
        return self.components[component_name].static

    @deprecated_in_next_major(
        details="Use `self.components.<component>.dynamic` instead.",
    )
    def pnl(self, component_name: str) -> Dict:
        """Alias for [`n.dynamic`][pypsa.Network.dynamic].

        Deprecation
        -----------
        Deprecated in [:material-tag-outline: v1.0](/release-notes/#v1.0.0) and will be
        removed in v2.0:

        Use the [Components Class][pypsa.Components] to access components attributes.
        As a drop in replacement you can use either
        [`n.components[<component>].dynamic`][pypsa.Components.dynamic] or
        `n.components.<component>.dynamic`. You can also use the alias [
        `n.c`][pypsa.Network.c] for [`n.components`][pypsa.Network.components].

        Parameters
        ----------
        component_name : string
            Name of the component.

        Returns
        -------
        dict of pandas.DataFrame
            Dynamic data of the component.

        """
        return self.components[component_name].dynamic

    @deprecated_in_next_major(
        details="Use `self.components.<component>.dynamic` instead.",
    )
    def dynamic(self, component_name: str) -> Dict:
        """Return the dictionary of DataFrames of varying components.

        Deprecation
        -----------
        Deprecated in [:material-tag-outline: v1.0](/release-notes/#v1.0.0) and will be
        removed in v2.0:

        Use the [Components Class][pypsa.Components] to access components attributes.
        As a drop in replacement you can use either
        [`n.components[<component>].dynamic`][pypsa.Components.dynamic] or
        `n.components.<component>.dynamic`. You can also use the alias [
        `n.c`][pypsa.Network.c] for [`n.components`][pypsa.Network.components].

        Parameters
        ----------
        component_name : string
            Name of the component.

        Returns
        -------
        dict of pandas.DataFrame
            Dynamic data of the component.

        """
        return self.components[component_name].dynamic
