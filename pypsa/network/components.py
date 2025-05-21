"""
Network components module.

Contains single helper class (_NetworkComponents) which is used to inherit
to Network class. Should not be used directly.

Adds all properties and access methods to the Components of a network. `n.components`
is already defined during the Network initialization and here just the access properties
are set.

Also See
--------
pypsa.network.index

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pypsa.network.abstract import _NetworkABC

if TYPE_CHECKING:
    from pypsa.components._types import (
        Buses,
        Carriers,
        Generators,
        GlobalConstraints,
        Lines,
        LineTypes,
        Links,
        Loads,
        Shapes,
        ShuntImpedances,
        StorageUnits,
        Stores,
        SubNetworks,
        Transformers,
        TransformerTypes,
    )
from pypsa._options import options

logger = logging.getLogger(__name__)

# TODO Change to UserWarning when they are all resolved and raised


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


class _NetworkComponents(_NetworkABC):
    """
    Helper class for components array methods.

    Class only inherits to Components and should not be used directly.
    """

    @property
    def sub_networks(self) -> SubNetworks:
        return (
            self.c.sub_networks.static
            if options.api.legacy_components
            else self.c.subnetworks
        )

    @sub_networks.setter
    def sub_networks(self, value: SubNetworks) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.sub_networks.static = value

    @property
    def sub_networks_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("sub_networks"))
        return self.c.sub_networks.dynamic

    @sub_networks_t.setter
    def sub_networks_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("sub_networks"))
        self.c.sub_networks.dynamic = value

    @property
    def buses(self) -> Buses:
        return self.c.buses.static if options.api.legacy_components else self.c.buses

    @buses.setter
    def buses(self, value: Buses) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.buses.static = value

    @property
    def buses_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("buses"))
        return self.c.buses.dynamic

    @buses_t.setter
    def buses_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("buses"))
        self.c.buses.dynamic = value

    @property
    def carriers(self) -> Carriers:
        return (
            self.c.carriers.static if options.api.legacy_components else self.c.carriers
        )

    @carriers.setter
    def carriers(self, value: Carriers) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.carriers.static = value

    @property
    def carriers_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("carriers"))
        return self.c.carriers.dynamic

    @carriers_t.setter
    def carriers_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("carriers"))
        self.c.carriers.dynamic = value

    @property
    def global_constraints(self) -> GlobalConstraints:
        return (
            self.c.global_constraints.static
            if options.api.legacy_components
            else self.c.global_constraints
        )

    @global_constraints.setter
    def global_constraints(self, value: GlobalConstraints) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.global_constraints.static = value

    @property
    def global_constraints_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(
                _DYNAMIC_GETTER_WARNING.format("global_constraints")
            )
        return self.c.global_constraints.dynamic

    @global_constraints_t.setter
    def global_constraints_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(
                _DYNAMIC_SETTER_WARNING.format("global_constraints")
            )
        self.c.global_constraints.dynamic = value

    @property
    def lines(self) -> Lines:
        return self.c.lines.static if options.api.legacy_components else self.c.lines

    @lines.setter
    def lines(self, value: Lines) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.lines.static = value

    @property
    def lines_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("lines"))
        return self.c.lines.dynamic

    @lines_t.setter
    def lines_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("lines"))
        self.c.lines.dynamic = value

    @property
    def line_types(self) -> LineTypes:
        return (
            self.c.line_types.static
            if options.api.legacy_components
            else self.c.line_types
        )

    @line_types.setter
    def line_types(self, value: LineTypes) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.line_types.static = value

    @property
    def line_types_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("line_types"))
        return self.c.line_types.dynamic

    @line_types_t.setter
    def line_types_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("line_types"))
        self.c.line_types.dynamic = value

    @property
    def transformers(self) -> Transformers:
        return (
            self.c.transformers.static
            if options.api.legacy_components
            else self.c.transformers
        )

    @transformers.setter
    def transformers(self, value: Transformers) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.transformers.static = value

    @property
    def transformers_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("transformers"))
        return self.c.transformers.dynamic

    @transformers_t.setter
    def transformers_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("transformers"))
        self.c.transformers.dynamic = value

    @property
    def transformer_types(self) -> TransformerTypes:
        return (
            self.c.transformer_types.static
            if options.api.legacy_components
            else self.c.transformer_types
        )

    @transformer_types.setter
    def transformer_types(self, value: TransformerTypes) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.transformer_types.static = value

    @property
    def transformer_types_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(
                _DYNAMIC_GETTER_WARNING.format("transformer_types")
            )
        return self.c.transformer_types.dynamic

    @transformer_types_t.setter
    def transformer_types_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(
                _DYNAMIC_SETTER_WARNING.format("transformer_types")
            )
        self.c.transformer_types.dynamic = value

    @property
    def links(self) -> Links:
        return self.c.links.static if options.api.legacy_components else self.c.links

    @links.setter
    def links(self, value: Links) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.links.static = value

    @property
    def links_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("links"))
        return self.c.links.dynamic

    @links_t.setter
    def links_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("links"))
        self.c.links.dynamic = value

    @property
    def loads(self) -> Loads:
        return self.c.loads.static if options.api.legacy_components else self.c.loads

    @loads.setter
    def loads(self, value: Loads) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.loads.static = value

    @property
    def loads_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("loads"))
        return self.c.loads.dynamic

    @loads_t.setter
    def loads_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("loads"))
        self.c.loads.dynamic = value

    @property
    def generators(self) -> Generators:
        return (
            self.c.generators.static
            if options.api.legacy_components
            else self.c.generators
        )

    @generators.setter
    def generators(self, value: Generators) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.generators.static = value

    @property
    def generators_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("generators"))
        return self.c.generators.dynamic

    @generators_t.setter
    def generators_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("generators"))
        self.c.generators.dynamic = value

    @property
    def storage_units(self) -> StorageUnits:
        return (
            self.c.storage_units.static
            if options.api.legacy_components
            else self.c.storage_units
        )

    @storage_units.setter
    def storage_units(self, value: StorageUnits) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.storage_units.static = value

    @property
    def storage_units_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("storage_units"))
        return self.c.storage_units.dynamic

    @storage_units_t.setter
    def storage_units_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("storage_units"))
        self.c.storage_units.dynamic = value

    @property
    def stores(self) -> Stores:
        return self.c.stores.static if options.api.legacy_components else self.c.stores

    @stores.setter
    def stores(self, value: Stores) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.stores.static = value

    @property
    def stores_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("stores"))
        return self.c.stores.dynamic

    @stores_t.setter
    def stores_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("stores"))
        self.c.stores.dynamic = value

    @property
    def shunt_impedances(self) -> ShuntImpedances:
        return (
            self.c.shunt_impedances.static
            if options.api.legacy_components
            else self.c.shunt_impedances
        )

    @shunt_impedances.setter
    def shunt_impedances(self, value: ShuntImpedances) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.shunt_impedances.static = value

    @property
    def shunt_impedances_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("shunt_impedances"))
        return self.c.shunt_impedances.dynamic

    @shunt_impedances_t.setter
    def shunt_impedances_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("shunt_impedances"))
        self.c.shunt_impedances.dynamic = value

    @property
    def shapes(self) -> Shapes:
        return self.c.shapes.static if options.api.legacy_components else self.c.shapes

    @shapes.setter
    def shapes(self, value: Shapes) -> None:
        if not options.api.legacy_components:
            raise AttributeError(_STATIC_SETTER_WARNING)
        self.c.shapes.static = value

    @property
    def shapes_t(self) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_GETTER_WARNING.format("shapes"))
        return self.c.shapes.dynamic

    @shapes_t.setter
    def shapes_t(self, value: None) -> None:
        if not options.api.legacy_components:
            raise DeprecationWarning(_DYNAMIC_SETTER_WARNING.format("shapes"))
        self.c.shapes.dynamic = value
