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
import warnings
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
# TODO Handle setters


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
            warnings.warn(
                "You are setting the `sub_networks` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.sub_networks.static = value

    @property
    def sub_networks_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.sub_networks.dynamic` as a "
                "drop-in replacement for `n.sub_networks_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.sub_networks.dynamic

    @property
    def buses(self) -> Buses:
        return self.c.buses.static if options.api.legacy_components else self.c.buses

    @buses.setter
    def buses(self, value: Buses) -> None:
        if not options.api.legacy_components:
            warnings.warn(
                "You are setting the `buses` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.buses.static = value

    @property
    def buses_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.buses.dynamic` as a "
                "drop-in replacement for `n.buses_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.buses.dynamic

    @property
    def carriers(self) -> Carriers:
        return (
            self.c.carriers.static if options.api.legacy_components else self.c.carriers
        )

    @carriers.setter
    def carriers(self, value: Carriers) -> None:
        if not options.api.legacy_components:
            warnings.warn(
                "You are setting the `carriers` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.carriers.static = value

    @property
    def carriers_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.carriers.dynamic` as a "
                "drop-in replacement for `n.carrier_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.carriers.dynamic

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
            warnings.warn(
                "You are setting the `global_constraints` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.global_constraints.static = value

    @property
    def global_constraints_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.global_constraints.dynamic` as a "
                "drop-in replacement for `n.global_constraint_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.global_constraints.dynamic

    @property
    def lines(self) -> Lines:
        return self.c.lines.static if options.api.legacy_components else self.c.lines

    @lines.setter
    def lines(self, value: Lines) -> None:
        if not options.api.legacy_components:
            warnings.warn(
                "You are setting the `lines` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.lines.static = value

    @property
    def lines_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.lines.dynamic` as a "
                "drop-in replacement for `n.lines_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.lines.dynamic

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
            warnings.warn(
                "You are setting the `line_types` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.line_types.static = value

    @property
    def line_types_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.line_types.dynamic` as a "
                "drop-in replacement for `n.line_types_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.line_types.dynamic

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
            warnings.warn(
                "You are setting the `transformers` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.transformers.static = value

    @property
    def transformers_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.transformers.dynamic` as a "
                "drop-in replacement for `n.transformers_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.transformers.dynamic

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
            warnings.warn(
                "You are setting the `transformer_types` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.transformer_types.static = value

    @property
    def transformer_types_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.transformer_types.dynamic` as a "
                "drop-in replacement for `n.transformer_types_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.transformer_types.dynamic

    @property
    def links(self) -> Links:
        return self.c.links.static if options.api.legacy_components else self.c.links

    @links.setter
    def links(self, value: Links) -> None:
        if not options.api.legacy_components:
            warnings.warn(
                "You are setting the `links` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.links.static = value

    @property
    def links_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.links.dynamic` as a "
                "drop-in replacement for `n.links_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.links.dynamic

    @property
    def loads(self) -> Loads:
        return self.c.loads.static if options.api.legacy_components else self.c.loads

    @loads.setter
    def loads(self, value: Loads) -> None:
        if not options.api.legacy_components:
            warnings.warn(
                "You are setting the `loads` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.loads.static = value

    @property
    def loads_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.loads.dynamic` as a "
                "drop-in replacement for `n.loads_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.loads.dynamic

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
            warnings.warn(
                "You are setting the `generators` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.generators.static = value

    @property
    def generators_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.generators.dynamic` as a "
                "drop-in replacement for `n.generators_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.generators.dynamic

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
            warnings.warn(
                "You are setting the `storage_units` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.storage_units.static = value

    @property
    def storage_units_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.storage_units.dynamic` as a "
                "drop-in replacement for `n.storage_units_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.storage_units.dynamic

    @property
    def stores(self) -> Stores:
        return self.c.stores.static if options.api.legacy_components else self.c.stores

    @stores.setter
    def stores(self, value: Stores) -> None:
        if not options.api.legacy_components:
            warnings.warn(
                "You are setting the `stores` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.stores.static = value

    @property
    def stores_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.stores.dynamic` as a "
                "drop-in replacement for `n.stores_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.stores.dynamic

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
            warnings.warn(
                "You are setting the `shunt_impedances` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.shunt_impedances.static = value

    @property
    def shunt_impedances_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.shunt_impedances.dynamic` as a "
                "drop-in replacement for `n.shunt_impedances_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.shunt_impedances.dynamic

    @property
    def shapes(self) -> Shapes:
        return self.c.shapes.static if options.api.legacy_components else self.c.shapes

    @shapes.setter
    def shapes(self, value: Shapes) -> None:
        if not options.api.legacy_components:
            warnings.warn(
                "You are setting the `shapes` attribute directly. This is not "
                "recommended, since it may lead to unexpected behavior. See #TODO for "
                "more information.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.c.shapes.static = value

    @property
    def shapes_t(self) -> None:
        if not options.api.legacy_components:
            msg = (
                "With PyPSA 1.0, the API for how to access components data has changed. "
                "See #TODO for more information. Use `n.shapes.dynamic` as a "
                "drop-in replacement for `n.shapes_t`."
            )
            raise DeprecationWarning(msg)
        return self.c.shapes.dynamic
