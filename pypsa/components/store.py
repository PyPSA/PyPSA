"""Components store module.

Contains store class which is used to store all different components in the network.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import TYPE_CHECKING, Any

from pypsa._options import options
from pypsa.deprecations import COMPONENT_ALIAS_DICT

if TYPE_CHECKING:
    from pypsa.components._types.buses import Buses
    from pypsa.components._types.carriers import Carriers
    from pypsa.components._types.generators import Generators
    from pypsa.components._types.global_constraints import GlobalConstraints
    from pypsa.components._types.line_types import LineTypes
    from pypsa.components._types.lines import Lines
    from pypsa.components._types.links import Links
    from pypsa.components._types.loads import Loads
    from pypsa.components._types.shapes import Shapes
    from pypsa.components._types.shunt_impedances import ShuntImpedances
    from pypsa.components._types.storage_units import StorageUnits
    from pypsa.components._types.stores import Stores
    from pypsa.components._types.sub_networks import SubNetworks
    from pypsa.components._types.transformer_types import TransformerTypes
    from pypsa.components._types.transformers import Transformers

logger = logging.getLogger(__name__)


class ComponentsStore(dict):
    """Component store for all components in the network."""

    buses: Buses
    carriers: Carriers
    generators: Generators
    global_constraints: GlobalConstraints
    line_types: LineTypes
    lines: Lines
    links: Links
    loads: Loads
    shapes: Shapes
    shunt_impedances: ShuntImpedances
    storage_units: StorageUnits
    stores: Stores
    sub_networks: SubNetworks
    transformer_types: TransformerTypes
    transformers: Transformers

    def __repr__(self) -> str:
        """Get representation of component store.

        Returns
        -------
        str
            Representation of component store

        Examples
        --------
        >>> n.components
        PyPSA Components Store
        ======================
        - 3 'SubNetwork' Components
        - 9 'Bus' Components
        - 6 'Carrier' Components
        - 1 'GlobalConstraint' Components
        - 7 'Line' Components
        - 36 'LineType' Components
        - 0 'Transformer' Components
        - 14 'TransformerType' Components
        - 4 'Link' Components
        - 6 'Load' Components
        - 6 'Generator' Components
        - 0 'StorageUnit' Components
        - 0 'Store' Components
        - 0 'ShuntImpedance' Components
        - 0 'Shape' Components

        """
        return "PyPSA Components Store\n======================\n- " + "\n- ".join(
            f"{len(value.static)} {value}" for value in self.values()
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Is invoked when object.member = value is called."""
        if hasattr(ComponentsStore, name):
            msg = f"'ComponentsStore' object attribute '{name}' can not be set."
            raise AttributeError(msg)
        self[name] = value

    def __getitem__(self, item: str | list | set) -> Any:
        """Index single and multiple items from the dictionary.

        Similar behavior to pandas.DataFrame.__getitem__.

        Examples
        --------
        >>> n.components
        PyPSA Components Store
        ======================
        - 3 'SubNetwork' Components
        - 9 'Bus' Components
        - 6 'Carrier' Components
        - 1 'GlobalConstraint' Components
        - 7 'Line' Components
        - 36 'LineType' Components
        - 0 'Transformer' Components
        - 14 'TransformerType' Components
        - 4 'Link' Components
        - 6 'Load' Components
        - 6 'Generator' Components
        - 0 'StorageUnit' Components
        - 0 'Store' Components
        - 0 'ShuntImpedance' Components
        - 0 'Shape' Components
        >>> n.components["generators"]
        'Generator' Components
        ----------------------
        Attached to PyPSA Network 'AC-DC-Meshed'
        Components: 6

        """
        if isinstance(item, (list | set)):
            return [self[key] for key in item]
        if item in COMPONENT_ALIAS_DICT:
            # TODO: Activate when changing logic
            # Accessing components in n.components using capitalized singular "
            # name is deprecated. Use lowercase list name instead: "
            # '{COMPONENT_ALIAS_DICT[item]}' instead of '{item}'.
            return super().__getitem__(COMPONENT_ALIAS_DICT[item])
        return super().__getitem__(item)

    def __getattr__(self, item: str) -> Any:
        """Get attribute from the dictionary.

        Examples
        --------
        >>> n.components.generators
        'Generator' Components
        ----------------------
        Attached to PyPSA Network 'AC-DC-Meshed'
        Components: 6

        """
        try:
            return self[item]
        except KeyError as e:
            msg = f"Network has no components '{item}'"
            raise AttributeError(msg) from e

    def __delattr__(self, name: str) -> None:
        """Is invoked when del object.member is called."""
        del self[name]

    _re_pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

    def __dir__(self) -> list[str]:
        """Return list of object attributes including dynamic ones from the keys."""
        dict_keys = [
            k for k in self.keys() if isinstance(k, str) and self._re_pattern.match(k)
        ]
        obj_attrs = list(dir(super()))
        return dict_keys + obj_attrs

    def __iter__(self) -> Any:
        """Value iterator over components in store."""
        if options.get_option("warnings.components_store_iter"):
            warnings.warn(
                "Iterating over `n.components` yields the values instead of keys from "
                "v0.33.0. This behavior might be breaking. Use `n.components.keys()` "
                "to iterate over the keys. To suppress this warning set "
                "`pypsa.options.warnings.components_store_iter = False`.",
                DeprecationWarning,
                stacklevel=2,
            )
        return iter(self.values())

    def __contains__(self, item: Any) -> bool:
        """Check if component is in store."""
        msg = (
            "Checking if a component is in `n.components` using the 'in' operator "
            "is deprecated. Use `item in n.components.keys()` to retain the old "
            "behavior. But with v0.33.0 custom components are deprecated and "
            "therefore keys in `n.components` never change. Check the release "
            "notes for more information."
        )
        raise DeprecationWarning(msg)
