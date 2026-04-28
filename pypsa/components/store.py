# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Components store module.

Contains store class which is used to store all different components in the network.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import TYPE_CHECKING, Any

from pypsa.components.types import all_components

_COMPONENT_ALIAS_DICT = {ct.name: ct.list_name for ct in all_components.values()}

if TYPE_CHECKING:
    from pypsa.components._types.buses import Buses
    from pypsa.components._types.carriers import Carriers
    from pypsa.components._types.generators import Generators
    from pypsa.components._types.global_constraints import GlobalConstraints
    from pypsa.components._types.line_types import LineTypes
    from pypsa.components._types.lines import Lines
    from pypsa.components._types.links import Links
    from pypsa.components._types.loads import Loads
    from pypsa.components._types.processes import Processes
    from pypsa.components._types.shapes import Shapes
    from pypsa.components._types.shunt_impedances import ShuntImpedances
    from pypsa.components._types.storage_units import StorageUnits
    from pypsa.components._types.stores import Stores
    from pypsa.components._types.sub_networks import SubNetworks
    from pypsa.components._types.transformer_types import TransformerTypes
    from pypsa.components._types.transformers import Transformers
    from pypsa.components.components import Components

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
    processes: Processes
    shapes: Shapes
    shunt_impedances: ShuntImpedances
    storage_units: StorageUnits
    stores: Stores
    sub_networks: SubNetworks
    transformer_types: TransformerTypes
    transformers: Transformers

    def __repr__(self) -> str:
        """Get representation of component store.

        Examples
        --------
        >>> n.components
        PyPSA Components Store
        ======================
        - 9 'Bus' Components
        - 6 'Carrier' Components
        - 6 'Generator' Components
        - 6 'Load' Components
        - 4 'Link' Components
        - 0 'Store' Components
        - 0 'StorageUnit' Components
        - 7 'Line' Components
        - 59 'LineType' Components
        - 0 'Process' Components
        - 0 'Transformer' Components
        - 14 'TransformerType' Components
        - 0 'ShuntImpedance' Components
        - 1 'GlobalConstraint' Components
        - 0 'Shape' Components
        - 3 'SubNetwork' Components

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

    def __getitem__(self, item: str | list | set | Components) -> Any:
        """Index single and multiple items from the dictionary.

        Similar behavior to pandas.DataFrame.__getitem__.

        Examples
        --------
        >>> n.components
        PyPSA Components Store
        ======================
        - 9 'Bus' Components
        - 6 'Carrier' Components
        - 6 'Generator' Components
        - 6 'Load' Components
        - 4 'Link' Components
        - 0 'Store' Components
        - 0 'StorageUnit' Components
        - 7 'Line' Components
        - 59 'LineType' Components
        - 0 'Process' Components
        - 0 'Transformer' Components
        - 14 'TransformerType' Components
        - 0 'ShuntImpedance' Components
        - 1 'GlobalConstraint' Components
        - 0 'Shape' Components
        - 3 'SubNetwork' Components
        >>> n.components["generators"]
        'Generator' Components
        ----------------------
        Attached to PyPSA Network 'AC-DC-Meshed'
        Components: 6

        """
        from pypsa.components.components import Components  # noqa: PLC0415

        if isinstance(item, Components):
            return item
        if isinstance(item, (list | set)):
            return [self[key] for key in item]
        if item in _COMPONENT_ALIAS_DICT:
            warnings.warn(
                f"Accessing components by PascalCase name '{item}' is deprecated. "
                f"Use '{_COMPONENT_ALIAS_DICT[item]}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return super().__getitem__(_COMPONENT_ALIAS_DICT[item])
        try:
            return super().__getitem__(item)
        except KeyError:
            msg = f"'{item}' is not a valid component type."
            raise ValueError(msg) from None

    def _get(self, item: str | Any) -> Any:
        """Look up a component without emitting a deprecation warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return self[item]

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
        except (KeyError, ValueError) as e:
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
        """Value iterator over components in store.

        Filters out empty components to maintain backward compatibility with
        n.iterate_components() behavior. For accessing all components including
        empty ones, use n.components.values() directly instead of iterating over the
        store.
        """
        # Filter to only return non-empty components (same behavior as iterate_components)
        return iter(c for c in self.values() if not c.empty)

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

    def filter(
        self,
        *,
        branch: bool | None = None,
        one_port: bool | None = None,
        passive: bool | None = None,
        controllable: bool | None = None,
        standard_type: bool | None = None,
    ) -> list:
        """Filter components by category.

        <!-- md:badge-version v1.3.0 -->

        Each keyword matches a boolean property on
        [`Components`][pypsa.Components]. Pass `True` to require the
        property, `False` to require its negation, or omit the keyword to
        ignore that category. Multiple keywords combine with AND.

        Parameters
        ----------
        branch : bool, optional
            Match [`Components.is_branch`][pypsa.Components.is_branch].
        one_port : bool, optional
            Match [`Components.is_one_port`][pypsa.Components.is_one_port].
        passive : bool, optional
            Match [`Components.is_passive`][pypsa.Components.is_passive].
        controllable : bool, optional
            Match [`Components.is_controllable`][pypsa.Components.is_controllable].
        standard_type : bool, optional
            Match [`Components.is_standard_type`][pypsa.Components.is_standard_type].

        Returns
        -------
        list
            Matching Components instances, in store order.

        Examples
        --------
        >>> for c in n.components.filter(branch=True):
        ...     print(c.list_name)
        >>> for c in n.components.filter(branch=True, controllable=True):
        ...     print(c.list_name)

        """
        kwarg_to_attr = {
            "branch": ("is_branch", branch),
            "one_port": ("is_one_port", one_port),
            "passive": ("is_passive", passive),
            "controllable": ("is_controllable", controllable),
            "standard_type": ("is_standard_type", standard_type),
        }
        predicates = [
            (attr, want) for attr, want in kwarg_to_attr.values() if want is not None
        ]
        return [
            c
            for c in self.values()
            if all(getattr(c, attr) is want for attr, want in predicates)
        ]
