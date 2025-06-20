"""Components transform module.

Contains single mixin class which is used to inherit to [pypsa.Components][] class.
Should not be used directly.

Transform methods are methods which modify and restructure data.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from pypsa.definitions.structures import Dict

logger = logging.getLogger(__name__)


class ComponentsTransformMixin:
    """Mixin class for components descriptors methods.

    Class only inherits to [pypsa.Components][] and should not be used directly.
    All attributes and methods can be used within any Components instance.
    """

    static: pd.DataFrame
    dynamic: Dict
    attached: Any
    n_save: Any
    name: Any

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str = "",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> pd.Index:
        """Add new components.

        Handles addition of single and multiple components along with their attributes.
        Pass a list of names to add multiple components at once or pass a single name
        to add a single component.

        When a single component is added, all non-scalar attributes are assumed to be
        time-varying and indexed by snapshots.
        When multiple components are added, all non-scalar attributes are assumed to be
        static and indexed by names. A single value sequence is treated as scalar and
        broadcasted to all components. It is recommended to explicitly pass a scalar
        instead.
        If you want to add time-varying attributes to multiple components, you can pass
        a 2D array/ DataFrame where the first dimension is snapshots and the second
        dimension is names.

        Any attributes which are not specified will be given the default
        value from :doc:`/user-guide/components`.

        Parameters
        ----------
        name : str or int or list of str or list of int
            Component name(s)
        suffix : str, default ""
            All components are named after name with this added suffix.
        overwrite : bool, default False
            If True, existing components with the same names as in `name` will be
            overwritten. Otherwise only new components will be added and others will be
            ignored.
        kwargs : Any
            Component attributes, e.g. x=[0.1, 0.2], can be list, pandas.Series
            of pandas.DataFrame for time-varying

        Returns
        -------
        new_names : pandas.index
            Names of new components (including suffix)

        Examples
        --------
        The example is shown for Generator component, but the same applies to all
        component types.

        >>> n = pypsa.Network()
        >>> c = n.components.generators
        >>> c
        Empty 'Generator' Components

        Add a single component:

        >>> c.add("my-generator-1", carrier="AC")

        A new generator is added to the components instance:
        >>> c
        'Generator' Components
        ----------------------
        Attached to PyPSA Network 'Unnamed Network'
        Components: 1

        With static data (and default values for all attributes):
        >>> c.static[["carrier", "p_nom"]]
                        carrier  p_nom
        Generator
        my-generator-1      AC    0.0

        Add multiple components with static attributes:

        >>> c.add(["my-generator-2", "my-generator-3"],
        ...       carrier=["AC", "DC"],
        ...       p_nom=10)

        A new generator is added to the components instance:
        >>> c
        'Generator' Components
        ----------------------
        Attached to PyPSA Network 'Unnamed Network'
        Components: 3

        With static data:
        >>> c.static[["carrier", "p_nom"]]
                    carrier  p_nom
        Generator
        my-generator-1      AC    0.0
        my-generator-2      AC   10.0
        my-generator-3      DC   10.0

        The single value for `p_nom` is broadcasted to all components. So you could also
        pass `[10, 10]` instead of `10`.

        See Also
        --------
        [pypsa.Network.add][] : Add components to the network instance.

        """
        if not self.attached:
            msg = (
                "Currently new components can only be added when the components "
                "are already attached to a network."
            )
            raise NotImplementedError(msg)

        self.n_save.add(
            self.name,
            name,
            suffix=suffix,
            overwrite=overwrite,
            **kwargs,
        )

    def rename_component_names(self, **kwargs: str) -> None:
        """Rename component names.

        Rename components and also update all cross-references of the component in
        the network.

        Parameters
        ----------
        **kwargs
            Mapping of old names to new names.

        Examples
        --------
        Define some network
        >>> import pypsa
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
            for component in self.n_save.components.values():
                col_name = self.name.lower()  # TODO: Generalize
                cols = [f"{col_name}{port}" for port in component.ports]
                if cols and not component.static.empty:
                    component.static[cols] = component.static[cols].replace(kwargs)
