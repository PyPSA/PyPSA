"""Network transform module.

Contains single mixin class which is used to inherit to [pypsa.Networks] class.
Should not be used directly.

Transform methods are methods which modify, restructure data and add or remove data.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from deprecation import deprecated

from pypsa.components.common import as_components
from pypsa.network.abstract import _NetworkABC
from pypsa.type_utils import is_1d_list_like

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from pypsa.components.components import Components
    from pypsa.networks import Network

logger = logging.getLogger(__name__)


class NetworkTransformMixin(_NetworkABC):
    """Mixin class for network transform methods.

    Class only inherits to [pypsa.Network][] and should not be used directly.
    All attributes and methods can be used within any Network instance.
    """

    def add(
        self,
        class_name: str,
        name: str | int | Sequence[int | str],
        suffix: str = "",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> pd.Index:
        """Add components to the network.

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
        class_name : str
            Component class name in ("Bus", "Generator", "Load", "StorageUnit",
            "Store", "ShuntImpedance", "Line", "Transformer", "Link").
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
            or pandas.DataFrame for time-varying

        Returns
        -------
        new_names : pandas.index
            Names of new components (including suffix)

        Examples
        --------
        Add a single component:

        >>> n = pypsa.Network()
        >>> n.add("Bus", "my_bus_0")
        Index(['my_bus_0'], dtype='object')
        >>> n.add("Bus", "my_bus_1", v_nom=380)
        Index(['my_bus_1'], dtype='object')
        >>> n.add("Line", "my_line_name", bus0="my_bus_0", bus1="my_bus_1", length=34, r=2, x=4)
        Index(['my_line_name'], dtype='object')

        Add multiple components with static attributes:

        >>> n.add("Load", ["load 1", "load 2"],
        ...       bus=["1", "2"],
        ...       p_set=np.random.rand(len(n.snapshots), 2))
        Index(['load 1', 'load 2'], dtype='object')

        Add multiple components with time-varying attributes:

        >>> import pandas as pd, numpy as np
        >>> buses = range(13)
        >>> snapshots = range(7)
        >>> n = pypsa.Network()
        >>> n.set_snapshots(snapshots)
        >>> n.add("Bus", buses)
        Index(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], dtype='object')
        >>> # add load as numpy array
        >>> n.add("Load",
        ...       n.buses.index + " load",
        ...       bus=buses,
        ...       p_set=np.random.rand(len(snapshots), len(buses)))
        Index(['0 load', '1 load', '2 load', '3 load', '4 load', '5 load', '6 load',
               '7 load', '8 load', '9 load', '10 load', '11 load', '12 load'],
              dtype='object', name='Bus')
        >>> # add wind availability as pandas DataFrame
        >>> wind = pd.DataFrame(np.random.rand(len(snapshots), len(buses)),
        ...        index=n.snapshots,
        ...        columns=buses)
        >>> # use a suffix to avoid boilerplate to rename everything
        >>> n.add("Generator",
        ...       buses,
        ...       suffix=' wind',
        ...       bus=buses,
        ...       p_nom_extendable=True,
        ...       capital_cost=1e5,
        ...       p_max_pu=wind)
        Index(['0 wind', '1 wind', '2 wind', '3 wind', '4 wind', '5 wind', '6 wind',
               '7 wind', '8 wind', '9 wind', '10 wind', '11 wind', '12 wind'],
              dtype='object')


        """
        c = as_components(self, class_name)
        # Process name/names to pandas.Index of strings and add suffix
        single_component = np.isscalar(name)
        names = pd.Index([name]) if single_component else pd.Index(name)
        names = names.astype(str) + suffix

        names_str = "name" if single_component else "names"
        # Read kwargs into static and time-varying attributes
        series = {}
        static = {}

        # Check if names are unique
        if not names.is_unique:
            msg = f"Names for {c.name} must be unique."
            raise ValueError(msg)

        for k, v in kwargs.items():
            # If index/ columnes are passed (pd.DataFrame or pd.Series)
            # - cast names index to string and add suffix
            # - check if passed index/ columns align
            msg = "{} has an index which does not align with the passed {}."
            if isinstance(v, pd.Series) and single_component:
                if not v.index.equals(self.snapshots):
                    raise ValueError(msg.format(f"Series {k}", "network snapshots"))
            elif isinstance(v, pd.Series):
                # Cast names index to string + suffix
                v = v.rename(
                    index=lambda s: str(s)
                    if str(s).endswith(suffix)
                    else str(s) + suffix
                )
                if not v.index.equals(names):
                    raise ValueError(msg.format(f"Series {k}", names_str))
            if isinstance(v, pd.DataFrame):
                # Cast names columns to string + suffix
                v = v.rename(
                    columns=lambda s: str(s)
                    if str(s).endswith(suffix)
                    else str(s) + suffix
                )
                if not v.index.equals(self.snapshots):
                    raise ValueError(msg.format(f"DataFrame {k}", "network snapshots"))
                if not v.columns.equals(names):
                    raise ValueError(msg.format(f"DataFrame {k}", names_str))

            # Convert list-like and 1-dim array to pandas.Series
            if is_1d_list_like(v):
                try:
                    if single_component:
                        v = pd.Series(v, index=self.snapshots)
                    else:
                        v = pd.Series(v)
                        if len(v) == 1:
                            v = v.iloc[0]
                            logger.debug(
                                "Single value sequence for %s is treated as a scalar "
                                "and broadcasted to all components. It is recommended "
                                "to explicitly pass a scalar instead.",
                                k,
                            )
                        else:
                            v.index = names
                except ValueError as e:
                    expec_str = (
                        f"{len(self.snapshots)} for each snapshot."
                        if single_component
                        else f"{len(names)} for each component name."
                    )
                    msg = f"Data for {k} has length {len(v)} but expected {expec_str}"
                    raise ValueError(msg) from e
            # Convert 2-dim array to pandas.DataFrame
            if isinstance(v, np.ndarray):
                if v.shape == (len(self.snapshots), len(names)):
                    v = pd.DataFrame(v, index=self.snapshots, columns=names)
                else:
                    msg = (
                        f"Array {k} has shape {v.shape} but expected "
                        f"({len(self.snapshots)}, {len(names)})."
                    )
                    raise ValueError(msg)

            if isinstance(v, dict):
                msg = (
                    "Dictionaries are not supported as attribute values. Please use "
                    "pandas.Series or pandas.DataFrame instead."
                )
                raise NotImplementedError(msg)

            # Handle addition of single component
            if single_component:
                # Read 1-dim data as time-varying attribute
                if isinstance(v, pd.Series):
                    series[k] = pd.DataFrame(
                        v.values, index=self.snapshots, columns=names
                    )
                # Read 0-dim data as static attribute
                else:
                    static[k] = v

            # Handle addition of multiple components
            elif not single_component:
                # Read 2-dim data as time-varying attribute
                if isinstance(v, pd.DataFrame):
                    series[k] = v
                # Read 1-dim data as static attribute
                elif isinstance(v, pd.Series):
                    static[k] = v.values
                # Read scalar data as static attribute
                else:
                    static[k] = v

        # Load static attributes as components
        if static:
            static_df = pd.DataFrame(static, index=names)
        else:
            static_df = pd.DataFrame(index=names)
        self._import_components_from_df(static_df, c.name, overwrite=overwrite)

        # Load time-varying attributes as components
        for k, v in series.items():
            self._import_series_from_df(v, c.name, k, overwrite=overwrite)

        return names

    def remove(
        self,
        class_name: str,
        name: str | int | Sequence[int | str],
        suffix: str = "",
    ) -> None:
        """Remove a single component or a list of components from the network.

        Removes it from component DataFrames.

        Parameters
        ----------
        class_name : str
            Component class name
        name : str, int, list-like or pandas.Index
            Component name(s)
        suffix : str, default=''
            Suffix to be added to the component name(s)

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.snapshots = pd.date_range("2015-01-01", freq="h", periods=2)
        >>> n.add("Bus", ["bus0", "bus1"])
        Index(['bus0', 'bus1'], dtype='object')
        >>> n.add("Bus", "bus2", p_min_pu=[1, 1])
        Index(['bus2'], dtype='object', name='Bus')
        >>> n.components.buses.static
              v_nom type    x    y  ... v_mag_pu_max control generator  sub_network
        Bus                         ...
        bus0    1.0       0.0  0.0  ...          inf      PQ
        bus1    1.0       0.0  0.0  ...          inf      PQ
        bus2    1.0       0.0  0.0  ...          inf      PQ
        <BLANKLINE>
        [3 rows x 13 columns]

        Remove a single component:
        >>> n.remove("Bus", "bus2")


        Any component data is dropped from the component DataFrames.
        >>> n.components.buses.static
            v_nom type    x    y  ... v_mag_pu_max control generator  sub_network
        Bus                         ...
        bus0    1.0       0.0  0.0  ...          inf      PQ
        bus1    1.0       0.0  0.0  ...          inf      PQ
        <BLANKLINE>
        [2 rows x 13 columns]
        >>> n.components.buses.dynamic.p_min_pu
        Empty DataFrame
        Columns: []
        Index: [2015-01-01 00:00:00, 2015-01-01 01:00:00]

        Remove multiple components:
        >>> n.remove("Bus", ["bus0", "bus1"])

        >>> n.components.buses.static
        Empty DataFrame
        Columns: [v_nom, type, x, y, carrier, unit, location, v_mag_pu_set, v_mag_pu_min, v_mag_pu_max, control, generator, sub_network]
        Index: []

        """
        c = as_components(self, class_name)

        # Process name/names to pandas.Index of strings and add suffix
        names = pd.Index([name]) if np.isscalar(name) else pd.Index(name)
        names = names.astype(str) + suffix

        # Drop from static components
        cls_static = self.static(c.name)
        cls_static.drop(names, inplace=True)

        # Drop from time-varying components
        dynamic = self.dynamic(c.name)
        for df in dynamic.values():
            df.drop(df.columns.intersection(names), axis=1, inplace=True)

    @deprecated(
        deprecated_in="0.31",
        removed_in="1.0",
        details="Use `n.add` as a drop-in replacement instead.",
    )
    def madd(
        self,
        class_name: str,
        names: Sequence,
        suffix: str = "",
        **kwargs: Any,
    ) -> pd.Index:
        """Add multiple components to the network, along with their attributes.

        .. deprecated:: 0.31
          ``n.madd`` is deprecated and will be removed in a future version. Use
            :py:meth:`pypsa.Network.add` instead. It can handle both single and multiple
            removal of components.

        Make sure when adding static attributes as pandas Series that they are indexed
        by names. Make sure when adding time-varying attributes as pandas DataFrames that
        their index is a superset of n.snapshots and their columns are a
        subset of names.

        Any attributes which are not specified will be given the default
        value from :doc:`/user-guide/components`.

        Parameters
        ----------
        class_name : string
            Component class name in ("Bus", "Generator", "Load", "StorageUnit",
            "Store", "ShuntImpedance", "Line", "Transformer", "Link").
        names : list-like or pandas.Index
            Component names
        suffix : string, default ''
            All components are named after names with this added suffix. It
            is assumed that all Series and DataFrames are indexed by the original names.
        kwargs
            Component attributes, e.g. x=[0.1, 0.2], can be list, pandas.Series
            of pandas.DataFrame for time-varying

        """
        return self.add(class_name=class_name, name=names, suffix=suffix, **kwargs)

    @deprecated(
        deprecated_in="0.31",
        removed_in="1.0",
        details="Use `n.remove` as a drop-in replacement instead.",
    )
    def mremove(self, class_name: str, names: Sequence) -> None:
        """Remove multiple components from the network.

        .. deprecated:: 0.31
          ``n.mremove`` is deprecated and will be removed in a future version. Use
            :py:meth:`pypsa.Network.remove` instead. It can handle both single and multiple
            removal of components.

        ``n.mremove`` is deprecated and will be removed in version 1.0. Use
        py:meth:`pypsa.Network.remove` instead. It can handle both single and multiple removal of
        components.

        Removes them from component DataFrames.

        Parameters
        ----------
        class_name : string
            Component class name
        names : list-like
            Component names

        """
        self.remove(class_name=class_name, name=names)

    def merge(
        self,
        other: Network,
        components_to_skip: Collection[str] | None = None,
        inplace: bool = False,
        with_time: bool = True,
    ) -> Any:
        """Merge the components of two networks.

        Requires disjunct sets of component indices and, if time-dependent data is
        merged, identical snapshots and snapshot weightings.

        If a component in ``other`` does not have values for attributes present in
        ``n``, default values are set.

        If a component in ``other`` has attributes which are not present in
        ``n`` these attributes are ignored.

        Parameters
        ----------
        n : pypsa.Network
            Network to add to.
        other : pypsa.Network
            Network to add from.
        components_to_skip : list-like, default None
            List of names of components which are not to be merged e.g. "Bus"
        inplace : bool, default False
            If True, merge into ``n`` in-place, otherwise a copy is made.
        with_time : bool, default True
            If False, only static data is merged.

        Returns
        -------
        receiving_n : pypsa.Network
            Merged network, or None if inplace=True

        """
        to_skip = {"Network", "SubNetwork", "LineType", "TransformerType"}
        if components_to_skip:
            to_skip.update(components_to_skip)
        to_iterate = other.all_components - to_skip
        # ensure buses are merged first
        to_iterate_list = ["Bus"] + sorted(to_iterate - {"Bus"})
        for c in other.iterate_components(to_iterate_list):
            if not c.static.index.intersection(self.static(c.name).index).empty:
                msg = f"Component {c.name} has overlapping indices, cannot merge networks."
                raise ValueError(msg)
        if with_time:
            snapshots_aligned = self.snapshots.equals(other.snapshots)
            if not snapshots_aligned:
                msg = "Snapshots do not agree, cannot merge networks."
                raise ValueError(msg)
            weightings_aligned = self.snapshot_weightings.equals(
                other.snapshot_weightings
            )
            if not weightings_aligned:
                # Check if only index order is different
                # TODO fix with #1128
                if self.snapshot_weightings.reindex(
                    sorted(self.snapshot_weightings.columns), axis=1
                ).equals(
                    other.snapshot_weightings.reindex(
                        sorted(other.snapshot_weightings.columns), axis=1
                    )
                ):
                    weightings_aligned = True
                else:
                    msg = "Snapshot weightings do not agree, cannot merge networks."
                    raise ValueError(msg)
        new = self if inplace else self.copy()
        if other.srid != new.srid:
            logger.warning(
                "Spatial Reference System Indentifier of networks do not agree: "
                "%s, %s. Assuming %s.",
                new.srid,
                other.srid,
                new.srid,
            )
        for c in other.iterate_components(to_iterate_list):
            new.add(c.name, c.static.index, **c.static)
            if with_time:
                for k, v in c.dynamic.items():
                    new._import_series_from_df(v, c.name, k)

        return None if inplace else new

    def rename_component_names(
        self, component: str | Components, **kwargs: str
    ) -> None:
        """Rename component names.

        Rename components of component type and also update all cross-references of
        the component in network.

        Parameters
        ----------
        component : str or pypsa.Components
            Component type or instance of pypsa.Components.
        **kwargs
            Mapping of old names to new names.


        Examples
        --------
        Define some network

        >>> n = pypsa.Network()
        >>> n.add("Bus", ["bus1"])
        Index(['bus1'], dtype='object')
        >>> n.add("Generator", ["gen1"], bus="bus1")
        Index(['gen1'], dtype='object')

        Now rename the bus component

        >>> n.rename_component_names("Bus", bus1="bus2")

        Which updates the bus components

        >>> n.buses.index
        Index(['bus2'], dtype='object', name='Bus')

        and all references in the network

        >>> n.generators.bus
        Generator
        gen1    bus2
        Name: bus, dtype: object

        """
        c = as_components(self, component)
        c.rename_component_names(**kwargs)
