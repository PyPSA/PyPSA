"""Power system components."""

from __future__ import annotations

import copy
import logging
import warnings
from typing import TYPE_CHECKING, Any
from weakref import ref

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path
import numpy as np
import pandas as pd
import pyproj
import validators
from deprecation import deprecated
from pyproj import CRS, Transformer
from scipy.sparse import csgraph

from pypsa.clustering import ClusteringAccessor
from pypsa.common import (
    as_index,
    deprecated_common_kwargs,
    deprecated_in_next_major,
    equals,
)
from pypsa.components.components import Components, SubNetworkComponents
from pypsa.components.store import ComponentsStore
from pypsa.consistency import NetworkConsistencyMixin
from pypsa.constants import DEFAULT_EPSG, DEFAULT_TIMESTAMP
from pypsa.definitions.structures import Dict
from pypsa.network.components import NetworkComponentsMixin
from pypsa.network.descriptors import NetworkDescriptorsMixin
from pypsa.network.graph import NetworkGraphMixin
from pypsa.network.index import NetworkIndexMixin
from pypsa.network.io import NetworkIOMixin
from pypsa.network.power_flow import (
    NetworkPowerFlowMixin,
    SubNetworkPowerFlowMixin,
    find_cycles,
)
from pypsa.network.transform import NetworkTransformMixin
from pypsa.optimization.optimize import OptimizationAccessor
from pypsa.plot.accessor import PlotAccessor
from pypsa.plot.maps import explore, iplot
from pypsa.statistics.abstract import AbstractStatisticsAccessor
from pypsa.statistics.expressions import StatisticsAccessor
from pypsa.version import __version_semver__

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Sequence

    import linopy
    from scipy.sparse import spmatrix

    from pypsa.components.legacy import Component


logger = logging.getLogger(__name__)


dir_name = Path(__file__).parent

standard_types_dir_name = "data/standard_types"


inf = float("inf")


class Network(
    NetworkComponentsMixin,
    NetworkDescriptorsMixin,
    NetworkTransformMixin,
    NetworkIndexMixin,
    NetworkConsistencyMixin,
    NetworkGraphMixin,
    NetworkPowerFlowMixin,
    NetworkIOMixin,
):
    """Network container for all buses, one-ports and branches."""

    # Optimization
    _multi_invest: int
    _linearized_uc: int
    iteration: int  # TODO Remove/ use property

    # ----------------
    # Dunder methods
    # ----------------

    def __init__(
        self,
        import_name: str | Path = "",
        name: str = "Unnamed Network",
        ignore_standard_types: bool = False,
        override_components: pd.DataFrame | None = None,
        override_component_attrs: Dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PyPSA Network.

        Parameters
        ----------
        import_name : string, Path
            Path to netCDF file, HDF5 .h5 store or folder of CSV files from which to
            import network data. The string could be a URL. If cloudpathlib is installed,
            the string could be a object storage URI with an `s3`, `gs` or `az` URI scheme.
        name : string, default: "Unnamed Network"
            Network name.
        ignore_standard_types : boolean, default False
            If True, do not read in PyPSA standard types into standard types
            DataFrames.
        override_components : pandas.DataFrame
            If you want to override the standard PyPSA components in
            :meth:`n.default_components <pypsa.Network.default_components>`, pass it a
            DataFrame indexed by component names and.
            See :doc:`/user-guide/components` for more information.
        override_component_attrs : pypsa.descriptors.Dict of pandas.DataFrame
            If you want to override
            :meth:`n.default_component_attrs <pypsa.Network.default_component_attrs>`.
            See :doc:`/user-guide/components` for more information.
        kwargs
            Any remaining attributes to set

        Deprecation
        ------------
        [:material-tag-outline: v0.33.0](/release-notes/#v0.33.0): Parameters
        `override_components` and `override_component_attrs` are deprecated and do not.
        Please check the release notes for more information.

        Examples
        --------
        >>> nw1 = pypsa.Network("network.nc") # doctest: +SKIP
        >>> nw2 = pypsa.Network("/my/folder") # doctest: +SKIP
        >>> nw3 = pypsa.Network("https://github.com/PyPSA/PyPSA/raw/master/examples/scigrid-de/scigrid-with-load-gen-trafos.nc") # doctest: +SKIP
        >>> nw4 = pypsa.Network("s3://my-bucket/my-network.nc") # doctest: +SKIP

        """
        if override_components is not None or override_component_attrs is not None:
            msg = (
                "Parameters `override_components` and `override_component_attrs` "
                "are deprecated. Please check the release notes: "
                "https://pypsa.readthedocs.io/en/latest/references/release-notes.html#v0-33-0."
                "Deprecated in version 0.33 and will be removed in version 1.0."
            )
            raise DeprecationWarning(msg)

        # Initialise root logger and set its level, if this has not been done before
        logging.basicConfig(level=logging.INFO)

        # Store PyPSA version
        self._pypsa_version: str = __version_semver__

        # Set attributes
        self._name = name
        self._meta: dict = {}
        self._crs: CRS = CRS.from_epsg(DEFAULT_EPSG)

        self._snapshots = pd.Index([DEFAULT_TIMESTAMP], name="snapshot")

        cols = ["objective", "stores", "generators"]
        self._snapshot_weightings = pd.DataFrame(1, index=self.snapshots, columns=cols)

        cols = ["objective", "years"]
        self._investment_period_weightings: pd.DataFrame = pd.DataFrame(
            index=self.investment_periods, columns=cols
        )

        self._model: linopy.Model | None = None
        self._objective: float | None = None
        self._objective_constant: float | None = None

        # Initialize accessors
        self.optimize: OptimizationAccessor = OptimizationAccessor(self)
        """Accessor to the network optimization functionality.

        See Also
        --------
        [pypsa.optimization.OptimizationAccessor][]
        """
        self.cluster: ClusteringAccessor = ClusteringAccessor(self)
        """Accessor to the network clustering functionality.

        See Also
        --------
        [pypsa.clustering.ClusteringAccessor][]
        """
        self.statistics: StatisticsAccessor = StatisticsAccessor(self)
        """Accessor to the network statistics functionality.

        See Also
        --------
        [pypsa.statistics.StatisticsAccessor][]
        """
        self.plot: PlotAccessor = PlotAccessor(self)
        """Accessor to the network plotting functionality.

        See Also
        --------
        [pypsa.plot.PlotAccessor][]
        """

        NetworkComponentsMixin.__init__(self)

        if not ignore_standard_types:
            self._read_in_default_standard_types()

        if import_name:
            if not validators.url(str(import_name)):
                import_name = Path(import_name)

            # Read specified file
            if str(import_name).endswith(".h5"):
                self.import_from_hdf5(import_name)
            elif str(import_name).endswith(".nc"):
                self.import_from_netcdf(import_name)
            elif str(import_name).endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
                self.import_from_excel(import_name)
            elif isinstance(import_name, Path) and import_name.is_dir():
                self.import_from_csv_folder(import_name)
            else:
                msg = f"import_name '{import_name}' is not a valid .h5 file, .nc file or directory."
                raise ValueError(msg)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        """Human Readable string representation of the network."""
        return f"PyPSA Network '{self.name}'"

    def __repr__(self) -> str:
        """Return a string representation for the REPL."""
        # TODO make this actually for the REPL
        header = f"{self}\n" + "-" * len(str(self))  # + "\n"
        comps = {
            c.name: f" - {c.name}: {len(c.static)}"
            for c in self.iterate_components()
            if "Type" not in c.name and len(c.static)
        }
        content = "\nComponents:"
        if comps:
            content += "\n" + "\n".join(comps[c] for c in sorted(comps))
        else:
            header = "Empty " + header
            content += " none"
        content += "\n"
        content += f"Snapshots: {len(self.snapshots)}"

        return header + content

    def __add__(self, other: Network) -> None:
        """Merge all components of two networks.

        Parameters
        ----------
        other : Network
            Network to merge into this one.

        See Also
        --------
        [pypsa.Network.merge][] : Merge second network into network.

        Examples
        --------
        >>> n1 = pypsa.Network()
        >>> n2 = pypsa.Network()
        >>> n1.add("Bus", "bus1")
        Index(['bus1'], dtype='object')
        >>> n2.add("Bus", "bus2")
        Index(['bus2'], dtype='object')
        >>> new_network = n1 + n2
        >>> len(new_network.buses)
        2

        """
        return self.__class__.merge(self, other)

    def __eq__(self, other: object) -> bool:
        """Check for equality of two networks.

        Parameters
        ----------
        other : Any
            The other network to compare with.

        Returns
        -------
        bool
            True if the networks are equal, False otherwise.

        See Also
        --------
        [pypsa.Network.equals][] : Check for equality of two networks.

        """
        return self.equals(other)

    @deprecated(
        deprecated_in="0.34",
        removed_in="1.0",
        details="Use `n.plot.iplot()` as a drop-in replacement instead.",
    )
    def iplot(self, *args: Any, **kwargs: Any) -> Any:
        """Plot the network on a map using Plotly.

        !!! warning "Deprecated in v0.34"
            Use `n.plot.iplot()` as a drop-in replacement instead.
        """
        return iplot(self, *args, **kwargs)

    @deprecated(
        deprecated_in="0.34",
        removed_in="1.0",
        details="Use `n.plot.explore()` as a drop-in replacement instead.",
    )
    def explore(self, *args: Any, **kwargs: Any) -> Any:
        """Plot the network on a map using Folium.

        !!! warning "Deprecated in v0.34"
            Use `n.plot.explore()` as a drop-in replacement instead.
        """
        return explore(self, *args, **kwargs)

    def equals(self, other: Any, log_mode: str = "silent") -> bool:
        """Check for equality of two networks.

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
            True if the networks are equal, False otherwise.

        Examples
        --------
        >>> n1 = pypsa.Network()
        >>> n2 = pypsa.Network()
        >>> n1.add("Bus", "bus1")
        Index(['bus1'], dtype='object')
        >>> n2.add("Bus", "bus2")
        Index(['bus2'], dtype='object')
        >>> n1.equals(n2)
        False

        """
        ignore = [
            OptimizationAccessor,
            ClusteringAccessor,
            StatisticsAccessor,
            PlotAccessor,
            AbstractStatisticsAccessor,
        ]
        not_equal = False
        if isinstance(other, self.__class__):
            for key, value in self.__dict__.items():
                if not equals(
                    value,
                    other.__dict__[key],
                    ignored_classes=ignore,
                    log_mode=log_mode,
                    path="n." + key,
                ):
                    logger.warning("Mismatch in attribute: %s", key)
                    not_equal = True
                    if not log_mode:
                        break
        else:
            logger.warning(
                "Can only compare two pypsa.Network objects with each other. Got %s.",
                type(other),
            )

            return False

        return not not_equal

    # ----------------
    # Meta data
    # ----------------

    @property
    def name(self) -> str:
        """Name of the network.

        The name is set when the network is created. It can also be changed by setting
        the `name` attribute. It is only descriptive and not used for any
        functionality.

        Examples
        --------
        >>> n.name
        'AC-DC-Meshed'

        >>> n = pypsa.Network(name='Unnamed Network')
        >>> n.name
        'Unnamed Network'

        >>> n.name = 'net'
        >>> n.name
        'net'

        """
        return self._name

    @name.setter
    def name(self, new: str) -> None:
        """Set the name of the network."""
        self._name = new

    @property
    def pypsa_version(self) -> str:
        """PyPSA version of the network.

        The PyPSA version is set when the network is created and cannot be changed
        manually. When a network of an older version is imported, the version is
        automatically updated to the current version.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.pypsa_version # doctest: +SKIP
        '1.0.0'

        """
        return self._pypsa_version

    @property
    def meta(self) -> dict:
        """Dictionary of the network meta data.

        Any additional meta data can be added to the network by setting the `meta`
        attribute. Meta data will be saved on export.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.meta['description'] = 'This is a test network'
        >>> n.meta['any_key'] = 'Any Key can be added'
        >>> n.meta
        {'description': 'This is a test network', 'any_key': 'Any Key can be added'}

        """
        return self._meta

    @meta.setter
    def meta(self, new: dict) -> None:
        """Set the network meta data."""
        if not isinstance(new, (dict | Dict)):
            msg = f"Meta must be a dictionary, received a {type(new)}"
            raise TypeError(msg)
        self._meta = new

    @property
    def model(self) -> linopy.Model:
        """Access to linopy model object.

        After optimizing a network, the linopy model object is stored in the network
        and can be accessed via this property. It cannot be set manually.

        Examples
        --------
        >>> n.model
        Linopy LP model
        ===============
        <BLANKLINE>
        Variables:
        ----------
        * Generator-p_nom (Generator-ext)
        * Line-s_nom (Line-ext)
        * Link-p_nom (Link-ext)
        * Generator-p (snapshot, Generator)
        * Line-s (snapshot, Line)
        * Link-p (snapshot, Link)
        * objective_constant
        <BLANKLINE>
        Constraints:
        ------------
        * Generator-ext-p_nom-lower (Generator-ext)
        * Generator-ext-p_nom-upper (Generator-ext)
        * Line-ext-s_nom-lower (Line-ext)
        * Line-ext-s_nom-upper (Line-ext)
        * Link-ext-p_nom-lower (Link-ext)
        * Link-ext-p_nom-upper (Link-ext)
        * Generator-ext-p-lower (snapshot, Generator-ext)
        * Generator-ext-p-upper (snapshot, Generator-ext)
        * Line-ext-s-lower (snapshot, Line-ext)
        * Line-ext-s-upper (snapshot, Line-ext)
        * Link-ext-p-lower (snapshot, Link-ext)
        * Link-ext-p-upper (snapshot, Link-ext)
        * Bus-nodal_balance (Bus, snapshot)
        * Kirchhoff-Voltage-Law (snapshot, cycles)
        * GlobalConstraint-co2_limit
        <BLANKLINE>
        Status:
        -------
        ok

        """
        if self._model is None:
            msg = "The network has not been optimized yet and no model is stored."
            raise ValueError(msg)
        return self._model

    @model.deleter
    def model(self) -> None:
        """Delete the model object."""
        self._model = None

    @property
    def objective(self) -> float:
        """Objective value of the solved network.

        The property yields the objective value of the solved network. It is set after
        optimizing the network points to the linopy solution (e.g. is an alias for
        `n.model.objective.value`). When loading a network from file and the model
        object is not loaded, the objective value is still available, as it is stored
        in the network object.

        When optimizing for system costs, the total system costs are the sum of the
        [pypsa.Network.objective][] and the [pypsa.Network.objective_constant][].

        Examples
        --------
        >>> n.objective # doctest: +ELLIPSIS
        -34742...

        >>> n.objective + n.objective_constant # doctest: +ELLIPSIS
        np.float64(18441...)

        """
        if self._objective is None:
            msg = "The network has not been optimized yet and no objective value is stored."
            raise ValueError(msg)
        return self._objective

    @objective.setter
    def objective(self, new: float) -> None:
        """Set the objective value of the network."""
        warnings.warn(
            "Setting the objective value via `n.objective = ...` is deprecated in 0.35 "
            "and will be removed in 1.0. Use `n.model.objective.value = ...` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._objective = new

    @property
    def objective_constant(self) -> float:
        """Objective constant of the network.

        The property yields the fixed part of the objective function. It is set after
        optimizing the network.

        When optimizing for system costs, the total system costs are the sum of the
        [pypsa.Network.objective][] and the [pypsa.Network.objective_constant][]. When
        loading a network from file and the model object is not loaded, the objective
        constant is still available, as it is stored in the network object.

        Examples
        --------
        >>> n.objective_constant # doctest: +ELLIPSIS
        np.float64(21915...)

        >>> n.objective + n.objective_constant # doctest: +ELLIPSIS
        np.float64(18441...)

        """
        if self._objective_constant is None:
            msg = "The network has not been optimized yet and no objective constant is stored."
            raise ValueError(msg)
        return self._objective_constant

    @objective_constant.setter
    def objective_constant(self, new: float) -> None:
        """Set the objective constant of the network."""
        warnings.warn(
            "Setting the objective constant via `n.objective_constant = ...` is deprecated in 0.35 "
            "and will be removed in 1.0. Use `n.model.objective.constant = ...` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._objective_constant = new

    @property
    def crs(self) -> Any:
        """Coordinate reference system of the network's geometries.

        Examples
        --------
        >>> n.crs
        <Geographic 2D CRS: EPSG:4326>
        Name: WGS 84
        Axis Info [ellipsoidal]:
        - Lat[north]: Geodetic latitude (degree)
        - Lon[east]: Geodetic longitude (degree)
        Area of Use:
        - name: World.
        - bounds: (-180.0, -90.0, 180.0, 90.0)
        Datum: World Geodetic System 1984 ensemble
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        """
        return self._crs

    @crs.setter
    def crs(self, new: Any) -> None:
        """Set the coordinate reference system of the network's geometries.

        See Also
        --------
        [pypsa.Network.srid][] : Spatial reference system identifier of the network's
            geometries.
        [pypsa.Network.shapes][] : Geometries of the network

        """
        self.components.shapes.static.set_crs(new)
        self._crs = self.components.shapes.static.crs

    def to_crs(self, new: int | str | pyproj.CRS) -> None:
        """Convert the network's geometries and bus coordinates to a new crs.

        See Also
        --------
        [pypsa.Network.crs][] : Coordinate reference system of the network's geometries
        [pypsa.Network.srid][] : Spatial reference system identifier of the network's
            geometries.
        [pypsa.Network.shapes][] : Geometries of the network

        """
        current = self.crs
        self.shapes.to_crs(new, inplace=True)
        self._crs = self.shapes.crs
        transformer = Transformer.from_crs(current, self.crs)
        self.buses["x"], self.buses["y"] = transformer.transform(
            self.buses["x"], self.buses["y"]
        )

    @property
    def srid(self) -> int:
        """Spatial reference system identifier of the network's geometries.

        Examples
        --------
        >>> n.srid
        4326

        See Also
        --------
        [pypsa.Network.crs][] : Coordinate reference system of the network's geometries
        [pypsa.Network.shapes][] : Geometries of the network

        """
        return self.crs.to_epsg()

    @srid.setter
    def srid(self, new: str | int) -> None:
        """Set the spatial reference system identifier of the network's geometries.

        See Also
        --------
        [pypsa.Network.crs][] : Coordinate reference system of the network's geometries
        [pypsa.Network.shapes][] : Geometries of the network

        """
        self.crs = pyproj.CRS.from_epsg(new)

    def copy(
        self,
        snapshots: Sequence | None = None,
        investment_periods: Sequence | None = None,
        ignore_standard_types: bool = False,
        with_time: bool | None = None,
    ) -> Network:
        """Return a deep copy of Network object.

        If only default arguments are passed, the copy will be created via
        :func:`copy.deepcopy` and will contain all components and time-varying data.
        For most networks this is the fastest way. However, if the network is very
        large, it might be better to filter snapshots and investment periods to reduce
        the size of the copy. In this case :func:`copy.deepcopy` is not used and only
        the selected snapshots and investment periods are copied to a new object.


        Parameters
        ----------
        snapshots : list or tuple or pd.Index , default self.snapshots
            A list of snapshots to copy, must be a subset of n.snapshots. Pass
            an empty list ignore all snapshots.
        investment_periods : list or tuple or pd.Index, default self.investment_period_weightings.index
            A list of investment periods to copy, must be a subset of n.investment_periods. Pass
        ignore_standard_types : boolean, default False
            Ignore the PyPSA standard types.
        with_time : boolean, default True
            Copy snapshots and time-varying n.component_names_t data too.

            .. deprecated:: 0.29.0
              The 'with_time' argument is deprecated in 0.29 and will be removed in a
              future version. Pass an empty list to 'snapshots' instead.

        Returns
        -------
        n : pypsa.Network
            The copied network object.

        Examples
        --------
        With a simple reference the network is not copied:
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> network_copy = n
        >>> id(network_copy) == id(n)
        True

        Use the copy method to create a new network object:
        >>> network_copy = n.copy()
        >>> id(network_copy) == id(n)
        False

        You can also filter on a subset of snapshots (or investment periods):
        >>> n.snapshots
        DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
                       '2015-01-01 02:00:00', '2015-01-01 03:00:00',
                       '2015-01-01 04:00:00', '2015-01-01 05:00:00',
                       '2015-01-01 06:00:00', '2015-01-01 07:00:00',
                       '2015-01-01 08:00:00', '2015-01-01 09:00:00'],
                      dtype='datetime64[ns]', name='snapshot', freq=None)
        >>> network_copy = n.copy(snapshots=n.snapshots[0])
        >>> network_copy.snapshots
        DatetimeIndex(['2015-01-01'], dtype='datetime64[ns]', name='snapshot', freq=None)

        """
        if self._model is not None and self._model.solver_model is not None:
            msg = "Copying solved networks is not supported yet."
            raise NotImplementedError(msg)

        # Use copy.deepcopy if no arguments are passed
        args = [snapshots, investment_periods, ignore_standard_types, with_time]
        if all(arg is None or arg is False for arg in args):
            return copy.deepcopy(self)

        # Convert to pandas.Index
        snapshots_ = as_index(self, snapshots, "snapshots")
        investment_periods_ = as_index(self, investment_periods, "investment_periods")

        # Deprecation warnings
        if with_time is not None:
            warnings.warn(
                "Argument 'with_time' is deprecated in 0.29 and will be "
                "removed in a future version. Pass an empty list to 'snapshots' instead."
                "Deprecated in version 0.29 and will be removed in version 1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            snapshots_ = pd.Index([], name="snapshot")

        # Setup new network
        n = self.__class__(ignore_standard_types=ignore_standard_types)

        # Copy components
        other_comps = sorted(self.all_components - {"Bus", "Carrier"})
        # Needs to copy buses and carriers first, since there are dependencies on them
        for component in self.iterate_components(["Bus", "Carrier"] + other_comps):
            # Drop the standard types to avoid them being read in twice
            if (
                not ignore_standard_types
                and component.name in self.standard_type_components
            ):
                static = component.static.drop(
                    n.components[component.name]["standard_types"].index
                )
            else:
                static = component.static
            n.add(component.name, static.index, **static)

        # Copy time-varying data, if given

        if len(snapshots_) > 0:
            n.set_snapshots(snapshots_)
            # Apply time-varying data
            for component in self.iterate_components():
                dynamic = getattr(n, component.list_name + "_t")
                for k in component.dynamic:
                    # Check if all snapshots_ are in the index
                    if set(snapshots_).issubset(component.dynamic[k].index):
                        dynamic[k] = component.dynamic[k].loc[snapshots_].copy()
                    else:
                        dynamic[k] = component.dynamic[k].reindex(snapshots_).copy()

            # Apply investment periods
            if not investment_periods_.empty:
                n.set_investment_periods(investment_periods_)

            # Add weightings
            n.snapshot_weightings = self.snapshot_weightings.loc[snapshots_].copy()
            n.investment_period_weightings = self.investment_period_weightings.loc[
                investment_periods_
            ].copy()

        # Catch all remaining attributes of network
        for attr in [
            "name",
            "srid",
            "_meta",
            "_linearized_uc",
            "_multi_invest",
            "_objective",
            "_objective_constant",
            "now",
        ]:
            if hasattr(self, attr):
                setattr(n, attr, getattr(self, attr))

        return n

    def __getitem__(self, key: str) -> Network:
        """Return a shallow slice of the Network object.

        A shallow slice will only include the selected buses and all the connected
        components.

        Parameters
        ----------
        key : indexer or tuple of indexer
            If only one indexer is provided it is used in the .loc
            indexer of the buses dataframe (refer also to the help for
            pd.DataFrame.loc). If a tuple of two indexers are provided,
            the first one is used to slice snapshots and the second
            one buses.

        Returns
        -------
        n : pypsa.Network

        Examples
        --------
        >>> sub_network_0 = n[n.buses.sub_network == "0"]

        """
        if isinstance(key, tuple):
            time_i, key = key
        else:
            time_i = slice(None)

        # Setup new network
        n = self.__class__()

        n.add(
            "Bus",
            pd.DataFrame(self.buses.loc[key]).assign(sub_network="").index,
            **pd.DataFrame(self.buses.loc[key]).assign(sub_network=""),
        )
        buses_i = n.buses.index

        rest_components = (
            self.all_components
            - self.standard_type_components
            - self.one_port_components
            - self.branch_components
        )
        for c in rest_components - {"Bus", "SubNetwork"}:
            n.add(c, pd.DataFrame(self.static(c)).index, **pd.DataFrame(self.static(c)))

        for c in self.standard_type_components:
            static = pd.DataFrame(
                self.static(c).drop(self.components[c]["standard_types"].index)
            )
            n.add(c, static.index, **static)

        for c in self.one_port_components:
            static = pd.DataFrame(self.static(c).loc[lambda df: df.bus.isin(buses_i)])
            n.add(c, static.index, **static)

        for c in self.branch_components:
            static = pd.DataFrame(
                self.static(c).loc[
                    lambda df: df.bus0.isin(buses_i) & df.bus1.isin(buses_i)
                ]
            )
            n.add(c, static.index, **static)

        n.set_snapshots(self.snapshots[time_i])
        for c in self.all_components:
            i = n.static(c).index
            try:
                ndynamic = n.dynamic(c)
                dynamic = self.dynamic(c)

                for k in dynamic:
                    ndynamic[k] = dynamic[k].loc[
                        time_i, i.intersection(dynamic[k].columns)
                    ]
            except AttributeError:
                pass

        # catch all remaining attributes of network
        for attr in ["name", "_crs"]:
            setattr(n, attr, getattr(self, attr))

        n.snapshot_weightings = self.snapshot_weightings.loc[time_i]

        return n

    # beware, this turns bools like s_nom_extendable into objects because of
    # presence of links without s_nom_extendable
    def branches(self) -> pd.DataFrame:
        """Get branches.

        Branches are Lines, Links and Transformers.

        !!! note
            This method will return a merged copy of all branches of the network.
            Changes to the returned DataFrame will not be reflected in the network.

        Examples
        --------
        >>> n.branches() # doctest: +ELLIPSIS
                                 active    b  b_pu  ...         x      x_pu  x_pu_eff
        component name                                  ...

        Line      0                    True  0.0   0.0  ...  0.796878  0.000006  0.000006
                  1                    True  0.0   0.0  ...  0.391560  0.000003  0.000003
                  2                    True  0.0   0.0  ...  0.000000  0.000000  0.000000
                  3                    True  0.0   0.0  ...  0.000000  0.000000  0.000000
                  4                    True  0.0   0.0  ...  0.000000  0.000000  0.000000
                  5                    True  0.0   0.0  ...  0.238800  0.000002  0.000002
                  6                    True  0.0   0.0  ...  0.400000  0.000003  0.000003
        Link      Norwich Converter    True  NaN   NaN  ...       NaN       NaN      NaN
                  Norway Converter     True  NaN   NaN  ...       NaN       NaN      NaN
                  Bremen Converter     True  NaN   NaN  ...       NaN       NaN      NaN
                  DC link              True  NaN   NaN  ...       NaN       NaN      NaN
        <BLANKLINE>
        [11 rows x 61 columns]

        See Also
        --------
        [pypsa.Network.passive_branches][]
        [pypsa.Network.controllable_branches][]

        """
        return pd.concat(
            (self.static(c) for c in self.branch_components),
            keys=self.branch_components,
            sort=True,
            names=["component", "name"],
        )

    def passive_branches(self) -> pd.DataFrame:
        """Get passive branches.

        Passive branches are Lines and Transformers.

        !!! note
            This method will return a merged copy of all passive branches of the network.
            Changes to the returned DataFrame will not be reflected in the network.

        Examples
        --------
        >>> n.passive_branches() # doctest: +ELLIPSIS
            active    b  b_pu  build_year  ...  v_nom         x      x_pu  x_pu_eff
        Line 0    True  0.0   0.0           0  ...  380.0  0.796878  0.000006  0.000006
             1    True  0.0   0.0           0  ...  380.0  0.391560  0.000003  0.000003
             2    True  0.0   0.0           0  ...  200.0  0.000000  0.000000  0.000000
             3    True  0.0   0.0           0  ...  200.0  0.000000  0.000000  0.000000
             4    True  0.0   0.0           0  ...  200.0  0.000000  0.000000  0.000000
             5    True  0.0   0.0           0  ...  380.0  0.238800  0.000002  0.000002
             6    True  0.0   0.0           0  ...  380.0  0.400000  0.000003  0.000003
        <BLANKLINE>
        [7 rows x 37 columns]

        """
        return pd.concat(
            (self.static(c) for c in self.passive_branch_components),
            keys=self.passive_branch_components,
            sort=True,
        )

    def controllable_branches(self) -> pd.DataFrame:
        """Get controllable branches.

        Controllable branches are Links.

        !!! note
            This method will return a merged copy of all controllable branches of the network.
            Changes to the returned DataFrame will not be reflected in the network.

        Examples
        --------
        >>> n.controllable_branches() # doctest: +ELLIPSIS
                                        active  build_year  ... type up_time_before
            Link                                   ...
        Link Norwich Converter    True           0  ...                   1
            Norway Converter     True           0  ...                   1
        ...

        See Also
        --------
        [pypsa.Network.branches][]
        [pypsa.Network.passive_branches][]

        """
        return pd.concat(
            (self.static(c) for c in self.controllable_branch_components),
            keys=self.controllable_branch_components,
            sort=True,
        )

    def determine_network_topology(
        self,
        investment_period: int | str | None = None,
        skip_isolated_buses: bool = False,
    ) -> Network:
        """Build sub_networks from topology.

        For the default case investment_period=None, it is not taken
        into account whether the branch components are active (based on
        build_year and lifetime). If the investment_period is specified,
        the network topology is determined on the basis of the active
        branches.
        """
        adjacency_matrix = self.adjacency_matrix(
            branch_components=self.passive_branch_components,
            investment_period=investment_period,
        )
        n_components, labels = csgraph.connected_components(
            adjacency_matrix, directed=False
        )

        # remove all old sub_networks
        for sub_network in self.sub_networks.index:
            obj = self.sub_networks.at[sub_network, "obj"]
            self.remove("SubNetwork", sub_network)
            del obj

        for i in np.arange(n_components):
            # index of first bus
            buses_i = (labels == i).nonzero()[0]

            if skip_isolated_buses and (len(buses_i) == 1):
                continue

            carrier = self.buses.carrier.iat[buses_i[0]]

            if carrier not in ["AC", "DC"] and len(buses_i) > 1:
                logger.warning(
                    "Warning, sub network %d is not electric but "
                    "contains multiple buses\nand branches. Passive "
                    "flows are not allowed for non-electric networks!",
                    i,
                )

            if (self.buses.carrier.iloc[buses_i] != carrier).any():
                logger.warning(
                    "Warning, sub network %d contains buses with "
                    "mixed carriers! Value counts:"
                    "\n%s",
                    i,
                    self.buses.carrier.iloc[buses_i].value_counts(),
                )

            self.add("SubNetwork", i, carrier=carrier)

        # add objects
        self.sub_networks["obj"] = [
            SubNetwork(self, name) for name in self.sub_networks.index
        ]

        self.buses.loc[:, "sub_network"] = labels.astype(str)

        for c in self.iterate_components(self.passive_branch_components):
            c.static["sub_network"] = c.static.bus0.map(self.buses["sub_network"])

            if investment_period is not None:
                active = self.get_active_assets(c.name, investment_period)
                # set non active assets to NaN
                c.static.loc[~active, "sub_network"] = np.nan

        for sub in self.sub_networks.obj:
            find_cycles(sub)
            sub.find_bus_controls()

        return self

    @deprecated_in_next_major(
        details="Use `n.components.<component>` instead.",
    )
    def component(self, c_name: str) -> Component:
        """Get a component from the network.

        !!! warning "Deprecated in v1.0"
            Use `n.components.<component>` or `n.components[component_name]` instead.

        Examples
        --------
        >>> n.component("Bus")
        'Bus' Components
        ----------------
        Attached to PyPSA Network 'AC-DC-Meshed'
        Components: 9

        """
        return self.components[c_name]

    @deprecated_in_next_major(details="Use `for component in n.components` instead.")
    def iterate_components(
        self, components: Collection[str] | None = None, skip_empty: bool = True
    ) -> Iterator[Component]:
        """Iterate over components.

        !!! warning "Deprecated in v1.0"
            Use `for component in n.components` instead.

        Examples
        --------
        >>> for component in n.iterate_components(): # doctest: +SKIP
        ...     print(component)
        ...     break
        'Bus' Components

        """
        if components is None:
            components = self.all_components

        return (
            self.component(c_name)
            for c_name in components
            if not (skip_empty and self.static(c_name).empty)
        )

    def __dir__(self) -> Iterable[str]:
        """Return a list of valid attributes and methods of the network.

        This method is used by dir() and help() to show available attributes.
        It filters out properties that would raise AttributeError when accessed,
        because they are not available in the network yet.

        Returns
        -------
        list[str]
            List of valid attribute and method names.

        """
        attrs = super().__dir__()

        # Filter out properties that would raise AttributeError
        if self._objective_constant is None:
            attrs = [attr for attr in attrs if attr != "objective_constant"]
        if self._objective is None:
            attrs = [attr for attr in attrs if attr != "objective"]
        if self._model is None:
            attrs = [attr for attr in attrs if attr != "model"]

        return attrs


class SubNetwork(NetworkGraphMixin, SubNetworkPowerFlowMixin):
    """SubNetwork for electric buses (AC or DC).

    SubNetworks are generated by `n.determine_network_topology()` for electric buses
    with passive flows or isolated non-electric buses.
    """

    # Type hints
    # ----------------

    buses_o: pd.Index
    pvpqs: pd.Index
    pqs: pd.Index
    pvs: pd.Index
    slack_bus: str
    B: spmatrix
    K: spmatrix
    C: spmatrix
    PTDF: spmatrix
    BODF: spmatrix

    list_name = "sub_networks"

    @deprecated_common_kwargs
    def __init__(self, n: Network, name: str) -> None:
        """Initialize a sub-network.

        Parameters
        ----------
        n : pypsa.Network
            The parent network of the sub-network.
        name : str
            The name of the sub-network.

        """
        self._n = ref(n)
        self.name = name

    @property
    @deprecated(
        deprecated_in="0.32", removed_in="1.0", details="Use the `n` property instead."
    )
    def network(self) -> Network:
        """Get the parent network of the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.n` instead.
        """
        return self._n()  # type: ignore

    @property
    def n(self) -> Network:
        """Get the parent network of the sub-network.

        Examples
        --------
        >>> sub_network.n # doctest: +ELLIPSIS
        PyPSA Network 'AC-DC-Meshed'
        ----------------------------
        Components:
         - Bus: 9
         ...

        """
        return self._n()  # type: ignore

    @property
    def components(self) -> ComponentsStore:
        """Get the components for the sub-network.

        Sub network components behave like Components in a basic pypsa.Network, but are
        a special class (SubNetworkComponents) to only return a view from the parent
        network.

        Examples
        --------
        Get single component:
        >>> sub_network.components.generators
        'Generator' SubNetworkComponents
        --------------------------------
        Attached to Sub-Network of PyPSA Network 'AC-DC-Meshed'
        Components: 6

        Getting a component is also possible via getitem:
        >>> sub_network.components['generators'] # doctest: +ELLIPSIS
        'Generator' SubNetworkComponents
        ...

        Or with the component name instead of list notation:
        >>> sub_network.components['Generator'] # doctest: +ELLIPSIS
        'Generator' SubNetworkComponents
        ...

        See Also
        --------
        [pypsa.Network.components][]

        """

        def filter_down(key: str, c: Components) -> Any:
            value = c[key]
            if key == "static":
                if c.name in {"Bus"} | self.n.passive_branch_components:
                    return value[value.sub_network == self.name]
                if c.name in self.n.one_port_components:
                    buses = self.buses_i()
                    return value[value.bus.isin(buses)]
                msg = f"Component {c.name} not supported for sub-networks"
                raise ValueError(msg)
            if key == "dynamic":
                dynamic = Dict()
                index = self.static(c.name).index
                for k, v in self.n.dynamic(c.name).items():
                    dynamic[k] = v[index.intersection(v.columns)]
                return dynamic
            return value

        return ComponentsStore(
            {
                key: SubNetworkComponents(value, filter_down)
                for key, value in self.n.components.items()
            }
        )

    @property
    def c(self) -> ComponentsStore:
        """Get the components for the sub-network.

        Alias for `sub_network.components`.

        See Also
        --------
        [pypsa.SubNetwork.components][]

        """
        return self.components

    @property
    def snapshots(self) -> pd.Index | pd.MultiIndex:
        """Get the snapshots for the sub-network.

        See Also
        --------
        [pypsa.Network.snapshots][]

        """
        return self.n.snapshots

    @property
    def snapshot_weightings(self) -> pd.DataFrame:
        """Get the snapshot weightings for the sub-network.

        See Also
        --------
        [pypsa.Network.snapshot_weightings][]

        """
        return self.n.snapshot_weightings

    @property
    def investment_periods(self) -> pd.Index:
        """Get the investment periods for the sub-network.

        See Also
        --------
        [pypsa.Network.investment_periods][]

        """
        return self.n.investment_periods

    @property
    def investment_period_weightings(self) -> pd.DataFrame:
        """Get the investment period weightings for the sub-network.

        See Also
        --------
        [pypsa.Network.investment_period_weightings][]

        """
        return self.n.investment_period_weightings

    def branches_i(self, active_only: bool = False) -> pd.MultiIndex:
        """Get the index of the branches in the sub-network.

        Parameters
        ----------
        active_only : bool, default False
            If True, only return the index of the active branches.

        Returns
        -------
        pd.MultiIndex
            The index of the branches in the sub-network.

        Examples
        --------
        >>> sub_network.branches_i()
        MultiIndex([('Line', '0'),
                    ('Line', '1'),
                    ('Line', '5')],
                    names=['type', 'name'])

        """
        types = []
        names = []
        for c in self.iterate_components(self.n.passive_branch_components):
            idx = c.static.query("active").index if active_only else c.static.index
            types += len(idx) * [c.name]
            names += list(idx)
        return pd.MultiIndex.from_arrays([types, names], names=("type", "name"))

    def branches(self) -> pd.DataFrame:
        """Get the branches in the sub-network.

        See Also
        --------
        [pypsa.Network.branches][]

        """
        branches = self.n.passive_branches()
        return branches[branches.sub_network == self.name]

    @deprecated_in_next_major(
        details="Use `sub_network.components.<c_name>` instead.",
    )
    def component(self, c_name: str) -> SubNetworkComponents:
        """Get a component from the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.<c_name>` instead.

        See Also
        --------
        [pypsa.Network.components][]

        """
        return self.components[c_name]

    @deprecated_in_next_major(
        details="Use `sub_network.components.<c_name>.static` instead.",
    )
    def df(self, c_name: str) -> pd.DataFrame:
        """Get a static component from the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.<c_name>.static` instead.

        See Also
        --------
        [pypsa.Network.static][]

        """
        return self.static(c_name)

    @deprecated_in_next_major(
        details="Use `sub_network.components.<c_name>.static` instead.",
    )
    def static(self, c_name: str) -> pd.DataFrame:
        """Get a static component from the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.<c_name>.static` instead.

        See Also
        --------
        [pypsa.Network.static][]

        """
        return self.components[c_name].static

    @deprecated_in_next_major(
        details="Use `sub_network.components.<c_name>.dynamic` instead.",
    )
    def pnl(self, c_name: str) -> Dict:
        """Get a dynamic component from the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.<c_name>.dynamic` instead.

        See Also
        --------
        [pypsa.Network.dynamic][]

        """
        return self.dynamic(c_name)

    @deprecated_in_next_major(
        details="Use `sub_network.components.<c_name>.dynamic` instead.",
    )
    def dynamic(self, c_name: str) -> Dict:
        """Get a dynamic component from the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.<c_name>.dynamic` instead.

        See Also
        --------
        [pypsa.Network.dynamic][]

        """
        return self.components[c_name].dynamic

    @deprecated_in_next_major(
        details="Use `sub_network.components.buses.static.index` instead.",
    )
    def buses_i(self) -> pd.Index:
        """Get the index of the buses in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.buses.static.index` instead.

        See Also
        --------
        [pypsa.Network.buses][]

        """
        return self.components.buses.static.index

    @deprecated_in_next_major(
        details="Use `sub_network.components.lines.static.index` instead.",
    )
    def lines_i(self) -> pd.Index:
        """Get the index of the lines in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.lines.static.index` instead.

        See Also
        --------
        [pypsa.Network.lines][]

        """
        return self.components.lines.static.index

    @deprecated_in_next_major(
        details="Use `sub_network.components.transformers.static.index` instead.",
    )
    def transformers_i(self) -> pd.Index:
        """Get the index of the transformers in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.transformers.static.index` instead.

        See Also
        --------
        [pypsa.Network.transformers][]

        """
        return self.components.transformers.static.index

    @deprecated_in_next_major(
        details="Use `sub_network.components.generators.static.index` instead.",
    )
    def generators_i(self) -> pd.Index:
        """Get the index of the generators in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.generators.static.index` instead.

        See Also
        --------
        [pypsa.Network.generators][]

        """
        return self.components.generators.static.index

    @deprecated_in_next_major(
        details="Use `sub_network.components.loads.static.index` instead.",
    )
    def loads_i(self) -> pd.Index:
        """Get the index of the loads in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.loads.static.index` instead.

        See Also
        --------
        [pypsa.Network.loads][]

        """
        return self.components.loads.static.index

    @deprecated_in_next_major(
        details="Use `sub_network.components.shunt_impedances.static.index` instead.",
    )
    def shunt_impedances_i(self) -> pd.Index:
        """Get the index of the shunt impedances in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.shunt_impedances.static.index` instead.

        See Also
        --------
        [pypsa.Network.shunt_impedances][]

        """
        return self.components.shunt_impedances.static.index

    @deprecated_in_next_major(
        details="Use `sub_network.components.storage_units.static.index` instead.",
    )
    def storage_units_i(self) -> pd.Index:
        """Get the index of the storage units in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.storage_units.static.index` instead.

        See Also
        --------
        [pypsa.Network.storage_units][]

        """
        return self.components.storage_units.static.index

    @deprecated_in_next_major(
        details="Use `sub_network.components.stores.index.static` instead.",
    )
    def stores_i(self) -> pd.Index:
        """Get the index of the stores in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.stores.static.index` instead.

        See Also
        --------
        [pypsa.Network.stores][]

        """
        return self.components.stores.static.index

    @deprecated_in_next_major(
        details="Use `sub_network.components.buses.static` instead.",
    )
    def buses(self) -> pd.DataFrame:
        """Get the buses in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.buses.static` instead.

        See Also
        --------
        [pypsa.Network.buses][]

        """
        return self.components.buses.static

    @deprecated_in_next_major(
        details="Use `sub_network.components.generators.static` instead.",
    )
    def generators(self) -> pd.DataFrame:
        """Get the generators in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.generators.static` instead.

        See Also
        --------
        [pypsa.Network.generators][]

        """
        return self.components.generators.static

    @deprecated_in_next_major(
        details="Use `sub_network.components.loads.static` instead.",
    )
    def loads(self) -> pd.DataFrame:
        """Get the loads in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.loads.static` instead.

        See Also
        --------
        [pypsa.Network.loads][]

        """
        return self.components.loads.static

    @deprecated_in_next_major(
        details="Use `sub_network.components.shunt_impedances.static` instead.",
    )
    def shunt_impedances(self) -> pd.DataFrame:
        """Get the shunt impedances in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.shunt_impedances.static` instead.

        See Also
        --------
        [pypsa.Network.shunt_impedances][]

        """
        return self.components.shunt_impedances.static

    @deprecated_in_next_major(
        details="Use `sub_network.components.storage_units.static` instead.",
    )
    def storage_units(self) -> pd.DataFrame:
        """Get the storage units in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.storage_units.static` instead.

        See Also
        --------
        [pypsa.Network.storage_units][]

        """
        return self.components.storage_units.static

    @deprecated_in_next_major(
        details="Use `!!! deprecated.components.stores.static` instead.",
    )
    def stores(self) -> pd.DataFrame:
        """Get the stores in the sub-network.

        !!! warning "Deprecated in v1.0"
            Use `sub_network.components.stores.static` instead.
        """
        return self.components.stores.static

    @deprecated_in_next_major(details="Use `self.components` instead.")
    # Deprecate: Use `self.iterate_components` instead
    def iterate_components(
        self, components: Collection[str] | None = None, skip_empty: bool = True
    ) -> Iterator[SubNetworkComponents]:
        """Iterate over components of the sub-network.

        Parameters
        ----------
        components : list-like, optional
            List of components ('Generator', 'Line', etc.) to iterate over,
            by default None
        skip_empty : bool, optional
            Whether to skip a components with no assigned assets,
            by default True

        See Also
        --------
        [pypsa.Network.iterate_components][]

        Yields
        ------
        Component
            Container for component data. See Component class for details.

        """
        if components is None:
            components = self.n.all_components

        return (
            self.components[c_name]
            for c_name in components
            if not (skip_empty and self.static(c_name).empty)
        )
