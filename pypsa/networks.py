"""Power system components."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any
from weakref import ref

from pypsa.common import deprecated_in_next_major, equals
from pypsa.components.components import Components
from pypsa.constants import DEFAULT_EPSG, DEFAULT_TIMESTAMP
from pypsa.statistics.abstract import AbstractStatisticsAccessor

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path

import functools

import linopy
import numpy as np
import pandas as pd
import pyproj
import validators
from pyproj import CRS, Transformer
from scipy.sparse import csgraph

from pypsa.clustering import ClusteringAccessor
from pypsa.common import (
    as_index,
)
from pypsa.components.components import SubNetworkComponents
from pypsa.components.store import ComponentsStore
from pypsa.consistency import NetworkConsistencyMixin
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
from pypsa.plot.maps import explore
from pypsa.statistics.expressions import StatisticsAccessor
from pypsa.version import __version_semver__

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Sequence

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
    iteration: int

    # ----------------
    # Dunder methods
    # ----------------

    def __init__(
        self,
        import_name: str | Path = "",
        name: str = "Unnamed Network",
        ignore_standard_types: bool = False,
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
        kwargs : Any
            Any remaining attributes to set

        Examples
        --------
        >>> nw1 = pypsa.Network("network.nc") # doctest: +SKIP
        >>> nw2 = pypsa.Network("/my/folder") # doctest: +SKIP
        >>> nw3 = pypsa.Network("https://github.com/PyPSA/PyPSA/raw/master/examples/scigrid-de/scigrid-with-load-gen-trafos.nc") # doctest: +SKIP
        >>> nw4 = pypsa.Network("s3://my-bucket/my-network.nc") # doctest: +SKIP

        """
        # Initialise root logger and set its level, if this has not been done before
        logging.basicConfig(level=logging.INFO)

        # Store PyPSA version
        self._pypsa_version: str = __version_semver__

        # Set attributes
        self._name = name
        self._meta: dict = {}
        self._crs: CRS = CRS.from_epsg(DEFAULT_EPSG)

        # Dimensions
        # Snapshots
        cols = ["objective", "stores", "generators"]
        index = pd.Index([DEFAULT_TIMESTAMP], name="snapshot")
        self._snapshots_data = pd.DataFrame(1, index=index, columns=cols)

        # Investment periods coordinate
        cols = ["objective", "years"]
        self._investment_periods_data = pd.DataFrame(index=self.periods, columns=cols)

        # Scenarios
        cols = ["weight"]
        index = pd.Index([], name="scenario")
        self._scenarios_data: pd.DataFrame = pd.DataFrame([], index=index, columns=cols)

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
        prefix = "Stochastic PyPSA Network" if self.has_scenarios else "PyPSA Network"
        return f"{prefix} '{self.name}'"

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
        content += "\n"

        if self.has_scenarios:
            content += f"Scenarios: {len(self.scenarios)}"

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
            linopy.Model,
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
        * Generator-p_nom (name)
        * Line-s_nom (name)
        * Link-p_nom (name)
        * Generator-p (snapshot, name)
        * Line-s (snapshot, name)
        * Link-p (snapshot, name)
        * objective_constant
        <BLANKLINE>
        Constraints:
        ------------
        * Generator-ext-p_nom-lower (name)
        * Line-ext-s_nom-lower (name)
        * Link-ext-p_nom-lower (name)
        * Generator-ext-p-lower (snapshot, name)
        * Generator-ext-p-upper (snapshot, name)
        * Line-ext-s-lower (snapshot, name)
        * Line-ext-s-upper (snapshot, name)
        * Link-ext-p-lower (snapshot, name)
        * Link-ext-p-upper (snapshot, name)
        * Bus-nodal_balance (name, snapshot)
        * Kirchhoff-Voltage-Law (snapshot, cycle)
        * GlobalConstraint-co2_limit
        <BLANKLINE>
        Status:
        -------
        ok

        """
        if self._model is None:
            logger.warning(
                "The network has not been optimized yet and no model is stored."
            )
        return self._model

    @model.deleter
    def model(self) -> None:
        """Delete the model object."""
        self._model = None

    @property
    def objective(self) -> float | None:
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
        >>> n.objective # doctest: +SKIP
        -47274166...

        >>> n.objective + n.objective_constant # doctest: +SKIP
        <xarray.DataArray ()> Size: 8B
        array(18441021...)

        """
        if self._objective is None:
            logger.warning(
                "The network has not been optimized yet and no objective value is stored."
            )
        return self._objective

    @property
    def objective_constant(self) -> float | None:
        """Objective constant of the network.

        The property yields the fixed part of the objective function. It is set after
        optimizing the network.

        When optimizing for system costs, the total system costs are the sum of the
        [pypsa.Network.objective][] and the [pypsa.Network.objective_constant][]. When
        loading a network from file and the model object is not loaded, the objective
        constant is still available, as it is stored in the network object.

        Examples
        --------
        >>> n.objective_constant # doctest: +SKIP
        <xarray.DataArray ()> Size: 8B
        array(65715187...)

        >>> n.objective + n.objective_constant # doctest: +SKIP
        <xarray.DataArray ()> Size: 8B
        array(18441021...)

        """
        if self._objective_constant is None:
            logger.warning(
                "The network has not been optimized yet and no objective constant is stored."
            )
        return self._objective_constant

    @property
    def is_solved(self) -> bool:
        """Check if the network has been solved.

        A solved network has an [objective][pypsa.Network.objective][] value assigned. A
        [model][pypsa.Network.model][] does not necessarily need to be stored in the
        network.

        Returns
        -------
        bool
            True if the network has been solved, False otherwise.

        Examples
        --------
        >>> n.is_solved
        True

        """
        return self._objective is not None

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
        self.c.shapes.static.to_crs(new, inplace=True)
        self._crs = self.c.shapes.static.crs
        transformer = Transformer.from_crs(current, self.crs)
        self.c.buses.static["x"], self.c.buses.static["y"] = transformer.transform(
            self.c.buses.static["x"], self.c.buses.static["y"]
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

    @functools.wraps(explore)
    def explore(self, *args: Any, **kwargs: Any) -> Any:
        """Interactive map plot method."""
        return explore(self, *args, **kwargs)

    def copy(
        self,
        snapshots: Sequence | None = None,
        investment_periods: Sequence | None = None,
        ignore_standard_types: bool = False,
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
        if (
            self.is_solved
            and hasattr(self._model, "solver_model")
            and self._model is not None
            and self._model.solver_model is not None
        ):
            msg = "Copying a solved network with an attached solver model is not supported."
            msg += " Please delete the model first using `n.model.solver_model = None`."
            raise ValueError(msg)

        # Use copy.deepcopy if no arguments are passed
        args = [snapshots, investment_periods, ignore_standard_types]
        if all(arg is None or arg is False for arg in args):
            copied_network = copy.deepcopy(self)
            return copied_network

        if self.has_scenarios:
            msg = (
                "Copying a stochastic network with a selection is currently not "
                "supported. Use `n.copy()` to copy the entire network."
            )
            raise NotImplementedError(msg)

        # Convert to pandas.Index
        snapshots_ = as_index(self, snapshots, "snapshots")
        investment_periods_ = as_index(self, investment_periods, "investment_periods")

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
            for component in self.components:
                dynamic = getattr(n.c, component.list_name).dynamic
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
            pd.DataFrame(self.c.buses.static.loc[key]).assign(sub_network="").index,
            **pd.DataFrame(self.c.buses.static.loc[key]).assign(sub_network=""),
        )
        buses_i = n.c.buses.static.index

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
    def _empty_components(self) -> list:
        """Get a list of all components that are empty.

        Returns
        -------
        list
            List of empty components.

        """
        return [c.name for c in self.iterate_components() if c.empty]

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
        comps = list(set(self.branch_components) - set(self._empty_components()))
        names = (
            ["component", "scenario", "name"]
            if self.has_scenarios
            else ["component", "name"]
        )
        return pd.concat(
            (self.static(c) for c in comps),
            keys=comps,
            sort=True,
            names=names,
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
                    active    b  b_pu  ...         x      x_pu  x_pu_eff
        component                     ...
        0            True  0.0   0.0  ...  0.796878  0.000006  0.000006
        1            True  0.0   0.0  ...  0.391560  0.000003  0.000003
        2            True  0.0   0.0  ...  0.000000  0.000000  0.000000
        3            True  0.0   0.0  ...  0.000000  0.000000  0.000000
        4            True  0.0   0.0  ...  0.000000  0.000000  0.000000
        5            True  0.0   0.0  ...  0.238800  0.000002  0.000002
        6            True  0.0   0.0  ...  0.400000  0.000003  0.000003
        <BLANKLINE>
        [7 rows x 37 columns]

        """
        comps = list(
            set(self.passive_branch_components) - set(self._empty_components())
        )
        names = (
            ["component", "scenario", "name"]
            if self.has_scenarios
            else ["component", "name"]
        )
        return pd.concat(
            (self.static(c) for c in comps),
            keys=comps,
            sort=True,
            names=names,
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
                        active    b  b_pu  ...         x      x_pu  x_pu_eff
        component name                     ...
        Line      0       True  0.0   0.0  ...  0.796878  0.000006  0.000006
                1       True  0.0   0.0  ...  0.391560  0.000003  0.000003
                2       True  0.0   0.0  ...  0.000000  0.000000  0.000000
                3       True  0.0   0.0  ...  0.000000  0.000000  0.000000
                4       True  0.0   0.0  ...  0.000000  0.000000  0.000000
                5       True  0.0   0.0  ...  0.238800  0.000002  0.000002
                6       True  0.0   0.0  ...  0.400000  0.000003  0.000003
        <BLANKLINE>
        [7 rows x 37 columns]

        See Also
        --------
        [pypsa.Network.branches][]
        [pypsa.Network.passive_branches][]

        """
        comps = list(
            set(self.passive_branch_components) - set(self._empty_components())
        )
        names = (
            ["component", "scenario", "name"]
            if self.has_scenarios
            else ["component", "name"]
        )
        return pd.concat(
            (self.static(c) for c in comps),
            keys=comps,
            sort=True,
            names=names,
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
            return_dataframe=True,
        )
        n_components, labels = csgraph.connected_components(
            adjacency_matrix.values, directed=False
        )
        labels = pd.Series(labels, adjacency_matrix.index, name="sub_network")
        sub_network_map = labels.astype(str)

        # remove all old sub_networks
        if not self.c.sub_networks.static.empty:
            # Delete sub-network objects first
            for sub_network in self.c.sub_networks.static.index:
                obj = self.c.sub_networks.static.at[sub_network, "obj"]
                del obj

            # Clear the sub_networks DataFrame completely
            # This handles both regular and stochastic cases
            self.c.sub_networks.static.drop(
                self.c.sub_networks.static.index, inplace=True
            )
            for dynamic in self.c.sub_networks.dynamic.values():
                dynamic.drop(dynamic.columns, inplace=True)

        if self.has_scenarios:
            bus_carrier = self.c.buses.static.carrier.xs(
                self.scenarios[0], level="scenario"
            )
        else:
            bus_carrier = self.c.buses.static.carrier

        for i in np.arange(n_components):
            # index of first bus
            buses = labels.index[labels == i]

            if skip_isolated_buses and (len(buses) == 1):
                continue

            carrier = bus_carrier.at[buses[0]]

            if carrier not in ["AC", "DC"] and len(buses) > 1:
                logger.warning(
                    "Warning, sub network %d is not electric but "
                    "contains multiple buses\nand branches. Passive "
                    "flows are not allowed for non-electric networks!",
                    i,
                )

            if (bus_carrier.loc[buses] != carrier).any():
                logger.warning(
                    "Warning, sub network %d contains buses with "
                    "mixed carriers! Value counts:"
                    "\n%s",
                    i,
                    bus_carrier.loc[buses].value_counts(),
                )

            self.add("SubNetwork", str(i), carrier=carrier)

        # add objects
        self.c.sub_networks.static["obj"] = [
            SubNetwork(self, name) for name in self.c.sub_networks.static.index
        ]

        self.c.buses.static = self.c.buses.static.drop(columns="sub_network").join(
            sub_network_map, "name"
        )[self.c.buses.static.columns]

        for c in self.iterate_components(self.passive_branch_components):
            c.static["sub_network"] = c.static.bus0.map(sub_network_map)

            if investment_period is not None:
                active = self.get_active_assets(c.name, investment_period)
                # set non active assets to NaN
                c.static.loc[~active, "sub_network"] = np.nan

        for sub in self.c.sub_networks.static.obj:
            find_cycles(sub)
            sub.find_bus_controls()

        return self

    def cycle_matrix(
        self, investment_period: str | int | None = None, apply_weights: bool = False
    ) -> pd.DataFrame:
        """Get the cycles in the network and represent them as a DataFrame.

        This function identifies a cycle basis of the network topology and
        returns a DataFrame representation of the cycle matrix. The cycles
        matrix is a sparse matrix with branches as rows and independent
        cycles as columns. An entry of +1 indicates the branch is traversed
        in the direction from bus0 to bus1 in that cycle, -1 indicates
        the opposite direction, and 0 indicates the branch is not part
        of the cycle.

        Parameters
        ----------
        investment_period : str or int, optional
            Investment period to use when determining network topology.
            If not given, all branches are considered regardless of
            build_year and lifetime.
        apply_weights : bool, default False
            Whether to apply weights (e.g., reactance for AC lines,
            resistance for DC lines) to the cycles.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with branches as rows (MultiIndex of (component, name))
            and cycles as columns. Each column represents an independent cycle
            in the network.

        """
        self.determine_network_topology(
            investment_period=investment_period, skip_isolated_buses=True
        )
        self.calculate_dependent_values()

        cycles = []

        # Process each sub-network to find its cycles
        for sub_network in self.c.sub_networks.static.obj:
            branches = sub_network.branches()

            if self.has_scenarios:
                branches = branches.xs(self.scenarios[0], level="scenario")

            branches_i = branches.index
            branches_i.names = ["type", "component"]
            if not hasattr(sub_network, "C") or not sub_network.C.size:
                continue

            # Convert sparse matrix to DataFrame
            C = pd.DataFrame(sub_network.C.todense(), index=branches_i)
            cycles.append(C)

        if not cycles:
            return pd.DataFrame()

        # Combine all cycles and fill missing values with 0
        cycles_df = pd.concat(cycles, axis=1, ignore_index=True).fillna(0)

        # Get all branch components
        existing_branch_components = cycles_df.index.unique("type")
        branches = self.branches()

        if self.has_scenarios:
            branches = branches.xs(self.scenarios[0], level="scenario")

        branches.index.names = ["type", "name"]
        branches_i = branches.loc[existing_branch_components].index

        if apply_weights:
            is_ac = branches.sub_network.map(self.c.sub_networks.static.carrier) == "AC"
            weights = branches.x_pu_eff.where(is_ac, branches.r_pu_eff)
            weights = weights[cycles_df.index]
            cycles_df = cycles_df.multiply(weights, axis=0)

        # Reindex to include all branches (even those not in cycles)
        return cycles_df.reindex(branches_i, fill_value=0).rename_axis(columns="cycle")

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

    # TODO assign __str__ and __repr__
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
                    buses = self.buses_i().unique("name")
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

    @property
    def scenarios(self) -> pd.Series:
        """Get the scenarios for the network.

        Returns
        -------
        pd.Series
            The scenarios for the network.

        """
        return self.n.scenarios

    @property
    def has_scenarios(self) -> bool:
        """Check if the network has scenarios.

        Returns
        -------
        bool
            True if the network has scenarios, False otherwise.

        """
        return self.n.has_scenarios

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
            static = c.static
            idx = static.query("active").index if active_only else static.index
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
