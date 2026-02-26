# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Power system components."""

from __future__ import annotations

import copy
import logging
import warnings
from typing import TYPE_CHECKING, Any
from weakref import ref

from deprecation import deprecated

from pypsa.common import equals
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
from pypsa.version import __version_base__

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
    """Network container for all [pypsa.Components][] and helds most of PyPSA's functionality.

    <!-- md:badge-version v0.1.0 --> | <!-- md:guide design.md -->
    """

    # Optimization
    _multi_invest: int
    _linearized_uc: int
    _committable_big_m: float | None
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
        import_name : string or Path
            Path to netCDF file, HDF5 .h5 store or folder of CSV files from which to
            import network data. The string could be a URL. If cloudpathlib is installed,
            the string could be a object storage URI with an `s3`, `gs` or `az` URI scheme.
        name : string
            Network name.
        ignore_standard_types : boolean
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
        self._pypsa_version: str = __version_base__

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

        # Risk preference
        self._risk_preference: dict[str, float] | None = None

        self._model: linopy.Model | None = None
        self._objective: float | None = None
        self._objective_constant: float | None = None
        self._multi_invest: int = 0
        self._committable_big_m: float | None = None

        # Initialize accessors
        self.optimize: OptimizationAccessor = OptimizationAccessor(self)
        """
        Network [optimization functionality][pypsa.optimization.OptimizationAccessor] accessor.
        """
        self.cluster: ClusteringAccessor = ClusteringAccessor(self)
        """
        Network [clustering functionality][pypsa.clustering.ClusteringAccessor] accessor.
        """
        self.statistics: StatisticsAccessor = StatisticsAccessor(self)
        """
        Network [statistics functionality][pypsa.statistics.StatisticsAccessor] accessor.
        """
        self.plot: PlotAccessor = PlotAccessor(self)
        """
        Network [plotting functionality][pypsa.plot.PlotAccessor] accessor.
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
        """Get a string representation of the network.

        <!-- md:badge-version v0.33.0 -->

        Examples
        --------
        >>> str(n)
        "PyPSA Network 'AC-DC-Meshed'"
        >>> str(n_stochastic)
        "Stochastic PyPSA Network 'Stochastic-Network'"

        """
        prefix = "Stochastic PyPSA Network" if self.has_scenarios else "PyPSA Network"
        return f"{prefix} '{self.name}'"

    def __repr__(self) -> str:
        """Get representation of the network.

        <!-- md:badge-version v0.3.0 -->

        Examples
        --------
        >>> n  # doctest: +ELLIPSIS
        PyPSA Network 'AC-DC-Meshed'
        ----------------------------
        Components:
         - Bus: 9
         - Carrier: ...
         - Generator: ...
         - GlobalConstraint: ...
         - Line: ...
         - Link: ...
         - Load: ...
         - SubNetwork: ...
        Snapshots: 10
        <BLANKLINE>

        """
        # TODO make this actually for the REPL
        header = f"{self}\n" + "-" * len(str(self))  # + "\n"
        comps = {
            c.name: f" - {c.name}: {len(c.static)}"
            for c in self.components
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

        <!-- md:badge-version v0.28.0 -->

        Parameters
        ----------
        other : Network
            Network to merge into this one.

        See Also
        --------
        [pypsa.Network.merge][]

        Examples
        --------
        >>> n1 = pypsa.Network()
        >>> n2 = pypsa.Network()
        >>> n1.add("Bus", "bus1")
        >>> n2.add("Bus", "bus2")
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
        [pypsa.Network.equals][]

        """
        return self.equals(other)

    def __getitem__(self, key: str) -> Network:
        """Return a shallow slice of the Network object.

        A network can be sliced in three different ways:
        1. If the key is a scenario name and the network has scenarios, the corresponding
           scenario is returned as a new network object.
           See [pypsa.Network.get_scenario][].
        2. If the network is a collection and the key is a name of one of the networks
           in the collection, that network is returned.
        3. If the key is a bus name or a boolean indexer for buses, a sliced copy of
           the network is returned, containing only the selected buses and all
           connected components. See [pypsa.Network.slice_network][].

        A warning will be issued if the key matches multiple of the above
        entities. In that case, the first match is returned. But it is recommended to
        use the explicit methods (e.g. get_scenario(), get_network()) or use unique
        scenario, collection and bus names to avoid ambiguity.

        Parameters
        ----------
        key : str or boolean mask
            The key or boolean mask to select a scenario, a network from a collection
            or slice the network based on buses.

        Returns
        -------
        n : pypsa.Network

        Examples
        --------
        Select single scenario from a stochastic network:

        >>> n_stochastic
        Stochastic PyPSA Network 'Stochastic-Network'
        ---------------------------------------------
        Components:
         - Bus: 3
         - Carrier: 18
         - Generator: 12
         - Load: 3
        Snapshots: 2920
        Scenarios: 3
        >>> n_stochastic["high"]
        PyPSA Network 'Stochastic-Network - Scenario 'high''
        ----------------------------------------------------
        Components:
         - Bus: 1
         - Carrier: 6
         - Generator: 4
         - Load: 1
        Snapshots: 2920

        Select single collection from a network collection:

        >>> nc
        NetworkCollection
        -----------------
        Networks: 2
        Index name: 'network'
        Entries: ['AC-DC-Meshed', 'AC-DC-Meshed-Shuffled-Load']

        >>> nc["AC-DC-Meshed"]
        PyPSA Network 'AC-DC-Meshed'
        ----------------------------
        Components:
         - Bus: 9
         - Carrier: 6
         - Generator: 6
         - GlobalConstraint: 1
         - Line: 7
         - Link: 4
         - Load: 6
         - SubNetwork: 3
        Snapshots: 10
        <BLANKLINE>

        Select a network slice based on buses:

        >>> n
        PyPSA Network 'AC-DC-Meshed'
        ----------------------------
        Components:
         - Bus: 9
         - Carrier: 6
         - Generator: 6
         - GlobalConstraint: 1
         - Line: 7
         - Link: 4
         - Load: 6
         - SubNetwork: 3
        Snapshots: 10
        >>> n["London"]
        PyPSA Network 'AC-DC-Meshed'
        ----------------------------
        Components:
        - Bus: 1
        - Carrier: 6
        - GlobalConstraint: 1
        - Load: 1
        Snapshots: 10

        or use the pandas `.loc` method to select multiple buses:

        >>> n[n.buses.carrier=='AC']
        PyPSA Network 'AC-DC-Meshed'
        ----------------------------
        Components:
        - Bus: 6
        - Carrier: 6
        - Generator: 6
        - GlobalConstraint: 1
        - Line: 4
        - Link: 1
        - Load: 6
        Snapshots: 10

        """
        # For a scalar key, selection is done either on scenarios, collection items or
        # a network slice
        if np.isscalar(key):
            results = []

            # Check scenarios
            if key in self.scenarios:
                results.append(("scenario", self.get_scenario(key)))

            # Check collection
            if self.is_collection and key in self.networks.index:  # type: ignore[attr-defined]
                results.append(("collection", self.networks[key]))  # type: ignore[attr-defined]

            # Check network slice
            if key in self.c.buses.names:
                results.append(("network", self.slice_network([key])))

            # Handle results
            if len(results) > 1:
                types = [r[0] for r in results]
                logger.warning(
                    "Key '%s' matches multiple entities: %s. Returning the "
                    "first match (%s). It is recommended to use explicit "
                    "methods, e.g. get_scenario() or get_network() to avoid "
                    "ambiguity or to use unique scenario, collection and bus names.",
                    key,
                    types,
                    types[0],
                )

            if results:
                return results[0][1]
            else:
                msg = f"Key '{key}' not found in scenarios, collection, or buses."
                raise KeyError(msg)
        else:
            # Check for deprecated tuple usage (buses, snapshots)
            if isinstance(key, tuple) and len(key) == 2:
                warnings.warn(
                    "Slicing by (buses, snapshots) tuples in __getitem__ is no longer supported. "
                    "Use the slice_network() method instead: "
                    "n.slice_network(buses=buses, snapshots=snapshots)",
                    DeprecationWarning,
                    stacklevel=2,
                )
                msg = "Tuple slicing is deprecated. Use slice_network(buses=..., snapshots=...) instead."
                raise NotImplementedError(msg)

            return self.slice_network(key)

    @property
    def stats(self) -> StatisticsAccessor:
        """Network [statistics functionality][pypsa.statistics.StatisticsAccessor] accessor (alias for [pypsa.Network.statistics][])."""
        return self.statistics

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
        >>> n2.add("Bus", "bus2")
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

        <!-- md:badge-version v0.1.0 -->

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

        <!-- md:badge-version v0.10.0 -->

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

        <!-- md:badge-version v0.20.0 -->

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

        <!-- md:badge-version v0.21.0 -->

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

        <!-- md:badge-version v0.21.0 -->

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

        <!-- md:badge-version v0.21.0 -->

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

        <!-- md:badge-version v0.35.0 -->

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

        <!-- md:badge-version v0.26.0 -->

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
        [pypsa.Network.srid][], [pypsa.Network.shapes][]

        """
        self.c.shapes.static.set_crs(new)
        self._crs = self.c.shapes.static.crs

    def to_crs(self, new: int | str | pyproj.CRS) -> None:
        """Convert the network's geometries and bus coordinates to a new crs.

        <!-- md:badge-version v0.26.0 -->

        See Also
        --------
        [pypsa.Network.crs][], [pypsa.Network.srid][], [pypsa.Network.shapes][]

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

        <!-- md:badge-version v0.26.0 -->

        Examples
        --------
        >>> n.srid
        4326

        See Also
        --------
        [pypsa.Network.crs][], [pypsa.Network.shapes][]

        """
        return self.crs.to_epsg()

    @srid.setter
    def srid(self, new: str | int) -> None:
        """Set the spatial reference system identifier of the network's geometries.

        See Also
        --------
        [pypsa.Network.crs][], [pypsa.Network.shapes][]

        """
        self.crs = pyproj.CRS.from_epsg(new)

    @functools.wraps(explore)
    def explore(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for [pypsa.Network.plot.explore][pypsa.plot.PlotAccessor.explore]."""
        return self.plot.explore(*args, **kwargs)

    def copy(
        self,
        snapshots: Sequence | None = None,
        investment_periods: Sequence | None = None,
        ignore_standard_types: bool = False,
    ) -> Network:
        """Return a deep copy of Network object.

        <!-- md:badge-version v0.4.0 -->

        If only default arguments are passed, the copy will be created via
        `copy.deepcopy` and will contain all components and time-varying data.
        For most networks this is the fastest way. However, if the network is very
        large, it might be better to filter snapshots and investment periods to reduce
        the size of the copy. In this case `copy.deepcopy` is not used and only
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
        for component in self.components[["Bus", "Carrier"] + other_comps]:
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
            "_committable_big_m",
            "_objective",
            "_objective_constant",
            "now",
        ]:
            if hasattr(self, attr):
                setattr(n, attr, getattr(self, attr))

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
        return [c.name for c in self.components if c.empty]

    def branches(self) -> pd.DataFrame:
        """Get branches.

        <!-- md:badge-version v0.3.0 -->

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
        [11 rows x 65 columns]

        See Also
        --------
        [pypsa.Network.passive_branches][],
        [pypsa.Network.controllable_branches][]

        """
        comps = list(set(self.branch_components) - set(self._empty_components()))
        names = (
            ["component", "scenario", "name"]
            if self.has_scenarios
            else ["component", "name"]
        )
        return pd.concat(
            (self.c[c].static for c in comps),
            keys=comps,
            sort=True,
            names=names,
        )

    def passive_branches(self) -> pd.DataFrame:
        """Get passive branches.

        <!-- md:badge-version v0.3.0 -->

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
        [7 rows x 41 columns]

        """
        comps = sorted(
            set(self.passive_branch_components) - set(self._empty_components())
        )
        names = (
            ["component", "scenario", "name"]
            if self.has_scenarios
            else ["component", "name"]
        )
        return pd.concat(
            (self.c[c].static for c in comps),
            keys=comps,
            sort=True,
            names=names,
        )

    def controllable_branches(self) -> pd.DataFrame:
        """Get controllable branches.

        <!-- md:badge-version v0.3.0 -->

        Controllable branches are Links.

        !!! note

            This method will return a merged copy of all controllable branches of the network.
            Changes to the returned DataFrame will not be reflected in the network.

        Examples
        --------
        >>> n.controllable_branches() # doctest: +ELLIPSIS
                active  build_year  ... type up_time_before
        component name                                   ...
        Link      Norwich Converter    True           0  ...                   1
                  Norway Converter     True           0  ...                   1
                  Bremen Converter     True           0  ...                   1
                  DC link              True           0  ...                   1
        <BLANKLINE>
        [4 rows x 41 columns]

        See Also
        --------
        [pypsa.Network.branches][],
        [pypsa.Network.passive_branches][]

        """
        comps = list(
            set(self.controllable_branch_components) - set(self._empty_components())
        )
        names = (
            ["component", "scenario", "name"]
            if self.has_scenarios
            else ["component", "name"]
        )
        return pd.concat(
            (self.c[c].static for c in comps),
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

        <!-- md:badge-version v0.3.0 -->

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

        for c in self.components:
            if c.name not in self.passive_branch_components:
                continue
            c.static["sub_network"] = c.static.bus0.map(sub_network_map)

            if investment_period is not None:
                active = c.get_active_assets(investment_period)
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

        <!-- md:badge-version v1.0.0 -->

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

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `n.components.<component>` instead.",
    )
    def component(self, c_name: str) -> Component:
        """Get a component from the network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `n.components.<component>` or `n.components[component_name]` instead.

        """
        return self.components[c_name]

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `for component in n.components` instead.",
    )
    def iterate_components(
        self, components: Collection[str] | None = None, skip_empty: bool = True
    ) -> Iterator[Component]:
        """Iterate over components.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

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
            self.c[c_name]
            for c_name in components
            if not (skip_empty and self.c[c_name].static.empty)
        )


class SubNetwork(NetworkGraphMixin, SubNetworkPowerFlowMixin):
    """SubNetwork for electric buses (AC or DC).

    <!-- md:badge-version v0.3.0 -->

    SubNetworks are generated by [pypsa.Network.determine_network_topology][] for
    electric buses with passive flows or isolated non-electric buses and stored in
    the [`n.components.sub_networks`][pypsa.components.SubNetworks] component.`
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
                    buses = self.c.buses.static.index.unique("name")
                    return value[value.bus.isin(buses)]
                msg = f"Component {c.name} not supported for sub-networks"
                raise ValueError(msg)
            if key == "dynamic":
                dynamic = Dict()
                index = c.static.index
                for k, v in c.dynamic.items():
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
        for c in self.components[sorted(self.n.passive_branch_components)]:
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

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.<c_name>` instead.",
    )
    def component(self, c_name: str) -> SubNetworkComponents:
        """Get a component from the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.<c_name>` instead.

        See Also
        --------
        [pypsa.Network.components][]

        """
        return self.components[c_name]

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.<c_name>.static` instead.",
    )
    def df(self, c_name: str) -> pd.DataFrame:
        """Get a static component from the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version-->"

            Use `sub_network.components.<c_name>.static` instead.

        See Also
        --------
        [pypsa.Network.static][]

        """
        return self.c[c_name].static

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.<c_name>.static` instead.",
    )
    def static(self, c_name: str) -> pd.DataFrame:
        """Get a static component from the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.<c_name>.static` instead.

        See Also
        --------
        [pypsa.Network.static][]

        """
        return self.components[c_name].static

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.<c_name>.dynamic` instead.",
    )
    def pnl(self, c_name: str) -> Dict:
        """Get a dynamic component from the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.<c_name>.dynamic` instead.

        See Also
        --------
        [pypsa.Network.dynamic][]

        """
        return self.c[c_name].dynamic

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.<c_name>.dynamic` instead.",
    )
    def dynamic(self, c_name: str) -> Dict:
        """Get a dynamic component from the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.<c_name>.dynamic` instead.

        See Also
        --------
        [pypsa.Network.dynamic][]

        """
        return self.components[c_name].dynamic

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.buses.static.index` instead.",
    )
    def buses_i(self) -> pd.Index:
        """Get the index of the buses in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.buses.static.index` instead.

        See Also
        --------
        [pypsa.Network.buses][]

        """
        return self.c.buses.static.index

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.lines.static.index` instead.",
    )
    def lines_i(self) -> pd.Index:
        """Get the index of the lines in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.lines.static.index` instead.

        See Also
        --------
        [pypsa.Network.lines][]

        """
        return self.c.lines.static.index

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.transformers.static.index` instead.",
    )
    def transformers_i(self) -> pd.Index:
        """Get the index of the transformers in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.transformers.static.index` instead.

        See Also
        --------
        [pypsa.Network.transformers][]

        """
        return self.c.transformers.static.index

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.generators.static.index` instead.",
    )
    def generators_i(self) -> pd.Index:
        """Get the index of the generators in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.generators.static.index` instead.

        See Also
        --------
        [pypsa.Network.generators][]

        """
        return self.c.generators.static.index

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.loads.static.index` instead.",
    )
    def loads_i(self) -> pd.Index:
        """Get the index of the loads in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.loads.static.index` instead.

        See Also
        --------
        [pypsa.Network.loads][]

        """
        return self.c.loads.static.index

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.shunt_impedances.static.index` instead.",
    )
    def shunt_impedances_i(self) -> pd.Index:
        """Get the index of the shunt impedances in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.shunt_impedances.static.index` instead.

        See Also
        --------
        [pypsa.Network.shunt_impedances][]

        """
        return self.c.shunt_impedances.static.index

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.storage_units.static.index` instead.",
    )
    def storage_units_i(self) -> pd.Index:
        """Get the index of the storage units in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.storage_units.static.index` instead.

        See Also
        --------
        [pypsa.Network.storage_units][]

        """
        return self.c.storage_units.static.index

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.stores.index.static` instead.",
    )
    def stores_i(self) -> pd.Index:
        """Get the index of the stores in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version

            Use `sub_network.components.stores.static.index` instead.

        See Also
        --------
        [pypsa.Network.stores][]

        """
        return self.c.stores.static.index

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.buses.static` instead.",
    )
    def buses(self) -> pd.DataFrame:
        """Get the buses in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.buses.static` instead.

        See Also
        --------
        [pypsa.Network.buses][]

        """
        return self.c.buses.static

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.generators.static` instead.",
    )
    def generators(self) -> pd.DataFrame:
        """Get the generators in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.generators.static` instead.

        See Also
        --------
        [pypsa.Network.generators][]

        """
        return self.c.generators.static

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.loads.static` instead.",
    )
    def loads(self) -> pd.DataFrame:
        """Get the loads in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.loads.static` instead.

        See Also
        --------
        [pypsa.Network.loads][]

        """
        return self.c.loads.static

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.shunt_impedances.static` instead.",
    )
    def shunt_impedances(self) -> pd.DataFrame:
        """Get the shunt impedances in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.shunt_impedances.static` instead.

        See Also
        --------
        [pypsa.Network.shunt_impedances][]

        """
        return self.c.shunt_impedances.static

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `sub_network.components.storage_units.static` instead.",
    )
    def storage_units(self) -> pd.DataFrame:
        """Get the storage units in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.storage_units.static` instead.

        See Also
        --------
        [pypsa.Network.storage_units][]

        """
        return self.c.storage_units.static

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `!!! deprecated.components.stores.static` instead.",
    )
    def stores(self) -> pd.DataFrame:
        """Get the stores in the sub-network.

        !!! warning "Deprecated in <!-- md:badge-version v1.0.0 -->"

            Use `sub_network.components.stores.static` instead.
        """
        return self.c.stores.static

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="2.0.0",
        details="Use `self.components` instead.",
    )
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
            if not (skip_empty and self.c[c_name].static.empty)
        )
