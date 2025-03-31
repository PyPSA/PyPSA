"""Power system components."""

from __future__ import annotations

import copy
import logging
import os
import warnings
from collections.abc import Collection, Iterator, Sequence
from typing import TYPE_CHECKING, Any
from weakref import ref

from deprecation import deprecated

from pypsa._options import option_context
from pypsa.common import equals, future_deprecation
from pypsa.components.abstract import Components
from pypsa.components.common import as_components
from pypsa.constants import DEFAULT_EPSG, DEFAULT_TIMESTAMP

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import validators
from pyproj import CRS, Transformer
from scipy.sparse import csgraph

from pypsa.clustering import ClusteringAccessor
from pypsa.common import as_index, deprecated_common_kwargs
from pypsa.components.abstract import SubNetworkComponents
from pypsa.components.components import Component
from pypsa.components.types import (
    component_types_df,
)
from pypsa.components.types import (
    get as get_component_type,
)
from pypsa.consistency import consistency_check
from pypsa.contingency import calculate_BODF, network_lpf_contingency
from pypsa.definitions.components import ComponentsStore
from pypsa.definitions.structures import Dict
from pypsa.descriptors import (
    get_active_assets,
    get_committable_i,
    get_extendable_i,
    get_non_extendable_i,
    get_switchable_as_dense,
)
from pypsa.graph import adjacency_matrix, graph, incidence_matrix
from pypsa.io import (
    _import_components_from_df,
    _import_series_from_df,
    export_to_csv_folder,
    export_to_excel,
    export_to_hdf5,
    export_to_netcdf,
    import_components_from_dataframe,
    import_from_csv_folder,
    import_from_excel,
    import_from_hdf5,
    import_from_netcdf,
    import_from_pandapower_net,
    import_from_pypower_ppc,
    import_series_from_dataframe,
    merge,
)
from pypsa.optimization.optimize import OptimizationAccessor
from pypsa.pf import (
    calculate_B_H,
    calculate_dependent_values,
    calculate_PTDF,
    calculate_Y,
    find_bus_controls,
    find_cycles,
    find_slack_bus,
    network_lpf,
    network_pf,
    sub_network_lpf,
    sub_network_pf,
)
from pypsa.plot.accessor import PlotAccessor
from pypsa.plot.maps import explore, iplot
from pypsa.statistics import StatisticsAccessor
from pypsa.typing import is_1d_list_like

if TYPE_CHECKING:
    import linopy
    from scipy.sparse import spmatrix

logger = logging.getLogger(__name__)
warnings.simplefilter("always", DeprecationWarning)


dir_name = os.path.dirname(__file__)

standard_types_dir_name = "data/standard_types"


inf = float("inf")


def create_component_property(property_type: str, component: str) -> property:
    def getter(self: Any) -> Any:
        return self.components.get(component).get(property_type)

    def setter(self: Any, value: Any) -> None:
        setattr(self.components[component], property_type, value)

    return property(getter, setter)


class Network:
    """
    Network container for all buses, one-ports and branches.

    Parameters
    ----------
    import_name : string, Path
        Path to netCDF file, HDF5 .h5 store or folder of CSV files from which to
        import network data. The string could be a URL. If cloudpathlib is installed,
        the string could be a object storage URI with an `s3`, `gs` or `az` URI scheme.
    name : string, default ""
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

    Returns
    -------
    None

    Examples
    --------
    >>> nw1 = pypsa.Network("network.nc") # doctest: +SKIP
    >>> nw2 = pypsa.Network("/my/folder") # doctest: +SKIP
    >>> nw3 = pypsa.Network("https://github.com/PyPSA/PyPSA/raw/master/examples/scigrid-de/scigrid-with-load-gen-trafos.nc") # doctest: +SKIP
    >>> nw4 = pypsa.Network("s3://my-bucket/my-network.nc") # doctest: +SKIP

    """

    # Type hints
    # ----------------

    # Core attributes
    name: str
    components: ComponentsStore
    sub_networks: pd.DataFrame

    # Component sets
    all_components: set[str]
    branch_components: set[str]
    passive_branch_components: set[str]
    passive_one_port_components: set[str]
    standard_type_components: set[str]
    controllable_branch_components: set[str]
    controllable_one_port_components: set[str]
    one_port_components: set[str]

    # Components
    buses: pd.DataFrame
    carriers: pd.DataFrame
    global_constraints: pd.DataFrame
    lines: pd.DataFrame
    line_types: pd.DataFrame
    transformers: pd.DataFrame
    transformer_types: pd.DataFrame
    links: pd.DataFrame
    loads: pd.DataFrame
    generators: pd.DataFrame
    storage_units: pd.DataFrame
    stores: pd.DataFrame
    shunt_impedances: pd.DataFrame
    shapes: pd.DataFrame

    # Components (time-dependent data)
    buses_t: Dict
    generators_t: Dict
    loads_t: Dict
    lines_t: Dict
    links_t: Dict
    transformers_t: Dict
    storage_units_t: Dict
    stores_t: Dict

    # Optimization
    model: linopy.Model
    _multi_invest: int
    _linearized_uc: int
    objective: float
    objective_constant: float
    iteration: int

    # Geospatial
    _crs = CRS.from_epsg(DEFAULT_EPSG)

    # Methods
    # -------

    # from pypsa.io
    import_from_csv_folder = import_from_csv_folder
    export_to_csv_folder = export_to_csv_folder
    import_from_excel = import_from_excel
    export_to_excel = export_to_excel
    import_from_hdf5 = import_from_hdf5
    export_to_hdf5 = export_to_hdf5
    import_from_netcdf = import_from_netcdf
    export_to_netcdf = export_to_netcdf
    import_from_pypower_ppc = import_from_pypower_ppc
    import_from_pandapower_net = import_from_pandapower_net
    merge = merge
    import_components_from_dataframe = import_components_from_dataframe  # Deprecated
    _import_series_from_df = _import_series_from_df
    import_series_from_dataframe = import_series_from_dataframe  # Deprecated

    # from pypsa.pf
    calculate_dependent_values = calculate_dependent_values
    lpf = network_lpf
    pf = network_pf

    # from pypsa.plot
    @deprecated(
        details="Use `n.plot.iplot()` as a drop-in replacement instead.",
    )
    def iplot(self, *args: Any, **kwargs: Any) -> Any:
        return iplot(self, *args, **kwargs)

    @deprecated(
        details="Use `n.plot.explore()` as a drop-in replacement instead.",
    )
    def explore(self, *args: Any, **kwargs: Any) -> Any:
        return explore(self, *args, **kwargs)

    # from pypsa.contingency
    lpf_contingency = network_lpf_contingency

    # from pypsa.graph
    graph = graph
    incidence_matrix = incidence_matrix
    adjacency_matrix = adjacency_matrix

    # from pypsa.descriptors
    get_committable_i = get_committable_i
    get_extendable_i = get_extendable_i
    get_switchable_as_dense = get_switchable_as_dense
    get_non_extendable_i = get_non_extendable_i
    get_active_assets = get_active_assets

    # from pypsa.consistency
    consistency_check = consistency_check

    # ----------------
    # Dunder methods
    # ----------------

    def __init__(
        self,
        import_name: str | Path = "",
        name: str = "",
        ignore_standard_types: bool = False,
        override_components: pd.DataFrame | None = None,
        override_component_attrs: Dict | None = None,
        **kwargs: Any,
    ) -> None:
        if override_components is not None or override_component_attrs is not None:
            msg = (
                "The arguments `override_components` and `override_component_attrs` "
                "are deprecated. Please check the release notes: "
                "https://pypsa.readthedocs.io/en/latest/references/release-notes.html#v0-33-0"
            )
            raise DeprecationWarning(msg)

        # Initialise root logger and set its level, if this has not been done before
        logging.basicConfig(level=logging.INFO)

        from pypsa import release_version as pypsa_version

        self.name: str = name

        # this will be saved on export
        self.pypsa_version: str = pypsa_version

        self._meta: dict = {}

        self._snapshots = pd.Index([DEFAULT_TIMESTAMP], name="snapshot")

        cols = ["objective", "stores", "generators"]
        self._snapshot_weightings = pd.DataFrame(1, index=self.snapshots, columns=cols)

        cols = ["objective", "years"]
        self._investment_period_weightings: pd.DataFrame = pd.DataFrame(
            index=self.investment_periods, columns=cols
        )

        # Initialize accessors
        self.optimize: OptimizationAccessor = OptimizationAccessor(self)
        self.cluster: ClusteringAccessor = ClusteringAccessor(self)
        self.statistics: StatisticsAccessor = StatisticsAccessor(self)
        self.plot: PlotAccessor = PlotAccessor(self)

        # Define component sets
        self._initialize_component_sets()

        self._initialize_components()

        if not ignore_standard_types:
            self.read_in_default_standard_types()

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
                raise ValueError(
                    f"import_name '{import_name}' is not a valid .h5 file, .nc file or directory."
                )

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return f"PyPSA Network '{self.name}'" if self.name else "Unnamed PyPSA Network"

    def __repr__(self) -> str:
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
        """Merge all components of two networks."""
        self.merge(other)

    def __eq__(self, other: Any) -> bool:
        """Check for equality of two networks."""
        ignore = [
            OptimizationAccessor,
            ClusteringAccessor,
            StatisticsAccessor,
            PlotAccessor,
        ]

        if isinstance(other, self.__class__):
            for key, value in self.__dict__.items():
                if not equals(value, other.__dict__[key], ignored_classes=ignore):
                    logger.warning("Mismatch in attribute: %s", key)
                    return False
        else:
            logger.warning(
                "Can only compare two pypsa.Network objects with each other. Got %s.",
                type(other),
            )

            return False
        return True

    # ----------------
    # Initialization
    # ----------------
    def _initialize_component_sets(self) -> None:
        # TODO merge with components.types
        for category in set(component_types_df.category.unique()):
            if not isinstance(category, float):
                setattr(
                    self,
                    category + "_components",
                    set(
                        component_types_df.index[
                            component_types_df.category == category
                        ]
                    ),
                )

        self.one_port_components = (
            self.passive_one_port_components | self.controllable_one_port_components
        )

        self.branch_components = (
            self.passive_branch_components | self.controllable_branch_components
        )

        self.all_components = set(component_types_df.index) - {"Network"}

    def _initialize_components(self) -> None:
        components = component_types_df.index.to_list()

        self.components = ComponentsStore()
        for c_name in components:
            ctype = get_component_type(c_name)

            self.components[ctype.list_name] = Component(ctype=ctype, n=self)

            setattr(
                type(self),
                ctype.list_name,
                create_component_property("static", ctype.list_name),
            )
            setattr(
                type(self),
                ctype.list_name + "_t",
                create_component_property("dynamic", ctype.list_name),
            )

    def read_in_default_standard_types(self) -> None:
        for std_type in self.standard_type_components:
            self.add(
                std_type,
                self.components[std_type].ctype.standard_types.index,
                **self.components[std_type].ctype.standard_types,
            )

    # ----------------
    # Components Store and Properties
    # ----------------

    @property
    def c(self) -> ComponentsStore:
        """
        Alias for network components.

        Access all components of the network via `n.c.<component>`. Same as
        :py:attr:`pypsa.Network.components`.

        Returns
        -------
        ComponentsStore

        """
        return self.components

    @future_deprecation(details="Use `self.components.<component>.dynamic` instead.")
    def df(self, component_name: str) -> pd.DataFrame:
        """
        Alias for :py:meth:`pypsa.Network.static`.

        Parameters
        ----------
        component_name : string

        Returns
        -------
        pandas.DataFrame

        """
        return self.static(component_name)

    @future_deprecation(details="Use `self.components.<component>.static` instead.")
    def static(self, component_name: str) -> pd.DataFrame:
        """
        Return the DataFrame of static components for component_name, i.e.
        n.component_names.

        Parameters
        ----------
        component_name : string

        Returns
        -------
        pandas.DataFrame

        """
        return self.components[component_name].static

    @future_deprecation(details="Use `self.components.<component>.dynamic` instead.")
    def pnl(self, component_name: str) -> Dict:
        """
        Alias for :py:meth:`pypsa.Network.dynamic`.

        Parameters
        ----------
        component_name : string

        Returns
        -------
        dict of pandas.DataFrame

        """
        return self.dynamic(component_name)

    @future_deprecation(details="Use `self.components.<component>.dynamic` instead.")
    def dynamic(self, component_name: str) -> Dict:
        """
        Return the dictionary of DataFrames of varying components for
        component_name, i.e. n.component_names_t.

        Parameters
        ----------
        component_name : string

        Returns
        -------
        dict of pandas.DataFrame

        """
        return self.components[component_name].dynamic

    @property
    @future_deprecation(details="Use `self.components.<component>.defaults` instead.")
    def component_attrs(self) -> pd.DataFrame:
        """
        Alias for :py:meth:`pypsa.Network.get`.

        Parameters
        ----------
        component_name : string

        Returns
        -------
        pandas.DataFrame

        """
        with option_context("warnings.components_store_iter", False):
            return Dict({value.name: value.defaults for value in self.components})

    # ----------------
    # Meta data
    # ----------------

    @property
    def meta(self) -> dict:
        """Dictionary of the network meta data."""
        return self._meta

    @meta.setter
    def meta(self, new: dict) -> None:
        if not isinstance(new, (dict | Dict)):
            raise TypeError(f"Meta must be a dictionary, received a {type(new)}")
        self._meta = new

    @property
    def crs(self) -> Any:
        """Coordinate reference system of the network's geometries (n.shapes)."""
        return self._crs

    @crs.setter
    def crs(self, new: Any) -> None:
        """
        Set the coordinate reference system of the network's geometries
        (n.shapes).
        """
        self.shapes.set_crs(new)
        self._crs = self.shapes.crs

    def to_crs(self, new: int | str | pyproj.CRS) -> None:
        """
        Convert the network's geometries and bus coordinates to a new
        coordinate reference system.
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
        """
        Spatial reference system identifier of the network's geometries
        (n.shapes).
        """
        return self.crs.to_epsg()

    @srid.setter
    def srid(self, new: str | int) -> None:
        """
        Set the spatial reference system identifier of the network's geometries
        (n.shapes).
        """
        self.crs = pyproj.CRS.from_epsg(new)

    # ----------------
    # Indexers
    # ----------------

    def set_snapshots(
        self,
        snapshots: Sequence,
        default_snapshot_weightings: float = 1.0,
        weightings_from_timedelta: bool = False,
    ) -> None:
        """
        Set the snapshots/time steps and reindex all time-dependent data.

        Snapshot weightings, typically representing the hourly length of each snapshot,
        is filled with the `default_snapshot_weighintgs` value, or uses the timedelta
        of the snapshots if `weightings_from_timedelta` flag is True, and snapshots are
        of type `pd.DatetimeIndex`.

        This will reindex all components time-dependent DataFrames
        (:py:meth:`pypsa.Network.dynamic`). NaNs are filled with the default value for
        that quantity.

        Parameters
        ----------
        snapshots : list, pandas.Index or pd.MultiIndex
            All time steps.
        default_snapshot_weightings: float
            The default weight for each snapshot. Defaults to 1.0.
        weightings_from_timedelta: bool
            Wheter to use the timedelta of `snapshots` as `snapshot_weightings` if
            `snapshots` is of type `pd.DatetimeIndex`.  Defaults to False.

        Returns
        -------
        None

        """
        # Check if snapshots contain timezones
        if isinstance(snapshots, pd.DatetimeIndex) and snapshots.tz is not None:
            msg = (
                "Numpy datetime64[ns] objects with timezones are not supported and are "
                "thus not allowed in snapshots. Please pass timezone-naive timestamps "
                "(e.g. via ds.values)."
            )
            raise ValueError(msg)

        if isinstance(snapshots, pd.MultiIndex):
            if snapshots.nlevels != 2:
                msg = "Maximally two levels of MultiIndex supported"
                raise ValueError(msg)
            sns = snapshots.rename(["period", "timestep"])
            sns.name = "snapshot"
            self._snapshots = sns
        else:
            self._snapshots = pd.Index(snapshots, name="snapshot")

        if len(self._snapshots) == 0:
            raise ValueError("Snapshots must not be empty.")

        self.snapshot_weightings = self.snapshot_weightings.reindex(
            self._snapshots, fill_value=default_snapshot_weightings
        )

        if isinstance(snapshots, pd.DatetimeIndex) and weightings_from_timedelta:
            hours_per_step = (
                snapshots.to_series()
                .diff(periods=1)
                .shift(-1)
                .ffill()  # fill last value by assuming same as the one before
                .apply(lambda x: x.total_seconds() / 3600)
            )
            self._snapshot_weightings = pd.DataFrame(
                {c: hours_per_step for c in self._snapshot_weightings.columns}
            )
        elif not isinstance(snapshots, pd.DatetimeIndex) and weightings_from_timedelta:
            logger.info(
                "Skipping `weightings_from_timedelta` as `snapshots`is not of type `pd.DatetimeIndex`."
            )

        for component in self.all_components:
            dynamic = self.dynamic(component)
            attrs = self.components[component]["attrs"]

            for k in dynamic.keys():
                if dynamic[k].empty:  # avoid expensive reindex operation
                    dynamic[k].index = self._snapshots
                elif k in attrs.default[attrs.varying]:
                    if isinstance(dynamic[k].index, pd.MultiIndex):
                        dynamic[k] = dynamic[k].reindex(
                            self._snapshots, fill_value=attrs.default[attrs.varying][k]
                        )
                    else:
                        # Make sure to keep timestep level in case of MultiIndex
                        dynamic[k] = dynamic[k].reindex(
                            self._snapshots,
                            fill_value=attrs.default[attrs.varying][k],
                            level="timestep",
                        )
                else:
                    dynamic[k] = dynamic[k].reindex(self._snapshots)

        # NB: No need to rebind dynamic to self, since haven't changed it

    @property
    def snapshots(self) -> pd.Index | pd.MultiIndex:
        """
        Snapshots dimension of the network.

        If snapshots are a pandas.MultiIndex, the first level are investment periods
        and the second level are timesteps. If snapshots are single indexed, the only
        level is timesteps.

        Returns
        -------
        pd.Index or pd.MultiIndex
            Snapshots of the network, either as a single index or a multi-index.

        See Also
        --------
        pypsa.networks.Network.timesteps : Get the timestep level only.
        pypsa.networks.Network.periods : Get the period level only.

        Notes
        -----
        Note that Snapshots are a dimension, while timesteps and and periods are
        only levels of the snapshots dimension, similar to coords in xarray.
        This is because timesteps and periods are not necessarily unique or complete
        across snapshots.
        """
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots: Sequence) -> None:
        """
        Setter for snapshots dimension.

        Parameters
        ----------
        snapshots : Sequence


        See Also
        --------
        pypsa.networks.Network.snapshots : Getter method
        pypsa.networks.Network.set_snapshots : Setter method
        """
        self.set_snapshots(snapshots)

    @property
    def timesteps(self) -> pd.Index:
        """
        Timestep level of snapshots dimension.

        If snapshots is single indexed, timesteps and snapshots yield the same result.
        Otherwise only the timestep level will be returned.

        Returns
        -------
        pd.Index
            Timesteps of the network.

        See Also
        --------
        pypsa.networks.Network.snapshots : Get the snapshots dimension.
        pypsa.networks.Network.periods : Get the period level only.

        """
        if "timestep" in self.snapshots.names:
            return self.snapshots.get_level_values("timestep").unique()
        else:
            return self.snapshots

    @timesteps.setter
    def timesteps(self, timesteps: Sequence) -> None:
        """
        Setter for timesteps level of snapshots dimension.

        .. warning::
            Setting `timesteps` is not supported. Please set `snapshots` instead.

        Parameters
        ----------
        timesteps : Sequence

        Also see
        --------
        pypsa.networks.Network.timesteps : Getter method
        """
        msg = "Setting `timesteps` is not supported. Please set `snapshots` instead."
        raise NotImplementedError(msg)

    @property
    def periods(self) -> pd.Index:
        """
        Periods level of snapshots dimension.

        If snapshots is single indexed, periods will always be empty, since there no
        investment periods without timesteps are defined. Otherwise only the period
        level will be returned.

        Returns
        -------
        pd.Index
            Periods of the network.

        See Also
        --------
        pypsa.networks.Network.snapshots : Get the snapshots dimension.
        pypsa.networks.Network.timesteps : Get the timestep level only.

        """
        if "period" in self.snapshots.names:
            return self.snapshots.get_level_values("period").unique()
        else:
            return pd.Index([], name="period")

    @periods.setter
    def periods(self, periods: Sequence) -> None:
        """
        Setter for periods level of snapshots dimension.

        Parameters
        ----------
        periods : Sequence

        Also see
        --------
        pypsa.networks.Network.periods : Getter method
        pypsa.networks.Network.set_investment_periods : Setter method
        """
        self.set_investment_periods(periods)

    @property
    def has_periods(self) -> bool:
        """
        Check if network has investment periods assigned to snapshots dimension.

        Returns
        -------
        bool
            True if network has investment periods, otherwise False.

        See Also
        --------
        pypsa.networks.Network.snapshots : Snapshots dimension of the network.
        pypsa.networks.Network.periods : Periods level of snapshots dimension.
        """
        return not self.periods.empty

    @property
    def investment_periods(self) -> pd.Index:
        """
        Periods level of snapshots dimension.

        If snapshots is single indexed, periods will always be empty, since there no
        investment periods without timesteps are defined. Otherwise only the period
        level will be returned.

        .. Note :: Alias for :py:meth:`pypsa.Network.periods`.

        Returns
        -------
        pd.Index
            Investment periods of the network.

        See Also
        --------
        pypsa.networks.Network.snapshots : Get the snapshots dimension.
        pypsa.networks.Network.periods : Get the snapshots dimension.
        pypsa.networks.Network.timesteps : Get the timestep level only.

        """
        return self.periods

    @investment_periods.setter
    def investment_periods(self, periods: Sequence) -> None:
        """
        Setter for periods level of snapshots dimension.

        .. Note :: Alias for :py:meth:`pypsa.Network.periods`.

        Parameters
        ----------
        periods : Sequence

        Also see
        --------
        pypsa.networks.Network.periods : Getter method
        pypsa.networks.Network.set_investment_periods : Setter method
        """
        self.periods = periods

    @property
    def has_investment_periods(self) -> bool:
        """
        Check if network has investment periods assigned to snapshots dimension.

        .. Note :: Alias for :py:meth:`pypsa.Network.has_periods`.

        Returns
        -------
        bool
            True if network has investment periods, otherwise False.

        See Also
        --------
        pypsa.networks.Network.snapshots : Snapshots dimension of the network.
        pypsa.networks.Network.periods : Periods level of snapshots dimension.
        """
        return self.has_periods

    @property
    def snapshot_weightings(self) -> pd.DataFrame:
        """
        Weightings applied to each snapshots during the optimization (LOPF).

        * Objective weightings multiply the operational cost in the
          objective function.

        * Generator weightings multiply the impact of all generators
          in global constraints, e.g. multiplier of GHG emmissions.

        * Store weightings define the elapsed hours for the charge, discharge
          standing loss and spillage of storage units and stores in order to
          determine the state of charge.
        """
        return self._snapshot_weightings

    @snapshot_weightings.setter
    def snapshot_weightings(self, df: pd.DataFrame) -> None:
        assert df.index.equals(self.snapshots), (
            "Weightings not defined for all snapshots."
        )
        if isinstance(df, pd.Series):
            logger.info("Applying weightings to all columns of `snapshot_weightings`")
            df = pd.DataFrame({c: df for c in self._snapshot_weightings.columns})
        self._snapshot_weightings = df

    def set_investment_periods(self, periods: Sequence) -> None:
        """
        Set the investment periods of the network.

        If the network snapshots are a pandas.MultiIndex, the investment periods
        have to be a subset of the first level. If snapshots are a single index,
        they and all time-series are repeated for all periods. This changes
        the network snapshots to be a MultiIndex (inplace operation) with the first
        level being the investment periods and the second level the snapshots.

        Parameters
        ----------
        n : pypsa.Network
        periods : list
            List of periods to be selected/initialized.

        Returns
        -------
        None.

        """
        periods_ = pd.Index(periods, name="period")
        if periods_.empty:
            return
        if not (
            pd.api.types.is_integer_dtype(periods_)
            and periods_.is_unique
            and periods_.is_monotonic_increasing
        ):
            raise ValueError(
                "Investment periods are not strictly increasing integers, "
                "which is required for multi-period investment optimisation."
            )
        if isinstance(self.snapshots, pd.MultiIndex):
            if not periods_.isin(self.snapshots.unique("period")).all():
                raise ValueError(
                    "Not all investment periods are in level `period` of snapshots."
                )
            if len(periods_) < len(self.snapshots.unique(level="period")):
                raise NotImplementedError(
                    "Investment periods do not equal first level values of snapshots."
                )
        else:
            # Convenience case:
            logger.info(
                "Repeating time-series for each investment period and "
                "converting snapshots to a pandas.MultiIndex."
            )
            names = ["period", "timestep"]
            for component in self.all_components:
                dynamic = self.dynamic(component)

                for k in dynamic.keys():
                    dynamic[k] = pd.concat(
                        {p: dynamic[k] for p in periods_}, names=names
                    )
                    dynamic[k].index.name = "snapshot"

            self._snapshots = pd.MultiIndex.from_product(
                [periods_, self.snapshots], names=names
            )
            self._snapshots.name = "snapshot"
            self._snapshot_weightings = pd.concat(
                {p: self.snapshot_weightings for p in periods_}, names=names
            )
            self._snapshot_weightings.index.name = "snapshot"

        self.investment_period_weightings = self.investment_period_weightings.reindex(
            self.periods, fill_value=1.0
        ).astype(float)

    @property
    def investment_period_weightings(self) -> pd.DataFrame:
        """
        Weightings applied to each investment period during the optimization
        (LOPF).

        * Objective weightings are multiplied with all cost coefficients in the
          objective function of the respective investment period
          (e.g. to include a social discount rate).

        * Years weightings denote the elapsed time until the subsequent investment period
          (e.g. used for global constraints CO2 emissions).
        """
        return self._investment_period_weightings

    @investment_period_weightings.setter
    def investment_period_weightings(self, df: pd.DataFrame) -> None:
        assert df.index.equals(self.investment_periods), (
            "Weightings not defined for all investment periods."
        )
        if isinstance(df, pd.Series):
            logger.info(
                "Applying weightings to all columns of `investment_period_weightings`"
            )
            df = pd.DataFrame(
                {c: df for c in self._investment_period_weightings.columns}
            )
        self._investment_period_weightings = df

    def add(
        self,
        class_name: str,
        name: str | int | Sequence[int | str],
        suffix: str = "",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> pd.Index:
        """
        Add components to the network.

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
            of pandas.DataFrame for time-varying

        Returns
        -------
        new_names : pandas.index
            Names of new components (including suffix)

        Examples
        --------
        Add a single component:

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
                                f"Single value sequence for {k} is treated as a scalar "
                                f"and broadcasted to all components. It is recommended "
                                f"to explicitly pass a scalar instead."
                            )
                        else:
                            v.index = names
                except ValueError:
                    expec_str = (
                        f"{len(self.snapshots)} for each snapshot."
                        if single_component
                        else f"{len(names)} for each component name."
                    )
                    msg = f"Data for {k} has length {len(v)} but expected {expec_str}"
                    raise ValueError(msg)
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
        _import_components_from_df(self, static_df, c.name, overwrite=overwrite)

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
        """
        Removes a single component or a list of components from the network.

        Removes it from component DataFrames.

        Parameters
        ----------
        class_name : str
            Component class name
        name : str, int, list-like or pandas.Index
            Component name(s)
        suffix : str, default ''


        Examples
        --------
        >>> n.remove("Line", "0")
        >>> n.remove("Line", ["1","2"])
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
        """
        Add multiple components to the network, along with their attributes.

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
        """
        Removes multiple components from the network.

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
        name : list-like
            Component names

        """
        self.remove(class_name=class_name, name=names)

    def copy(
        self,
        snapshots: Sequence | None = None,
        investment_periods: Sequence | None = None,
        ignore_standard_types: bool = False,
        with_time: bool | None = None,
    ) -> Network:
        """
        Returns a deep copy of Network objec    t.

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
        >>> network_copy = n.copy()

        """
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
                "removed in a future version. Pass an empty list to 'snapshots' instead.",
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
                for k in component.dynamic.keys():
                    try:
                        dynamic[k] = component.dynamic[k].loc[snapshots_].copy()
                    except KeyError:
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
            "objective",
            "objective_constant",
            "now",
        ]:
            try:
                setattr(n, attr, getattr(self, attr))
            except AttributeError:
                pass

        return n

    def __getitem__(self, key: str) -> Network:
        """
        Returns a shallow slice of the Network object containing only the
        selected buses and all the connected components.

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
        return pd.concat(
            (self.static(c) for c in self.branch_components),
            keys=self.branch_components,
            sort=True,
            names=["component", "name"],
        )

    def passive_branches(self) -> pd.DataFrame:
        return pd.concat(
            (self.static(c) for c in self.passive_branch_components),
            keys=self.passive_branch_components,
            sort=True,
        )

    def controllable_branches(self) -> pd.DataFrame:
        return pd.concat(
            (self.static(c) for c in self.controllable_branch_components),
            keys=self.controllable_branch_components,
            sort=True,
        )

    def determine_network_topology(
        self,
        investment_period: int | str | None = None,
        skip_isolated_buses: bool = False,
    ) -> None:
        """
        Build sub_networks from topology.

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
                    f"Warning, sub network {i} is not electric but "
                    "contains multiple buses\nand branches. Passive "
                    "flows are not allowed for non-electric networks!"
                )

            if (self.buses.carrier.iloc[buses_i] != carrier).any():
                logger.warning(
                    f"Warning, sub network {i} contains buses with "
                    "mixed carriers! Value counts:"
                    f"\n{self.buses.carrier.iloc[buses_i].value_counts()}"
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
                active = get_active_assets(self, c.name, investment_period)
                # set non active assets to NaN
                c.static.loc[~active, "sub_network"] = np.nan

        for sub in self.sub_networks.obj:
            find_cycles(sub)
            sub.find_bus_controls()

    @future_deprecation(details="Use `self.components.<component>` instead.")
    def component(self, c_name: str) -> Component:
        return self.components[c_name]

    @future_deprecation(details="Use `self.components` instead.")
    def iterate_components(
        self, components: Collection[str] | None = None, skip_empty: bool = True
    ) -> Iterator[Component]:
        if components is None:
            components = self.all_components

        return (
            self.component(c_name)
            for c_name in components
            if not (skip_empty and self.static(c_name).empty)
        )

    def rename_component_names(
        self, component: str | Components, **kwargs: str
    ) -> None:
        """
        Rename component names.

        Rename components of component type and also update all cross-references of
        the component in network.

        Parameters
        ----------
        component : str or pypsa.Components
            Component type or instance of pypsa.Components.
        **kwargs
            Mapping of old names to new names.

        Returns
        -------
        None

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


class SubNetwork:
    """
    Connected network of electric buses (AC or DC) with passive flows or
    isolated non-electric buses.

    Generated by n.determine_network_topology().
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

    # Methods
    # ------------------

    # from pypsa.pf
    lpf = sub_network_lpf
    pf = sub_network_pf
    find_bus_controls = find_bus_controls
    find_slack_bus = find_slack_bus
    calculate_Y = calculate_Y
    calculate_PTDF = calculate_PTDF
    calculate_B_H = calculate_B_H

    # from pypsa.contingency
    calculate_BODF = calculate_BODF

    # from pypsa.graph
    graph = graph
    incidence_matrix = incidence_matrix
    adjacency_matrix = adjacency_matrix

    @deprecated_common_kwargs
    def __init__(self, n: Network, name: str) -> None:
        self._n = ref(n)
        self.name = name

    @property
    @deprecated(details="Use the `n` property instead.")
    def network(self) -> Network:
        return self._n()  # type: ignore

    @property
    def n(self) -> Network:
        return self._n()  # type: ignore

    @property
    def components(self) -> ComponentsStore:
        def filter_down(key: str, c: Components) -> Any:
            value = c[key]
            if key == "static":
                if c.name in {"Bus"} | self.n.passive_branch_components:
                    return value[value.sub_network == self.name]
                elif c.name in self.n.one_port_components:
                    buses = self.buses_i()
                    return value[value.bus.isin(buses)]
                else:
                    raise ValueError(
                        f"Component {c.name} not supported for sub-networks"
                    )
            elif key == "dynamic":
                dynamic = Dict()
                index = self.static(c.name).index
                for k, v in self.n.dynamic(c.name).items():
                    dynamic[k] = v[index.intersection(v.columns)]
                return dynamic
            else:
                return value

        return ComponentsStore(
            {
                key: SubNetworkComponents(value, filter_down)
                for key, value in self.n.components.items()
            }
        )

    @property
    def c(self) -> ComponentsStore:
        return self.components

    @property
    def snapshots(self) -> pd.Index | pd.MultiIndex:
        return self.n.snapshots

    @property
    def snapshot_weightings(self) -> pd.DataFrame:
        return self.n.snapshot_weightings

    @property
    def investment_periods(self) -> pd.Index:
        return self.n.investment_periods

    @property
    def investment_period_weightings(self) -> pd.DataFrame:
        return self.n.investment_period_weightings

    def branches_i(self, active_only: bool = False) -> pd.MultiIndex:
        types = []
        names = []
        for c in self.iterate_components(self.n.passive_branch_components):
            idx = c.static.query("active").index if active_only else c.static.index
            types += len(idx) * [c.name]
            names += list(idx)
        return pd.MultiIndex.from_arrays([types, names], names=("type", "name"))

    def branches(self) -> pd.DataFrame:
        branches = self.n.passive_branches()
        return branches[branches.sub_network == self.name]

    @future_deprecation(details="Use `self.components.<c_name>` instead.")
    def component(self, c_name: str) -> SubNetworkComponents:
        return self.components[c_name]

    @future_deprecation(details="Use `self.components.<c_name>.static` instead.")
    def df(self, c_name: str) -> pd.DataFrame:
        return self.static(c_name)

    @future_deprecation(details="Use `self.components.<c_name>.static` instead.")
    def static(self, c_name: str) -> pd.DataFrame:
        return self.components[c_name].static

    @future_deprecation(details="Use `self.components.<c_name>.dynamic` instead.")
    def pnl(self, c_name: str) -> Dict:
        return self.dynamic(c_name)

    @future_deprecation(details="Use `self.components.<c_name>.dynamic` instead.")
    def dynamic(self, c_name: str) -> Dict:
        return self.components[c_name].dynamic

    @future_deprecation(details="Use `self.components.buses.static.index` instead.")
    def buses_i(self) -> pd.Index:
        return self.components.buses.static.index

    @future_deprecation(details="Use `self.components.lines.static.index` instead.")
    def lines_i(self) -> pd.Index:
        return self.components.lines.static.index

    @future_deprecation(
        details="Use `self.components.transformers.static.index` instead."
    )
    def transformers_i(self) -> pd.Index:
        return self.components.transformers.static.index

    @future_deprecation(
        details="Use `self.components.generators.static.index` instead."
    )
    def generators_i(self) -> pd.Index:
        return self.components.generators.static.index

    @future_deprecation(details="Use `self.components.loads.static.index` instead.")
    def loads_i(self) -> pd.Index:
        return self.components.loads.static.index

    @future_deprecation(
        details="Use `self.components.shunt_impedances.static.index` instead."
    )
    def shunt_impedances_i(self) -> pd.Index:
        return self.components.shunt_impedances.static.index

    @future_deprecation(
        details="Use `self.components.storage_units.static.index` instead."
    )
    def storage_units_i(self) -> pd.Index:
        return self.components.storage_units.static.index

    @future_deprecation(details="Use `self.components.stores.index.static` instead.")
    def stores_i(self) -> pd.Index:
        return self.components.stores.static.index

    @future_deprecation(details="Use `self.components.buses.static` instead.")
    def buses(self) -> pd.DataFrame:
        return self.components.buses.static

    @future_deprecation(details="Use `self.components.generators.static` instead.")
    def generators(self) -> pd.DataFrame:
        return self.components.generators.static

    @future_deprecation(details="Use `self.components.loads.static` instead.")
    def loads(self) -> pd.DataFrame:
        return self.components.loads.static

    @future_deprecation(
        details="Use `self.components.shunt_impedances.static` instead."
    )
    def shunt_impedances(self) -> pd.DataFrame:
        return self.components.shunt_impedances.static

    @future_deprecation(details="Use `self.components.storage_units.static` instead.")
    def storage_units(self) -> pd.DataFrame:
        return self.components.storage_units.static

    @future_deprecation(details="Use `self.components.stores.static` instead.")
    def stores(self) -> pd.DataFrame:
        return self.components.stores.static

    @future_deprecation(details="Use `self.components` instead.")
    # Deprecate: Use `self.iterate_components` instead
    def iterate_components(
        self, components: Collection[str] | None = None, skip_empty: bool = True
    ) -> Iterator[SubNetworkComponents]:
        """
        Iterate over components of the sub-network and extract corresponding
        data.

        Parameters
        ----------
        components : list-like, optional
            List of components ('Generator', 'Line', etc.) to iterate over,
            by default None
        skip_empty : bool, optional
            Whether to skip a components with no assigned assets,
            by default True

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
