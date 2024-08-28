"""Power system components."""

from __future__ import annotations

import copy
import logging
import os
import warnings
from collections import namedtuple
from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from weakref import ref

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import validators
from pyproj import CRS, Transformer
from scipy.sparse import csgraph

from pypsa.clustering import ClusteringAccessor
from pypsa.consistency import (
    check_assets,
    check_dtypes_,
    check_for_disconnected_buses,
    check_for_unknown_buses,
    check_for_unknown_carriers,
    check_for_zero_impedances,
    check_for_zero_s_nom,
    check_generators,
    check_investment_periods,
    check_nans_for_component_default_attrs,
    check_shapes,
    check_static_power_attributes,
    check_time_series,
    check_time_series_power_attributes,
)
from pypsa.contingency import calculate_BODF, network_lpf_contingency
from pypsa.descriptors import (
    Dict,
    get_active_assets,
    get_committable_i,
    get_extendable_i,
    get_non_extendable_i,
    get_switchable_as_dense,
    update_linkports_component_attrs,
)
from pypsa.graph import adjacency_matrix, graph, incidence_matrix
from pypsa.io import (
    export_to_csv_folder,
    export_to_hdf5,
    export_to_netcdf,
    import_components_from_dataframe,
    import_from_csv_folder,
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
from pypsa.plot import explore, iplot, plot  # type: ignore
from pypsa.statistics import StatisticsAccessor
from pypsa.utils import as_index

if TYPE_CHECKING:
    import linopy
    from scipy.sparese import spmatrix

logger = logging.getLogger(__name__)
warnings.simplefilter("always", DeprecationWarning)


dir_name = os.path.dirname(__file__)
component_attrs_dir_name = "component_attrs"

standard_types_dir_name = "standard_types"


inf = float("inf")


components = pd.read_csv(os.path.join(dir_name, "components.csv"), index_col=0)

component_attrs = Dict()

for component in components.index:
    file_name = os.path.join(
        dir_name,
        component_attrs_dir_name,
        components.at[component, "list_name"] + ".csv",
    )
    component_attrs[component] = pd.read_csv(file_name, index_col=0, na_values="n/a")

del component


class Basic:
    """Common to every object."""

    name = ""

    def __init__(self, name: str = ""):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.name}"


class Common(Basic):
    """Common to all objects inside Network object."""

    def __init__(self, network: Network, name: str = ""):
        Basic.__init__(self, name)
        self._network = ref(network)

    @property
    def network(self) -> Network:
        return self._network()  # type: ignore


Component = namedtuple("Component", ["name", "list_name", "attrs", "df", "pnl", "ind"])


class Network(Basic):
    """
    Network container for all buses, one-ports and branches.

    Parameters
    ----------
    import_name : string, Path
        Path to netCDF file, HDF5 .h5 store or folder of CSV files from which to
        import network data. The string could be a URL.
    name : string, default ""
        Network name.
    ignore_standard_types : boolean, default False
        If True, do not read in PyPSA standard types into standard types
        DataFrames.
    override_components : pandas.DataFrame
        If you want to override the standard PyPSA components in
        pypsa.components.components, pass it a DataFrame with index of component
        name and columns of list_name and description, following the format of
        pypsa.components.components. See git repository examples/new_components/.
    override_component_attrs : pypsa.descriptors.Dict of pandas.DataFrame
        If you want to override pypsa.component_attrs, follow its format.
        See :doc:`/user-guide/components` for more information.
    kwargs
        Any remaining attributes to set

    Returns
    -------
    None

    Examples
    --------
    >>> nw1 = pypsa.Network("my_store.h5")
    >>> nw2 = pypsa.Network("/my/folder")
    >>> nw3 = pypsa.Network("https://github.com/PyPSA/PyPSA/raw/master/examples/scigrid-de/scigrid-with-load-gen-trafos.nc")

    """

    # Type hints
    # ----------------

    # Core attributes
    snapshots: pd.Index | pd.MultiIndex
    components: Dict
    component_attrs: Dict
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
    _crs = CRS.from_epsg(4326)

    # Methods
    # -------

    # from pypsa.io
    import_from_csv_folder = import_from_csv_folder
    export_to_csv_folder = export_to_csv_folder
    import_from_hdf5 = import_from_hdf5
    export_to_hdf5 = export_to_hdf5
    import_from_netcdf = import_from_netcdf
    export_to_netcdf = export_to_netcdf
    import_from_pypower_ppc = import_from_pypower_ppc
    import_from_pandapower_net = import_from_pandapower_net
    import_components_from_dataframe = import_components_from_dataframe
    merge = merge
    import_series_from_dataframe = import_series_from_dataframe

    # from pypsa.pf
    calculate_dependent_values = calculate_dependent_values
    lpf = network_lpf
    pf = network_pf

    # from pypsa.plot
    plot = plot
    iplot = iplot
    explore = explore

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

    def __init__(
        self,
        import_name: str | Path = "",
        name: str = "",
        ignore_standard_types: bool = False,
        override_components: pd.DataFrame | None = None,
        override_component_attrs: Dict | None = None,
        **kwargs: Any,
    ):
        # Initialise root logger and set its level, if this has not been done before
        logging.basicConfig(level=logging.INFO)

        from pypsa import release_version as pypsa_version

        Basic.__init__(self, name)

        # this will be saved on export
        self.pypsa_version: str = pypsa_version

        self._meta: dict = {}

        self._snapshots = pd.Index(["now"])

        cols = ["objective", "stores", "generators"]
        self._snapshot_weightings = pd.DataFrame(1, index=self.snapshots, columns=cols)

        self._investment_periods: pd.Index = pd.Index([])

        cols = ["objective", "years"]
        self._investment_period_weightings: pd.DataFrame = pd.DataFrame(columns=cols)

        self.optimize: OptimizationAccessor = OptimizationAccessor(self)

        self.cluster: ClusteringAccessor = ClusteringAccessor(self)

        if override_components is None:
            self.components = components
        else:
            self.components = override_components

        if override_component_attrs is None:
            self.component_attrs = component_attrs
        else:
            self.component_attrs = override_component_attrs

        for c_type in set(self.components.type.unique()):
            if not isinstance(c_type, float):
                setattr(
                    self,
                    c_type + "_components",
                    set(self.components.index[self.components.type == c_type]),
                )

        self.one_port_components = (
            self.passive_one_port_components | self.controllable_one_port_components
        )

        self.branch_components = (
            self.passive_branch_components | self.controllable_branch_components
        )

        self.all_components = set(self.components.index) - {"Network"}

        self.components = Dict(self.components.T.to_dict())

        self.statistics: StatisticsAccessor = StatisticsAccessor(self)

        for component in self.components:
            # make copies to prevent unexpected sharing of variables
            attrs = self.component_attrs[component].copy()

            attrs["default"] = attrs.default.astype(object)
            attrs["static"] = attrs["type"] != "series"
            attrs["varying"] = attrs["type"].isin({"series", "static or series"})
            attrs["typ"] = (
                attrs["type"]
                .map(
                    {"boolean": bool, "int": int, "string": str, "geometry": "geometry"}
                )
                .fillna(float)
            )
            attrs["dtype"] = (
                attrs["type"]
                .map(
                    {
                        "boolean": np.dtype(bool),
                        "int": np.dtype(int),
                        "string": np.dtype("O"),
                    }
                )
                .fillna(np.dtype(float))
            )

            bool_b = attrs.type == "boolean"
            if bool_b.any():
                attrs.loc[bool_b, "default"] = attrs.loc[bool_b, "default"].isin(
                    {True, "True"}
                )

            # exclude Network because it's not in a DF and has non-typical attributes
            if component != "Network":
                str_b = attrs.typ.apply(lambda x: x is str)
                attrs.loc[str_b, "default"] = attrs.loc[str_b, "default"].fillna("")
                for typ in (str, float, int):
                    typ_b = attrs.typ == typ
                    attrs.loc[typ_b, "default"] = attrs.loc[typ_b, "default"].astype(
                        typ
                    )

            self.component_attrs[component] = attrs
            self.components[component]["attrs"] = attrs

        self._build_dataframes()

        if not ignore_standard_types:
            self.read_in_default_standard_types()

        if import_name:
            if not validators.url(str(import_name)):
                import_name = Path(import_name)
            if str(import_name).endswith(".h5"):
                self.import_from_hdf5(import_name)
            elif str(import_name).endswith(".nc"):
                self.import_from_netcdf(import_name)
            elif isinstance(import_name, Path) and import_name.is_dir():
                self.import_from_csv_folder(import_name)
            else:
                raise ValueError(
                    f"import_name '{import_name}' is not a valid .h5 file, .nc file or directory."
                )

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        header = "PyPSA Network" + (f" '{self.name}'" if self.name else "")
        comps = {
            c.name: f" - {c.name}: {len(c.df)}"
            for c in self.iterate_components()
            if "Type" not in c.name and len(c.df)
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

    # def __getattr__(self, name: str) -> Any:
    #     return self[name]

    def __add__(self, other: Network) -> None:
        """Merge all components of two networks."""
        self.merge(other)

    def __eq__(self, other: Any) -> bool:
        """Check for equality of two networks."""

        def equals(a: Any, b: Any) -> bool:
            assert isinstance(a, type(b)), f"Type mismatch: {type(a)} != {type(b)}"
            # Classes with equality methods
            if isinstance(a, np.ndarray):
                if not np.array_equal(a, b):
                    return False
            elif isinstance(a, (pd.DataFrame, pd.Series, pd.Index)):
                if not a.equals(b):
                    return False
            # Iterators
            elif isinstance(a, (dict, Dict)):
                for k, v in a.items():
                    if not equals(v, b[k]):
                        return False
            elif isinstance(a, (list, tuple)):
                for i, v in enumerate(a):
                    if not equals(v, b[i]):
                        return False
            # Ignore for now
            elif isinstance(
                value, (OptimizationAccessor, ClusteringAccessor, StatisticsAccessor)
            ):
                pass
            # Nans
            elif pd.isna(a) and pd.isna(b):
                pass
            else:
                if a != b:
                    return False

            return True

        if isinstance(other, self.__class__):
            for key, value in self.__dict__.items():
                if not equals(value, other.__dict__[key]):
                    return False
        else:
            logger.warning(
                "Can only compare two pypsa.Network objects with each other. Got %s.",
                type(other),
            )

            return False
        return True

    def _build_dataframes(self) -> None:
        """
        Function called when network is created to build component
        pandas.DataFrames.
        """
        for component in self.all_components:
            attrs = self.components[component]["attrs"]

            static_dtypes = attrs.loc[attrs.static, "dtype"].drop(["name"])

            if component == "Shape":
                df = gpd.GeoDataFrame(
                    {k: gpd.GeoSeries(dtype=d) for k, d in static_dtypes.items()},
                    columns=static_dtypes.index,
                    crs=self.srid,
                )
            else:
                df = pd.DataFrame(
                    {k: pd.Series(dtype=d) for k, d in static_dtypes.items()},
                    columns=static_dtypes.index,
                )

            df.index.name = component
            setattr(self, self.components[component]["list_name"], df)

            # it's currently hard to imagine non-float series,
            # but this could be generalised
            pnl = Dict()
            for k in attrs.index[attrs.varying]:
                df = pd.DataFrame(index=self.snapshots, columns=[], dtype=float)
                df.index.name = "snapshot"
                df.columns.name = component
                pnl[k] = df

            setattr(self, self.components[component]["list_name"] + "_t", pnl)

    def read_in_default_standard_types(self) -> None:
        for std_type in self.standard_type_components:
            list_name = self.components[std_type]["list_name"]

            file_name = os.path.join(
                dir_name, standard_types_dir_name, list_name + ".csv"
            )

            self.components[std_type]["standard_types"] = pd.read_csv(
                file_name, index_col=0
            )

            self.import_components_from_dataframe(
                self.components[std_type]["standard_types"], std_type
            )

    def df(self, component_name: str) -> pd.DataFrame:
        """
        Return the DataFrame of static components for component_name, i.e.
        network.component_names.

        Parameters
        ----------
        component_name : string

        Returns
        -------
        pandas.DataFrame

        """
        return getattr(self, self.components[component_name]["list_name"])

    def pnl(self, component_name: str) -> Dict:
        """
        Return the dictionary of DataFrames of varying components for
        component_name, i.e. network.component_names_t.

        Parameters
        ----------
        component_name : string

        Returns
        -------
        dict of pandas.DataFrame

        """
        return getattr(self, self.components[component_name]["list_name"] + "_t")

    @property
    def meta(self) -> dict:
        """Dictionary of the network meta data."""
        return self._meta

    @meta.setter
    def meta(self, new: dict) -> None:
        if not isinstance(new, (dict, Dict)):
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
        (:py:meth:`pypsa.network.pnl`). NaNs are filled with the default value for that quantity.

        Parameters
        ----------
        snapshots : list, pandas.Index or pd.MultiIndex
            All time steps.
        default_snapshot_weightings: float
            The default weight for each snapshot. Defaults to 1.0.
        weightings_from_timedelta: bool
            Wheter to use the timedelta of `snapshots` as `snapshot_weightings` if `snapshots` is of type `pd.DatetimeIndex`.  Defaults to False.

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
            pnl = self.pnl(component)
            attrs = self.components[component]["attrs"]

            for k in pnl.keys():
                if pnl[k].empty:  # avoid expensive reindex operation
                    pnl[k].index = self._snapshots
                elif k in attrs.default[attrs.varying]:
                    pnl[k] = pnl[k].reindex(
                        self._snapshots, fill_value=attrs.default[attrs.varying][k]
                    )
                else:
                    pnl[k] = pnl[k].reindex(self._snapshots)

        # NB: No need to rebind pnl to self, since haven't changed it

    snapshots = property(
        lambda self: self._snapshots, set_snapshots, doc="Time steps of the network"
    )

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
        assert df.index.equals(
            self.snapshots
        ), "Weightings not defined for all snapshots."
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
        periods_ = pd.Index(periods)
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
                    "Not all investment periods are in level `period` " "of snapshots."
                )
            if len(periods_) < len(self.snapshots.unique(level="period")):
                raise NotImplementedError(
                    "Investment periods do not equal first level "
                    "values of snapshots."
                )
        else:
            # Convenience case:
            logger.info(
                "Repeating time-series for each investment period and "
                "converting snapshots to a pandas.MultiIndex."
            )
            names = ["period", "timestep"]
            for component in self.all_components:
                pnl = self.pnl(component)

                for k in pnl.keys():
                    pnl[k] = pd.concat({p: pnl[k] for p in periods_}, names=names)
                    pnl[k].index.name = "snapshot"

            self._snapshots = pd.MultiIndex.from_product(
                [periods_, self.snapshots], names=names
            )
            self._snapshots.name = "snapshot"
            self._snapshot_weightings = pd.concat(
                {p: self.snapshot_weightings for p in periods_}, names=names
            )
            self._snapshot_weightings.index.name = "snapshot"

        self._investment_periods = periods_
        self.investment_period_weightings = self.investment_period_weightings.reindex(
            periods_, fill_value=1.0
        ).astype(float)

    investment_periods = property(
        lambda self: self._investment_periods,
        set_investment_periods,
        doc="Investment steps during the optimization.",
    )

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
        assert df.index.equals(
            self.investment_periods
        ), "Weightings not defined for all investment periods."
        if isinstance(df, pd.Series):
            logger.info(
                "Applying weightings to all columns of `investment_period_weightings`"
            )
            df = pd.DataFrame(
                {c: df for c in self._investment_period_weightings.columns}
            )
        self._investment_period_weightings = df

    def add(self, class_name: str, name: str, **kwargs: Any) -> None:
        """
        Add a single component to the network.

        Adds it to component DataFrame.

        Any attributes which are not specified will be given the default
        value from :doc:`/user-guide/components`.

        This method is slow for many components; instead use ``madd`` or
        ``import_components_from_dataframe`` (see below).

        Parameters
        ----------
        class_name : string
            Component class name in ("Bus", "Generator", "Load", "StorageUnit",
            "Store", "ShuntImpedance", "Line", "Transformer", "Link").
        name : string
            Component name
        kwargs
            Component attributes, e.g. x=0.1, length=123

        Examples
        --------
        >>> network.add("Bus", "my_bus_0")
        >>> network.add("Bus", "my_bus_1", v_nom=380)
        >>> network.add("Line", "my_line_name", bus0="my_bus_0", bus1="my_bus_1", length=34, r=2, x=4)

        """
        if class_name not in self.components:
            msg = f"Component class {class_name} not found"
            raise ValueError(msg)

        cls_df = self.df(class_name)
        cls_pnl = self.pnl(class_name)

        name = str(name)

        if name in cls_df.index:
            msg = (
                f"Failed to add {class_name} component {name} because there is "
                f"already an object with this name in "
                f"{self.components[class_name]['list_name']}"
            )
            raise ValueError(msg)

        if class_name == "Link":
            update_linkports_component_attrs(self, kwargs.keys())

        attrs = self.components[class_name]["attrs"]

        static_attrs = attrs[attrs.static].drop("name")

        # This guarantees that the correct attribute type is maintained
        obj_df = pd.DataFrame(
            data=[static_attrs.default], index=[name], columns=static_attrs.index
        )

        # Remove all NaN columns before concatenating
        obj_df = obj_df.dropna(axis=1, how="all")
        new_df = pd.concat([cls_df, obj_df], sort=False)

        new_df.index.name = class_name
        setattr(self, self.components[class_name]["list_name"], new_df)

        for k, v in kwargs.items():
            if k not in attrs.index:
                logger.warning(
                    f"{class_name} has no attribute {k}, " "ignoring this passed value."
                )
                continue
            typ = attrs.at[k, "typ"]
            if not attrs.at[k, "varying"]:
                new_df.at[name, k] = typ(v) if typ != "geometry" else v
            elif attrs.at[k, "static"] and not isinstance(
                v, (pd.Series, pd.DataFrame, np.ndarray, list)
            ):
                new_df.at[name, k] = typ(v)
            else:
                cls_pnl[k][name] = pd.Series(data=v, index=self.snapshots, dtype=typ)

        for attr in [attr for attr in new_df if attr.startswith("bus")]:
            bus_name = new_df.at[name, attr]
            # allow empty buses for multi-ports
            port = int(attr[-1]) if attr[-1].isdigit() else 0
            if bus_name not in self.buses.index and not (bus_name == "" and port > 1):
                logger.warning(
                    f"The bus name `{bus_name}` given for {attr} "
                    f"of {class_name} `{name}` does not appear "
                    "in network.buses"
                )

    def remove(self, class_name: str, name: str) -> None:
        """
        Removes a single component from the network.

        Removes it from component DataFrames.

        Parameters
        ----------
        class_name : string
            Component class name
        name : string
            Component name

        Examples
        --------
        >>> network.remove("Line", "my_line 12345")

        """
        if class_name not in self.components:
            logger.error(f"Component class {class_name} not found")
            return None

        cls_df = self.df(class_name)

        cls_df.drop(name, inplace=True)

        pnl = self.pnl(class_name)

        for df in pnl.values():
            if name in df:
                df.drop(name, axis=1, inplace=True)

    def madd(
        self,
        class_name: str,
        names: Sequence,
        suffix: str = "",
        **kwargs: Any,
    ) -> pd.Index:
        """
        Add multiple components to the network, along with their attributes.

        Make sure when adding static attributes as pandas Series that they are indexed
        by names. Make sure when adding time-varying attributes as pandas DataFrames that
        their index is a superset of network.snapshots and their columns are a
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

        Returns
        -------
        new_names : pandas.index
            Names of new components (including suffix)

        Examples
        --------
        Short Example:

        >>> network.madd("Load", ["load 1", "load 2"],
        ...        bus=["1", "2"],
        ...        p_set=np.random.rand(len(network.snapshots), 2))

        Long Example:

        >>> import pandas as pd, numpy as np
        >>> buses = range(13)
        >>> snapshots = range(7)
        >>> n = pypsa.Network()
        >>> n.set_snapshots(snapshots)
        >>> n.madd("Bus", buses)
        >>> # add load as numpy array
        >>> n.madd("Load",
        ...        n.buses.index + " load",
        ...        bus=buses,
        ...        p_set=np.random.rand(len(snapshots), len(buses)))
        >>> # add wind availability as pandas DataFrame
        >>> wind = pd.DataFrame(np.random.rand(len(snapshots), len(buses)),
        ...        index=n.snapshots,
        ...        columns=buses)
        >>> #use a suffix to avoid boilerplate to rename everything
        >>> n.madd("Generator",
        ...        buses,
        ...        suffix=' wind',
        ...        bus=buses,
        ...        p_nom_extendable=True,
        ...        capital_cost=1e5,
        ...        p_max_pu=wind)

        """
        if class_name not in self.components:
            logger.error(f"Component class {class_name} not found")
            return None

        if not isinstance(names, pd.Index):
            names = pd.Index(names)

        new_names = names.astype(str) + suffix

        static = {}
        series = {}
        for k, v in kwargs.items():
            if isinstance(v, pd.DataFrame):
                series[k] = v.rename(columns=lambda i: str(i) + suffix)
            elif isinstance(v, pd.Series):
                static[k] = v.rename(lambda i: str(i) + suffix)
            elif isinstance(v, np.ndarray) and v.shape == (
                len(self.snapshots),
                len(names),
            ):
                series[k] = pd.DataFrame(v, index=self.snapshots, columns=new_names)
            else:
                static[k] = v

        self.import_components_from_dataframe(
            pd.DataFrame(static, index=new_names), class_name
        )

        for k, v in series.items():
            self.import_series_from_dataframe(v, class_name, k)

        return new_names

    def mremove(self, class_name: str, names: Sequence) -> None:
        """
        Removes multiple components from the network.

        Removes them from component DataFrames.

        Parameters
        ----------
        class_name : string
            Component class name
        name : list-like
            Component names

        Examples
        --------
        >>> network.mremove("Line", ["line x", "line y"])

        """
        if class_name not in self.components:
            logger.error(f"Component class {class_name} not found")
            return None

        if not isinstance(names, pd.Index):
            names = pd.Index(names)

        cls_df = self.df(class_name)

        cls_df.drop(names, inplace=True)

        pnl = self.pnl(class_name)

        for df in pnl.values():
            df.drop(df.columns.intersection(names), axis=1, inplace=True)

    def _retrieve_overridden_components(self) -> tuple[pd.DataFrame, Dict]:
        components_index = list(self.components.keys())

        cols = ["list_name", "description", "type"]

        override_components = pd.DataFrame(
            [[self.components[i][c] for c in cols] for i in components_index],
            columns=cols,
            index=components_index,
        )

        override_component_attrs = Dict(
            {i: self.component_attrs[i].copy() for i in components_index}
        )

        return override_components, override_component_attrs

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
            A list of snapshots to copy, must be a subset of network.snapshots. Pass
            an empty list ignore all snapshots.
        investment_periods : list or tuple or pd.Index, default self.investment_period_weightings.index
            A list of investment periods to copy, must be a subset of network.investment_periods. Pass
        ignore_standard_types : boolean, default False
            Ignore the PyPSA standard types.
        with_time : boolean, default True
            Copy snapshots and time-varying network.component_names_t data too.

            .. deprecated:: 0.29.0
              Will be removed in a future version. Pass an empty list to 'snapshots'
              instead.

        Returns
        -------
        network : pypsa.Network
            The copied network object.

        Examples
        --------
        >>> network_copy = network.copy()

        """

        # Use copy.deepcopy if no arguments are passed
        args = [snapshots, investment_periods, ignore_standard_types, with_time]
        if all(arg is None or arg is False for arg in args):
            return copy.deepcopy(self)

        # Convert to pandas.Index
        snapshots_ = as_index(self, snapshots, "snapshots", "snapshot")
        investment_periods_ = as_index(
            self, investment_periods, "investment_periods", None
        )

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
        (
            override_components,
            override_component_attrs,
        ) = self._retrieve_overridden_components()

        network = self.__class__(
            ignore_standard_types=ignore_standard_types,
            override_components=override_components,
            override_component_attrs=override_component_attrs,
        )

        # Copy components
        other_comps = sorted(self.all_components - {"Bus", "Carrier"})
        # Needs to copy buses and carriers first, since there are dependencies on them
        for component in self.iterate_components(["Bus", "Carrier"] + other_comps):
            # Drop the standard types to avoid them being read in twice
            if (
                not ignore_standard_types
                and component.name in self.standard_type_components
            ):
                df = component.df.drop(
                    network.components[component.name]["standard_types"].index
                )
            else:
                df = component.df

            import_components_from_dataframe(network, df, component.name)

        # Copy time-varying data, if given

        if len(snapshots_) > 0:
            network.set_snapshots(snapshots_)
            # Apply time-varying data
            for component in self.iterate_components():
                pnl = getattr(network, component.list_name + "_t")
                for k in component.pnl.keys():
                    try:
                        pnl[k] = component.pnl[k].loc[snapshots_].copy()
                    except KeyError:
                        pnl[k] = component.pnl[k].reindex(snapshots_).copy()

            # Apply investment periods
            if not investment_periods_.empty:
                network.set_investment_periods(investment_periods_)

            # Add weightings
            network.snapshot_weightings = self.snapshot_weightings.loc[
                snapshots_
            ].copy()
            network.investment_period_weightings = (
                self.investment_period_weightings.loc[investment_periods_].copy()
            )

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
                setattr(network, attr, getattr(self, attr))
            except AttributeError:
                pass

        return network

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
        network : pypsa.Network

        Examples
        --------
        >>> sub_network_0 = network[network.buses.sub_network = "0"]

        >>> sub_network_0_with_only_10_snapshots = network[:10, network.buses.sub_network = "0"]

        """
        if isinstance(key, tuple):
            time_i, key = key
        else:
            time_i = slice(None)

        (
            override_components,
            override_component_attrs,
        ) = self._retrieve_overridden_components()
        n = self.__class__(
            override_components=override_components,
            override_component_attrs=override_component_attrs,
        )
        n.import_components_from_dataframe(
            pd.DataFrame(self.buses.loc[key]).assign(sub_network=""), "Bus"
        )
        buses_i = n.buses.index

        rest_components = (
            self.all_components
            - self.standard_type_components
            - self.one_port_components
            - self.branch_components
        )
        for c in rest_components - {"Bus", "SubNetwork"}:
            n.import_components_from_dataframe(pd.DataFrame(self.df(c)), c)

        for c in self.standard_type_components:
            df = self.df(c).drop(self.components[c]["standard_types"].index)
            n.import_components_from_dataframe(pd.DataFrame(df), c)

        for c in self.one_port_components:
            df = self.df(c).loc[lambda df: df.bus.isin(buses_i)]
            n.import_components_from_dataframe(pd.DataFrame(df), c)

        for c in self.branch_components:
            df = self.df(c).loc[
                lambda df: df.bus0.isin(buses_i) & df.bus1.isin(buses_i)
            ]
            n.import_components_from_dataframe(pd.DataFrame(df), c)

        n.set_snapshots(self.snapshots[time_i])
        for c in self.all_components:
            i = n.df(c).index
            try:
                npnl = n.pnl(c)
                pnl = self.pnl(c)

                for k in pnl:
                    npnl[k] = pnl[k].loc[time_i, i.intersection(pnl[k].columns)]
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
            (self.df(c) for c in self.branch_components),
            keys=self.branch_components,
            sort=True,
            names=["component", "name"],
        )

    def passive_branches(self) -> pd.DataFrame:
        return pd.concat(
            (self.df(c) for c in self.passive_branch_components),
            keys=self.passive_branch_components,
            sort=True,
        )

    def controllable_branches(self) -> pd.DataFrame:
        return pd.concat(
            (self.df(c) for c in self.controllable_branch_components),
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
            c.df["sub_network"] = c.df.bus0.map(self.buses["sub_network"])

            if investment_period is not None:
                active = get_active_assets(self, c.name, investment_period)
                # set non active assets to NaN
                c.df.loc[~active, "sub_network"] = np.nan

        for sub in self.sub_networks.obj:
            find_cycles(sub)
            sub.find_bus_controls()

    def iterate_components(
        self, components: Collection[str] | None = None, skip_empty: bool = True
    ) -> Iterator[Component]:
        if components is None:
            components = self.all_components

        return (
            Component(
                name=c,
                list_name=self.components[c]["list_name"],
                attrs=self.components[c]["attrs"],
                df=self.df(c),
                pnl=self.pnl(c),
                ind=None,
            )
            for c in components
            if not (skip_empty and self.df(c).empty)
        )

    def consistency_check(self, check_dtypes: bool = False) -> None:
        """
        Checks the network for consistency; e.g. that all components are
        connected to existing buses and that no impedances are singular.

        Prints warnings if anything is potentially inconsistent.

        Examples
        --------
        >>> network.consistency_check()

        """
        self.calculate_dependent_values()

        # TODO: Check for bidirectional links with efficiency < 1.
        # TODO: Warn if any ramp limits are 0.

        # Per component checks
        for c in self.iterate_components():
            # Checks all components
            check_for_unknown_buses(self, c)
            check_for_unknown_carriers(self, c)
            check_time_series(self, c)
            check_static_power_attributes(self, c)
            check_time_series_power_attributes(self, c)
            check_nans_for_component_default_attrs(self, c)
            # Checks passive_branch_components
            check_for_zero_impedances(self, c)
            # Checks transformers
            check_for_zero_s_nom(c)
            # Checks generators and links
            check_assets(self, c)
            # Checks generators
            check_generators(c)

            if check_dtypes:
                check_dtypes_(c)

        # Combined checks
        check_for_disconnected_buses(self)
        check_investment_periods(self)
        check_shapes(self)


class SubNetwork(Common):
    """
    Connected network of electric buses (AC or DC) with passive flows or
    isolated non-electric buses.

    Generated by network.determine_network_topology().
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

    def buses_i(self) -> pd.Index:
        return self.network.buses.index[self.network.buses.sub_network == self.name]

    def lines_i(self) -> pd.Index:
        return self.network.lines.index[self.network.lines.sub_network == self.name]

    def transformers_i(self) -> pd.Index:
        return self.network.transformers.index[
            self.network.transformers.sub_network == self.name
        ]

    def branches_i(self) -> pd.MultiIndex:
        types = []
        names = []
        for c in self.iterate_components(self.network.passive_branch_components):
            types += len(c.ind) * [c.name]
            names += list(c.ind)
        return pd.MultiIndex.from_arrays([types, names], names=("type", "name"))

    def branches(self) -> pd.DataFrame:
        branches = self.network.passive_branches()
        return branches[branches.sub_network == self.name]

    def generators_i(self) -> pd.Index:
        sub_networks = self.network.generators.bus.map(self.network.buses.sub_network)
        return self.network.generators.index[sub_networks == self.name]

    def loads_i(self) -> pd.Index:
        sub_networks = self.network.loads.bus.map(self.network.buses.sub_network)
        return self.network.loads.index[sub_networks == self.name]

    def shunt_impedances_i(self) -> pd.Index:
        sub_networks = self.network.shunt_impedances.bus.map(
            self.network.buses.sub_network
        )
        return self.network.shunt_impedances.index[sub_networks == self.name]

    def storage_units_i(self) -> pd.Index:
        sub_networks = self.network.storage_units.bus.map(
            self.network.buses.sub_network
        )
        return self.network.storage_units.index[sub_networks == self.name]

    def stores_i(self) -> pd.Index:
        sub_networks = self.network.stores.bus.map(self.network.buses.sub_network)
        return self.network.stores.index[sub_networks == self.name]

    def buses(self) -> pd.DataFrame:
        return self.network.buses.loc[self.buses_i()]

    def generators(self) -> pd.DataFrame:
        return self.network.generators.loc[self.generators_i()]

    def loads(self) -> pd.DataFrame:
        return self.network.loads.loc[self.loads_i()]

    def shunt_impedances(self) -> pd.DataFrame:
        return self.network.shunt_impedances.loc[self.shunt_impedances_i()]

    def storage_units(self) -> pd.DataFrame:
        return self.network.storage_units.loc[self.storage_units_i()]

    def iterate_components(
        self, components: Collection[str] | None = None, skip_empty: bool = True
    ) -> Iterator[Component]:
        """
        Iterate over components of the subnetwork and extract corresponding
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
        for c in self.network.iterate_components(
            components=components, skip_empty=False
        ):
            ind = getattr(self, c.list_name + "_i")()
            pnl = Dict()
            for k, v in c.pnl.items():
                pnl[k] = v[ind.intersection(v.columns)]
            c = Component(c.name, c.list_name, c.attrs, c.df.loc[ind], pnl, ind)
            if not (skip_empty and len(ind) == 0):
                yield c
