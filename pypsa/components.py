# -*- coding: utf-8 -*-
"""
Power system components.
"""


from weakref import ref

from pypsa.clustering import ClusteringAccessor

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2024 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import os
import sys
from collections import namedtuple
from pathlib import Path
from typing import List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import validators
from deprecation import deprecated
from pyproj import CRS, Transformer
from scipy.sparse import csgraph

from pypsa.contingency import calculate_BODF, network_lpf_contingency, network_sclopf
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
from pypsa.plot import iplot, plot
from pypsa.statistics import StatisticsAccessor

if sys.version_info.major >= 3:
    from pypsa.linopf import network_lopf as network_lopf_lowmem

if sys.version_info < (3, 12):
    from pypsa.opf import network_lopf, network_opf
else:

    def network_lopf(*args, **kwargs):
        raise NotImplementedError(
            "Function `network_lopf` not available from Python 3.12."
        )

    def network_opf(*args, **kwargs):
        raise NotImplementedError(
            "Function `network_opf` not available from Python 3.12."
        )


import logging

logger = logging.getLogger(__name__)


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


class Basic(object):
    """
    Common to every object.
    """

    name = ""

    def __init__(self, name=""):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name}"


class Common(Basic):
    """
    Common to all objects inside Network object.
    """

    network = None

    def __init__(self, network, name=""):
        Basic.__init__(self, name)
        self._network = ref(network)

    @property
    def network(self):
        return self._network()


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
        See git repository examples/new_components/.
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

    _crs = CRS.from_epsg(4326)

    # methods imported from other sub-modules
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

    lpf = network_lpf

    pf = network_pf

    #    lopf = network_lopf

    opf = network_opf

    plot = plot

    iplot = iplot

    calculate_dependent_values = calculate_dependent_values

    lpf_contingency = network_lpf_contingency

    sclopf = network_sclopf

    graph = graph

    incidence_matrix = incidence_matrix

    adjacency_matrix = adjacency_matrix

    get_switchable_as_dense = get_switchable_as_dense

    get_extendable_i = get_extendable_i

    get_non_extendable_i = get_non_extendable_i

    get_committable_i = get_committable_i

    get_active_assets = get_active_assets

    def __init__(
        self,
        import_name=None,
        name="",
        ignore_standard_types=False,
        override_components=None,
        override_component_attrs=None,
        **kwargs,
    ):
        # Initialise root logger and set its level, if this has not been done before
        logging.basicConfig(level=logging.INFO)

        from pypsa import __version__ as pypsa_version

        Basic.__init__(self, name)

        # this will be saved on export
        self.pypsa_version = pypsa_version

        self._meta = {}

        self._snapshots = pd.Index(["now"])

        cols = ["objective", "stores", "generators"]
        self._snapshot_weightings = pd.DataFrame(1, index=self.snapshots, columns=cols)

        self._investment_periods = pd.Index([])

        cols = ["objective", "years"]
        self._investment_period_weightings = pd.DataFrame(columns=cols)

        self.optimize = OptimizationAccessor(self)

        self.cluster = ClusteringAccessor(self)

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

        self.statistics = StatisticsAccessor(self)

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
                str_b = attrs.typ == str
                attrs.loc[str_b, "default"] = attrs.loc[str_b, "default"].fillna("")
                for typ in (str, float, int):
                    typ_b = attrs.typ == typ
                    attrs.loc[typ_b, "default"] = attrs.loc[typ_b, "default"].astype(
                        typ
                    )

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
            elif import_name.is_dir():
                self.import_from_csv_folder(import_name)
            else:
                raise ValueError(
                    f"import_name '{import_name}' is not a valid .h5 file, .nc file or directory."
                )

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
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

    def __add__(self, other):
        """
        Merge all components of two networks.
        """
        self.merge(other)

    def _build_dataframes(self):
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

    def read_in_default_standard_types(self):
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

    def df(self, component_name):
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

    def pnl(self, component_name):
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
    def meta(self):
        """
        Dictionary of the network meta data.
        """
        return self._meta

    @meta.setter
    def meta(self, new):
        if not isinstance(new, (dict, Dict)):
            raise TypeError(f"Meta must be a dictionary, received a {type(new)}")
        self._meta = new

    @property
    def crs(self):
        """
        Coordinate reference system of the network's geometries (n.shapes).
        """
        return self._crs

    @crs.setter
    def crs(self, new):
        """
        Set the coordinate reference system of the network's geometries
        (n.shapes).
        """
        self.shapes.crs = new
        self._crs = self.shapes.crs

    def to_crs(self, new):
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
    def srid(self):
        """
        Spatial reference system identifier of the network's geometries
        (n.shapes).
        """
        return self.crs.to_epsg()

    @srid.setter
    def srid(self, new):
        """
        Set the spatial reference system identifier of the network's geometries
        (n.shapes).
        """
        self.crs = pyproj.CRS.from_epsg(new)

    def set_snapshots(
        self,
        snapshots: Union[List, pd.Index, pd.MultiIndex, pd.DatetimeIndex],
        default_snapshot_weightings: float = 1.0,
        weightings_from_timedelta: bool = False,
    ) -> None:
        """
        Set the snapshots/time steps and reindex all time-dependent data.

        Snapshot weightings, typically representing the hourly length of each snapshot, is filled with the `default_snapshot_weighintgs` value,
        or uses the timedelta of the snapshots if `weightings_from_timedelta` flag is True, and snapshots are of type `pd.DatetimeIndex`.

        This will reindex all components time-dependent DataFrames (:role:`Network.pnl`); NaNs are filled
        with the default value for that quantity.

        Parameters
        ----------
        snapshots : list, pandas.Index, pd.MultiIndex or pd.DatetimeIndex
            All time steps.
        default_snapshot_weightings: float
            The default weight for each snapshot. Defaults to 1.0.
        weightings_from_timedelta: bool
            Wheter to use the timedelta of `snapshots` as `snapshot_weightings` if `snapshots` is of type `pd.DatetimeIndex`.  Defaults to False.

        Returns
        -------
        None
        """
        if isinstance(snapshots, pd.MultiIndex):
            assert (
                snapshots.nlevels == 2
            ), "Maximally two levels of MultiIndex supported"
            snapshots = snapshots.rename(["period", "timestep"])
            snapshots.name = "snapshot"
            self._snapshots = snapshots
        else:
            self._snapshots = pd.Index(snapshots, name="snapshot")

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

            for k, default in attrs.default[attrs.varying].items():
                if pnl[k].empty:  # avoid expensive reindex operation
                    pnl[k].index = self._snapshots
                else:
                    pnl[k] = pnl[k].reindex(self._snapshots, fill_value=default)

        # NB: No need to rebind pnl to self, since haven't changed it

    snapshots = property(
        lambda self: self._snapshots, set_snapshots, doc="Time steps of the network"
    )

    @property
    def snapshot_weightings(self):
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
    def snapshot_weightings(self, df):
        assert df.index.equals(
            self.snapshots
        ), "Weightings not defined for all snapshots."
        if isinstance(df, pd.Series):
            logger.info("Applying weightings to all columns of `snapshot_weightings`")
            df = pd.DataFrame({c: df for c in self._snapshot_weightings.columns})
        self._snapshot_weightings = df

    def set_investment_periods(self, periods):
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
        periods = pd.Index(periods)
        if not (
            pd.api.types.is_integer_dtype(periods)
            and periods.is_unique
            and periods.is_monotonic_increasing
        ):
            raise ValueError(
                "Investment periods are not strictly increasing integers, "
                "which is required for multi-period investment optimisation."
            )
        if isinstance(self.snapshots, pd.MultiIndex):
            if not periods.isin(self.snapshots.unique("period")).all():
                raise ValueError(
                    "Not all investment periods are in level `period` " "of snapshots."
                )
            if len(periods) < len(self.snapshots.unique(level="period")):
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
                attrs = self.components[component]["attrs"]

                for k, default in attrs.default[attrs.varying].items():
                    pnl[k] = pd.concat({p: pnl[k] for p in periods}, names=names)
                    pnl[k].index.name = "snapshot"

            self._snapshots = pd.MultiIndex.from_product(
                [periods, self.snapshots], names=names
            )
            self._snapshots.name = "snapshot"
            self._snapshot_weightings = pd.concat(
                {p: self.snapshot_weightings for p in periods}, names=names
            )
            self._snapshot_weightings.index.name = "snapshot"

        self._investment_periods = periods
        self.investment_period_weightings = self.investment_period_weightings.reindex(
            periods, fill_value=1.0
        ).astype(float)

    investment_periods = property(
        lambda self: self._investment_periods,
        set_investment_periods,
        doc="Investment steps during the optimization.",
    )

    @property
    def investment_period_weightings(self):
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
    def investment_period_weightings(self, df):
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

    @deprecated(
        deprecated_in="0.24",
        removed_in="1.0",
        details="Use linopy-based function ``n.optimize()`` instead. Migrate extra functionalities: https://pypsa.readthedocs.io/en/latest/examples/optimization-with-linopy-migrate-extra-functionalities.html.",
    )
    def lopf(
        self,
        snapshots=None,
        pyomo=False,
        solver_name="glpk",
        solver_options={},
        solver_logfile=None,
        formulation="kirchhoff",
        transmission_losses=0,
        keep_files=False,
        extra_functionality=None,
        multi_investment_periods=False,
        **kwargs,
    ):
        """
        Linear optimal power flow for a group of snapshots.

        Parameters
        ----------
        snapshots : list or index slice
            A list of snapshots to optimise, must be a subset of
            network.snapshots, defaults to network.snapshots
        pyomo : bool, default False
            Whether to use pyomo for building and solving the model, setting
            this to False saves a lot of memory and time.
        solver_name : string
            Must be a solver name that pyomo recognises and that is
            installed, e.g. "glpk", "gurobi"
        solver_options : dictionary
            A dictionary with additional options that get passed to the solver.
            (e.g. {'threads':2} tells gurobi to use only 2 cpus)
        solver_logfile : None|string
            If not None, sets the logfile option of the solver.
        keep_files : bool, default False
            Keep the files that pyomo constructs from OPF problem
            construction, e.g. .lp file - useful for debugging
        formulation : string
            Formulation of the linear power flow equations to use; must be
            one of ["angles", "cycles", "kirchhoff", "ptdf"]
        transmission_losses : int
            Whether an approximation of transmission losses should be included
            in the linearised power flow formulation. A passed number will denote
            the number of tangents used for the piecewise linear approximation.
            Defaults to 0, which ignores losses.
        extra_functionality : callable function
            This function must take two arguments
            `extra_functionality(network, snapshots)` and is called after
            the model building is complete, but before it is sent to the
            solver. It allows the user to
            add/change constraints and add/change the objective function.
        multi_investment_periods : bool, default False
            Whether to optimise as a single investment period or to optimise in multiple
            investment periods. Then, snapshots should be a ``pd.MultiIndex``.

        Other Parameters
        ----------------
        ptdf_tolerance : float
            Only taking effect when pyomo is True.
            Value below which PTDF entries are ignored
        free_memory : set, default {'pyomo'}
            Only taking effect when pyomo is True.
            Any subset of {'pypsa', 'pyomo'}. Allows to stash `pypsa` time-series
            data away while the solver runs (as a pickle to disk) and/or free
            `pyomo` data after the solution has been extracted.
        solver_io : string, default None
            Only taking effect when pyomo is True.
            Solver Input-Output option, e.g. "python" to use "gurobipy" for
            solver_name="gurobi"
        skip_pre : bool, default False
            Only taking effect when pyomo is True.
            Skip the preliminary steps of computing topology, calculating
            dependent values and finding bus controls.
        extra_postprocessing : callable function
            Only taking effect when pyomo is True.
            This function must take three arguments
            `extra_postprocessing(network, snapshots, duals)` and is called after
            the model has solved and the results are extracted. It allows the user
            to extract further information about the solution, such as additional
            shadow prices.
        skip_objective : bool, default False
            Only taking effect when pyomo is False.
            Skip writing the default objective function. If False, a custom
            objective has to be defined via extra_functionality.
        warmstart : bool or string, default False
            Only taking effect when pyomo is False.
            Use this to warmstart the optimization. Pass a string which gives
            the path to the basis file. If set to True, a path to
            a basis file must be given in network.basis_fn.
        store_basis : bool, default True
            Only taking effect when pyomo is False.
            Whether to store the basis of the optimization results. If True,
            the path to the basis file is saved in network.basis_fn. Note that
            a basis can only be stored if simplex, dual-simplex, or barrier
            *with* crossover is used for solving.
        keep_references : bool, default False
            Only taking effect when pyomo is False.
            Keep the references of variable and constraint names withing the
            network. These can be looked up in `n.vars` and `n.cons` after solving.
        keep_shadowprices : bool or list of component names
            Only taking effect when pyomo is False.
            Keep shadow prices for all constraints, if set to True. If a list
            is passed the shadow prices will only be parsed for those constraint
            names. Defaults to ['Bus', 'Line', 'GlobalConstraint'].
            After solving, the shadow prices can be retrieved using
            :func:`pypsa.linopt.get_dual` with corresponding name
        solver_dir : str, default None
            Only taking effect when pyomo is False.
            Path to directory where necessary files are written, default None leads
            to the default temporary directory used by tempfile.mkstemp().

        Returns
        -------
        status : str
            Status of optimization.
            Either "ok" if solution is optimal, or "warning" if not.
        termination_condition : str
            More information on how the solver terminated.
            One of "optimal", "suboptimal" (in which case a solution is still
            provided), "infeasible", "infeasible or unbounded", or "other".
        """
        args = {
            "snapshots": snapshots,
            "keep_files": keep_files,
            "solver_options": solver_options,
            "formulation": formulation,
            "transmission_losses": transmission_losses,
            "extra_functionality": extra_functionality,
            "multi_investment_periods": multi_investment_periods,
            "solver_name": solver_name,
            "solver_logfile": solver_logfile,
        }
        args.update(kwargs)

        if not self.shunt_impedances.empty:
            logger.warning(
                "You have defined one or more shunt impedances. "
                "Shunt impedances are ignored by the linear optimal "
                "power flow (LOPF)."
            )

        if pyomo:
            return network_lopf(self, **args)
        return network_lopf_lowmem(self, **args)

    def add(self, class_name, name, **kwargs):
        """
        Add a single component to the network.

        Adds it to component DataFrame.

        Any attributes which are not specified will be given the default
        value from :doc:`components`.

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
        assert class_name in self.components, f"Component class {class_name} not found"

        cls_df = self.df(class_name)
        cls_pnl = self.pnl(class_name)

        name = str(name)

        assert name not in cls_df.index, (
            f"Failed to add {class_name} component {name} because there is already "
            f"an object with this name in {self.components[class_name]['list_name']}"
        )

        if class_name == "Link":
            update_linkports_component_attrs(self, kwargs.keys())

        attrs = self.components[class_name]["attrs"]

        static_attrs = attrs[attrs.static].drop("name")

        # This guarantees that the correct attribute type is maintained
        obj_df = pd.DataFrame(
            data=[static_attrs.default], index=[name], columns=static_attrs.index
        )
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

        for attr in ["bus", "bus0", "bus1"]:
            if attr in new_df.columns:
                bus_name = new_df.at[name, attr]
                if bus_name not in self.buses.index:
                    logger.warning(
                        f"The bus name `{bus_name}` given for {attr} "
                        f"of {class_name} `{name}` does not appear "
                        "in network.buses"
                    )

    def remove(self, class_name, name):
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

    def madd(self, class_name, names, suffix="", **kwargs):
        """
        Add multiple components to the network, along with their attributes.

        Make sure when adding static attributes as pandas Series that they are indexed
        by names. Make sure when adding time-varying attributes as pandas DataFrames that
        their index is a superset of network.snapshots and their columns are a
        subset of names.

        Any attributes which are not specified will be given the default
        value from :doc:`components`.

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
        --------
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

    def mremove(self, class_name, names):
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

    def _retrieve_overridden_components(self):
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
        with_time=True,
        snapshots=None,
        investment_periods=None,
        ignore_standard_types=False,
    ):
        """
        Returns a deep copy of the Network object with all components and time-
        dependent data.

        Returns
        --------
        network : pypsa.Network

        Parameters
        ----------
        with_time : boolean, default True
            Copy snapshots and time-varying network.component_names_t data too.
        snapshots : list or index slice
            A list of snapshots to copy, must be a subset of
            network.snapshots, defaults to network.snapshots
        ignore_standard_types : boolean, default False
            Ignore the PyPSA standard types.

        Examples
        --------
        >>> network_copy = network.copy()
        """
        (
            override_components,
            override_component_attrs,
        ) = self._retrieve_overridden_components()

        network = self.__class__(
            ignore_standard_types=ignore_standard_types,
            override_components=override_components,
            override_component_attrs=override_component_attrs,
        )

        other_comps = sorted(self.all_components - {"Bus", "Carrier"})
        for component in self.iterate_components(["Bus", "Carrier"] + other_comps):
            df = component.df
            # drop the standard types to avoid them being read in twice
            if (
                not ignore_standard_types
                and component.name in self.standard_type_components
            ):
                df = component.df.drop(
                    network.components[component.name]["standard_types"].index
                )

            import_components_from_dataframe(network, df, component.name)

        if with_time:
            if snapshots is None:
                snapshots = self.snapshots
            if investment_periods is None:
                investment_periods = self.investment_period_weightings.index
            network.set_snapshots(snapshots)
            if not investment_periods.empty:
                network.set_investment_periods(investment_periods)
            for component in self.iterate_components():
                pnl = getattr(network, component.list_name + "_t")
                for k in component.pnl.keys():
                    pnl[k] = component.pnl[k].loc[snapshots].copy()
            network.snapshot_weightings = self.snapshot_weightings.loc[snapshots].copy()
            network.investment_period_weightings = (
                self.investment_period_weightings.loc[investment_periods].copy()
            )

        # catch all remaining attributes of network
        for attr in ["name", "srid"]:
            setattr(network, attr, getattr(self, attr))

        return network

    def __getitem__(self, key):
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
        --------
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
    def branches(self):
        return pd.concat(
            (self.df(c) for c in self.branch_components),
            keys=self.branch_components,
            sort=True,
            names=["component", "name"],
        )

    def passive_branches(self):
        return pd.concat(
            (self.df(c) for c in self.passive_branch_components),
            keys=self.passive_branch_components,
            sort=True,
        )

    def controllable_branches(self):
        return pd.concat(
            (self.df(c) for c in self.controllable_branch_components),
            keys=self.controllable_branch_components,
            sort=True,
        )

    def determine_network_topology(
        self, investment_period=None, skip_isolated_buses=False
    ):
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

    def iterate_components(self, components=None, skip_empty=True):
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

    def consistency_check(self, check_dtypes=False):
        """
        Checks the network for consistency; e.g. that all components are
        connected to existing buses and that no impedances are singular.

        Prints warnings if anything is potentially inconsistent.

        Examples
        --------
        >>> network.consistency_check()
        """
        # TODO: Check for bidirectional links with efficiency < 1.
        # TODO: Warn if any ramp limits are 0.

        self.calculate_dependent_values()

        def bus_columns(df):
            return df.columns[df.columns.str.startswith("bus")]

        # check for unknown buses
        for c in self.iterate_components():
            for attr in bus_columns(c.df):
                missing = ~c.df[attr].isin(self.buses.index)
                # if bus2, bus3... contain empty strings do not warn
                if c.name in self.branch_components:
                    if int(attr[-1]) > 1:
                        missing &= c.df[attr] != ""
                if missing.any():
                    msg = "The following %s have %s which are not defined:\n%s"
                    logger.warning(msg, c.list_name, attr, c.df.index[missing])

        # check for disconnected buses
        connected_buses = set()
        for c in self.iterate_components():
            for attr in bus_columns(c.df):
                connected_buses.update(c.df[attr])

        disconnected_buses = set(self.buses.index) - connected_buses
        if disconnected_buses:
            msg = "The following buses have no attached components, which can break the lopf:\n%s"
            logger.warning(msg, disconnected_buses)

        for c in self.iterate_components(self.passive_branch_components):
            for attr in ["x", "r"]:
                bad = c.df[attr] == 0
                if bad.any():
                    msg = (
                        "The following %s have zero %s, which "
                        "could break the linear load flow:\n%s"
                    )
                    logger.warning(msg, c.list_name, attr, c.df.index[bad])

        for c in self.iterate_components({"Transformer"}):
            bad = c.df["s_nom"] == 0
            if bad.any():
                logger.warning(
                    "The following %s have zero s_nom, which is used "
                    "to define the impedance and will thus break "
                    "the load flow:\n%s",
                    c.list_name,
                    c.df.index[bad],
                )

        for c in self.iterate_components():
            for attr in c.attrs.index[c.attrs.varying & c.attrs.static]:
                attr_df = c.pnl[attr]

                diff = attr_df.columns.difference(c.df.index)
                if len(diff):
                    logger.warning(
                        "The following %s have time series defined "
                        "for attribute %s in network.%s_t, but are "
                        "not defined in network.%s:\n%s",
                        c.list_name,
                        attr,
                        c.list_name,
                        c.list_name,
                        diff,
                    )

                if not self.snapshots.equals(attr_df.index):
                    logger.warning(
                        "The index of the time-dependent Dataframe for attribute "
                        "%s of network.%s_t is not aligned with network snapshots",
                        attr,
                        c.list_name,
                    )

        static_attrs = ["p_nom", "s_nom", "e_nom"]
        varying_attrs = ["p_max_pu", "e_max_pu"]
        for c in self.iterate_components(self.all_components - {"TransformerType"}):
            varying_attr = c.attrs.query("varying").index.intersection(varying_attrs)
            static_attr = c.attrs.query("static").index.intersection(static_attrs)

            if len(static_attr):
                attr = static_attr[0]
                bad = c.df[attr + "_max"] < c.df[attr + "_min"]
                if bad.any():
                    logger.warning(
                        "The following %s have smaller maximum than "
                        "minimum expansion limit which can lead to "
                        "infeasibilty:\n%s",
                        c.list_name,
                        c.df.index[bad],
                    )

                attr = static_attr[0]
                for col in [attr + "_min", attr + "_max"]:
                    if c.df[col][c.df[attr + "_extendable"]].isnull().any():
                        logger.warning(
                            "Encountered nan's in column %s of component '%s'.",
                            col,
                            c.name,
                        )

            if len(varying_attr):
                attr = varying_attr[0][0]
                max_pu = self.get_switchable_as_dense(c.name, attr + "_max_pu")
                min_pu = self.get_switchable_as_dense(c.name, attr + "_min_pu")

                # check for NaN values:
                if max_pu.isnull().values.any():
                    for col in max_pu.columns[max_pu.isnull().any()]:
                        logger.warning(
                            "The attribute %s of element %s of %s has "
                            "NaN values for the following snapshots:\n%s",
                            attr + "_max_pu",
                            col,
                            c.list_name,
                            max_pu.index[max_pu[col].isnull()],
                        )
                if min_pu.isnull().values.any():
                    for col in min_pu.columns[min_pu.isnull().any()]:
                        logger.warning(
                            "The attribute %s of element %s of %s has "
                            "NaN values for the following snapshots:\n%s",
                            attr + "_min_pu",
                            col,
                            c.list_name,
                            min_pu.index[min_pu[col].isnull()],
                        )

                # check for infinite values
                if np.isinf(max_pu).values.any():
                    for col in max_pu.columns[np.isinf(max_pu).any()]:
                        logger.warning(
                            "The attribute %s of element %s of %s has "
                            "infinite values for the following snapshots:\n%s",
                            attr + "_max_pu",
                            col,
                            c.list_name,
                            max_pu.index[np.isinf(max_pu[col])],
                        )
                if np.isinf(min_pu).values.any():
                    for col in min_pu.columns[np.isinf(min_pu).any()]:
                        logger.warning(
                            "The attribute %s of element %s of %s has "
                            "infinite values for the following snapshots:\n%s",
                            attr + "_min_pu",
                            col,
                            c.list_name,
                            min_pu.index[np.isinf(min_pu[col])],
                        )

                diff = max_pu - min_pu
                diff = diff[diff < 0].dropna(axis=1, how="all")
                for col in diff.columns:
                    logger.warning(
                        "The element %s of %s has a smaller maximum "
                        "than minimum operational limit which can "
                        "lead to infeasibility for the following snapshots:\n%s",
                        col,
                        c.list_name,
                        diff[col].dropna().index,
                    )

            if c.name in {"Generator", "Link"}:
                committables = self.get_committable_i(c.name)
                extendables = self.get_extendable_i(c.name)
                intersection = committables.intersection(extendables)
                if not intersection.empty:
                    logger.warning(
                        "Assets can only be committable or extendable. Found "
                        f"assets in component {c} which are both:"
                        f"\n\n\t{', '.join(intersection)}"
                    )

            if c.name in {"Generator"}:
                bad_uc_gens = c.df.index[
                    c.df.committable
                    & (c.df.up_time_before > 0)
                    & (c.df.down_time_before > 0)
                ]
                if not bad_uc_gens.empty:
                    logger.warning(
                        "The following committable generators were both up and down"
                        f" before the simulation: {bad_uc_gens}."
                        " This could cause an infeasibility."
                    )

        # check all dtypes of component attributes

        if check_dtypes:
            for c in self.iterate_components():
                # first check static attributes

                dtypes_soll = c.attrs.loc[c.attrs["static"], "dtype"].drop("name")
                unmatched = c.df.dtypes[dtypes_soll.index] != dtypes_soll

                if unmatched.any():
                    logger.warning(
                        "The following attributes of the dataframe %s "
                        "have the wrong dtype:\n%s\n"
                        "They are:\n%s\nbut should be:\n%s",
                        c.list_name,
                        unmatched.index[unmatched],
                        c.df.dtypes[dtypes_soll.index[unmatched]],
                        dtypes_soll[unmatched],
                    )

                # now check varying attributes

                types_soll = c.attrs.loc[c.attrs["varying"], ["typ", "dtype"]]

                for attr, typ, dtype in types_soll.itertuples():
                    if c.pnl[attr].empty:
                        continue

                    unmatched = c.pnl[attr].dtypes != dtype

                    if unmatched.any():
                        logger.warning(
                            "The following columns of time-varying attribute "
                            "%s in %s_t have the wrong dtype:\n%s\n"
                            "They are:\n%s\nbut should be:\n%s",
                            attr,
                            c.list_name,
                            unmatched.index[unmatched],
                            c.pnl[attr].dtypes[unmatched],
                            typ,
                        )

        constraint_periods = set(
            self.global_constraints.investment_period.dropna().unique()
        )
        if isinstance(self.snapshots, pd.MultiIndex):
            if not constraint_periods.issubset(self.snapshots.unique("period")):
                raise ValueError(
                    "The global constraints contain investment periods which are "
                    "not in the set of optimized snapshots."
                )
        else:
            if constraint_periods:
                raise ValueError(
                    "The global constraints contain investment periods but snapshots are "
                    "not multi-indexed."
                )

        shape_components = self.shapes.component.unique()
        for c in set(shape_components) & set(self.all_components):
            geos = self.shapes.query("component == @c")
            not_included = geos.index[~geos.idx.isin(self.df(c).index)]

            if not not_included.empty:
                logger.warning(
                    f"The following shapes are related to component {c} and have"
                    f" idx values that are not included in the component's index:\n"
                    f"{not_included}"
                )


class SubNetwork(Common):
    """
    Connected network of electric buses (AC or DC) with passive flows or
    isolated non-electric buses.

    Generated by network.determine_network_topology().
    """

    list_name = "sub_networks"

    lpf = sub_network_lpf

    pf = sub_network_pf

    find_bus_controls = find_bus_controls

    find_slack_bus = find_slack_bus

    calculate_Y = calculate_Y

    calculate_PTDF = calculate_PTDF

    calculate_B_H = calculate_B_H

    calculate_BODF = calculate_BODF

    graph = graph

    incidence_matrix = incidence_matrix

    adjacency_matrix = adjacency_matrix

    def buses_i(self):
        return self.network.buses.index[self.network.buses.sub_network == self.name]

    def lines_i(self):
        return self.network.lines.index[self.network.lines.sub_network == self.name]

    def transformers_i(self):
        return self.network.transformers.index[
            self.network.transformers.sub_network == self.name
        ]

    def branches_i(self):
        types = []
        names = []
        for c in self.iterate_components(self.network.passive_branch_components):
            types += len(c.ind) * [c.name]
            names += list(c.ind)
        return pd.MultiIndex.from_arrays([types, names], names=("type", "name"))

    def branches(self):
        branches = self.network.passive_branches()
        return branches[branches.sub_network == self.name]

    def generators_i(self):
        sub_networks = self.network.generators.bus.map(self.network.buses.sub_network)
        return self.network.generators.index[sub_networks == self.name]

    def loads_i(self):
        sub_networks = self.network.loads.bus.map(self.network.buses.sub_network)
        return self.network.loads.index[sub_networks == self.name]

    def shunt_impedances_i(self):
        sub_networks = self.network.shunt_impedances.bus.map(
            self.network.buses.sub_network
        )
        return self.network.shunt_impedances.index[sub_networks == self.name]

    def storage_units_i(self):
        sub_networks = self.network.storage_units.bus.map(
            self.network.buses.sub_network
        )
        return self.network.storage_units.index[sub_networks == self.name]

    def stores_i(self):
        sub_networks = self.network.stores.bus.map(self.network.buses.sub_network)
        return self.network.stores.index[sub_networks == self.name]

    def buses(self):
        return self.network.buses.loc[self.buses_i()]

    def generators(self):
        return self.network.generators.loc[self.generators_i()]

    def loads(self):
        return self.network.loads.loc[self.loads_i()]

    def shunt_impedances(self):
        return self.network.shunt_impedances.loc[self.shunt_impedances_i()]

    def storage_units(self):
        return self.network.storage_units.loc[self.storage_units_i()]

    def iterate_components(self, components=None, skip_empty=True):
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
