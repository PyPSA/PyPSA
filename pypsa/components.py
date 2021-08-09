
## Copyright 2015-2021 PyPSA Developers

## You can find the list of PyPSA Developers at
## https://pypsa.readthedocs.io/en/latest/developers.html

## PyPSA is released under the open source MIT License, see
## https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt

"""Power system components.
"""


from weakref import ref

__author__ = "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
__copyright__ = ("Copyright 2015-2021 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
                 "MIT License")

import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from collections import namedtuple
import os


from .descriptors import (Dict, get_switchable_as_dense, get_active_assets,
                          get_extendable_i, get_non_extendable_i)

from .io import (export_to_csv_folder, import_from_csv_folder,
                 export_to_hdf5, import_from_hdf5,
                 export_to_netcdf, import_from_netcdf,
                 import_from_pypower_ppc, import_components_from_dataframe,
                 import_series_from_dataframe, import_from_pandapower_net)

from .pf import (network_lpf, sub_network_lpf, network_pf,
                 sub_network_pf, find_bus_controls, find_slack_bus, find_cycles,
                 calculate_Y, calculate_PTDF, calculate_B_H,
                 calculate_dependent_values)

from .contingency import (calculate_BODF, network_lpf_contingency,
                          network_sclopf)


from .opf import network_lopf, network_opf

from .plot import plot, iplot

from .graph import graph, incidence_matrix, adjacency_matrix

import sys

if sys.version_info.major >= 3:
    from .linopf import network_lopf as network_lopf_lowmem

import logging
logger = logging.getLogger(__name__)



dir_name = os.path.dirname(__file__)
component_attrs_dir_name = "component_attrs"

standard_types_dir_name = "standard_types"


inf = float("inf")


components = pd.read_csv(os.path.join(dir_name,
                                      "components.csv"),
                         index_col=0)

component_attrs = Dict()

for component in components.index:
    file_name = os.path.join(dir_name,
                             component_attrs_dir_name,
                             components.at[component,"list_name"] + ".csv")
    component_attrs[component] = pd.read_csv(file_name, index_col=0, na_values="n/a")

del component

class Basic(object):
    """Common to every object."""

    name = ""


    def __init__(self, name=""):
        self.name = name

    def __repr__(self):
        return "%s %s" % (self.__class__.__name__, self.name)




class Common(Basic):
    """Common to all objects inside Network object."""
    network = None

    def __init__(self, network, name=""):
        Basic.__init__(self, name)
        self._network = ref(network)

    @property
    def network(self):
        return self._network()


Component = namedtuple("Component",
                       ['name', 'list_name', 'attrs', 'df', 'pnl', 'ind'])

class Network(Basic):
    """
    Network container for all buses, one-ports and branches.

    Parameters
    ----------
    import_name : string
        Name of netCDF file, HDF5 .h5 store or folder from which to import CSVs
        of network data.
    name : string, default ""
        Network name.
    ignore_standard_types : boolean, default False
        If True, do not read in PyPSA standard types into standard types
        DataFrames.
    csv_folder_name : string
        Name of folder from which to import CSVs of network data. Overrides
        import_name.
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

    """

    #Spatial Reference System Identifier (SRID), defaults to longitude and latitude
    srid = 4326

    #methods imported from other sub-modules

    import_from_csv_folder = import_from_csv_folder

    export_to_csv_folder = export_to_csv_folder

    import_from_hdf5 = import_from_hdf5

    export_to_hdf5 = export_to_hdf5

    import_from_netcdf = import_from_netcdf

    export_to_netcdf = export_to_netcdf

    import_from_pypower_ppc = import_from_pypower_ppc

    import_from_pandapower_net = import_from_pandapower_net

    import_components_from_dataframe = import_components_from_dataframe

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

    get_active_assets = get_active_assets



    def __init__(self, import_name=None, name="", ignore_standard_types=False,
                 override_components=None, override_component_attrs=None,
                 **kwargs):

        # Initialise root logger and set its level, if this has not been done before
        logging.basicConfig(level=logging.INFO)

        from . import __version__ as pypsa_version

        Basic.__init__(self, name)

        #this will be saved on export
        self.pypsa_version = pypsa_version

        self._snapshots = pd.Index(["now"])

        cols = ["objective", "stores", "generators"]
        self._snapshot_weightings = pd.DataFrame(1, index=self.snapshots, columns=cols)

        self._investment_periods = pd.Index([])

        cols = ["objective", "years"]
        self._investment_period_weightings = pd.DataFrame(columns=cols)

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
                setattr(self, c_type + "_components",
                        set(self.components.index[self.components.type == c_type]))

        self.one_port_components = self.passive_one_port_components|self.controllable_one_port_components

        self.branch_components = self.passive_branch_components|self.controllable_branch_components

        self.all_components = set(self.components.index) - {"Network"}

        self.components = Dict(self.components.T.to_dict())

        for component in self.components:
            #make copies to prevent unexpected sharing of variables
            attrs = self.component_attrs[component].copy()

            attrs['static'] = (attrs['type'] != 'series')
            attrs['varying'] = attrs['type'].isin({'series', 'static or series'})
            attrs['typ'] = attrs['type'].map({'boolean': bool, 'int': int, 'string': str}).fillna(float)
            attrs['dtype'] = attrs['type'].map({'boolean': np.dtype(bool), 'int': np.dtype(int),
                                                'string': np.dtype('O')}).fillna(np.dtype(float))

            bool_b = attrs.type == 'boolean'
            attrs.loc[bool_b, 'default'] = attrs.loc[bool_b].isin({True, 'True'})

            #exclude Network because it's not in a DF and has non-typical attributes
            if component != "Network":
                attrs.loc[attrs.typ == str, "default"] = (
                    attrs.loc[attrs.typ == str, "default"].replace({np.nan: ""}))
                for typ in (str, float, int):
                    attrs.loc[attrs.typ == typ, "default"] = (
                        attrs.loc[attrs.typ == typ, "default"].astype(typ))

            self.components[component]["attrs"] = attrs

        self._build_dataframes()

        if not ignore_standard_types:
            self.read_in_default_standard_types()

        if import_name is not None:
            if import_name[-3:] == ".h5":
                self.import_from_hdf5(import_name)
            elif import_name[-3:] == ".nc":
                self.import_from_netcdf(import_name)
            else:
                self.import_from_csv_folder(import_name)

        for key, value in kwargs.items():
            setattr(self, key, value)


    def _build_dataframes(self):
        """Function called when network is created to build component pandas.DataFrames."""

        for component in self.all_components:

            attrs = self.components[component]["attrs"]

            static_dtypes = attrs.loc[attrs.static, "dtype"].drop(["name"])

            df = pd.DataFrame({k: pd.Series(dtype=d) for k, d in static_dtypes.iteritems()},
                              columns=static_dtypes.index)

            df.index.name = "name"

            setattr(self,self.components[component]["list_name"],df)

            #it's currently hard to imagine non-float series,
            # but this could be generalised
            pnl = Dict({k : pd.DataFrame(index=self.snapshots,
                                         columns=[],
                                         dtype=np.dtype(float))
                        for k in attrs.index[attrs.varying]})

            setattr(self,self.components[component]["list_name"]+"_t",pnl)


    def read_in_default_standard_types(self):

        for std_type in self.standard_type_components:

            list_name = self.components[std_type]["list_name"]

            file_name = os.path.join(dir_name,
                                     standard_types_dir_name,
                                     list_name + ".csv")

            self.components[std_type]["standard_types"] = (
                pd.read_csv(file_name, index_col=0))

            self.import_components_from_dataframe(
                self.components[std_type]["standard_types"], std_type)


    def df(self, component_name):
        """
        Return the DataFrame of static components for component_name,
        i.e. network.component_names

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
        Return the dictionary of DataFrames of varying components for component_name,
        i.e. network.component_names_t

        Parameters
        ----------
        component_name : string

        Returns
        -------
        dict of pandas.DataFrame
        """
        return getattr(self, self.components[component_name]["list_name"]+"_t")



    def set_snapshots(self, value):
        """
        Set the snapshots and reindex all time-dependent data.

        This will reindex all pandas.Panels of time-dependent data; NaNs are filled
        with the default value for that quantity.

        Parameters
        ----------
        snapshots : list or pandas.Index
            All time steps.

        Returns
        -------
        None
        """
        if isinstance(value, pd.MultiIndex):
            assert value.nlevels == 2, "Maximally two levels of MultiIndex supported"
            self._snapshots = value.rename(['period', 'snapshot'])
        else:
            self._snapshots = pd.Index(value, name='snapshot')

        self.snapshot_weightings = (
            self.snapshot_weightings.reindex(self._snapshots, fill_value=1.))

        for component in self.all_components:
            pnl = self.pnl(component)
            attrs = self.components[component]["attrs"]

            for k,default in attrs.default[attrs.varying].iteritems():
                pnl[k] = pnl[k].reindex(self._snapshots).fillna(default)

        #NB: No need to rebind pnl to self, since haven't changed it

    snapshots = property(lambda self: self._snapshots, set_snapshots,
                         doc="Time steps of the network")


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
        assert df.index.equals(self.snapshots), "Weightings not defined for all snapshots."
        if isinstance(df, pd.Series):
            logger.info('Applying weightings to all columns of `snapshot_weightings`')
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
        if not (periods.is_integer() and periods.is_unique and periods.is_monotonic_increasing):
            raise ValueError("Investment periods are not strictly increasing integers, "
                        "which is required for multi-period investment optimisation.")
        if isinstance(self.snapshots, pd.MultiIndex):
            if not periods.isin(self.snapshots.unique('period')).all():
                raise ValueError("Not all investment periods are in level `period` "
                                 "of snapshots.")
            if len(periods) < len(self.snapshots.levels[0]):
                raise NotImplementedError("Investment periods do not equal first level "
                                          "values of snapshots.")
        else:
            # Convenience case:
            logger.info("Repeating time-series for each investment period and "
                        "converting snapshots to a pandas.MultiIndex.")
            for component in self.all_components:
                pnl = self.pnl(component)
                attrs = self.components[component]["attrs"]

                for k,default in attrs.default[attrs.varying].iteritems():
                    pnl[k] = pd.concat({p: pnl[k] for p in periods})

            self._snapshots = pd.MultiIndex.from_product([periods, self.snapshots],
                                                      names=['period', 'snapshot'])
            self._snapshot_weightings = pd.concat({p: self.snapshot_weightings for p in periods})

        self._investment_periods = periods
        self.investment_period_weightings = (
            self.investment_period_weightings.reindex(periods, fill_value=1.).astype(float))



    investment_periods = property(lambda self: self._investment_periods,
                                  set_investment_periods,
                                  doc="Investment steps during the optimization.")


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
        assert df.index.equals(self.investment_periods), (
                "Weightings not defined for all investment periods.")
        if isinstance(df, pd.Series):
            logger.info('Applying weightings to all columns of `investment_period_weightings`')
            df = pd.DataFrame({c: df for c in self._investment_period_weightings.columns})
        self._investment_period_weightings = df



    def lopf(self, snapshots=None, pyomo=True, solver_name="glpk",
             solver_options={}, solver_logfile=None, formulation="kirchhoff",
             keep_files=False, extra_functionality=None,
             multi_investment_periods=False,  **kwargs):
        """
        Linear optimal power flow for a group of snapshots.

        Parameters
        ----------
        snapshots : list or index slice
            A list of snapshots to optimise, must be a subset of
            network.snapshots, defaults to network.snapshots
        pyomo : bool, default True
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
            one of ["angles","cycles","kirchhoff","ptdf"]
        extra_functionality : callable function
            This function must take two arguments
            `extra_functionality(network,snapshots)` and is called after
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
            `extra_postprocessing(network,snapshots,duals)` and is called after
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
        args = {'snapshots': snapshots, 'keep_files': keep_files,
                'solver_options': solver_options, 'formulation': formulation,
                'extra_functionality': extra_functionality,
                'multi_investment_periods': multi_investment_periods,
                'solver_name': solver_name, 'solver_logfile': solver_logfile}
        args.update(kwargs)

        if not self.shunt_impedances.empty:
            logger.warning("You have defined one or more shunt impedances. "
                           "Shunt impedances are ignored by the linear optimal "
                           "power flow (LOPF).")

        if pyomo:
            return network_lopf(self, **args)
        else:
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
        >>> network.add("Bus","my_bus_0")
        >>> network.add("Bus","my_bus_1",v_nom=380)
        >>> network.add("Line","my_line_name",bus0="my_bus_0",bus1="my_bus_1",length=34,r=2,x=4)
        """

        assert class_name in self.components, "Component class {} not found".format(class_name)

        cls_df = self.df(class_name)
        cls_pnl = self.pnl(class_name)

        name = str(name)

        assert name not in cls_df.index, (
            f"Failed to add {class_name} component {name} because there is already "
            f"an object with this name in {self.components[class_name]['list_name']}")

        attrs = self.components[class_name]["attrs"]

        static_attrs = attrs[attrs.static].drop("name")

        #This guarantees that the correct attribute type is maintained
        obj_df = pd.DataFrame(data=[static_attrs.default], index=[name],
                              columns=static_attrs.index)
        new_df = cls_df.append(obj_df, sort=False)

        setattr(self, self.components[class_name]["list_name"], new_df)

        for k,v in kwargs.items():
            if k not in attrs.index:
                logger.warning(f"{class_name} has no attribute {k}, "
                               "ignoring this passed value.")
                continue
            typ = attrs.at[k, "typ"]
            if not attrs.at[k,"varying"]:
                new_df.at[name,k] = typ(v)
            elif attrs.at[k,"static"] and not isinstance(v, (pd.Series, pd.DataFrame, np.ndarray, list)):
                new_df.at[name,k] = typ(v)
            else:
                cls_pnl[k][name] = pd.Series(data=v, index=self.snapshots, dtype=typ)


        for attr in ["bus","bus0","bus1"]:
            if attr in new_df.columns:
                bus_name = new_df.at[name,attr]
                if bus_name not in self.buses.index:
                    logger.warning(f"The bus name `{bus_name}` given for {attr} "
                                   f"of {class_name} `{name}` does not appear "
                                   "in network.buses")


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
        >>> network.remove("Line","my_line 12345")

        """

        if class_name not in self.components:
            logger.error("Component class {} not found".format(class_name))
            return None

        cls_df = self.df(class_name)

        cls_df.drop(name, inplace=True)

        pnl = self.pnl(class_name)

        for df in pnl.values():
            if name in df:
                df.drop(name, axis=1, inplace=True)



    def madd(self, class_name, names, suffix='', **kwargs):
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
            Component attributes, e.g. x=[0.1,0.2], can be list, pandas.Series
            of pandas.DataFrame for time-varying

        Returns
        --------
        new_names : pandas.index
            Names of new components (including suffix)

        Examples
        --------

        Short Example:

        >>> network.madd("Load", ["load 1", "load 2"],
        ...        bus=["1","2"],
        ...        p_set=np.random.rand(len(network.snapshots),2))

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
        ...        p_set=np.random.rand(len(snapshots),len(buses)))
        >>> # add wind availability as pandas DataFrame
        >>> wind = pd.DataFrame(np.random.rand(len(snapshots),len(buses)),
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
            logger.error("Component class {} not found".format(class_name))
            return None

        if not isinstance(names, pd.Index):
            names = pd.Index(names)

        new_names = names.astype(str) + suffix

        static = {}; series = {}
        for k, v in kwargs.items():
            if isinstance(v, pd.DataFrame):
                series[k] = v.rename(columns=lambda i: str(i)+suffix)
            elif isinstance(v, pd.Series):
                static[k] = v.rename(lambda i: str(i)+suffix)
            elif (isinstance(v, np.ndarray) and
                  v.shape == (len(self.snapshots), len(names))):
                series[k] = pd.DataFrame(v, index=self.snapshots, columns=new_names)
            else:
                static[k] = v

        self.import_components_from_dataframe(
            pd.DataFrame(static, index=new_names), class_name)

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
            logger.error("Component class {} not found".format(class_name))
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

        cols = ["list_name","description","type"]

        override_components = pd.DataFrame([[self.components[i][c] for c in cols]
                                            for i in components_index],
                                           columns=cols,
                                           index=components_index)

        override_component_attrs = Dict({i : self.component_attrs[i].copy()
                                         for i in components_index})

        return override_components, override_component_attrs


    def copy(self, with_time=True, snapshots=None, investment_periods=None,
             ignore_standard_types=False):
        """
        Returns a deep copy of the Network object with all components and
        time-dependent data.

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
        override_components, override_component_attrs = self._retrieve_overridden_components()

        network = self.__class__(ignore_standard_types=ignore_standard_types,
                                 override_components=override_components,
                                 override_component_attrs=override_component_attrs)

        other_comps = sorted(self.all_components - {"Bus","Carrier"})
        for component in self.iterate_components(["Bus", "Carrier"] + other_comps):
            df = component.df
            #drop the standard types to avoid them being read in twice
            if not ignore_standard_types and component.name in self.standard_type_components:
                df = component.df.drop(network.components[component.name]["standard_types"].index)

            import_components_from_dataframe(network, df, component.name)

        if with_time:
            if snapshots is None:
                snapshots = self.snapshots
            if investment_periods is None:
                investment_periods = self.investment_period_weightings.index
            network.set_snapshots(snapshots)
            if not investment_periods.empty:
                network.set_investment_periods(self.investment_periods)
            for component in self.iterate_components():
                pnl = getattr(network, component.list_name+"_t")
                for k in component.pnl.keys():
                    pnl[k] = component.pnl[k].loc[snapshots].copy()
            network.snapshot_weightings = self.snapshot_weightings.loc[snapshots].copy()
            network.investment_period_weightings = self.investment_period_weightings.loc[investment_periods].copy()


        #catch all remaining attributes of network
        for attr in ["name", "srid"]:
            setattr(network,attr,getattr(self,attr))

        return network

    def __getitem__(self, key):
        """
        Returns a shallow slice of the Network object containing only
        the selected buses and all the connected components.

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

        override_components, override_component_attrs = (
            self._retrieve_overridden_components())
        n = self.__class__(override_components=override_components,
                           override_component_attrs=override_component_attrs)
        n.import_components_from_dataframe(
            pd.DataFrame(self.buses.loc[key]).assign(sub_network=""),
            "Bus"
        )
        buses_i = n.buses.index

        rest_components = (self.all_components - self.standard_type_components -
                           self.one_port_components - self.branch_components)
        for c in rest_components - {"Bus", "SubNetwork"}:
            n.import_components_from_dataframe(pd.DataFrame(self.df(c)), c)

        for c in self.standard_type_components:
            df = self.df(c).drop(self.components[c]["standard_types"].index)
            n.import_components_from_dataframe(pd.DataFrame(df), c)

        for c in self.one_port_components:
            df = self.df(c).loc[lambda df: df.bus.isin(buses_i)]
            n.import_components_from_dataframe(pd.DataFrame(df), c)

        for c in self.branch_components:
            df = self.df(c).loc[lambda df: df.bus0.isin(buses_i) & df.bus1.isin(buses_i)]
            n.import_components_from_dataframe(pd.DataFrame(df), c)

        n.set_snapshots(self.snapshots[time_i])
        for c in self.all_components:
            i = n.df(c).index
            try:
                npnl = n.pnl(c)
                pnl = self.pnl(c)

                for k in pnl:
                    npnl[k] = pnl[k].loc[time_i,i.intersection(pnl[k].columns)]
            except AttributeError:
                pass

        # catch all remaining attributes of network
        for attr in ["name", "srid"]:
            setattr(n,attr,getattr(self, attr))

        n.snapshot_weightings = self.snapshot_weightings.loc[time_i]

        return n


    #beware, this turns bools like s_nom_extendable into objects because of
    #presence of links without s_nom_extendable
    def branches(self):
        return pd.concat((self.df(c) for c in self.branch_components),
                         keys=self.branch_components, sort=True,
                         names=['component', 'name'])

    def passive_branches(self):
        return pd.concat((self.df(c) for c in self.passive_branch_components),
                         keys=self.passive_branch_components, sort=True)

    def controllable_branches(self):
        return pd.concat((self.df(c) for c in self.controllable_branch_components),
                         keys=self.controllable_branch_components, sort=True)

    def determine_network_topology(self, investment_period=None):
        """
        Build sub_networks from topology.

        For the default case investment_period=None, it is not taken into
        account whether the branch components are active
        (based on build_year and lifetime).
        If the investment_period is specified, the network topology is
        determined on the basis of the active branches.
        """

        adjacency_matrix = self.adjacency_matrix(branch_components=self.passive_branch_components,
                                                 investment_period=investment_period)
        n_components, labels = csgraph.connected_components(adjacency_matrix, directed=False)

        # remove all old sub_networks
        for sub_network in self.sub_networks.index:
            obj = self.sub_networks.at[sub_network,"obj"]
            self.remove("SubNetwork", sub_network)
            del obj

        for i in np.arange(n_components):
            # index of first bus
            buses_i = (labels == i).nonzero()[0]
            carrier = self.buses.carrier.iat[buses_i[0]]

            if carrier not in ["AC","DC"] and len(buses_i) > 1:
                logger.warning(f"Warning, sub network {i} is not electric but "
                               "contains multiple buses\nand branches. Passive "
                               "flows are not allowed for non-electric networks!")

            if (self.buses.carrier.iloc[buses_i] != carrier).any():
                logger.warning(f"Warning, sub network {i} contains buses with "
                               "mixed carriers! Value counts:"
                               f"\n{self.buses.carrier.iloc[buses_i].value_counts()}")

            self.add("SubNetwork", i, carrier=carrier)

        #add objects
        self.sub_networks["obj"] = [SubNetwork(self, name) for name in self.sub_networks.index]

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

        return (Component(name=c,
                          list_name=self.components[c]["list_name"],
                          attrs=self.components[c]["attrs"],
                          df=self.df(c),
                          pnl=self.pnl(c),
                          ind=None)
                for c in components
                if not (skip_empty and self.df(c).empty))


    def consistency_check(self):
        """
        Checks the network for consistency; e.g.
        that all components are connected to existing buses and
        that no impedances are singular.

        Prints warnings if anything is potentially inconsistent.

        Examples
        --------
        >>> network.consistency_check()

        """


        for c in self.iterate_components(self.one_port_components):
            missing = c.df.index[~c.df.bus.isin(self.buses.index)]
            if len(missing) > 0:
                logger.warning("The following %s have buses which are not defined:\n%s",
                               c.list_name, missing)

        for c in self.iterate_components(self.branch_components):
            for attr in ["bus0","bus1"]:
                missing = c.df.index[~c.df[attr].isin(self.buses.index)]
                if len(missing) > 0:
                    logger.warning("The following %s have %s which are not defined:\n%s",
                                   c.list_name, attr, missing)

        def bad_by_type(branch, attr):
            if branch.type not in self.line_types.index:
                return True
            elif self.line_types.loc[branch.type, attr+'_per_length'] * branch.length == 0.:
                return True
            else:
                return False

        for c in self.iterate_components(self.passive_branch_components):
            for attr in ["x","r"]:
                bad = c.df.index[(c.df[attr] == 0.) &
                                 c.df.apply(bad_by_type, args=(attr,), axis=1)]
                if len(bad) > 0:
                    logger.warning("The following %s have zero %s, which "
                                   "could break the linear load flow:\n%s",
                                   c.list_name, attr, bad)

            bad = c.df.index[(c.df["x"] == 0.) & (c.df["r"] == 0.) &
                             c.df.apply(bad_by_type, args=('x',), axis=1) &
                             c.df.apply(bad_by_type, args=('r',), axis=1)]
            if len(bad) > 0:
                logger.warning("The following %s have zero series impedance, "
                               "which will break the load flow:\n%s",
                               c.list_name, bad)


        for c in self.iterate_components({"Transformer"}):
            bad = c.df.index[c.df["s_nom"] == 0.]
            if len(bad) > 0:
                logger.warning("The following %s have zero s_nom, which is used "
                               "to define the impedance and will thus break "
                               "the load flow:\n%s",
                               c.list_name, bad)


        for c in self.iterate_components(self.all_components):
            for attr in c.attrs.index[c.attrs.varying & c.attrs.static]:
                attr_df = c.pnl[attr]

                diff = attr_df.columns.difference(c.df.index)
                if len(diff) > 0:
                    logger.warning("The following %s have time series defined "
                                   "for attribute %s in network.%s_t, but are "
                                   "not defined in network.%s:\n%s",
                                   c.list_name, attr, c.list_name, c.list_name, diff)

                diff = self.snapshots.difference(attr_df.index)
                if len(diff) > 0:
                    logger.warning("In the time-dependent Dataframe for attribute "
                                   "%s of network.%s_t the following snapshots "
                                   "are missing:\n%s",
                                   attr, c.list_name, diff)

                diff = attr_df.index.difference(self.snapshots)
                if len(diff) > 0:
                    logger.warning("In the time-dependent Dataframe for attribute "
                                   "%s of network.%s_t the following snapshots "
                                   "are defined which are not in network.snapshots:\n%s",
                                   attr, c.list_name, diff)

        static_attrs = ['p_nom', 's_nom', 'e_nom']
        varying_attrs = ['p_max_pu', 'e_max_pu']
        for c in self.iterate_components(self.all_components - {'TransformerType'}):
            varying_attr = c.attrs.index[c.attrs.varying].intersection(varying_attrs)
            static_attr = c.attrs.index[c.attrs.static].intersection(static_attrs)

            if len(static_attr):
                diff = (getattr(self, c.list_name)[static_attr[0] + "_max"] -
                        getattr(self, c.list_name)[static_attr[0] + "_min"])
                if not diff[diff < 0].empty:
                    logger.warning("The following %s have smaller maximum than "
                                   "minimum expansion limit which can lead to "
                                   "infeasibilty:\n%s",
                                   c.list_name, diff[diff < 0].index)

            if len(varying_attr):
                max_pu = get_switchable_as_dense(self, c.name,
                                                 varying_attr[0][0] + "_max_pu")
                min_pu = get_switchable_as_dense(self, c.name,
                                                 varying_attr[0][0] + "_min_pu")

                # check for NaN values:
                if max_pu.isnull().values.any():
                    for col in max_pu.columns[max_pu.isnull().any()]:
                        logger.warning("The attribute %s of element %s of %s has "
                                       "NaN values for the following snapshots:\n%s",
                                       varying_attr[0][0] + "_max_pu", col,
                                       c.list_name,
                                       max_pu.index[max_pu[col].isnull()])
                if min_pu.isnull().values.any():
                    for col in min_pu.columns[min_pu.isnull().any()]:
                        logger.warning("The attribute %s of element %s of %s has "
                                       "NaN values for the following snapshots:\n%s",
                                       varying_attr[0][0] + "_min_pu", col,
                                       c.list_name, min_pu.index[min_pu[col].isnull()])

                # check for infinite values
                if np.isinf(max_pu).values.any():
                    for col in max_pu.columns[np.isinf(max_pu).any()]:
                        logger.warning("The attribute %s of element %s of %s has "
                                       "infinite values for the following snapshots:\n%s",
                                       varying_attr[0][0] + "_max_pu", col,
                                       c.list_name, max_pu.index[np.isinf(max_pu[col])])
                if np.isinf(min_pu).values.any():
                    for col in min_pu.columns[np.isinf(min_pu).any()]:
                        logger.warning("The attribute %s of element %s of %s has "
                                       "infinite values for the following snapshots:\n%s",
                                       varying_attr[0][0] + "_min_pu", col,
                                       c.list_name, min_pu.index[np.isinf(min_pu[col])])

                diff = max_pu - min_pu
                diff = diff[diff < 0].dropna(axis=1, how='all')
                for col in diff.columns:
                    logger.warning("The element %s of %s has a smaller maximum "
                                   "than minimum operational limit which can "
                                   "lead to infeasibility for the following snapshots:\n%s",
                                   col, c.list_name, diff[col].dropna().index)

        #check all dtypes of component attributes

        for c in self.iterate_components():

            #first check static attributes

            dtypes_soll = c.attrs.loc[c.attrs["static"], "dtype"].drop("name")
            unmatched = (c.df.dtypes[dtypes_soll.index] != dtypes_soll)

            if unmatched.any():
                logger.warning("The following attributes of the dataframe %s "
                               "have the wrong dtype:\n%s\n"
                               "They are:\n%s\nbut should be:\n%s",
                               c.list_name,
                               unmatched.index[unmatched],
                               c.df.dtypes[dtypes_soll.index[unmatched]],
                               dtypes_soll[unmatched])

            #now check varying attributes

            types_soll = c.attrs.loc[c.attrs["varying"], ["typ", "dtype"]]

            for attr, typ, dtype in types_soll.itertuples():
                if c.pnl[attr].empty:
                    continue

                unmatched = (c.pnl[attr].dtypes != dtype)

                if unmatched.any():
                    logger.warning("The following columns of time-varying attribute "
                                   "%s in %s_t have the wrong dtype:\n%s\n"
                                   "They are:\n%s\nbut should be:\n%s",
                                   attr,c.list_name,
                                   unmatched.index[unmatched],
                                   c.pnl[attr].dtypes[unmatched],
                                   typ)

class SubNetwork(Common):
    """
    Connected network of electric buses (AC or DC) with passive flows
    or isolated non-electric buses.

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
        return self.network.transformers.index[self.network.transformers.sub_network == self.name]

    def branches_i(self):
        types = []
        names = []
        for c in self.iterate_components(self.network.passive_branch_components):
            types += len(c.ind) * [c.name]
            names += list(c.ind)
        return pd.MultiIndex.from_arrays([types, names], names=('type', 'name'))

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
        sub_networks = self.network.shunt_impedances.bus.map(self.network.buses.sub_network)
        return self.network.shunt_impedances.index[sub_networks == self.name]

    def storage_units_i(self):
        sub_networks = self.network.storage_units.bus.map(self.network.buses.sub_network)
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
        for c in self.network.iterate_components(components=components, skip_empty=False):
            c = Component(*c[:-1], ind=getattr(self, c.list_name + '_i')())
            if not (skip_empty and len(c.ind) == 0):
                yield c
