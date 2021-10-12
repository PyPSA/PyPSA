
"""Functions for importing and exporting data.
"""

import logging
logger = logging.getLogger(__name__)

import os
from textwrap import dedent
from glob import glob

import pandas as pd
import numpy as np
import math

try:
    import xarray as xr
    has_xarray = True
except ImportError:
    has_xarray = False

class ImpExper(object):
    ds = None

    def __enter__(self):
        if self.ds is not None:
            self.ds = self.ds.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finish()

        if self.ds is not None:
            self.ds.__exit__(exc_type, exc_val, exc_tb)

    def finish(self):
        pass

class Exporter(ImpExper):
    def remove_static(self, list_name):
        pass

    def remove_series(self, list_name, attr):
        pass

class Importer(ImpExper):
    pass

class ExporterCSV(Exporter):
    def __init__(self, csv_folder_name, encoding):
        self.csv_folder_name = csv_folder_name
        self.encoding = encoding

        #make sure directory exists
        if not os.path.isdir(csv_folder_name):
            logger.warning("Directory {} does not exist, creating it"
                           .format(csv_folder_name))
            os.mkdir(csv_folder_name)

    def save_attributes(self, attrs):
        name = attrs.pop('name')
        df = pd.DataFrame(attrs, index=pd.Index([name], name='name'))
        fn = os.path.join(self.csv_folder_name, "network.csv")
        df.to_csv(fn, encoding=self.encoding)

    def save_snapshots(self, snapshots):
        fn = os.path.join(self.csv_folder_name, "snapshots.csv")
        snapshots.to_csv(fn, encoding=self.encoding)

    def save_investment_periods(self, investment_periods):
        fn = os.path.join(self.csv_folder_name, "investment_periods.csv")
        investment_periods.to_csv(fn, encoding=self.encoding)

    def save_static(self, list_name, df):
        fn = os.path.join(self.csv_folder_name, list_name + ".csv")
        df.to_csv(fn, encoding=self.encoding)

    def save_series(self, list_name, attr, df):
        fn = os.path.join(self.csv_folder_name, list_name + "-" + attr + ".csv")
        df.to_csv(fn, encoding=self.encoding)

    def remove_static(self, list_name):
        fns = glob(os.path.join(self.csv_folder_name, list_name) + "*.csv")
        if fns:
            for fn in fns: os.unlink(fn)
            logger.warning("Stale csv file(s) {} removed".format(', '.join(fns)))

    def remove_series(self, list_name, attr):
        fn = os.path.join(self.csv_folder_name, list_name + "-" + attr + ".csv")
        if os.path.exists(fn):
            os.unlink(fn)

def import_from_csv_folder(network, csv_folder_name, encoding=None, skip_time=False):
    """
    Import network data from CSVs in a folder.
    The CSVs must follow the standard form, see ``pypsa/examples``.
    Parameters
    ----------
    csv_folder_name : string
        Name of folder
    encoding : str, default None
        Encoding to use for UTF when reading (ex. 'utf-8'). `List of Python standard encodings
        <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
    skip_time : bool, default False
        Skip reading in time dependent attributes
    Examples
    ----------
    >>> network.import_from_csv_folder(csv_folder_name)
    """

    basename = os.path.basename(csv_folder_name)
    with ImporterCSV(csv_folder_name, encoding=encoding) as importer:
        _import_from_importer(network, importer, basename=basename, skip_time=skip_time)

class ImporterCSV(Importer):

    def __init__(self, csv_folder_name, encoding):
        self.csv_folder_name = csv_folder_name
        self.encoding = encoding

        assert os.path.isdir(csv_folder_name), f"Directory {csv_folder_name} does not exist."

    def get_attributes(self):
        fn = os.path.join(self.csv_folder_name, "network.csv")
        if not os.path.isfile(fn):
            return None
        
        return dict(pd.read_csv(fn, encoding=self.encoding).iloc[0])

    def get_snapshots(self):
        fn = os.path.join(self.csv_folder_name, "snapshots.csv")
        if not os.path.isfile(fn): return None
        df = pd.read_csv(fn, index_col=0, encoding=self.encoding, parse_dates=True)
        if "snapshot" in df:
            df["snapshot"] = pd.to_datetime(df.snapshot)
        return df

    def get_investment_periods(self):
        fn = os.path.join(self.csv_folder_name, "investment_periods.csv")
        if not os.path.isfile(fn): return None
        return pd.read_csv(fn, index_col=0, encoding=self.encoding)

    def get_static(self, list_name):
        fn = os.path.join(self.csv_folder_name, list_name + ".csv")
        return (pd.read_csv(fn, index_col=0, encoding=self.encoding)
                if os.path.isfile(fn) else None)

    def get_series(self, list_name):
        for fn in os.listdir(self.csv_folder_name):
            if fn.startswith(list_name+"-") and fn.endswith(".csv"):
                attr = fn[len(list_name)+1:-4]
                df = pd.read_csv(os.path.join(self.csv_folder_name, fn),
                                 index_col=0, encoding=self.encoding, parse_dates=True)
                yield attr, df

def _import_from_importer(network, importer, basename, skip_time=False):
    """
    Import network data from importer.
    Parameters
    ----------
    skip_time : bool
        Skip importing time
    """

    # If network.csv exists, attributes loaded as dict from csv:
    # name,now,srid,pypsa_version, e.g. AC-DC,now,4326,0.10.0

    attrs = importer.get_attributes()

    current_pypsa_version = [int(s) for s in network.pypsa_version.split(".")]
    pypsa_version = None

    if attrs is not None:
        network.name = attrs.pop('name')
        try:
            pypsa_version = [int(s) for s in attrs.pop("pypsa_version").split(".")]
        except KeyError:
            pypsa_version = None

        for attr, val in attrs.items():
            setattr(network, attr, val)

    ##https://docs.python.org/3/tutorial/datastructures.html#comparing-sequences-and-other-types
    if pypsa_version is None or pypsa_version < current_pypsa_version:
        logger.warning(dedent("""
                Importing PyPSA from older version of PyPSA than current version {}.
                Please read the release notes at https://pypsa.org/doc/release_notes.html
                carefully to prepare your network for import.
        """).format(network.pypsa_version))

    importer.pypsa_version = pypsa_version
    importer.current_pypsa_version = current_pypsa_version

    # if there is snapshots.csv, read in snapshot data
    df = importer.get_snapshots()

    if df is not None:

        # check if imported snapshots have MultiIndex
        snapshot_levels = set(["period", "snapshot"]).intersection(df.columns)
        if snapshot_levels:
            df.set_index(sorted(snapshot_levels), inplace=True)
        network.set_snapshots(df.index)

        cols = ['objective', 'generators', 'stores']
        if not df.columns.intersection(cols).empty:
            network.snapshot_weightings = df.reindex(index=network.snapshots,
                                                     columns=cols)
        elif "weightings" in df.columns:
            network.snapshot_weightings = df["weightings"].reindex(network.snapshots)

        network.set_snapshots(df.index)

    # read in investment period weightings
    periods = importer.get_investment_periods()

    if periods is not None:
        network._investment_periods = periods.index

        network._investment_period_weightings = (
            periods.reindex(network.investment_periods))


    imported_components = []

    # now read in other components; make sure buses and carriers come first
    for component in ["Bus", "Carrier"] + sorted(network.all_components - {"Bus", "Carrier", "SubNetwork"}):
        list_name = network.components[component]["list_name"]

        df = importer.get_static(list_name)
        if df is None:
            if component == "Bus":
                logger.error("Error, no buses found")
                return
            else:
                continue

        import_components_from_dataframe(network, df, component)

        if not skip_time:
            for attr, df in importer.get_series(list_name):
                df.set_index(network.snapshots, inplace=True)
                import_series_from_dataframe(network, df, component, attr)

        logger.debug(getattr(network,list_name))

        imported_components.append(list_name)

    logger.info("Imported network{} has {}".format(" " + basename, ", ".join(imported_components)))

def export_to_csv_folder(network, csv_folder_name, encoding=None, export_standard_types=False):
    """
    Export network and components to a folder of CSVs.
    Both static and series attributes of all components are exported, but only
    if they have non-default values.
    If ``csv_folder_name`` does not already exist, it is created.
    Static attributes are exported in one CSV file per component,
    e.g. ``generators.csv``.
    Series attributes are exported in one CSV file per component per
    attribute, e.g. ``generators-p_set.csv``.
    Parameters
    ----------
    csv_folder_name : string
        Name of folder to which to export.
    encoding : str, default None
        Encoding to use for UTF when reading (ex. 'utf-8'). `List of Python
        standard encodings
        <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
    export_standard_types : boolean, default False
        If True, then standard types are exported too (upon reimporting you
        should then set "ignore_standard_types" when initialising the network).
    Examples
    --------
    >>> network.export_to_csv_folder(csv_folder_name)
    """

    basename = os.path.basename(csv_folder_name)
    with ExporterCSV(csv_folder_name=csv_folder_name, encoding=encoding) as exporter:
        _export_to_exporter(network, exporter, basename=basename,
                            export_standard_types=export_standard_types)

def _export_to_exporter(network, exporter, basename, export_standard_types=False):
    """
    Export to exporter.
    Both static and series attributes of components are exported, but only
    if they have non-default values.
    Parameters
    ----------
    exporter : Exporter
        Initialized exporter instance
    basename : str
        Basename, used for logging
    export_standard_types : boolean, default False
        If True, then standard types are exported too (upon reimporting you
        should then set "ignore_standard_types" when initialising the netowrk).
    """

    #exportable component types
    #what about None???? - nan is float?
    allowed_types = (float, int, bool, str) + tuple(np.sctypeDict.values())

    #first export network properties
    attrs = dict((attr, getattr(network, attr))
                 for attr in dir(network)
                 if (not attr.startswith("__") and
                     isinstance(getattr(network,attr), allowed_types)))
    exporter.save_attributes(attrs)

    #now export snapshots
    if isinstance(network.snapshot_weightings.index, pd.MultiIndex):
        network.snapshot_weightings.index.rename(["period", "snapshot"], inplace=True)
    else:
        network.snapshot_weightings.index.rename("snapshot", inplace=True)
    snapshots = network.snapshot_weightings.reset_index()
    exporter.save_snapshots(snapshots)

    # export investment period weightings
    investment_periods = network.investment_period_weightings
    exporter.save_investment_periods(investment_periods)

    exported_components = []
    for component in network.all_components - {"SubNetwork"}:

        list_name = network.components[component]["list_name"]
        attrs = network.components[component]["attrs"]

        df = network.df(component)
        pnl = network.pnl(component)

        if not export_standard_types and component in network.standard_type_components:
            df = df.drop(network.components[component]["standard_types"].index)

        # first do static attributes
        df.index.name = "name"
        if df.empty:
            exporter.remove_static(list_name)
            continue

        col_export = []
        for col in df.columns:
            # do not export derived attributes
            if col in ["sub_network", "r_pu", "x_pu", "g_pu", "b_pu"]:
                continue
            if col in attrs.index and pd.isnull(attrs.at[col, "default"]) and pd.isnull(df[col]).all():
                continue
            if (col in attrs.index
                and df[col].dtype == attrs.at[col, 'dtype']
                and (df[col] == attrs.at[col, "default"]).all()):
                continue

            col_export.append(col)

        exporter.save_static(list_name, df[col_export])

        #now do varying attributes
        for attr in pnl:
            if attr not in attrs.index:
                col_export = pnl[attr].columns
            else:
                default = attrs.at[attr, "default"]

                if pd.isnull(default):
                    col_export = pnl[attr].columns[(~pd.isnull(pnl[attr])).any()]
                else:
                    col_export = pnl[attr].columns[(pnl[attr] != default).any()]

            if len(col_export) > 0:
                df = pnl[attr].reset_index()[col_export]
                exporter.save_series(list_name, attr, df)
            else:
                exporter.remove_series(list_name, attr)

        exported_components.append(list_name)

    logger.info("Exported network {} has {}".format(basename, ", ".join(exported_components)))

def import_components_from_dataframe(network, dataframe, cls_name):
    """
    Import components from a pandas DataFrame.
    If columns are missing then defaults are used.
    If extra columns are added, these are left in the resulting component dataframe.
    Parameters
    ----------
    dataframe : pandas.DataFrame
        A DataFrame whose index is the names of the components and
        whose columns are the non-default attributes.
    cls_name : string
        Name of class of component, e.g. ``"Line","Bus","Generator", "StorageUnit"``
    Examples
    --------
    >>> import pandas as pd
    >>> buses = ['Berlin', 'Frankfurt', 'Munich', 'Hamburg']
    >>> network.import_components_from_dataframe(
            pd.DataFrame({"v_nom" : 380, "control" : 'PV'},
			index=buses),
			"Bus")
    >>> network.import_components_from_dataframe(
            pd.DataFrame({"carrier" : "solar", "bus" : buses, "p_nom_extendable" : True},
			index=[b+" PV" for b in buses]),
			"Generator")
    See Also
    --------
    pypsa.Network.madd
    """

    attrs = network.components[cls_name]["attrs"]

    static_attrs = attrs[attrs.static].drop("name")
    non_static_attrs = attrs[~attrs.static]

    # Clean dataframe and ensure correct types
    dataframe = pd.DataFrame(dataframe)
    dataframe.index = dataframe.index.astype(str)

    for k in static_attrs.index:
        if k not in dataframe.columns:
            dataframe[k] = static_attrs.at[k, "default"]
        else:
            if static_attrs.at[k, "type"] == 'string':
                dataframe[k] = dataframe[k].replace({np.nan: ""})

            dataframe[k] = dataframe[k].astype(static_attrs.at[k, "typ"])

    #check all the buses are well-defined
    for attr in ["bus", "bus0", "bus1"]:
        if attr in dataframe.columns:
            missing = dataframe.index[~dataframe[attr].isin(network.buses.index)]
            if len(missing) > 0:
                logger.warning("The following %s have buses which are not defined:\n%s",
                               cls_name, missing)

    non_static_attrs_in_df = non_static_attrs.index.intersection(dataframe.columns)
    old_df = network.df(cls_name)
    new_df = dataframe.drop(non_static_attrs_in_df, axis=1)
    if not old_df.empty:
        new_df = pd.concat((old_df, new_df), sort=False)

    if not new_df.index.is_unique:
        logger.error("Error, new components for {} are not unique".format(cls_name))
        return

    setattr(network, network.components[cls_name]["list_name"], new_df)

    #now deal with time-dependent properties

    pnl = network.pnl(cls_name)

    for k in non_static_attrs_in_df:
        #If reading in outputs, fill the outputs
        pnl[k] = pnl[k].reindex(columns=new_df.index,
                                fill_value=non_static_attrs.at[k, "default"])
        pnl[k].loc[:,dataframe.index] = dataframe.loc[:,k].values

    setattr(network,network.components[cls_name]["list_name"]+"_t",pnl)

def import_series_from_dataframe(network, dataframe, cls_name, attr):
    """
    Import time series from a pandas DataFrame.
    Parameters
    ----------
    dataframe : pandas.DataFrame
        A DataFrame whose index is ``network.snapshots`` and
        whose columns are a subset of the relevant components.
    cls_name : string
        Name of class of component
    attr : string
        Name of time-varying series attribute
    Examples
    --------
    >>> import numpy as np
    >>> network.set_snapshots(range(10))
    >>> network.import_series_from_dataframe(
            pd.DataFrame(np.random.rand(10,4),
                columns=network.generators.index,
			    index=range(10)),
			"Generator",
			"p_max_pu")
    See Also
    --------
    pypsa.Network.madd()
    """

    df = network.df(cls_name)
    pnl = network.pnl(cls_name)
    list_name = network.components[cls_name]["list_name"]

    diff = dataframe.columns.difference(df.index)
    if len(diff) > 0:
        logger.warning(f"Components {diff} for attribute {attr} of {cls_name} "
                       f"are not in main components dataframe {list_name}")

    attrs = network.components[cls_name]['attrs']
    expected_attrs = attrs[lambda ds: ds.type.str.contains('series')].index
    if attr not in expected_attrs:
        pnl[attr] = dataframe
        return

    attr_series = attrs.loc[attr]
    default = attr_series.default
    columns = dataframe.columns

    diff = network.snapshots.difference(dataframe.index)
    if len(diff):
        logger.warning(f"Snapshots {diff} are missing from {attr} of {cls_name}."
                       f" Filling with default value '{default}'")
        dataframe = dataframe.reindex(network.snapshots, fill_value=default)

    if not attr_series.static:
        pnl[attr] = pnl[attr].reindex(columns=df.index.union(columns), fill_value=default)
    else:
        pnl[attr] = pnl[attr].reindex(columns=(pnl[attr].columns.union(columns)))

    pnl[attr].loc[network.snapshots, columns] = dataframe.loc[network.snapshots, columns]