
## Copyright 2015-2021 PyPSA Developers

## You can find the list of PyPSA Developers at
## https://pypsa.readthedocs.io/en/latest/developers.html

## PyPSA is released under the open source MIT License, see
## https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt

"""Functions for importing and exporting data.
"""

__author__ = "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
__copyright__ = ("Copyright 2015-2021 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
                 "MIT License")

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

class ImporterCSV(Importer):
    def __init__(self, csv_folder_name, encoding):
        self.csv_folder_name = csv_folder_name
        self.encoding = encoding

        assert os.path.isdir(csv_folder_name), "Directory {} does not exist.".format(csv_folder_name)

    def get_attributes(self):
        fn = os.path.join(self.csv_folder_name, "network.csv")
        if not os.path.isfile(fn): return None
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

class ImporterHDF5(Importer):
    def __init__(self, path):
        self.ds = pd.HDFStore(path, mode='r')
        self.index = {}

    def get_attributes(self):
        return dict(self.ds["/network"].reset_index().iloc[0])

    def get_snapshots(self):
        return self.ds["/snapshots"] if "/snapshots" in self.ds else None

    def get_investment_periods(self):
        return self.ds["/investment_periods"] if "/investment_periods" in self.ds else None

    def get_static(self, list_name):
        if "/" + list_name not in self.ds:
            return None

        if self.pypsa_version is None or self.pypsa_version < [0, 13, 1]:
            df = self.ds["/" + list_name]
        else:
            df = self.ds["/" + list_name].set_index('name')

        self.index[list_name] = df.index
        return df

    def get_series(self, list_name):
        for tab in self.ds:
            if tab.startswith('/' + list_name + '_t/'):
                attr = tab[len('/' + list_name + '_t/'):]
                df = self.ds[tab]
                if self.pypsa_version is not None and self.pypsa_version > [0, 13, 0]:
                    df.columns = self.index[list_name][df.columns]
                yield attr, df

class ExporterHDF5(Exporter):
    def __init__(self, path, **kwargs):
        self.ds = pd.HDFStore(path, mode='w', **kwargs)
        self.index = {}

    def save_attributes(self, attrs):
        name = attrs.pop('name')
        self.ds.put('/network',
                    pd.DataFrame(attrs, index=pd.Index([name], name='name')),
                    format='table', index=False)

    def save_snapshots(self, snapshots):
        self.ds.put('/snapshots', snapshots, format='table', index=False)

    def save_investment_periods(self, investment_periods):
        self.ds.put('/investment_periods', investment_periods, format='table', index=False)

    def save_static(self, list_name, df):
        df.index.name = 'name'
        self.index[list_name] = df.index
        df = df.reset_index()
        self.ds.put('/' + list_name, df, format='table', index=False)

    def save_series(self, list_name, attr, df):
        df.columns = self.index[list_name].get_indexer(df.columns)
        self.ds.put('/' + list_name + '_t/' + attr, df, format='table', index=False)

if has_xarray:
    class ImporterNetCDF(Importer):
        def __init__(self, path):
            self.path = path
            if isinstance(path, str):
                self.ds = xr.open_dataset(path)
            else:
                self.ds = path

        def __enter__(self):
            if isinstance(self.path, str):
                super(ImporterNetCDF, self).__init__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if isinstance(self.path, str):
                super(ImporterNetCDF, self).__exit__(exc_type, exc_val, exc_tb)

        def get_attributes(self):
            return {attr[len('network_'):]: val
                    for attr, val in self.ds.attrs.items()
                    if attr.startswith('network_')}

        def get_snapshots(self):
            return self.get_static('snapshots', 'snapshots')

        def get_investment_periods(self):
            return self.get_static('investment_periods', 'investment_periods')

        def get_static(self, list_name, index_name=None):
            t = list_name + '_'
            i = len(t)
            if index_name is None:
                index_name = list_name + '_i'
            if index_name not in self.ds.coords:
                return None
            index = self.ds.coords[index_name].to_index().rename('name')
            df = pd.DataFrame(index=index)
            for attr in self.ds.data_vars.keys():
                if attr.startswith(t) and attr[i:i+2] != 't_':
                    df[attr[i:]] = self.ds[attr].to_pandas()
            return df

        def get_series(self, list_name):
            t = list_name + '_t_'
            for attr in self.ds.data_vars.keys():
                if attr.startswith(t):
                    df = self.ds[attr].to_pandas()
                    df.index.name = 'name'
                    df.columns.name = 'name'
                    yield attr[len(t):], df

    class ExporterNetCDF(Exporter):
        def __init__(self, path, least_significant_digit=None):
            self.path = path
            self.least_significant_digit = least_significant_digit
            self.ds = xr.Dataset()

        def save_attributes(self, attrs):
            self.ds.attrs.update(('network_' + attr, val)
                                 for attr, val in attrs.items())

        def save_snapshots(self, snapshots):
            snapshots.index.name = 'snapshots'
            for attr in snapshots.columns:
                self.ds['snapshots_' + attr] = snapshots[attr]

        def save_investment_periods(self, investment_periods):
            investment_periods.index.rename("investment_periods", inplace=True)
            for attr in investment_periods.columns:
                self.ds['investment_periods_' + attr] = investment_periods[attr]

        def save_static(self, list_name, df):
            df.index.name = list_name + '_i'
            self.ds[list_name + '_i'] = df.index
            for attr in df.columns:
                self.ds[list_name + '_' + attr] = df[attr]

        def save_series(self, list_name, attr, df):
            df.index.name = 'snapshots'
            df.columns.name = list_name + '_t_' + attr + '_i'
            self.ds[list_name + '_t_' + attr] = df
            if self.least_significant_digit is not None:
                print(self.least_significant_digit)
                self.ds.encoding.update({
                    'zlib': True,
                    'least_significant_digit': self.least_significant_digit
                })

        def finish(self):
            if self.path is not None:
                self.ds.to_netcdf(self.path)

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

def import_from_csv_folder(network, csv_folder_name, encoding=None, skip_time=False):
    """
    Import network data from CSVs in a folder.

    The CSVs must follow the standard form, see ``pypsa/examples``.

    Parameters
    ----------
    csv_folder_name : string
        Name of folder
    encoding : str, default None
        Encoding to use for UTF when reading (ex. 'utf-8'). `List of Python
        standard encodings
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

def import_from_hdf5(network, path, skip_time=False):
    """
    Import network data from HDF5 store at `path`.

    Parameters
    ----------
    path : string
        Name of HDF5 store
    skip_time : bool, default False
        Skip reading in time dependent attributes
    """

    basename = os.path.basename(path)
    with ImporterHDF5(path) as importer:
        _import_from_importer(network, importer, basename=basename, skip_time=skip_time)

def export_to_hdf5(network, path, export_standard_types=False, **kwargs):
    """
    Export network and components to an HDF store.

    Both static and series attributes of components are exported, but only
    if they have non-default values.

    If path does not already exist, it is created.

    Parameters
    ----------
    path : string
        Name of hdf5 file to which to export (if it exists, it is overwritten)
    export_standard_types : boolean, default False
        If True, then standard types are exported too (upon reimporting you
        should then set "ignore_standard_types" when initialising the network).
    **kwargs
        Extra arguments for pd.HDFStore to specify f.i. compression
        (default: complevel=4)

    Examples
    --------
    >>> network.export_to_hdf5(filename)
    """

    kwargs.setdefault('complevel', 4)

    basename = os.path.basename(path)
    with ExporterHDF5(path, **kwargs) as exporter:
        _export_to_exporter(network, exporter, basename=basename,
                            export_standard_types=export_standard_types)

def import_from_netcdf(network, path, skip_time=False):
    """
    Import network data from netCDF file or xarray Dataset at `path`.

    Parameters
    ----------
    path : string|xr.Dataset
        Path to netCDF dataset or instance of xarray Dataset
    skip_time : bool, default False
        Skip reading in time dependent attributes
    """

    assert has_xarray, "xarray must be installed for netCDF support."

    basename = os.path.basename(path) if isinstance(path, str) else None
    with ImporterNetCDF(path=path) as importer:
        _import_from_importer(network, importer, basename=basename,
                              skip_time=skip_time)

def export_to_netcdf(network, path=None, export_standard_types=False,
                     least_significant_digit=None):
    """Export network and components to a netCDF file.

    Both static and series attributes of components are exported, but only
    if they have non-default values.

    If path does not already exist, it is created.

    If no path is passed, no file is exported, but the xarray.Dataset
    is still returned.

    Be aware that this cannot export boolean attributes on the Network
    class, e.g. network.my_bool = False is not supported by netCDF.

    Parameters
    ----------
    path : string|None
        Name of netCDF file to which to export (if it exists, it is overwritten);
        if None is passed, no file is exported.
    export_standard_types : boolean, default False
        If True, then standard types are exported too (upon reimporting you
        should then set "ignore_standard_types" when initialising the network).
    least_significant_digit
        This is passed to the netCDF exporter, but currently makes no difference
        to file size or float accuracy. We're working on improving this...

    Returns
    -------
    ds : xarray.Dataset

    Examples
    --------
    >>> network.export_to_netcdf("my_file.nc")

    """

    assert has_xarray, "xarray must be installed for netCDF support."

    basename = os.path.basename(path) if path is not None else None
    with ExporterNetCDF(path, least_significant_digit) as exporter:
        _export_to_exporter(network, exporter, basename=basename,
                            export_standard_types=export_standard_types)
        return exporter.ds

def _import_from_importer(network, importer, basename, skip_time=False):
    """
    Import network data from importer.

    Parameters
    ----------
    skip_time : bool
        Skip importing time
    """

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
                Importing PyPSA from older version of PyPSA than current version.
                Please read the release notes at https://pypsa.org/doc/release_notes.html
                carefully to prepare your network for import. 
                Currently used PyPSA version {}, imported network file PyPSA version {}.
        """).format(current_pypsa_version, pypsa_version))

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



def import_from_pypower_ppc(network, ppc, overwrite_zero_s_nom=None):
    """
    Import network from PYPOWER PPC dictionary format version 2.

    Converts all baseMVA to base power of 1 MVA.

    For the meaning of the pypower indices, see also pypower/idx_*.

    Parameters
    ----------
    ppc : PYPOWER PPC dict
    overwrite_zero_s_nom : Float or None, default None

    Examples
    --------
    >>> from pypower.api import case30
    >>> ppc = case30()
    >>> network.import_from_pypower_ppc(ppc)
    """


    version = ppc["version"]
    if int(version) != 2:
        logger.warning("Warning, importing from PYPOWER may not work if PPC version is not 2!")

    logger.warning("Warning: Note that when importing from PYPOWER, some PYPOWER features not supported: areas, gencosts, component status")


    baseMVA = ppc["baseMVA"]

    #dictionary to store pandas DataFrames of PyPower data
    pdf = {}


    # add buses

    #integer numbering will be bus names
    index = np.array(ppc['bus'][:,0],dtype=int)

    columns = ["type","Pd","Qd","Gs","Bs","area","v_mag_pu_set","v_ang_set","v_nom","zone","v_mag_pu_max","v_mag_pu_min"]

    pdf["buses"] = pd.DataFrame(index=index,columns=columns,data=ppc['bus'][:,1:len(columns)+1])

    if (pdf["buses"]["v_nom"] == 0.).any():
        logger.warning("Warning, some buses have nominal voltage of 0., setting the nominal voltage of these to 1.")
        pdf['buses'].loc[pdf['buses']['v_nom'] == 0.,'v_nom'] = 1.


    #rename controls
    controls = ["","PQ","PV","Slack"]
    pdf["buses"]["control"] = pdf["buses"].pop("type").map(lambda i: controls[int(i)])

    #add loads for any buses with Pd or Qd
    pdf['loads'] = pdf["buses"].loc[pdf["buses"][["Pd","Qd"]].any(axis=1), ["Pd","Qd"]]
    pdf['loads']['bus'] = pdf['loads'].index
    pdf['loads'].rename(columns={"Qd" : "q_set", "Pd" : "p_set"}, inplace=True)
    pdf['loads'].index = ["L"+str(i) for i in range(len(pdf['loads']))]


    #add shunt impedances for any buses with Gs or Bs

    shunt = pdf["buses"].loc[pdf["buses"][["Gs","Bs"]].any(axis=1), ["v_nom","Gs","Bs"]]

    #base power for shunt is 1 MVA, so no need to rebase here
    shunt["g"] = shunt["Gs"]/shunt["v_nom"]**2
    shunt["b"] = shunt["Bs"]/shunt["v_nom"]**2
    pdf['shunt_impedances'] = shunt.reindex(columns=["g","b"])
    pdf['shunt_impedances']["bus"] = pdf['shunt_impedances'].index
    pdf['shunt_impedances'].index = ["S"+str(i) for i in range(len(pdf['shunt_impedances']))]

    #add gens

    #it is assumed that the pypower p_max is the p_nom

    #could also do gen.p_min_pu = p_min/p_nom

    columns = "bus, p_set, q_set, q_max, q_min, v_set_pu, mva_base, status, p_nom, p_min, Pc1, Pc2, Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf".split(", ")

    index = ["G"+str(i) for i in range(len(ppc['gen']))]

    pdf['generators'] = pd.DataFrame(index=index,columns=columns,data=ppc['gen'][:,:len(columns)])


    #make sure bus name is an integer
    pdf['generators']['bus'] = np.array(ppc['gen'][:,0],dtype=int)

    #add branchs
    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax

    columns = 'bus0, bus1, r, x, b, s_nom, rateB, rateC, tap_ratio, phase_shift, status, v_ang_min, v_ang_max'.split(", ")


    pdf['branches'] = pd.DataFrame(columns=columns,data=ppc['branch'][:,:len(columns)])

    pdf['branches']['original_index'] = pdf['branches'].index

    pdf['branches']["bus0"] = pdf['branches']["bus0"].astype(int)
    pdf['branches']["bus1"] = pdf['branches']["bus1"].astype(int)

    # s_nom = 0 indicates an unconstrained line
    zero_s_nom = pdf['branches']["s_nom"] == 0.
    if zero_s_nom.any():
        if overwrite_zero_s_nom is not None:
            pdf['branches'].loc[zero_s_nom, "s_nom"] = overwrite_zero_s_nom
        else:
            logger.warning("Warning: there are {} branches with s_nom equal to zero, "
                  "they will probably lead to infeasibilities and should be "
                  "replaced with a high value using the `overwrite_zero_s_nom` "
                  "argument.".format(zero_s_nom.sum()))

    # determine bus voltages of branches to detect transformers
    v_nom = pdf['branches'].bus0.map(pdf['buses'].v_nom)
    v_nom_1 = pdf['branches'].bus1.map(pdf['buses'].v_nom)

    # split branches into transformers and lines
    transformers = ((v_nom != v_nom_1)
                    | ((pdf['branches'].tap_ratio != 0.) & (pdf['branches'].tap_ratio != 1.)) #NB: PYPOWER has strange default of 0. for tap ratio
                    | (pdf['branches'].phase_shift != 0))
    pdf['transformers'] = pd.DataFrame(pdf['branches'][transformers])
    pdf['lines'] = pdf['branches'][~ transformers].drop(["tap_ratio", "phase_shift"], axis=1)

    #convert transformers from base baseMVA to base s_nom
    pdf['transformers']['r'] = pdf['transformers']['r']*pdf['transformers']['s_nom']/baseMVA
    pdf['transformers']['x'] = pdf['transformers']['x']*pdf['transformers']['s_nom']/baseMVA
    pdf['transformers']['b'] = pdf['transformers']['b']*baseMVA/pdf['transformers']['s_nom']

    #correct per unit impedances
    pdf['lines']["r"] = v_nom**2*pdf['lines']["r"]/baseMVA
    pdf['lines']["x"] = v_nom**2*pdf['lines']["x"]/baseMVA
    pdf['lines']["b"] = pdf['lines']["b"]*baseMVA/v_nom**2


    if (pdf['transformers']['tap_ratio'] == 0.).any():
        logger.warning("Warning, some transformers have a tap ratio of 0., setting the tap ratio of these to 1.")
        pdf['transformers'].loc[pdf['transformers']['tap_ratio'] == 0.,'tap_ratio'] = 1.


    #name them nicely
    pdf['transformers'].index = ["T"+str(i) for i in range(len(pdf['transformers']))]
    pdf['lines'].index = ["L"+str(i) for i in range(len(pdf['lines']))]

    #TODO

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0

    for component in ["Bus","Load","Generator","Line","Transformer","ShuntImpedance"]:
        import_components_from_dataframe(network,pdf[network.components[component]["list_name"]],component)

    network.generators["control"] = network.generators.bus.map(network.buses["control"])

    #for consistency with pypower, take the v_mag set point from the generators
    network.buses.loc[network.generators.bus,"v_mag_pu_set"] = np.asarray(network.generators["v_set_pu"])




def import_from_pandapower_net(network, net, extra_line_data=False):
    """
    Import network from pandapower net.

    Importing from pandapower is still in beta;
    not all pandapower data is supported.

    Unsupported features include:
    - three-winding transformers
    - switches
    - in_service status,
    - shunt impedances, and
    -  tap positions of transformers."

    Parameters
    ----------
    net : pandapower network
    extra_line_data : boolean, default: False
        if True, the line data for all parameters is imported instead of only
        the type

    Examples
    --------
    >>> network.import_from_pandapower_net(net)
    OR
    >>> import pandapower as pp
    >>> import pandapower.networks as pn
    >>> net = pn.create_cigre_network_mv(with_der='all')
    >>> pp.runpp(net)
    >>> network.import_from_pandapower_net(net, extra_line_data=True)
    """
    logger.warning("Warning: Importing from pandapower is still in beta; not all pandapower data is supported.\nUnsupported features include: three-winding transformers, switches, in_service status, shunt impedances and tap positions of transformers.")

    d = {}

    d["Bus"] = pd.DataFrame({"v_nom" : net.bus.vn_kv.values,
                             "v_mag_pu_set" : 1.},
                            index=net.bus.name)

    d["Load"] = pd.DataFrame({"p_set" : (net.load.scaling*net.load.p_mw).values,
                              "q_set" : (net.load.scaling*net.load.q_mvar).values,
                              "bus" : net.bus.name.loc[net.load.bus].values},
                             index=net.load.name)

    #deal with PV generators
    d["Generator"] = pd.DataFrame({"p_set" : -(net.gen.scaling*net.gen.p_mw).values,
                                   "q_set" : 0.,
                                   "bus" : net.bus.name.loc[net.gen.bus].values,
                                   "control" : "PV"},
                                  index=net.gen.name)

    d["Bus"].loc[net.bus.name.loc[net.gen.bus].values,"v_mag_pu_set"] = net.gen.vm_pu.values


    #deal with PQ "static" generators
    d["Generator"] = pd.concat((d["Generator"],pd.DataFrame({"p_set" : -(net.sgen.scaling*net.sgen.p_mw).values,
                                                             "q_set" : -(net.sgen.scaling*net.sgen.q_mvar).values,
                                                             "bus" : net.bus.name.loc[net.sgen.bus].values,
                                                             "control" : "PQ"},
                                                            index=net.sgen.name)), sort=False)

    d["Generator"] = pd.concat((d["Generator"],pd.DataFrame({"control" : "Slack",
                                                             "p_set" : 0.,
                                                             "q_set" : 0.,
                                                             "bus" : net.bus.name.loc[net.ext_grid.bus].values},
                                                            index=net.ext_grid.name.fillna("External Grid"))), sort=False)

    d["Bus"].loc[net.bus.name.loc[net.ext_grid.bus].values,"v_mag_pu_set"] = net.ext_grid.vm_pu.values

    if extra_line_data == False:
        d["Line"] = pd.DataFrame({"type": net.line.std_type.values,
                                  "bus0": net.bus.name.loc[net.line.from_bus].values,
                                  "bus1": net.bus.name.loc[net.line.to_bus].values,
                                  "length": net.line.length_km.values,
                                  "num_parallel": net.line.parallel.values},
                                 index=net.line.name)
    else:
        r = net.line.r_ohm_per_km.values * net.line.length_km.values
        x = net.line.x_ohm_per_km.values * net.line.length_km.values
        # capacitance values from pandapower in nF; transformed here:
        f = net.f_hz
        b = net.line.c_nf_per_km.values * net.line.length_km.values*1e-9
        b = b*2*math.pi*f

        u = net.bus.vn_kv.loc[net.line.from_bus].values
        s_nom = u*net.line.max_i_ka.values

        d["Line"] = pd.DataFrame({"r" : r,
                                  "x" : x,
                                  "b" : b,
                                  "s_nom" : s_nom,
                                  "bus0" : net.bus.name.loc[net.line.from_bus].values,
                                  "bus1" : net.bus.name.loc[net.line.to_bus].values,
                                  "length" : net.line.length_km.values,
                                  "num_parallel" : net.line.parallel.values},
                                 index=net.line.name)

    # check, if the trafo is based on a standard-type:
    if net.trafo.std_type.any():
        d["Transformer"] = pd.DataFrame({"type" : net.trafo.std_type.values,
                                         "bus0" : net.bus.name.loc[net.trafo.hv_bus].values,
                                         "bus1" : net.bus.name.loc[net.trafo.lv_bus].values,
                                         "tap_position" : net.trafo.tap_pos.values},
                                        index=net.trafo.name)
        d["Transformer"] = d["Transformer"].fillna(0)

    # if it's not based on a standard-type - get the included values:
    else:
        s_nom = net.trafo.sn_mva.values/1000.

        r = net.trafo.vkr_percent.values/100.
        x = np.sqrt((net.trafo.vk_percent.values/100.)**2 - r**2)
        # NB: b and g are per unit of s_nom
        g = net.trafo.pfe_kw.values/(1000. * s_nom)

        # for some bizarre reason, some of the standard types in pandapower have i0^2 < g^2
        b = - np.sqrt(((net.trafo.i0_percent.values/100.)**2 - g**2).clip(min=0))

        d["Transformer"] = pd.DataFrame({"phase_shift" : net.trafo.shift_degree.values,
                                         "s_nom" : s_nom,
                                         "bus0" : net.bus.name.loc[net.trafo.hv_bus].values,
                                         "bus1" : net.bus.name.loc[net.trafo.lv_bus].values,
                                         "r" : r,
                                         "x" : x,
                                         "g" : g,
                                         "b" : b,
                                         "tap_position" : net.trafo.tap_pos.values},
                                        index=net.trafo.name)
        d["Transformer"] = d["Transformer"].fillna(0)

    for c in ["Bus","Load","Generator","Line","Transformer"]:
        network.import_components_from_dataframe(d[c],c)

    #amalgamate buses connected by closed switches

    bus_switches = net.switch[(net.switch.et=="b") & net.switch.closed]

    bus_switches["stays"] = bus_switches.bus.map(net.bus.name)
    bus_switches["goes"] = bus_switches.element.map(net.bus.name)

    to_replace = pd.Series(bus_switches.stays.values,bus_switches.goes.values)

    for i in to_replace.index:
        network.remove("Bus",i)

    for c in network.iterate_components({"Load","Generator"}):
        c.df.bus.replace(to_replace,inplace=True)

    for c in network.iterate_components({"Line","Transformer"}):
        c.df.bus0.replace(to_replace,inplace=True)
        c.df.bus1.replace(to_replace,inplace=True)
