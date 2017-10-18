## Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Functions for importing and exporting data.
"""


# make the code as Python 3 compatible as possible
from __future__ import division, absolute_import
from six import iteritems
from six.moves import filter, range


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"

import logging
logger = logging.getLogger(__name__)

from textwrap import dedent
import os

import pandas as pd
import pypsa
import numpy as np



def export_to_csv_folder(network, csv_folder_name, encoding=None, export_standard_types=False):
    """
    Export network and components to a folder of CSVs.

    Both static and series attributes of components are exported, but only
    if they have non-default values.

    If csv_folder_name does not already exist, it is created.

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
        should then set "ignore_standard_types" when initialising the netowrk).

    Examples
    --------
    >>> export_to_csv(network,csv_folder_name)
    OR
    >>> network.export_to_csv(csv_folder_name)
    """


    #exportable component types
    #what about None???? - nan is float?
    allowed_types = [float,int,str,bool] + list(np.typeDict.values())

    #make sure directory exists
    if not os.path.isdir(csv_folder_name):
        logger.warning("Directory {} does not exist, creating it".format(csv_folder_name))
        os.mkdir(csv_folder_name)


    #first export network properties

    columns = [attr for attr in dir(network) if type(getattr(network,attr)) in allowed_types and attr != "name" and attr[:2] != "__"]
    index = [network.name]
    df = pd.DataFrame(index=index,columns=columns,data = [[getattr(network,col) for col in columns]])
    df.index.name = "name"

    df.to_csv(os.path.join(csv_folder_name,"network.csv"),encoding=encoding)

    #now export snapshots

    df = pd.DataFrame(index=network.snapshots)
    df["weightings"] = network.snapshot_weightings
    df.index.name = "name"

    df.to_csv(os.path.join(csv_folder_name,"snapshots.csv"),encoding=encoding)

    #now export all other components

    exported_components = []

    for component in pypsa.components.all_components - {"SubNetwork"}:

        list_name = network.components[component]["list_name"]
        attrs = network.components[component]["attrs"]
        df = network.df(component)
        pnl = network.pnl(component)


        if not export_standard_types and component in pypsa.components.standard_types:
            df = df.drop(network.components[component]["standard_types"].index)


        #first do static attributes
        filename = os.path.join(csv_folder_name,list_name+".csv")
        df.index.name = "name"
        if df.empty:
            if os.path.exists(filename):
                os.unlink(filename)

                fns = [os.path.basename(filename)]
                for attr in attrs.index[attrs.varying]:
                    fn = os.path.join(csv_folder_name,list_name+'-'+attr+'.csv')
                    if os.path.exists(fn):
                        os.unlink(fn)
                        fns.append(os.path.basename(fn))

                logger.warning("Stale csv file(s) {} removed".format(', '.join(fns)))

            continue

        col_export = []
        for col in df.columns:
            #do not export derived attributes
            if col in ["sub_network","r_pu","x_pu","g_pu","b_pu"]:
                continue
            if col in attrs.index and pd.isnull(attrs.at[col,"default"]) and pd.isnull(df[col]).all():
                continue
            if (col in attrs.index
                and df[col].dtype == attrs.at[col, 'dtype']
                and (df[col] == attrs.at[col,"default"]).all()):
                continue

            col_export.append(col)

        df[col_export].to_csv(filename,encoding=encoding)


        #now do varying attributes
        for attr in pnl:
            if attr not in attrs.index:
                col_export = pnl[attr].columns
            else:
                default = attrs.at[attr,"default"]

                if pd.isnull(default):
                    col_export = pnl[attr].columns[(~pd.isnull(pnl[attr])).any()]
                else:
                    col_export = pnl[attr].columns[(pnl[attr] != default).any()]

            filename = os.path.join(csv_folder_name,list_name+"-" + attr + ".csv")
            if len(col_export) > 0:
                pnl[attr].loc[:,col_export].to_csv(filename,encoding=encoding)
            else:
                if os.path.exists(filename):
                    os.unlink(filename)
                    logger.warning("Stale csv file {} removed"
                                   .format(os.path.basename(filename)))

        exported_components.append(list_name)

    logger.info("Exported network {} has {}".format(os.path.basename(csv_folder_name), ", ".join(exported_components)))

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
    **kwargs
        Extra arguments for pd.HDFStore to specify f.i. compression
        (default: complevel=4)

    Examples
    --------
    >>> export_to_hdf5(network, filename)
    OR
    >>> network.export_to_hdf5(filename)
    """

    kwargs.setdefault('complevel', 4)

    with pd.HDFStore(path, mode='w', **kwargs) as store:
        #first export network properties

        #exportable component types
        #what about None???? - nan is float?
        allowed_types = [float,int,str,bool] + list(np.typeDict.values())

        columns = [attr for attr in dir(network)
                   if (attr != "name" and attr[:2] != "__" and
                       type(getattr(network,attr)) in allowed_types)]
        index = pd.Index([network.name], name="name")
        store.put('/network',
                  pd.DataFrame(index=index, columns=columns,
                               data=[[getattr(network, col) for col in columns]]),
                  format='table', index=False)

        #now export snapshots

        store.put('/snapshots',
                  pd.DataFrame(dict(weightings=network.snapshot_weightings),
                               index=pd.Index(network.snapshots, name="name")),
                  format='table', index=False)

        #now export all other components

        exported_components = []
        for component in pypsa.components.all_components - {"SubNetwork"}:

            list_name = network.components[component]["list_name"]
            attrs = network.components[component]["attrs"]

            df = network.df(component)
            pnl = network.pnl(component)

            if not export_standard_types and component in pypsa.components.standard_types:
                df = df.drop(network.components[component]["standard_types"].index)

            #first do static attributes
            df.index.name = "name"
            if df.empty:
                continue

            col_export = []
            for col in df.columns:
                #do not export derived attributes
                if col in ["sub_network", "r_pu", "x_pu", "g_pu", "b_pu"]:
                    continue
                if col in attrs.index and pd.isnull(attrs.at[col, "default"]) and pd.isnull(df[col]).all():
                    continue
                if (col in attrs.index
                    and df[col].dtype == attrs.at[col, 'dtype']
                    and (df[col] == attrs.at[col, "default"]).all()):
                    continue

                col_export.append(col)

            store.put('/' + list_name, df[col_export], format='table', index=False)

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

                df = pnl[attr][col_export]
                if not df.empty:
                    store.put('/' + list_name + '_t/' + attr, df, format='table', index=False)

            exported_components.append(list_name)

    logger.info("Exported network {} has {}".format(os.path.basename(path), ", ".join(exported_components)))

def import_from_hdf5(network, path, skip_time=False):
    """
    Import network data from HDF5 store at `path`.

    Parameters
    ----------
    path : string
        Name of HDF5 store
    """

    with pd.HDFStore(path, mode='r') as store:
        df = store['/network']
        logger.debug("/network")
        logger.debug(df)
        network.name = df.index[0]

        ##https://docs.python.org/3/tutorial/datastructures.html#comparing-sequences-and-other-types
        current_pypsa_version = [int(s) for s in network.pypsa_version.split(".")]
        try:
            pypsa_version = [int(s) for s in df.at[network.name, 'pypsa_version'].split(".")]
            df = df.drop('pypsa_version', axis=1)
        except KeyError:
            pypsa_version = None

        if pypsa_version is None or pypsa_version < current_pypsa_version:
            logger.warning(dedent("""
                Importing PyPSA from older version of PyPSA than current version {}.
                Please read the release notes at https://pypsa.org/doc/release_notes.html
                carefully to prepare your network for import.
            """).format(network.pypsa_version))

        for col in df.columns:
            setattr(network, col, df[col][network.name])

        #if there is snapshots.csv, read in snapshot data

        if '/snapshots' in store:
            df = store['/snapshots']

            network.set_snapshots(df.index)
            if "weightings" in df.columns:
                network.snapshot_weightings = df["weightings"].reindex(network.snapshots)

        imported_components = []

        #now read in other components; make sure buses and carriers come first
        for component in ["Bus", "Carrier"] + sorted(pypsa.components.all_components - {"Bus", "Carrier", "SubNetwork"}):
            list_name = network.components[component]["list_name"]

            if '/' + list_name not in store:
                if component == "Bus":
                    logger.error("Error, no buses found")
                    return
                else:
                    continue

            df = store['/' + list_name]
            import_components_from_dataframe(network, df, component)

            if not skip_time:
                for attr in store:
                    if attr.startswith('/' + list_name + '_t/'):
                        attr_name = attr[len('/' + list_name + '_t/'):]
                        import_series_from_dataframe(network, store[attr], component, attr_name)

            logger.debug(getattr(network,list_name))

            imported_components.append(list_name)

    logger.info("Imported network {} has {}".format(os.path.basename(path), ", ".join(imported_components)))

def import_components_from_dataframe(network, dataframe, cls_name):
    """
    Import components from a pandas DataFrame.

    If columns are missing then defaults are used.

    If extra columns are added, these are left in the resulting component dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
    cls_name : string
        Name of class of component

    Examples
    --------
    >>> network.import_components_from_dataframe(dataframe,"Line")
    """

    if cls_name == "Generator" and "source" in dataframe.columns:
        logger.warning("'source' for generators is deprecated, use 'carrier' instead.")
    if cls_name == "Generator" and "dispatch" in dataframe.columns:
        logger.warning("'dispatch' for generators is deprecated, use time-varing 'p_max_pu' for 'variable' and static 'p_max_pu' for 'flexible'.")
    if cls_name in ["Generator","StorageUnit"] and "p_max_pu_fixed" in dataframe.columns:
        logger.warning("'p_max_pu_fixed' for generators is deprecated, use static 'p_max_pu' instead.")
    if cls_name in ["Generator","StorageUnit"] and "p_min_pu_fixed" in dataframe.columns:
        logger.warning("'p_min_pu_fixed' for generators is deprecated, use static 'p_min_pu' instead.")
    if cls_name == "Bus" and "current_type" in dataframe.columns:
        logger.warning("'current_type' for buses is deprecated, use 'carrier' instead.")
    if cls_name == "Link" and "s_nom" in dataframe.columns:
        logger.warning("'s_nom*' for links is deprecated, use 'p_nom*' instead.")

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
    new_df = pd.concat((network.df(cls_name), dataframe.drop(non_static_attrs_in_df, axis=1)))

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
    cls_name : string
        Name of class of component
    attr : string
        Name of series attribute

    Examples
    --------
    >>> import_series_from_dataframe(dataframe,"Load","p_set")
    """

    df = network.df(cls_name)
    pnl = network.pnl(cls_name)
    list_name = network.components[cls_name]["list_name"]

    diff = dataframe.columns.difference(df.index)
    if len(diff) > 0:
        logger.warning("Components {} for attribute {} of {} are not in main components dataframe {}".format(diff,attr,cls_name,list_name))

    diff = network.snapshots.difference(dataframe.index)
    if len(diff):
        logger.warning("Snapshots {} are missing from {} of {}".format(diff,attr,cls_name))


    attr_series = network.components[cls_name]["attrs"].loc[attr]
    columns = dataframe.columns

    if not attr_series.static:
        pnl[attr] = pnl[attr].reindex(columns=df.index|columns, fill_value=attr_series.default)
    else:
        pnl[attr] = pnl[attr].reindex(columns=(pnl[attr].columns | columns))

    pnl[attr].loc[network.snapshots, columns] = dataframe.loc[network.snapshots, columns]



def import_from_csv_folder(network, csv_folder_name, encoding=None, skip_time=False):
    """
    Import network data from CSVs in a folder.

    The CSVs must follow the standard form, see pypsa/examples.

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
    """

    if not os.path.isdir(csv_folder_name):
        logger.error("Directory {} does not exist.".format(csv_folder_name))
        return

    #if there is network.csv, read in network data

    file_name = os.path.join(csv_folder_name,"network.csv")

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name,index_col=0,encoding=encoding)
        logger.debug("networks.csv:")
        logger.debug(df)
        network.name = df.index[0]

        ##https://docs.python.org/3/tutorial/datastructures.html#comparing-sequences-and-other-types
        current_pypsa_version = [int(s) for s in network.pypsa_version.split(".")]
        pypsa_version = None
        for col in df.columns:
            if col == "pypsa_version":
                pypsa_version = [int(s) for s in df.at[network.name,"pypsa_version"].split(".")]
            else:
                setattr(network,col,df[col][network.name])

        if pypsa_version is None or pypsa_version < current_pypsa_version:
            logger.warning("Importing PyPSA from older version of PyPSA than current version {}.\n\
            Please read the release notes at https://pypsa.org/doc/release_notes.html\n\
            carefully to prepare your network for import.".format(network.pypsa_version))

    #if there is snapshots.csv, read in snapshot data

    file_name = os.path.join(csv_folder_name,"snapshots.csv")

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name, index_col=0, encoding=encoding, parse_dates=True)
        network.set_snapshots(df.index)
        if "weightings" in df.columns:
            network.snapshot_weightings = df["weightings"].reindex(network.snapshots)

    imported_components = []

    #now read in other components; make sure buses and carriers come first
    for component in ["Bus", "Carrier"] + sorted(pypsa.components.all_components - {"Bus","Carrier","SubNetwork"}):

        list_name = network.components[component]["list_name"]

        file_name = os.path.join(csv_folder_name,list_name+".csv")

        if not os.path.isfile(file_name):
            if component == "Bus":
                logger.error("Error, no buses found")
                return
            else:
                continue

        df = pd.read_csv(file_name,index_col=0,encoding=encoding)

        import_components_from_dataframe(network,df,component)

        if not skip_time:
            file_attrs = [n for n in os.listdir(csv_folder_name) if n.startswith(list_name+"-") and n.endswith(".csv")]

            for file_name in file_attrs:
                df = pd.read_csv(os.path.join(csv_folder_name,file_name), index_col=0, encoding=encoding, parse_dates=True)
                import_series_from_dataframe(network,df,component,file_name[len(list_name)+1:-4])

        logger.debug(getattr(network,list_name))

        imported_components.append(list_name)

    logger.info("Imported network {} has {}".format(os.path.basename(csv_folder_name), ", ".join(imported_components)))

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




def import_from_pandapower_net(network, net):
    """
    Import network from pandapower net.

    This import function is not yet finished (see warning below).

    Parameters
    ----------
    net : pandapower network

    Examples
    --------
    >>> network.import_from_pandapower_net(net)
    """
    logger.warning("Warning: Importing from pandapower is still in beta; not all pandapower data is supported.\nUnsupported features include: three-winding transformers, switches, in_service status, shunt impedances and tap positions of transformers.")

    d = {}

    d["Bus"] = pd.DataFrame({"v_nom" : net.bus.vn_kv.values,
                             "v_mag_pu_set" : 1.},
                            index=net.bus.name)

    d["Load"] = pd.DataFrame({"p_set" : (net.load.scaling*net.load.p_kw).values/1e3,
                              "q_set" : (net.load.scaling*net.load.q_kvar).values/1e3,
                              "bus" : net.bus.name.loc[net.load.bus].values},
                             index=net.load.name)

    #deal with PV generators
    d["Generator"] = pd.DataFrame({"p_set" : -(net.gen.scaling*net.gen.p_kw).values/1e3,
                                   "q_set" : 0.,
                                   "bus" : net.bus.name.loc[net.gen.bus].values,
                                   "control" : "PV"},
                                  index=net.gen.name)

    d["Bus"].loc[net.bus.name.loc[net.gen.bus].values,"v_mag_pu_set"] = net.gen.vm_pu.values


    #deal with PQ "static" generators
    d["Generator"] = pd.concat((d["Generator"],pd.DataFrame({"p_set" : -(net.sgen.scaling*net.sgen.p_kw).values/1e3,
                                                             "q_set" : -(net.sgen.scaling*net.sgen.q_kvar).values/1e3,
                                                             "bus" : net.bus.name.loc[net.sgen.bus].values,
                                                             "control" : "PQ"},
                                                            index=net.sgen.name)))

    d["Generator"] = pd.concat((d["Generator"],pd.DataFrame({"control" : "Slack",
                                                             "p_set" : 0.,
                                                             "q_set" : 0.,
                                                             "bus" : net.bus.name.loc[net.ext_grid.bus].values},
                                                            index=net.ext_grid.name.fillna("External Grid"))))

    d["Bus"].loc[net.bus.name.loc[net.ext_grid.bus].values,"v_mag_pu_set"] = net.ext_grid.vm_pu.values

    d["Line"] = pd.DataFrame({"type" : net.line.std_type.values,
                              "bus0" : net.bus.name.loc[net.line.from_bus].values,
                              "bus1" : net.bus.name.loc[net.line.to_bus].values,
                              "length" : net.line.length_km.values,
                              "num_parallel" : net.line.parallel.values},
                             index=net.line.name)

    d["Transformer"] = pd.DataFrame({"type" : net.trafo.std_type.values,
                                     "bus0" : net.bus.name.loc[net.trafo.hv_bus].values,
                                     "bus1" : net.bus.name.loc[net.trafo.lv_bus].values,
                                     "tap_position" : net.trafo.tp_pos.values},
                                    index=net.trafo.name)

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
