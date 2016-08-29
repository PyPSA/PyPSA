## Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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
from __future__ import print_function, division
from __future__ import absolute_import
from six import iteritems
from six.moves import filter, range


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"



import pandas as pd

import os

import pypsa

import numpy as np



def export_to_csv_folder(network,csv_folder_name,verbose=True,encoding=None):
    """
    Export network and components to a folder of CSVs.

    Both static and series attributes of components are exported, but only
    if they have non-default values.

    If csv_folder_name does not already exist, it is created.

    Parameters
    ----------
    csv_folder_name : string
        Name of folder to which to export.
    verbose : boolean, default True
    encoding : str, default None
        Encoding to use for UTF when reading (ex. 'utf-8'). `List of Python
        standard encodings
        <https://docs.python.org/3/library/codecs.html#standard-encodings>`_

    Examples
    --------
    >>> export_to_csv(network,csv_folder_name)
    OR
    >>> network.export_to_csv(csv_folder_name)
    """


    #exportable component types
    #what about None???? - nan is float?
    allowed_types = [float,int,str,bool] + list(np.typeDict.values())

    #derived components are excluded from the export
    excluded_components = ["branches","sub_networks"]

    #make sure directory exists
    if not os.path.isdir(csv_folder_name):
        if verbose:
            "Directory {} does not exist, creating it".format(csv_folder_name)
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

    #now export all other components static attributes

    for key, od in network.component_simple_descriptors.items():
        if key.list_name in excluded_components:
            continue
        df = getattr(network,key.list_name)
        if df.empty:
            if verbose:
                print("No",key.list_name)
            continue
        col_export = []
        for col in df.columns:
            #do not export derived attributes
            if col in ["obj","sub_network","r_pu","x_pu","g_pu","b_pu"]:
                continue
            if col in od and pd.isnull(od[col].default) and pd.isnull(df[col]).all():
                continue
            if col in od and (df[col] == od[col].default).all():
                continue

            col_export.append(col)

        df[col_export].to_csv(os.path.join(csv_folder_name,key.list_name+".csv"),encoding=encoding)


    #now export all other components series attributes

    for key, od in network.component_series_descriptors.items():
        if key.list_name in excluded_components:
            continue
        pnl = getattr(network,key.list_name+"_t")
        if pnl.empty:
            if verbose:
                print("No",key.list_name+"_t")
            continue
        for attr in pnl:
            if attr not in od:
                col_export = pnl.minor_axis
            else:
                default = od[attr].default

                if pd.isnull(default):
                    col_export = pnl.minor_axis[(~pd.isnull(pnl[attr])).any()]
                else:
                    col_export = pnl.minor_axis[(pnl[attr] != default).any()]

            if len(col_export) > 0:
                pnl.loc[attr,:,col_export].to_csv(os.path.join(csv_folder_name,key.list_name+"-" + attr + ".csv"),encoding=encoding)





def import_components_from_dataframe(network,dataframe,cls_name):
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
    >>> import_components_from_dataframe(dataframe,"Line")
    """


    dataframe.index = [str(i) for i in dataframe.index]

    cls = getattr(pypsa.components,cls_name)

    simple_attrs = set(network.component_simple_descriptors[cls].keys()) & set(dataframe.columns)
    series_attrs = set(network.component_series_descriptors[cls].keys()) & set(dataframe.columns)
    string_attrs = {"bus","bus0","bus1","carrier"} & set(dataframe.columns)

    for col in string_attrs:
        dataframe[col] = [str(v) for v in dataframe[col]]


    old_df = getattr(network,cls.list_name)

    new_df = pd.concat((old_df,dataframe.drop(series_attrs,axis=1)))

    if not new_df.index.is_unique:
        print("Error, new components for",cls_name,"are not unique")
        return

    for k,v in network.component_simple_descriptors[cls].items():
        if k not in simple_attrs:
            new_df.loc[dataframe.index,k] = v.default

        #This is definitely necessary to avoid boolean bugs - should
        #we also do this for other types?
        if v.typ == bool and new_df[k].dtype is not np.dtype(v.typ):
            new_df.loc[:,k] = new_df.loc[:,k].astype(v.typ)

    new_df.loc[dataframe.index,"obj"] = [cls(network,str(i)) for i in dataframe.index]

    setattr(network,cls.list_name,new_df)


    #now deal with time-dependent properties

    pnl = getattr(network,cls.list_name+"_t")

    pnl = pnl.reindex(minor_axis=pnl.minor_axis.append(dataframe.index))

    for k,v in network.component_series_descriptors[cls].items():
        pnl.loc[k,:,dataframe.index] = v.default
        if k in series_attrs:
            pnl[k].loc[:,dataframe.index] = dataframe.loc[:,k].values

    setattr(network,cls.list_name+"_t",pnl)




def import_series_from_dataframe(network,dataframe,cls_name,attr):
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

    cls = getattr(pypsa.components,cls_name)

    pnl = getattr(network,cls.list_name+"_t")

    pnl.loc[attr].loc[:,dataframe.columns] = dataframe

def import_from_csv_folder(network,csv_folder_name,verbose=False,encoding=None):
    """
    Import network data from CSVs in a folder.

    The CSVs must follow the standard form, see pypsa/examples.

    Parameters
    ----------
    csv_folder_name : string
        Name of folder
    verbose : boolean, default True
    encoding : str, default None
        Encoding to use for UTF when reading (ex. 'utf-8'). `List of Python
        standard encodings
        <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
    """

    if not os.path.isdir(csv_folder_name):
        print("Directory {} does not exist.".format(csv_folder_name))
        return

    #if there is network.csv, read in network data

    file_name = os.path.join(csv_folder_name,"network.csv")

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name,index_col=0,encoding=encoding)
        if verbose:
            print(df)
        network.name = df.index[0]
        for col in df.columns:
            setattr(network,col,df[col][network.name])

    #if there is snapshots.csv, read in snapshot data

    file_name = os.path.join(csv_folder_name,"snapshots.csv")

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name,index_col=0,encoding=encoding)
        network.set_snapshots(df.index)
        if "weightings" in df.columns:
            network.snapshot_weightings = df["weightings"].reindex(network.snapshots)

    #now read in other components
    for cls in pypsa.components.component_types - {pypsa.components.SubNetwork}:

        if verbose:
            print(cls)

        list_name = cls.list_name

        file_name = os.path.join(csv_folder_name,list_name+".csv")

        if not os.path.isfile(file_name):
            if cls.__name__ == "Bus":
                print("Error, no buses found")
                return
            else:
                if verbose:
                    print("No",list_name+".csv","found.")
                continue

        df = pd.read_csv(file_name,index_col=0,encoding=encoding)

        import_components_from_dataframe(network,df,cls.__name__)

        file_attrs = [n for n in os.listdir(csv_folder_name) if n.startswith(list_name+"-") and n.endswith(".csv")]

        for file_name in file_attrs:
            df = pd.read_csv(os.path.join(csv_folder_name,file_name),index_col=0,encoding=encoding)
            import_series_from_dataframe(network,df,cls.__name__,file_name[len(list_name)+1:-4])

        if verbose:
            print(getattr(network,list_name))




def import_from_pypower_ppc(network, ppc, verbose=True, overwrite_zero_s_nom=None):
    """
    Import network from PYPOWER PPC dictionary format version 2.

    Converts all baseMVA to base power of 1 MVA.

    For the meaning of the pypower indices, see also pypower/idx_*.

    Parameters
    ----------
    ppc : PYPOWER PPC dict
    verbose : bool, default True
    overwrite_zero_s_nom : Float or None, default None

    Examples
    --------
    >>> network.import_from_pypower_ppc(ppc)
    """


    version = ppc["version"]
    if int(version) != 2:
        print("Warning, importing from PYPOWER may not work if PPC version is not 2!")

    print("Warning: Note that when importing from PYPOWER, some PYPOWER features not supported: areas, gencosts, component status")


    baseMVA = ppc["baseMVA"]

    #dictionary to store pandas DataFrames of PyPower data
    pdf = {}


    # add buses

    #integer numbering will be bus names
    index = np.array(ppc['bus'][:,0],dtype=int)

    columns = ["type","Pd","Qd","Gs","Bs","area","v_mag_pu_set","v_ang_set","v_nom","zone","v_mag_pu_max","v_mag_pu_min"]

    pdf["buses"] = pd.DataFrame(index=index,columns=columns,data=ppc['bus'][:,1:len(columns)+1])

    if (pdf["buses"]["v_nom"] == 0.).any():
        print("Warning, some buses have nominal voltage of 0., setting the nominal voltage of these to 1.")
        pdf['buses'].loc[pdf['buses']['v_nom'] == 0.,'v_nom'] = 1.


    #rename controls
    controls = ["","PQ","PV","Slack"]
    pdf["buses"]["control"] = pdf["buses"]["type"].map(lambda i: controls[int(i)])

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

    #could also do gen.p_min_pu_fixed = p_min/p_nom

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
            print("Warning: there are {} branches with s_nom equal to zero, "
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
        print("Warning, some transformers have a tap ratio of 0., setting the tap ratio of these to 1.")
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
        cls = getattr(pypsa.components,component)
        import_components_from_dataframe(network,pdf[cls.list_name],component)

    for gen in network.generators.obj:
        gen.control = network.buses.control[gen.bus]

    #for consistency with pypower, take the v_mag set point from the generators
    network.buses_t.v_mag_pu_set.loc[network.now,network.generators.bus] = np.asarray(network.generators["v_set_pu"])
