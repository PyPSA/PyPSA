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
from six.moves import filter


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"



import pandas as pd

import os

import pypsa

import numpy as np


def export_to_csv_folder(network,csv_folder_name,time_series={},verbose=True):
    """
    Export network and components to a folder of CSVs.

    Parameters
    ----------
    csv_folder_name : string
        Name of folder to which to export.
    time_series : dictionary of callable functions per component type, defaults to {}
        This allows you to select which components' time series are exported with
        the format {list_name : {attribute_name : filter/None}}
    verbose : boolean, default True

    Examples
    --------
    >>> export_to_csv(network,csv_folder_name,time_series={"generators" : {"p_max_pu" : lambda g: g.dispatch == "variable"},
    "loads" : {"p_set" : None}})
    """


    #exportable component types
    #what about None????
    allowed_types = [float,int,str,bool] + list(np.typeDict.values())

    #first export network properties

    columns = [attr for attr in dir(network) if type(getattr(network,attr)) in allowed_types and attr != "name" and attr[:2] != "__"]
    index = [network.name]
    df = pd.DataFrame(index=index,columns=columns,data = [[getattr(network,col) for col in columns]])
    df.index.name = "name"

    if verbose:
        print("\n"*3+"network\n",df)
    df.to_csv(os.path.join(csv_folder_name,"network.csv"))

    #now export snapshots

    df = pd.DataFrame(index=network.snapshots)
    df["weightings"] = network.snapshot_weightings
    df.index.name = "name"

    if verbose:
        print("\n"*3+"snapshots\n",df)
    df.to_csv(os.path.join(csv_folder_name,"snapshots.csv"))


    #now export all other components

    for list_name in ["buses","generators","storage_units","loads","shunt_impedances","transport_links","lines","transformers","converters","sources"]:
        list_df = getattr(network,list_name)
        if len(list_df) == 0:
            if verbose:
                print("No",list_name)
            continue

        if verbose:
            print("\n"*3+list_name+"\n",list_df)

        list_df.to_csv(os.path.join(csv_folder_name,list_name+".csv"))

    for list_name in time_series:

        if verbose:
            print("\n"*3 + "Exporting time series for:",list_name)

        pnl = getattr(network,list_name+"_t")

        for attr in time_series[list_name]:
            if verbose:
                print(attr)
            filter_f = time_series[list_name][attr]

            sub_selection = list(filter(filter_f,getattr(network,list_name).obj))

            df = pnl.loc[attr,:,[s.name for s in sub_selection]]

            df.to_csv(os.path.join(csv_folder_name,list_name+"-" + attr + ".csv"))
            if verbose:
                print(df)






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
    string_attrs = {"bus","bus0","bus1","source"} & set(dataframe.columns)

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

def import_from_csv_folder(network,csv_folder_name,verbose=False):
    """
    Import network data from CSVs in a folder.

    The CSVs must follow the standard form, see pypsa/examples.

    Parameters
    ----------
    csv_folder_name : string
        Name of folder
    """

    if not os.path.isdir(csv_folder_name):
        print("Directory {} does not exist.".format(csv_folder_name))
        return

    #if there is network.csv, read in network data

    file_name = os.path.join(csv_folder_name,"network.csv")

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name,index_col=0)
        if verbose:
            print(df)
        network.name = df.index[0]
        for col in df.columns:
            setattr(network,col,df[col][network.name])

    #if there is snapshots.csv, read in snapshot data

    file_name = os.path.join(csv_folder_name,"snapshots.csv")

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name,index_col=0)
        network.set_snapshots(df.index)
        if "weightings" in df.columns:
            network.snapshot_weightings = df["weightings"].reindex(network.snapshots)

    #now read in other components

    for class_name in ["Bus","Source","Generator","StorageUnit","Load","ShuntImpedance","TransportLink","Line","Transformer","Converter"]:
        cls = getattr(pypsa.components,class_name)
        if verbose:
            print(class_name,cls)
        list_name = cls.list_name

        file_name = os.path.join(csv_folder_name,list_name+".csv")

        if not os.path.isfile(file_name):
            if class_name == "Buses":
                print("Error, no buses found")
                return
            else:
                if verbose:
                    print("No",list_name+".csv","found.")
                continue

        df = pd.read_csv(file_name,index_col=0)

        import_components_from_dataframe(network,df,cls.__name__)

        file_attrs = [n for n in os.listdir(csv_folder_name) if n.startswith(list_name+"-") and n.endswith(".csv")]

        for file_name in file_attrs:
            df = pd.read_csv(os.path.join(csv_folder_name,file_name),index_col=0)
            import_series_from_dataframe(network,df,class_name,file_name[len(list_name)+1:-4])

        if verbose:
            print(getattr(network,list_name))




def import_from_pypower_ppc(network,ppc,verbose=True):
    """
    Import network from PYPOWER PPC dictionary format version 2.

    Converts all baseMVA to base power of 1 MVA.

    For the meaning of the pypower indices, see also pypower/idx_*.

    Parameters
    ----------
    ppc : PYPOWER PPC dict
    verbose : bool, default True

    Examples
    --------
    >>> network.import_from_pypower_ppc(ppc)
    """


    version = ppc["version"]
    if int(version) != 2:
        print("Warning, importing from PYPOWER may not work if PPC version is not 2!")

    print("Warning: some PYPOWER features not supported: areas, gencosts, component status, branch: ratio, phase angle")


    baseMVA = ppc["baseMVA"]

    #dictionary to store pandas DataFrames of PyPower data
    pdf = {}


    # add buses

    #integer numbering will be bus names
    index = np.array(ppc['bus'][:,0],dtype=int)

    columns = ["type","Pd","Qd","Gs","Bs","area","v_mag_pu_set","v_ang_set","v_nom","zone","v_mag_pu_max","v_mag_pu_min"]

    pdf["buses"] = pd.DataFrame(index=index,columns=columns,data=ppc['bus'][:,1:])


    #rename controls
    controls = ["","PQ","PV","Slack"]
    pdf['buses']["control"] = [controls[int(pdf["buses"]["type"][i])] for i in pdf["buses"].index]



    #add loads for any buses with Pd or Qd
    pdf['loads'] = pdf["buses"][["Pd","Qd"]][pdf["buses"][["Pd","Qd"]].any(axis=1)]

    pdf['loads']['bus'] = pdf['loads'].index

    pdf['loads'].rename(columns={"Qd" : "q_set", "Pd" : "p_set"}, inplace=True)

    pdf['loads'].index = ["L"+str(i) for i in range(len(pdf['loads']))]



    #add shunt impedances for any buses with Gs or Bs

    shunt = pdf["buses"][["v_nom","Gs","Bs"]][pdf["buses"][["Gs","Bs"]].any(axis=1)]

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

    index = ["G"+str(i) for i in range(ppc['gen'].shape[0])]

    pdf['generators'] = pd.DataFrame(index=index,columns=columns,data=ppc['gen'])


    #make sure bus name is an integer
    pdf['generators']['bus'] = np.array(ppc['gen'][:,0],dtype=int)

    #add branchs
    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax

    columns = 'bus0, bus1, r, x, b, s_nom, rateB, rateC, tap_ratio, phase_shift, status, v_ang_min, v_ang_max'.split(", ")


    pdf['branches'] = pd.DataFrame(columns=columns,data=ppc['branch'])

    pdf['branches']['original_index'] = pdf['branches'].index

    pdf['branches']["bus0"] = pdf['branches']["bus0"].astype(int)
    pdf['branches']["bus1"] = pdf['branches']["bus1"].astype(int)

    # determine bus voltages of branches to detect transformers
    v_nom = pdf['branches'].bus0.map(pdf['buses'].v_nom)
    v_nom_1 = pdf['branches'].bus1.map(pdf['buses'].v_nom)

    # split branches into transformers and lines
    transformers = (v_nom != v_nom_1) | (pdf['branches'].tap_ratio != 0) | (pdf['branches'].phase_shift != 0)
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
