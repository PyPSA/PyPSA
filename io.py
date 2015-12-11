

## Copyright 2015 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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

"""Python for Power Systems Analysis (PyPSA)

Grid calculation library.
"""


# make the code as Python 3 compatible as possible
from __future__ import print_function, division


__version__ = "0.1"
__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"



import pandas as pd

import os

import pypsa

import numpy as np


def get_cls_from_list_name(list_name):

    for k,v in vars(pypsa.components).iteritems():
        if hasattr(v,"list_name") and v.list_name == list_name:
            return v


def export_to_csv_folder(network,csv_folder_name,time_series={}):
    """Export network and components to csv_folder_name. Include also
    time_series for components in the dictionary time_series in the
    format:

    list_name : {attribute_name : filter/None}

    e.g.

    export_to_csv(network,csv_folder_name,time_series={"generators" : {"p_max_pu" : lambda g: g.dispatch == "variable"},
    "loads" : {"p_set" : None}})

    """


    #exportable component types
    #what about None????
    allowed_types = [float,int,str,bool]

    #first export network properties

    columns = [attr for attr in dir(network) if type(getattr(network,attr)) in allowed_types and attr != "name" and attr[:2] != "__"]
    index = [network.name]
    df = pd.DataFrame(index=index,columns=columns,data = [[getattr(network,col) for col in columns]])
    df.index.name = "name"

    print("\n"*3+"network\n",df)
    df.to_csv(os.path.join(csv_folder_name,"network.csv"))

    #now export snapshots

    df = pd.DataFrame(index=network.snapshots)
    df["weightings"] = network.snapshot_weightings
    df.index.name = "name"

    print("\n"*3+"snapshots\n",df)
    df.to_csv(os.path.join(csv_folder_name,"snapshots.csv"))


    #now export all other components

    for list_name in ["buses","generators","storage_units","loads","transport_links","lines","converters","sources"]:
        od = getattr(network,list_name)
        if len(od) == 0:
            print("No",list_name)
            continue

        index = od.keys()

        df = pd.DataFrame(index=index)

        df.index.name = "name"

        first = next(od.itervalues())

        for attr in dir(first):
            if attr in ["list_name","name"] or attr[:1] == "_":
                continue
            elif "source" in attr or "bus" in attr:
                df[attr] = [getattr(o,attr).name for o in od.itervalues()]
            elif type(getattr(first,attr)) in allowed_types:
                df[attr] = [getattr(o,attr) for o in od.itervalues()]

        print("\n"*3+list_name+"\n",df)

        df.to_csv(os.path.join(csv_folder_name,list_name+".csv"))

    for list_name in time_series:
        print("\n"*3 + "Exporting time series for:",list_name)

        for attr in time_series[list_name]:
            print(attr)
            filter_f = time_series[list_name][attr]

            sub_selection = filter(filter_f,getattr(network,list_name).itervalues())

            df = pd.DataFrame(index=network.snapshots)

            df.index.name = "snapshots"

            for item in sub_selection:
                df[item.name] = getattr(item,attr)

            df.to_csv(os.path.join(csv_folder_name,list_name+"-" + attr + ".csv"))

            print(df)






def import_components_from_dataframe(network,dataframe,cls_name):

    if not dataframe.index.is_unique:
        print("Warning! Dataframe for",cls_name,"does not have a unique index!")


    for i in dataframe.index:
        obj = network.add(cls_name,i)
        for attr in dataframe.columns:
            if attr == "source":
                setattr(obj,attr,network.sources[str(dataframe[attr][i])])
            elif "bus" in attr:
                setattr(obj,attr,network.buses[str(dataframe[attr][i])])
                #add oneports to bus lists
                if attr == "bus":
                    getattr(obj.bus,obj.__class__.list_name)[obj.name] = obj
            else:
                setattr(obj,attr,dataframe[attr][i])


def import_series_from_dataframe(network,dataframe,list_name,attr):

    od = getattr(network,list_name)

    cls = get_cls_from_list_name(list_name)

    for col in dataframe:
        setattr(od[col],attr,dataframe[col])



def import_from_csv_folder(network,csv_folder_name):


    if not os.path.isdir(csv_folder_name):
        print("Directory {} does not exist.".format(csv_folder_name))
        return

    #if there is network.csv, read in network data

    file_name = os.path.join(csv_folder_name,"network.csv")

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name,index_col=0)
        print(df)
        network.name = df.index[0]
        for col in df.columns:
            setattr(network,col,df[col][0])

    #if there is snapshots.csv, read in snapshot data

    file_name = os.path.join(csv_folder_name,"snapshots.csv")

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name,index_col=0)
        network.set_snapshots(df.index)
        if "weightings" in df.columns:
            network.snapshot_weightings = df["weightings"].reindex(network.snapshots)

    #now read in other components

    for class_name in ["Bus","Source","Generator","StorageUnit","Load","TransportLink","Line","Converter"]:
        cls = getattr(pypsa.components,class_name)

        print(class_name,cls)
        list_name = cls.list_name

        file_name = os.path.join(csv_folder_name,list_name+".csv")

        if not os.path.isfile(file_name):
            if class_name == "Buses":
                print("Error, no buses found")
                return
            else:
                print("No",list_name+".csv","found.")
                continue

        df = pd.read_csv(file_name,index_col=0)

        import_components_from_dataframe(network,df,cls.__name__)

        file_attrs = filter(lambda n: n.startswith(list_name+"-") and n.endswith(".csv"),os.listdir(csv_folder_name))

        for file_name in file_attrs:
            df = pd.read_csv(os.path.join(csv_folder_name,file_name),index_col=0)
            import_series_from_dataframe(network,df,list_name,file_name[len(list_name)+1:-4])


        print(getattr(network,list_name))




def import_from_pypower_ppc(network,ppc):
    """Imports data from a pypower ppc dictionary to a PyPSA network
    object."""


    version = ppc["version"]
    if int(version) != 2:
        print("Warning, importing from PYPOWER may not work if PPC version is not 2!")

    print("Warning: some PYPOWER features not supported: areas, gencosts, baseMVA, shunt Z, component status, branch: ratio, phase angle")

    #dictionary to store pandas DataFrames of PyPower data
    pdf = {}


    # add buses

    #integer numbering will be bus names
    index = np.array(ppc['bus'][:,0],dtype=int)

    columns = ["type","Pd","Qd","Gs","Bs","area","v_mag_set","v_ang_set","v_nom","zone","Vmax","Vmin"]

    pdf["buses"] = pd.DataFrame(index=index,columns=columns,data=ppc['bus'][:,1:])


    #rename controls
    controls = ["","PQ","PV","Slack"]
    pdf['buses']["control"] = [controls[int(pdf["buses"]["type"][i])] for i in pdf["buses"].index]



    #add loads for any buses with Pd or Qd

    if pdf["buses"][["Gs","Bs"]].any().any():
        print("Warning, shunt Z at buses not yet supported!")

    pdf['loads'] = pdf["buses"][["Pd","Qd"]][pdf["buses"][["Pd","Qd"]].any(axis=1)]

    pdf['loads']['bus'] = pdf['loads'].index

    pdf['loads'].rename(columns={"Qd" : "q_set", "Pd" : "p_set"}, inplace=True)



    #add gens

    columns = "bus, p_set, q_set, q_max, q_min, Vg, mBase, status, p_max, p_min, Pc1, Pc2, Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf".split(", ")

    index = np.array(ppc['gen'][:,0],dtype=int)

    pdf['generators'] = pd.DataFrame(index=index,columns=columns,data=ppc['gen'])

    #make sure bus name is an integer
    pdf['generators']['bus'] = index

    #add branchs
    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax

    columns = 'bus0, bus1, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax'.split(", ")


    pdf['branches'] = pd.DataFrame(columns=columns,data=ppc['branch'])

    pdf['branches']["bus0"] = np.array(pdf['branches']["bus0"],dtype=int)
    pdf['branches']["bus1"] = np.array(pdf['branches']["bus1"],dtype=int)


    #TODO

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0

    import_components_from_dataframe(network,pdf["buses"],"Bus")
    import_components_from_dataframe(network,pdf["loads"],"Load")
    import_components_from_dataframe(network,pdf["generators"],"Generator")
    import_components_from_dataframe(network,pdf["branches"],"Line")


    return pdf
