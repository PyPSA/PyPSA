

#Optimise the dispatch and capacities of the network in
#opf-storage-data.

# make the code as Python 3 compatible as possible
from __future__ import print_function, division
from __future__ import absolute_import

import pypsa

import datetime
import pandas as pd

import networkx as nx

import numpy as np

from itertools import chain

import os


from distutils.spawn import find_executable




csv_folder_name = "opf-storage-data"
network = pypsa.Network(csv_folder_name=csv_folder_name)
print(network,network.co2_limit)


network.build_graph()

network.determine_network_topology()

print("Connected networks:\n",network.sub_networks)


solver_search_order = ["glpk","gurobi"]

solver_executable = {"glpk" : "glpsol", "gurobi" : "gurobi_cl"}

solver_name = None

for s in solver_search_order:
    if find_executable(solver_executable[s]) is not None:
        solver_name = s
        break

if solver_name is None:
    print("No known solver found, quitting.")
    sys.exit()

print("Using solver:",solver_name)


snapshots = network.snapshots

network.lopf(snapshots=snapshots,solver_name=solver_name)

print("Generator and storage capacities:\n")

print(pd.concat((network.generators["p_nom"],network.storage_units["p_nom"])))

print("\n\nBranch capacities:\n")

print(network.branches.s_nom)

for snapshot in snapshots:

    print("\n"*2+"For time",snapshot,":\nBus injections:")

    print(network.buses.p.loc[snapshot])

    print("Total:",network.buses.p.loc[snapshot].sum())


network.now = network.snapshots[0]


for branch in network.branches.obj:
    print(branch,branch.p1[network.now])


print("Comparing bus injection to branch outgoing for %s:" % network.now)

for sub_network in network.sub_networks.obj:

    print("\n\nConsidering sub network",sub_network,":")

    for bus in sub_network.buses.obj:

        print("\n%s" % bus)

        print("power injection (generators - loads + Transport feed-in):",bus.p[network.now])

        print("generators - loads:",sum([g.sign*g.p[network.now] for g in bus.generators.obj])  + sum([l.sign*l.p[network.now] for l in bus.loads.obj]))

        total = 0.0

        for branch in sub_network.branches.obj:
            if bus.name == branch.bus0:
                print("from branch:",branch,branch.p0[network.now])
                total -=branch.p0[network.now]
            elif bus.name == branch.bus1:
                print("to branch:",branch,branch.p1[network.now])
                total -=branch.p1[network.now]
        print("branch injection:",total)


for su in network.storage_units.obj:
    if su.p_nom > 1e-5:
        print(su,su.p_nom,"\n\nState of Charge:\n",su.state_of_charge,"\n\nDispatch:\n",su.p)


for gen in network.generators.obj:
    print(gen,network.sources.loc[gen.source,"co2_emissions"]*(1/gen.efficiency))




results_folder_name = os.path.join(csv_folder_name,"results")


network.export_to_csv_folder(results_folder_name,time_series={"generators" : {"p" : None},
                                                              "storage_units" : {"p" : None},
                                                              "transport_links": {"p0" : None},
                                                              "lines": {"p0" : None}
},verbose=False)
