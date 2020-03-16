

#Optimise the dispatch and capacities of the network in
#opf-storage-data.

import pypsa

import pandas as pd

import numpy as np

import os

from distutils.spawn import find_executable




csv_folder_name = "opf-storage-data"
network = pypsa.Network(csv_folder_name=csv_folder_name)
print(network,network.global_constraints)

#useful for debugging
network.opf_keep_files = True

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


network.lopf(solver_name=solver_name)

print("Generator and storage capacities:\n")

print(pd.concat((network.generators["p_nom"],network.storage_units["p_nom"])))

print("\n\nBranch capacities:\n")

print(network.branches().s_nom)

for snapshot in network.snapshots:

    print("\n"*2+"For time",snapshot,":\nBus injections:")

    print(network.buses_t.p.loc[snapshot])

    print("Total:",network.buses_t.p.loc[snapshot].sum())


now = network.snapshots[0]


for branch in network.branches().index:
    print(branch,getattr(network,network.components[branch[0]]["list_name"]+"_t").p1.loc[now,branch[1]])


print("Comparing bus injection to branch outgoing for %s:" % now)


for sub_network in network.sub_networks.obj:

    print("\n\nConsidering sub network",sub_network,":")

    for bus in sub_network.buses().index:

        print("\n%s" % bus)

        print("power injection (generators - loads + link feed-in):",network.buses_t.p.loc[now,bus])

        generators = sum(network.generators_t.p.loc[now,network.generators.bus==bus])
        loads = sum(network.loads_t.p.loc[now,network.loads.bus==bus])
        storages = sum(network.storage_units_t.p.loc[now,network.storage_units.bus==bus])

        print("generators - loads:",generators+ storages - loads)


        p0 = 0.
        p1 = 0.

        for c in network.iterate_components(network.branch_components):

            bs = (c.df.bus0 == bus)

            if bs.any():
                print(c,"\n",c.pnl.p0.loc[now,bs])
                p0 += c.pnl.p0.loc[now,bs].sum()

            bs = (c.df.bus1 == bus)

            if bs.any():
                print(c,"\n",c.pnl.p1.loc[now,bs])
                p1 += c.pnl.p1.loc[now,bs].sum()


        print("Branch injections:",p0+p1)

        np.testing.assert_allclose(p0+p1,generators+ storages - loads)

for su in network.storage_units.index:
    suo = network.storage_units.loc[su]
    if suo.p_nom > 1e-5:
        print(su,suo.p_nom,"\n\nState of Charge:\n",network.storage_units_t.state_of_charge.loc[:,su],"\n\nDispatch:\n",network.storage_units_t.p.loc[:,su])


for gen in network.generators.index:
    print(gen,network.carriers.loc[network.generators.loc[gen,"carrier"],"co2_emissions"]*(1/network.generators.loc[gen,"efficiency"]))




results_folder_name = os.path.join(csv_folder_name,"results")


network.export_to_csv_folder(results_folder_name)
