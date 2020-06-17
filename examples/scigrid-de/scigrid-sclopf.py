## Security-Constrained LOPF with SciGRID
#
#This Jupyter Notebook is also available to download at: <http://www.pypsa.org/examples/scigrid-sclopf.ipynb>  and can be viewed as an HTML page at: <http://pypsa.org/examples/scigrid-sclopf.html>.
#
#In this example, the dispatch of generators is optimised using the security-constrained linear OPF, to guaranteed that no branches are overloaded by certain branch outages.
#
#The data files for this example are in the examples folder of the github repository: <https://github.com/PyPSA/PyPSA>.
#
### Data sources and health warnings
#
#See the separate notebook at <http://www.pypsa.org/examples/add_load_gen_trafos_to_scigrid.ipynb>.

import pypsa, os

csv_folder_name = os.path.dirname(pypsa.__file__) + "/../examples/scigrid-de/scigrid-with-load-gen-trafos/"

network = pypsa.Network(csv_folder_name=csv_folder_name)

#There are some infeasibilities without line extensions                                                                                  
for line_name in ["316","527","602"]:
    network.lines.loc[line_name,"s_nom"] = 1200


now = network.snapshots[0]

branch_outages = network.lines.index[:15]

print("Performing security-constrained linear OPF:")

network.sclopf(now,branch_outages=branch_outages)
print("Objective:",network.objective)



#For the PF, set the P to the optimised P
network.generators_t.p_set = network.generators_t.p_set.reindex(columns=network.generators.index)
network.generators_t.p_set.loc[now] = network.generators_t.p.loc[now]
network.storage_units_t.p_set = network.storage_units_t.p_set.reindex(columns=network.storage_units.index)
network.storage_units_t.p_set.loc[now] = network.storage_units_t.p.loc[now]

#Check no lines are overloaded with the linear contingency analysis

p0_test = network.lpf_contingency(now,branch_outages=branch_outages)

p0_test

#check loading as per unit of s_nom in each contingency

max_loading = abs(p0_test.divide(network.passive_branches().s_nom,axis=0)).describe().loc["max"]

print(max_loading)

import numpy as np
np.testing.assert_array_almost_equal(max_loading,np.ones((len(max_loading))))

