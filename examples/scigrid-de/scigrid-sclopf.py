

from __future__ import print_function, division, absolute_import

import pypsa, os

csv_folder_name = os.path.dirname(pypsa.__file__) + "/../examples/scigrid-de/scigrid-with-load-gen/"

network = pypsa.Network(csv_folder_name=csv_folder_name)


#There are some infeasibilities at the edge of the network where loads
#are supplied by foreign lines - just add some extra capacity in Germany
for line_name in ["350","583"]:
    network.lines.loc[line_name,"s_nom"] += 500


network.now = network.snapshots[0]

branch_outages = network.lines.index[:15]

print("Performing security-constrained linear OPF:")

network.sclopf(branch_outages=branch_outages)


#For the PF, set the P to the optimised P
network.generators_t.p_set.loc[network.now] = network.generators_t.p.loc[network.now]
network.storage_units_t.p_set.loc[network.now] = network.storage_units_t.p.loc[network.now]


p0_test = network.lpf_contingency(branch_outages=branch_outages)


max_loading = abs(p0_test.divide(network.passive_branches().s_nom,axis=0)).describe().loc["max"]

print(max_loading)
