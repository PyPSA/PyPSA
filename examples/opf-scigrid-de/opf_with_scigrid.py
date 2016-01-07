
# coding: utf-8

# # Script to perform linear OPF with SciGRID data
# 
# With load and generators attached by script in this directory.

# In[1]:


# make the code as Python 3 compatible as possible                                                                                          
from __future__ import print_function, division


import pypsa


# In[2]:

network = pypsa.Network(csv_folder_name="scigrid-with-load-gen/")


# In[15]:

network.build_graph()

network.determine_network_topology()

snapshots = network.snapshots[:2]

solver_name = "glpk"

network.lopf(snapshots,solver_name)


# In[35]:

for snapshot in snapshots:
    print("\n"*2,"For snapshot",snapshot,":\n")
    network.now = snapshot
    print("power injections sum:",network.buses.p.loc[network.now].sum())
    print("generators sum:",network.generators.p.loc[network.now].sum())
    print("loads sum:",network.loads.p.loc[network.now].sum())


# In[21]:

network.lines["max_loading"] = abs(network.lines.p0.loc[snapshots]).max(axis=0)


# In[34]:

network.lines["loading_pu"] = network.lines["max_loading"]/network.lines['s_nom']

print("Statistics of maximum line loading (per unit of capacity) for snapshots:\n")
print(network.lines["loading_pu"].describe())


# In[ ]:




# In[ ]:



