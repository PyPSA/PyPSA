
# coding: utf-8

# In[16]:


# make the code as Python 3 compatible as possible                                                                                          
from __future__ import print_function, division


import pypsa

import pandas as pd


# In[17]:

#note that some columns have 'quotes because of fields containing commas'
vertices = pd.read_csv("vertices_de_power_151109.csvdata",sep=",",quotechar="'",index_col=0)

vertices.rename(columns={"lon":"x","lat":"y","name":"osm_name"},inplace=True)


# In[18]:

print(vertices.columns)


# In[19]:

links = pd.read_csv("links_de_power_151109.csvdata",sep=",",quotechar="'",index_col=0)
links.rename(columns={"v_id_1":"bus0","v_id_2":"bus1","name":"osm_name"},inplace=True)

links["cables"].fillna(3,inplace=True)
links["wires"].fillna(2,inplace=True)

links["length"] = links["length_m"]/1000.

default = dict(wires_typical=2.0, r=0.08, x=0.32, c=11.5, i=1.3)

coeffs = {
        220000: dict(wires_typical=2.0, r=0.08, x=0.32, c=11.5, i=1.3),
        380000: dict(wires_typical=4.0, r=0.025, x=0.25, c=13.7, i=2.6)
    }

links["r"] = [row["length"]*coeffs.get(row["voltage"],default)["r"]/(row["wires"]/coeffs.get(row["voltage"],default)["wires_typical"])/(row["cables"]/3.)  for i,row in links.iterrows()]

links["x"] = [row["length"]*coeffs.get(row["voltage"],default)["x"]/(row["wires"]/coeffs.get(row["voltage"],default)["wires_typical"])/(row["cables"]/3.)  for i,row in links.iterrows()]

links["s_nom"] = [3.**0.5*row["voltage"]/1000.*coeffs.get(row["voltage"],default)["i"]*(row["wires"]/coeffs.get(row["voltage"],default)["wires_typical"])*(row["cables"]/3.)  for i,row in links.iterrows()]


# In[21]:

print(links.columns)


# In[22]:

links["voltage"].value_counts(dropna=False)


# In[7]:

print(links[links["length_m"] <=0])


# In[23]:

print(links[(links["voltage"] != 220000) & (links["voltage"] != 380000)])


# In[28]:

print(links[pd.isnull(links.cables)])


# In[29]:

network = pypsa.Network()

pypsa.io.import_components_from_dataframe(network,vertices,"Bus")

pypsa.io.import_components_from_dataframe(network,links,"Line")


# In[31]:

network.build_graph()

network.determine_network_topology()


#remove small networks, which break the load flow
for sn in network.sub_networks.itervalues():
    print(sn,len(sn.buses))
    if len(sn.buses) < 5:
        print(sn.branches,sn.buses)
        for bus in sn.buses.values():
            network.remove(bus)
        for branch in sn.branches.values():
            network.remove(branch)
                


# In[32]:

network.lpf()


# In[ ]:



