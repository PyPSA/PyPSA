
# coding: utf-8

# # Non-linear power flow with SciGRID
# 
# 
# In this example, the dispatch of generators is optimised using the linear OPF, then a non-linear power flow is run on the resulting dispatch.

# In[1]:

#make the code as Python 3 compatible as possible                                                                               \
                                                                                                                                  
from __future__ import print_function, division
from __future__ import absolute_import

import pypsa

from scipy.sparse.linalg import spsolve

from scipy.sparse import csr_matrix,csc_matrix

import numpy as np

import pandas as pd

import os


# In[2]:

csv_folder_name = os.path.dirname(pypsa.__file__) + "/../examples/opf-scigrid-de/scigrid-with-load-gen/"

network = pypsa.Network(csv_folder_name=csv_folder_name)


# In[3]:


#Add dummy generators to add reactive power where necessary,
#since we don't now where there are reactive power assets in the grid

for bus in network.buses.obj:
    if len(bus.generators()) == 0:
        network.add("Generator",bus.name,bus=bus.name,control="PV")


# In[4]:

#allow lines to be extended (otherwise LOPF is infeasible)
network.lines.s_nom_extendable = True

#keep a record of the original line capacity
network.lines["s_nom_old"] = network.lines["s_nom"]

#don't allow the optimisation to reduce line capacity
network.lines["s_nom_min"] = network.lines["s_nom"]

#give the lines some cost
network.lines["capital_cost"] = 400*network.lines["length"]


# In[5]:

network.now = network.snapshots[0]


# In[6]:

print("Performing linear OPF:")

network.lopf()


# In[7]:

network.lines["s_nom"] = network.lines["s_nom_opt"]

print("The following lines were extended:")

print(network.lines[["bus0","bus1","s_nom_old","s_nom"]][abs(network.lines["s_nom"] - network.lines["s_nom_old"]) > 0.1])


# In[8]:

print("With the non-linear load flow, there is the following per unit overloading:")
print((network.lines_t.p0.loc[network.now]/network.lines.s_nom).describe())


# In[9]:

#For the PF, set the P to the optimised P
network.generators_t.p_set.loc[network.now] = network.generators_t.p.loc[network.now] 


# In[10]:


#set all buses to PV, since we don't know what Q set points are
network.generators.control = "PV"

#set slack
#network.generators.loc["1 Coal","control"] = "Slack"


#Need some PQ buses so that Jacobian doesn't break
f = network.generators[network.generators.bus == "492"]
network.generators.loc[f.index,"control"] = "PQ"


# In[11]:


print("Performing non-linear PF on results of LOPF:")

network.pf()


# In[12]:

print("With the non-linear load flow, there is the following per unit overloading:")
print((network.lines_t.p0.loc[network.now]/network.lines.s_nom).describe())


# In[13]:

df = network.lines.copy()

for b in ["bus0","bus1"]:
    df = pd.merge(df,network.buses_t.v_ang.loc[[network.now]].T,how="left",
         left_on=b,right_index=True)

s = df[str(network.now)+"_x"]- df[str(network.now)+"_y"]


# In[14]:

print("The voltage angle differences across the lines have (in degrees):")
print((s*180/np.pi).describe())


# In[ ]:



