
# coding: utf-8

# In[1]:

# make the code as Python 3 compatible as possible
from __future__ import print_function, division

import pypsa

from pypsa.dicthelpers import attrfilter

import datetime
import pandas as pd

import networkx as nx

import numpy as np

from itertools import chain


# In[2]:

csv_folder_name = "opf-storage-data"
network = pypsa.Network(csv_folder_name=csv_folder_name)
print(network,network.co2_limit)


# In[3]:

network.build_graph()


# In[4]:

network.determine_network_topology()


# In[5]:

print(network.sub_networks)


# In[6]:

from distutils.spawn import find_executable

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


# In[7]:

snapshots = network.snapshots[:4]
network.lopf(snapshots=snapshots,solver_name=solver_name)


# In[8]:

print("Generator and storage capacities:\n")

for one_port in chain(network.generators.itervalues(),network.storage_units.itervalues()):
    print(one_port,one_port.p_nom)

print("\n\nBranch capacities:\n")

for branch in network.branches.itervalues():
    print(branch,branch.s_nom)

for snapshot in snapshots:

    print("\n"*2+"For time",snapshot,":\nBus injections:")

    for bus in network.buses.itervalues():
        print(bus,bus.p[snapshot])
    print("Total:",sum([bus.p[snapshot] for bus in network.buses.itervalues()]))


# In[9]:


network.now = network.snapshots[0]


for branch in network.branches.itervalues():
    print(branch,branch.p1[network.now])


# In[10]:

print("Comparing bus injection to branch outgoing for %s:",network.now)

for sub_network in network.sub_networks.itervalues():

    print("\n\nConsidering sub network",sub_network,":")

    for bus in sub_network.buses.itervalues():

        print("\n%s" % bus)

        print("power injection (generators - loads + Transport feed-in):",bus.p[network.now])

        print("generators - loads:",sum([g.sign*g.p[network.now] for g in bus.generators.itervalues()])                        + sum([l.sign*l.p[network.now] for l in bus.loads.itervalues()]))

        total = 0.0

        for branch in sub_network.branches.itervalues():
            if bus == branch.bus0:
                print("from branch:",branch,branch.p0[network.now])
                total +=branch.p0[network.now]
            elif bus == branch.bus1:
                print("to branch:",branch,branch.p1[network.now])
                total +=branch.p1[network.now]
        print("branch injection:",total)


# In[11]:

for su in network.storage_units.values():
    print(su,su.p_nom,"\n",su.state_of_charge,"\n",su.p)


# In[12]:

for gen in network.generators.itervalues():
    print(gen,gen.source.co2_emissions*(1/gen.efficiency))
