
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


# In[2]:

#Build the Network object, which stores all other objects

network = pypsa.Network()

#Build the snapshots we consider for the first T hours in 2015

T = 10

network.index = pd.to_datetime([datetime.datetime(2015,1,1) + datetime.timedelta(hours=i) for i in range(T)])

network.now = network.index[0]

print("network:",network)
print("index:",network.index)
print("current snapshot:",network.now)


# In[3]:

#The network is two three-node AC networks connected by 2 point-to-point DC links

#building block
n = 3

#copies
c = 2


#add buses
for i in range(n*c):
    network.add("Bus",i,v_nom="380")

#add lines
for i in range(n*c):
    network.add("Line",i,
                bus0=network.buses[str(i)],
                bus1=network.buses[str(n*(i // n)+ (i+1) % n)],
                x=np.random.random(),
                s_nom=0,
                capital_cost=0.0,
                s_nom_min=0,
                s_nom_extendable=True)

#add HVDC lines
for i in range(2):
    network.add("TransportLink","TL %d" % (i),
                bus0=network.buses[str(i)],
                bus1=network.buses[str(3+i)],
                p_nom=1000,
                p_max=900,
                p_min=-900,
                s_nom=0,
                capital_cost=0.0,
                s_nom_min=0,
                s_nom_extendable=True)
    

#add loads
for i in range(n*c):
    network.add("Load",i,bus=network.buses[str(i)])

#add some generators
for i in range(n*c):
    #gas generator
    network.add("StorageUnit","Storage %d" % (i),
                bus=network.buses[str(i)],
                p_nom=0,source="storage",
                marginal_cost=0,
                capital_cost=5,
                p_nom_extendable=True,
                p_max_pu_fixed=1,
                p_min_pu_fixed=-1,
                max_hours=6)
    #wind generator
    network.add("Generator","Wind %d" % (i),bus=network.buses[str(i)],
                p_nom=100,source="wind",dispatch="variable",
                marginal_cost=0,
                capital_cost=10,
                p_nom_extendable=True,
                p_nom_max=400,
                p_nom_min=100)


# In[4]:

#now attach some time series
  
network.load_series = pd.DataFrame(index = network.index,
                                       columns = [load_name for load_name in network.loads],
                                       data = 1000*np.random.rand(len(network.index), len(network.loads)))

for load in network.loads.itervalues():
    load.p_set = network.load_series[load.name]

    

wind_generators = attrfilter(network.generators,source="wind")

network.wind_series = pd.DataFrame(index = network.index,
                                       columns = [gen.name for gen in wind_generators],
                                       data = np.random.rand(len(network.index), len(wind_generators)))


for generator in wind_generators:
    generator.p_set = network.wind_series[generator.name]*generator.p_nom
    generator.p_max_pu = network.wind_series[generator.name]
    
for su in network.storage_units.itervalues():
    su.state_of_charge[network.index[0]] = 0.0


for transport_link in network.transport_links.itervalues():
    transport_link.p_set = pd.Series(index = network.index, data=(200*np.random.rand(len(network.index))-100))


# In[5]:

network.build_graph()


# In[6]:

network.determine_network_topology()


# In[7]:

print(network.sub_networks)


# In[8]:

subindex = network.index[:2]
network.lopf(subindex=subindex)


# In[ ]:

now = network.index[1]

i = subindex.get_loc(now)

print(now,i)

previous = subindex[i-1]

print(previous)


# In[ ]:

print("Bus injections:")

for bus in network.buses.itervalues():
    print(bus,bus.p[network.now])
print("Total:",sum([bus.p[network.now] for bus in network.buses.itervalues()]))


# In[ ]:

for branch in network.branches.itervalues():
    print(branch,branch.p1[network.now])


# In[ ]:

network.now = network.index[0]

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


# In[ ]:

now = network.now
sub_network = network.sub_networks["1"]

print(sub_network.buses)


for t in sub_network.network.transport_links.itervalues():
    if t.bus0.name in sub_network.buses:
        t.bus0.p[now] += t.p0[now]
        print(t,"leaves",t.bus0,t.p0[now])
    if t.bus1.name in sub_network.buses:
        t.bus1.p[now] += t.p1[now]
        print(t,"arrives",t.bus1,t.p1[now])


# In[ ]:

now = network.now
print(now)

for generator in network.generators.itervalues():
    print(generator,generator.p[now],generator.p_nom)


# In[ ]:

for load in network.loads.itervalues():
    print(load,load.p[network.now])


# In[ ]:

for branch in network.branches.itervalues():
    print(branch,branch.s_nom)


# In[ ]:

for v in network.model.branch_s_nom:
    print(v)


# In[ ]:

s = pd.Series(data=np.nan,index=network.index,dtype=float)


# In[ ]:

s


# In[ ]:

dt = datetime.timedelta(hours=4)


# In[ ]:

dt.total_seconds


# In[ ]:



