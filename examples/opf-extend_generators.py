
# coding: utf-8

# In[14]:

# make the code as Python 3 compatible as possible                                                                       
from __future__ import print_function, division

import pypsa

from pypsa.dicthelpers import attrfilter

import datetime
import pandas as pd

import networkx as nx

import numpy as np


# In[27]:

#Build the Network object, which stores all other objects

network = pypsa.Network()

#Build the snapshots we consider for the first T hours in 2015

T = 10

network.index = pd.to_datetime([datetime.datetime(2015,1,1) + datetime.timedelta(hours=i) for i in range(T)])

network.now = network.index[0]

print("network:",network)
print("index:",network.index)
print("current snapshot:",network.now)


# In[28]:

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
                s_nom=1500)

#add HVDC lines
for i in range(2):
    network.add("TransportLink","TL %d" % (i),
                bus0=network.buses[str(i)],
                bus1=network.buses[str(3+i)],
                p_nom=1000,
                p_max=900,
                p_min=-900)
    

#add loads
for i in range(n*c):
    network.add("Load",i,bus=network.buses[str(i)])

#add some generators
for i in range(n*c):
    #gas generator
    network.add("Generator","Gas %d" % (i),bus=network.buses[str(i)],p_nom=1000,source="gas",dispatch="flexible",marginal_cost=50)
    #wind generator
    network.add("Generator","Wind %d" % (i),bus=network.buses[str(i)],
                p_nom=100,source="wind",dispatch="variable",
                marginal_cost=0,
                capital_cost=10,
                p_nom_extendable=True,
                p_nom_max=200,
                p_nom_min=100)


# In[29]:

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
    
gas_generators = attrfilter(network.generators,source="gas")

for generator in gas_generators:
    generator.p_set = pd.Series([1.]*len(network.index),network.index)


for transport_link in network.transport_links.itervalues():
    transport_link.p_set = pd.Series(index = network.index, data=(200*np.random.rand(len(network.index))-100))


# In[30]:

network.build_graph()


# In[31]:

network.determine_network_topology()


# In[32]:

print(network.sub_networks)


# In[33]:

network.lopf(subindex=network.index[:2])


# In[34]:

print("Bus injections:")

for bus in network.buses.itervalues():
    print(bus,bus.p[network.now])
print("Total:",sum([bus.p[network.now] for bus in network.buses.itervalues()]))


# In[35]:

for branch in network.branches.itervalues():
    print(branch,branch.p1[network.now])


# In[36]:

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


# In[37]:

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


# In[38]:

now = network.now
print(now)

for generator in network.generators.itervalues():
    print(generator,generator.p[now],generator.p_nom)


# In[14]:

for load in network.loads.itervalues():
    print(load,load.p[network.now])


# In[ ]:



