

#In this example the network is built up by hand...

# coding: utf-8

# In[1]:

import pypsa

import datetime,pandas

import networkx as nx

import numpy as np


# In[2]:

#Build the Network object, which stores all other objects

network = pypsa.Network()

#Build the snapshots we consider for the first T hours in 2015

T = 10

network.index = [datetime.datetime(2015,1,1) + datetime.timedelta(hours=i) for i in range(T)]

network.i = network.index[0]

print network
print network.index
print network.i


# In[10]:

#Build buses

#building block
n = 3

#copies
c = 2

network.buses = [network.add("Bus",i) for i in range(n*c)]

network.lines = [network.add("Line",i) for i in range(n*c)]
    
    
for i,line in enumerate(network.lines):
    line.bus0 = network.buses[i]
    line.bus1 = network.buses[n*(i // n)+ (i+1) % n]
    
network.branches = network.lines

#add loads with random values

network.loads = [network.add("Load",i) for i in range(n*c)]

for i,load in enumerate(network.loads):
    load.bus = network.buses[i]
    load.bus.loads = [load]
    
network.load_series = pandas.DataFrame(index = network.index,
                                       columns = [load.name for load in network.loads],
                                       data = -100*np.random.rand(len(network.index), len(network.loads)))
for load in network.loads:
    load.p_set_series = network.load_series[load.name]
    
#add generators

network.generators = [network.add("Generator",i) for i in range(n*c)]

for i,generator in enumerate(network.generators):
    generator.bus = network.buses[i]
    generator.bus.generators =  [generator]
    generator.p_nom = 100
    generator.source = "gas"
    generator.dispatch = "flexible"


# In[11]:

generator.p_nom


# In[12]:

network.build_graph()


# In[13]:

network.determine_network_topology()


# In[14]:

network.sub_networks


for sn in network.sub_networks:
    print "\nsub network",sn,"contains:"
    print "buses:",sn.buses
    print "branches:",sn.branches


# In[15]:

sn = network.sub_networks[0]
sn.pf()
