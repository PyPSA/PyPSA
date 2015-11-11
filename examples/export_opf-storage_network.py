# make the code as Python 3 compatible as possible
from __future__ import print_function, division

import pypsa

from pypsa.dicthelpers import attrfilter

import datetime
import pandas as pd

import networkx as nx

import numpy as np

from itertools import chain

import inspect

import os



csv_folder_name = 'opf-storage-data'



#Build the Network object, which stores all other objects

network = pypsa.Network()

#Build the snapshots we consider for the first T hours in 2015

T = 10

network.snapshots = pd.to_datetime([datetime.datetime(2015,1,1) + datetime.timedelta(hours=i) for i in range(T)])

print("network:",network)
print("snapshots:",network.snapshots)


#add fuel types
network.add("Source","gas",co2_emissions=0.24)
network.add("Source","wind")
network.add("Source","battery")

network.co2_limit = 1000


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
                capital_cost=0.1,
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
                capital_cost=0.1,
                s_nom_min=0,
                s_nom_extendable=True)


#add loads
for i in range(n*c):
    network.add("Load",i,bus=network.buses[str(i)])

#add some generators
for i in range(n*c):
    #storage
    network.add("StorageUnit","Storage %d" % (i),
                bus=network.buses[str(i)],
                p_nom=0,source=network.sources["battery"],
                marginal_cost=2,
                capital_cost=1000,
                p_nom_extendable=True,
                p_max_pu_fixed=1,
                p_min_pu_fixed=-1,
                efficiency_store=0.9,
                efficiency_dispatch=0.95,
                standing_loss=0.01,
                max_hours=6)
    #wind generator
    network.add("Generator","Wind %d" % (i),bus=network.buses[str(i)],
                p_nom=100,source=network.sources["wind"],dispatch="variable",
                marginal_cost=0,
                capital_cost=1000,
                p_nom_extendable=True,
                p_nom_max=None,
                p_nom_min=100)
    #gas generator
    network.add("Generator","Gas %d" % (i),bus=network.buses[str(i)],
                p_nom=0,source=network.sources["gas"],dispatch="flexible",
                marginal_cost=2,
                capital_cost=100,
                efficiency=0.35,
                p_nom_extendable=True,
                p_nom_max=None,
                p_nom_min=0)


#now attach some time series

network.load_series = pd.DataFrame(index = network.snapshots,
                                       columns = [load_name for load_name in network.loads],
                                       data = 1000*np.random.rand(len(network.snapshots), len(network.loads)))

for load in network.loads.itervalues():
    load.p_set = network.load_series[load.name]



wind_generators = attrfilter(network.generators,source=network.sources["wind"])

network.wind_series = pd.DataFrame(index = network.snapshots,
                                       columns = [gen.name for gen in wind_generators],
                                       data = np.random.rand(len(network.snapshots), len(wind_generators)))


for generator in wind_generators:
    generator.p_set = network.wind_series[generator.name]*generator.p_nom
    generator.p_max_pu = network.wind_series[generator.name]

for su in network.storage_units.itervalues():
    su.state_of_charge[network.snapshots[0]] = 0.0


for transport_link in network.transport_links.itervalues():
    transport_link.p_set = pd.Series(index = network.snapshots, data=(200*np.random.rand(len(network.snapshots))-100))



#now attach some time series

network.load_series = pd.DataFrame(index = network.snapshots,
                                       columns = [load_name for load_name in network.loads],
                                       data = 1000*np.random.rand(len(network.snapshots), len(network.loads)))

for load in network.loads.itervalues():
    load.p_set = network.load_series[load.name]



wind_generators = attrfilter(network.generators,source=network.sources["wind"])

network.wind_series = pd.DataFrame(index = network.snapshots,
                                       columns = [gen.name for gen in wind_generators],
                                       data = np.random.rand(len(network.snapshots), len(wind_generators)))


for generator in wind_generators:
    generator.p_set = network.wind_series[generator.name]*generator.p_nom
    generator.p_max_pu = network.wind_series[generator.name]

for su in network.storage_units.itervalues():
    su.state_of_charge[network.snapshots[0]] = 0.0


for transport_link in network.transport_links.itervalues():
    transport_link.p_set = pd.Series(index = network.snapshots, data=(200*np.random.rand(len(network.snapshots))-100))



time_series = {"generators" : {"p_max_pu" : lambda g: g.dispatch == "variable"}, "loads" : {"p_set" : None}}


network.name = "Test 6 bus"

network.export_to_csv_folder(csv_folder_name,time_series)
