
# coding: utf-8

# # Script to add load and generators to SciGRID
# 
# Load based on Landkreise, power plants from the BNetzA list.

# In[1]:


# make the code as Python 3 compatible as possible                                                                                          
from __future__ import print_function, division


import pypsa

import pandas as pd


# In[2]:

#note that some columns have 'quotes because of fields containing commas'
vertices = pd.read_csv("scigrid-151109/vertices_de_power_151109.csvdata",sep=",",quotechar="'",index_col=0)

vertices["v_nom"] = 380.

vertices.rename(columns={"lon":"x","lat":"y","name":"osm_name"},inplace=True)


# In[3]:

links = pd.read_csv("scigrid-151109/links_de_power_151109.csvdata",sep=",",quotechar="'",index_col=0)
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


# In[4]:

print(links["voltage"].value_counts(dropna=False))


# In[5]:

print(links[links["length_m"] <=0])


# In[6]:

print(links[(links["voltage"] != 220000) & (links["voltage"] != 380000)])


# In[7]:

print(links[pd.isnull(links.cables)])


# In[8]:

network = pypsa.Network()

pypsa.io.import_components_from_dataframe(network,vertices,"Bus")

pypsa.io.import_components_from_dataframe(network,links,"Line")


# In[9]:

network.build_graph()

network.determine_network_topology()


#remove small isolated networks
for sn in network.sub_networks.obj:
    print(sn,len(sn.buses))
    if len(sn.buses) < 5:
        print(sn.branches,sn.buses)
        for bus in sn.buses.obj:
            network.remove(bus)
        for branch in sn.branches.obj:
            network.remove(branch)
network.build_graph()

network.determine_network_topology()                


# In[10]:

#import FIAS libraries for attaching data - sorry, not free software yet

try:
    import vresutils
except:
    print("Oh dear! You don't have vresutils, so you cannot add load :-(")


# In[11]:

from vresutils import graph as vgraph
from vresutils import shapes as vshapes
from vresutils import grid as vgrid
from vresutils import dispatch as vdispatch
from shapely.geometry import Polygon


# In[12]:


#bounding poly for Germany
poly = Polygon([[5.8,47.],[5.8,55.5],[15.2,55.5],[15.2,47.]])


# In[13]:

import numpy as np

for bus in network.buses.obj:
    network.graph.node[bus.name]["pos"] = np.array([bus.x,bus.y])


# In[14]:

region = vshapes.germany()
#print(region.convex_hull)


# In[15]:

vgraph.voronoi_partition(network.graph, poly)


# In[16]:

network.graph.node[network.buses.index[0]]["region"]


# In[17]:

import load.germany as DEload


# In[18]:

type(network.graph)


# In[19]:

import networkx
#DEload only works on non-MultiGraph
g= networkx.Graph(network.graph)


# In[20]:

load = DEload.timeseries(g, years=[2011, 2012, 2013, 2014])



# In[21]:

import datetime
start = datetime.datetime(2011,1,1)
n_hours = 24
network.set_snapshots([start + datetime.timedelta(hours=i) for i in range(n_hours)])

network.now = network.snapshots[0]

print(network.snapshots)


# In[22]:

load[:len(network.snapshots),2].shape


# In[23]:

for i,bus in enumerate(network.buses.obj):
    network.add("Load",bus.name,bus=bus.name,p_set = pd.Series(data=1000*load[:len(network.snapshots),i],index=network.snapshots))


# In[24]:

print(len(network.loads))


# In[25]:

get_ipython().magic(u'matplotlib inline')


# In[26]:

pd.DataFrame(load.sum(axis=1)).plot()


# In[27]:

[k.osm_name for k,v in network.graph.node.iteritems() if 'region' not in v]


# In[28]:

#cap = vdispatch.backup_capacity_german_grid(network.graph)


# In[29]:

import random

def backup_capacity_german_grid(G):
    from shapely.geometry import Point

    plants = pd.read_csv("/home/vres/data/tom/playground/bnetza.csv")
    plants = plants[plants["Kraftwerksstatus"] == u"in Betrieb"]
    cells = {n: d["region"]
             for n, d in G.nodes_iter(data=True)}

    def nodeofaplant(x):
        if np.isnan(x["lon"]) or np.isnan(x["lat"]):
            return random.choice(cells.keys())
        p = Point(x["lon"], x["lat"])
        for n, cell in cells.iteritems():
            if cell.contains(p):
                return n
        else:
            return min(cells, key=lambda n: cells[n].distance(p))
    nodes = plants.apply(nodeofaplant, axis=1)

    capacity = plants['Netto-Nennleistung'].groupby((nodes, plants[u'Type'])).sum() / 1e3
    capacity.name = 'Capacity'

    return capacity


# In[30]:

cap = backup_capacity_german_grid(network.graph)


# In[31]:

cap.describe(),cap.sum(),type(cap)


# In[32]:

print(cap[pd.isnull(cap)])


# In[33]:

cap.fillna(0.1,inplace=True)


# In[34]:


cap.index.levels[1]


# In[35]:

m_costs = {"Gas" : 14.,
           "Coal" : 9.,
           "Oil" : 30.,
           "Nuclear" : 4.}



# In[36]:

for (bus_name,tech_name) in cap.index:
    print(bus_name,tech_name,cap[(bus_name,tech_name)])
    network.add("Generator",bus_name + " " + tech_name,bus=bus_name,p_nom=1000*cap[(bus_name,tech_name)],marginal_cost=m_costs.get(tech_name,1.))


# In[69]:

csv_folder_name = "scigrid-with-load-gen"

time_series = {"loads" : {"p_set" : None}}




network.export_to_csv_folder(csv_folder_name,time_series,verbose=False)

