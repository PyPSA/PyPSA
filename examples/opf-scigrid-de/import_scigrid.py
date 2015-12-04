
# coding: utf-8

# In[184]:


# make the code as Python 3 compatible as possible                                                                                          
from __future__ import print_function, division


import pypsa

import pandas as pd


# In[185]:

#note that some columns have 'quotes because of fields containing commas'
vertices = pd.read_csv("vertices_de_power_151109.csvdata",sep=",",quotechar="'",index_col=0)

vertices["v_nom"] = 380.

vertices.rename(columns={"lon":"x","lat":"y","name":"osm_name"},inplace=True)


# In[186]:

print(vertices.columns)


# In[187]:

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


# In[188]:

print(links.columns)


# In[189]:

print(links["voltage"].value_counts(dropna=False))


# In[190]:

print(links[links["length_m"] <=0])


# In[191]:

print(links[(links["voltage"] != 220000) & (links["voltage"] != 380000)])


# In[192]:

print(links[pd.isnull(links.cables)])


# In[193]:

network = pypsa.Network()

pypsa.io.import_components_from_dataframe(network,vertices,"Bus")

pypsa.io.import_components_from_dataframe(network,links,"Line")


# In[194]:

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
                


# In[195]:

network.lpf()


# In[196]:

try:
    import vresutils
except:
    print("Oh dear! You don't have vresutils, so you cannot add load :-(")


# In[197]:

from vresutils import graph as vgraph
from vresutils import shapes as vshapes
from vresutils import grid as vgrid
from vresutils import dispatch as vdispatch
from shapely.geometry import Polygon


# In[198]:

poly = Polygon([[5.8,47.],[5.8,55.5],[15.2,55.5],[15.2,47.]])


# In[199]:

poly


# In[200]:

import numpy as np

for bus in network.buses.itervalues():
    network.graph.node[bus]["pos"] = np.array([bus.x,bus.y])


# In[201]:

region = vshapes.germany()
#print(region.convex_hull)


# In[202]:

vgraph.voronoi_partition(network.graph, poly)


# In[203]:

network.graph.node[network.buses.itervalues().next()]["region"]


# In[204]:

import load.germany as DEload


# In[205]:

load = DEload.timeseries(network.graph, years=[2011, 2012, 2013, 2014])



# In[206]:

import datetime
start = datetime.datetime(2011,1,1)
n_hours = 24
network.snapshots = [start + datetime.timedelta(hours=i) for i in range(n_hours)]

network.now = network.snapshots[0]

network.snapshots


# In[207]:

load[:len(network.snapshots),2].shape


# In[208]:

for i,bus in enumerate(network.buses.values()):
    network.add("Load",bus.name,bus=bus,p_set = pd.Series(data=1000*load[:len(network.snapshots),i],index=network.snapshots))


# In[209]:

print(len(network.loads))


# In[210]:

get_ipython().magic(u'matplotlib inline')


# In[211]:

pd.DataFrame(load.sum(axis=1)).plot()


# In[212]:

[k.osm_name for k,v in network.graph.node.iteritems() if 'region' not in v]


# In[213]:

#cap = vdispatch.backup_capacity_german_grid(network.graph)


# In[214]:

import random

def backup_capacity_german_grid(G):
    from shapely.geometry import Point

    plants = pd.read_csv("/home/tom/fias-shared/playground/bnetza.csv")
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


# In[215]:

cap = backup_capacity_german_grid(network.graph)


# In[216]:

cap.describe(),cap.sum(),type(cap)


# In[217]:

print(cap[pd.isnull(cap)])


# In[218]:

cap.fillna(0.1,inplace=True)


# In[219]:


cap.index.levels[1]


# In[220]:

m_costs = {"Gas" : 14.,
           "Coal" : 9.,
           "Oil" : 30.,
           "Nuclear" : 4.}



# In[221]:

for (bus,tech_name) in cap.index:
    print(bus,tech_name,cap[(bus,tech_name)])
    network.add("Generator",bus.name + " " + tech_name,bus=bus,p_nom=1000*cap[(bus,tech_name)],marginal_cost=m_costs.get(tech_name,1.))


# In[222]:

cap.values.shape


# In[223]:

##plot graph!!!


# In[224]:

print(len(network.generators))


# In[225]:

slack = next(network.generators.itervalues())
slack.bus.control = "Slack"


# In[226]:

network.lpf()


# In[227]:

print(sum(g.p[network.now] for g in network.generators.values()))


# In[228]:

print(sum(g.p_nom for g in network.generators.values()))


# In[229]:

print(sum(l.p[network.now] for l in network.loads.values()))


# In[230]:

loading = pd.Series([b.p0[network.now] for b in network.branches.values()])
caps = pd.Series([b.s_nom for b in network.branches.values()])


# In[231]:

loading.describe()


# In[232]:

caps.describe()


# In[233]:

loading_pu = abs(loading)/caps

loading_pu.describe()


# In[234]:

angles = pd.Series([b.v_ang[network.now] for b in network.buses.values()])
angles.describe()


# In[235]:

len(network.buses)


# In[236]:

for b in network.branches.values():
    print(b,b.s_nom)
    b.s_nom_old = b.s_nom
    b.s_nom_extendable = True
    b.s_nom_min = b.s_nom
    b.capital_cost = 10000.


# In[237]:

network.lopf(network.snapshots[:2])


# In[254]:

for i,b in enumerate(network.branches.values()):
    
    if abs(b.s_nom - b.s_nom_old) > 0.1:
        print(i,b,b.s_nom,b.s_nom_old,b.bus0.x,b.bus0.y,b.length,b.voltage)


# In[239]:

import networkx as nx

import matplotlib.pyplot as plt

import vresutils.plot as vplot


# In[240]:

segments = np.array([[1.] for bus in network.buses.values()])

segments.shape


# In[241]:

len(network.graph.edges())


# In[251]:

fig, ax = plt.subplots(figsize=(10,10))


nx.draw_networkx_edges(network.graph, pos = {bus : [bus.x,bus.y] for bus in network.buses.values()}, edgelist = [(b.bus0,b.bus1,b) for b in network.branches.values()],edge_color=['r' if abs(b.s_nom-b.s_nom_old) > 0.1 else "c" for b in network.branches.values()],
                     width=np.array([b.s_nom/1000. for b in network.branches.values()]), ax=ax)

nx.draw_networkx_nodes(network.graph, pos=nx.get_node_attributes(network.graph, 'pos'), node_size=[load.p_set[network.now]/10. for load in network.loads.values()], ax=ax)

ax.set_xlim([8,10])
ax.set_ylim([47,50.5])


# In[243]:


#Line 350 285.826818133 247.683265482 9.42551238607 48.0643507981 132.974
#Line 583 969.851606077 495.366530965 8.95464412134 50.0860157333 28.905


# In[244]:

colors = ['r' if abs(b.s_nom-b.s_nom_old) > 0.1 else "c" for b in network.branches.values()]


# In[245]:

for i in range(len(colors)):
    if colors[i] == "r":
        print(i)


# In[246]:

len(network.branches)


# In[247]:

len(network.graph.edges())


# In[248]:

for i in range(822):
    print(network.graph.edges(keys=True)[i],network.branches.values()[i])


# In[249]:

get_ipython().magic(u'pinfo network.graph.edges')


# In[250]:

network.graph.g


# In[ ]:

b = next(network.branches.itervalues())

print(b,b.bus0,b.bus1)

network.graph.edge[b.bus0][b.bus1][b]


# # Look around specific nodes

# In[261]:

name = "Conneforde"
#name = "Dipperz"


buses = list(filter(lambda b: type(b.osm_name) == str and name in b.osm_name,network.buses.values()))
for bus in buses:
    print(bus,bus.osm_name,bus.voltage)
    
bus = network.buses[str(159)]

print(bus,bus.osm_name,bus.voltage)


# In[262]:

lines = list(filter(lambda l: l.bus0 == bus or l.bus1 == bus,network.lines.values()))


# In[263]:




for l in lines:
    print(l,l.bus0.osm_name,l.bus1.osm_name,l.x,l.voltage,l.length)


# In[ ]:

ed

