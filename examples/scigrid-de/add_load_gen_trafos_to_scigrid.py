## Investigate trafo situation
#
#Which nodes are within 0.5 km of each other but not connected? ant which voltage levels?

# make the code as Python 3 compatible as possible
from __future__ import print_function, division,absolute_import

import pypsa

import pandas as pd

import numpy as np

from six import iteritems
from six.moves import range

import os

import matplotlib.pyplot as plt

#%matplotlib inline

#You may have to adjust this path to where 
#you downloaded the github repository
#https://github.com/FRESNA/PyPSA

folder_prefix = os.path.dirname(pypsa.__file__) + "/../examples/opf-scigrid-de/"

#note that some columns have 'quotes because of fields containing commas'
vertices = pd.read_csv(folder_prefix+"scigrid-151109/vertices_de_power_151109.csvdata",sep=",",quotechar="'",index_col=0)

vertices.rename(columns={"lon":"x","lat":"y","name":"osm_name"},inplace=True)

links = pd.read_csv(folder_prefix+"scigrid-151109/links_de_power_151109.csvdata",sep=",",quotechar="'",index_col=0)
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

# if g = 0, b = 2*pi*f*C; C is in nF
links["b"] = [2*np.pi*50*1e-9*row["length"]*coeffs.get(row["voltage"],default)["c"]*(row["wires"]/coeffs.get(row["voltage"],default)["wires_typical"])*(row["cables"]/3.)  for i,row in links.iterrows()]

links["s_nom"] = [3.**0.5*row["voltage"]/1000.*coeffs.get(row["voltage"],default)["i"]*(row["wires"]/coeffs.get(row["voltage"],default)["wires_typical"])*(row["cables"]/3.)  for i,row in links.iterrows()]

print(vertices["voltage"].value_counts(dropna=False))

print(links["voltage"].value_counts(dropna=False))

## Drop the DC lines

for voltage in [300000,400000,450000]:
    links.drop(links[links.voltage == voltage].index,inplace=True)

network = pypsa.Network()

pypsa.io.import_components_from_dataframe(network,vertices,"Bus")

pypsa.io.import_components_from_dataframe(network,links,"Line")

network.build_graph()

network.determine_network_topology()


#remove small isolated networks
for sn in network.sub_networks.obj:
    buses = sn.buses()
    branches = sn.branches()
    print(sn,len(buses))
    if len(buses) < 5:
        print(branches,sn.buses)
        for bus in buses.obj:
            network.remove("Bus",bus.name)
        for branch in branches.obj:
            network.remove("Line",branch.name)

network.build_graph()

network.determine_network_topology()

colors = network.lines.voltage.map(lambda v: "g" if v == 220000 else "r" if v == 380000 else "c")

network.plot(line_colors=colors)

#how many vertices are within x km of each other

x = 0.2 #km

count = 0

for v in network.buses.index:
    lon = np.deg2rad(network.buses["x"])
    lat = np.deg2rad(network.buses["y"])
    
    lon_v = np.deg2rad(network.buses.at[v,"x"])
    lat_v = np.deg2rad(network.buses.at[v,"y"])
    
    a = np.sin((lat-lat_v)/2.)**2 + np.cos(lat_v) * np.cos(lat) * np.sin((lon_v - lon)/2.)**2
   
    dist_km = 6371.000 * 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )

    near = dist_km[dist_km < x]
    
    near = near.drop(v)
    
    for w in near.index:
        if w not in network.graph.adj[v]:
            print(v,w,near[w])
    
    if len(near) > 0:
        count +=1
print(count)

### Split buses with more than one voltage; add trafos between
#
#This code splits the buses where you have 220 and 380 kV lines landing.

buses_by_voltage = {}

for voltage in network.lines.voltage.value_counts().index:
    buses_by_voltage[voltage] = set(network.lines[network.lines.voltage == voltage].bus0)\
                                | set(network.lines[network.lines.voltage == voltage].bus1)

network.buses.loc[buses_by_voltage[220000],"v_nom"] = 220
network.buses.loc[buses_by_voltage[380000],"v_nom"] = 380

overlap = buses_by_voltage[220000] & buses_by_voltage[380000]
len(overlap)

## build up new buses and transformers to import


buses_to_split = [str(i) for i in sorted([int(item) for item in overlap])]
buses_to_split_df = network.buses.loc[buses_to_split]
#displace the 220 kV buses slightly to the right so that
#Voronoi partition will split them
buses_to_split_df.x+=0.005

buses_to_split_df.v_nom = 220

buses_to_split_220kV = [name + "_220kV" for name in buses_to_split_df.index]

buses_to_split_df.index = buses_to_split_220kV

trafos_df = pd.DataFrame(index=buses_to_split)
trafos_df["bus0"] = buses_to_split
trafos_df["bus1"] = buses_to_split_220kV
trafos_df["x"] = 0.1
#This high a nominal power is required for feasibility in LOPF
trafos_df["s_nom"] = 2000

pypsa.io.import_components_from_dataframe(network,buses_to_split_df,"Bus")
pypsa.io.import_components_from_dataframe(network,trafos_df,"Transformer")

##reconnect lines to the correct voltage bus

for line in network.lines.index:
    bus0 = network.lines.at[line,"bus0"]
    bus1 = network.lines.at[line,"bus1"]
    v0 = network.buses.at[bus0,"v_nom"]
    v1 = network.buses.at[bus1,"v_nom"]
    v = network.lines.at[line,"voltage"]
    if v0 != v/1000.:
        print(line,v0,v)
        network.lines.at[line,"bus0"] = bus0+"_220kV"
    if v1 != v/1000.:
        network.lines.at[line,"bus1"] = bus1+"_220kV"

network.build_graph()

network.determine_network_topology()


#remove small isolated networks
for sn in network.sub_networks.obj:
    buses = sn.buses()
    branches = sn.branches()
    print(sn,len(buses))
    if len(buses) < 5:
        print(branches,sn.buses)
        for bus in buses.obj:
            network.remove("Bus",bus.name)
        for branch in branches.obj:
            network.remove("Line",branch.name)
network.build_graph()

network.determine_network_topology()                

## Attach the load

#import FIAS libraries for attaching data - sorry, not free software yet

try:
    import vresutils, load
except:
    print("Oh dear! You don't have FIAS libraries, so you cannot add load :-(")

import load

from vresutils import graph as vgraph
from vresutils import shapes as vshapes
from vresutils import grid as vgrid
from vresutils import dispatch as vdispatch
from shapely.geometry import Polygon
from load import germany as DEload


#bounding poly for Germany for the Voronoi - necessary
#because some SciGRID points lie outside border vshapes.germany()
poly = Polygon([[5.8,47.],[5.8,55.5],[15.2,55.5],[15.2,47.]])


#add positions to graph for voronoi cell computation
for bus in network.buses.obj:
    network.graph.node[bus.name]["pos"] = np.array([bus.x,bus.y])

network.graph.name = "scigrid-with_trafos"

vgraph.voronoi_partition(network.graph, poly)

#NB: starts at midnight CET, 23:00 UTC
load = DEload.timeseries(network.graph, years=[2011, 2012, 2013, 2014])

#Kill the Timezone information to avoid pandas bugs
load.index = load.index.values

#Take the first day (in UTC time - we don't set time zone because of a Pandas bug)
network.set_snapshots(pd.date_range("2011-01-01 00:00","2011-01-01 23:00",freq="H"))

network.now = network.snapshots[0]

print(network.snapshots)

for bus in network.buses.obj:
    network.add("Load",bus.name,bus=bus.name,
                p_set = pd.Series(data=1000*load.loc[load.index[1:25],bus.name],index=network.snapshots))

#%matplotlib inline

pd.DataFrame(load.sum(axis=1)).plot()

load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()

network.plot(bus_sizes=load_distribution)

## Attach conventional generators from BNetzA list

from vresutils import shapes as vshapes

def read_kraftwerksliste(with_latlon=True):                                                                              
                                                                                                              
    kraftwerke = pd.read_csv('../../lib/vresutils/data/Kraftwerksliste_CSV_deCP850ed.csv',                                         
                             delimiter=';', encoding='utf-8', thousands='.', decimal=',')                                
    def sanitize_names(x):                                                                                               
        try:                                                                                                             
            x = x[:x.index('(')]                                                                                         
        except ValueError:                                                                                               
            pass                                                                                                         
        return x.replace(u'\n', u' ').strip()
    kraftwerke.columns = kraftwerke.columns.map(sanitize_names)
    
    def sanitize_plz(x):
        try:
            x = x.strip()
            if len(x) > 5:
                x = x[:5]
            return float(x)
        except (ValueError, AttributeError):
            return np.NAN
    kraftwerke.PLZ = kraftwerke.PLZ.apply(sanitize_plz)
    if with_latlon:
        postcodes = {pc: sh.centroid
                     for pc, sh in iteritems(vshapes.postcodeareas())
                     if sh is not None}
        kraftwerke['lon'] = kraftwerke.PLZ.map({pc: c.x for pc, c in iteritems(postcodes)})
        kraftwerke['lat'] = kraftwerke.PLZ.map({pc: c.y for pc, c in iteritems(postcodes)})
        #kraftwerke.dropna(subset=('lon','lat'), inplace=True)                                                           

    kraftwerke[u'Type'] = kraftwerke[u"Auswertung Energieträger"].map({
        u'Erdgas': u'Gas',
        u'Grubengas': u'Gas',
        u'Laufwasser': u'Run of River',
        u'Pumpspeicher': u'Pumped Hydro',
        u'Speicherwasser (ohne Pumpspeicher)': u'Storage Hydro',
        u'Mineralölprodukte': u'Oil',
        u'Steinkohle': u'Hard Coal',
        u'Braunkohle': u'Brown Coal',
        u'Abfall': u'Waste',
        u'Kernenergie': u'Nuclear',
        u'Sonstige Energieträger\n(nicht erneuerbar) ': u'Other',
        u'Mehrere Energieträger\n(nicht erneuerbar)': u'Multiple',
        u'Biomasse' : u'Biomass',
        u'Deponiegas' : u'Gas',
        u'Klärgas' : u'Gas',
        u'Geothermie' : u'Geothermal',
        u'Windenergie (Onshore-Anlage)' : u'Wind Onshore',
        u'Windenergie (Offshore-Anlage)' : u'Wind Offshore',
        u'Solare Strahlungsenergie' : u'Solar',
        u'Unbekannter Energieträger\n(nicht erneuerbar)' : u'Other'
    })

    return kraftwerke

power_plants = read_kraftwerksliste()

power_plants[power_plants[u"Unternehmen"] == "EEG-Anlagen < 10 MW"].groupby(u"Type").sum()

import random

#NB: bnetza extracted from BNetzA using

#./Kraftwerksdaten.ipynb


def backup_capacity_german_grid(G):   

    from shapely.geometry import Point

    plants = power_plants
    plants = plants[plants["Kraftwerksstatus"] == u"in Betrieb"]
    
    #remove EEG-receiving power plants - except biomass, these will be added later
    
    #it's necessary to remove biomass because we don't have coordinates for it
    
    for tech in ["Solar","Wind Onshore","Wind Offshore","Biomass"]:
        plants = plants[plants['Type'] != tech]
    
    cells = {n: d["region"]
             for n, d in G.nodes_iter(data=True)}

    def nodeofaplant(x):
        if np.isnan(x["lon"]) or np.isnan(x["lat"]):
            return random.choice(list(cells.keys()))
        p = Point(x["lon"], x["lat"])
        for n, cell in iteritems(cells):
            if cell.contains(p):
                return n
        else:
            return min(cells, key=lambda n: cells[n].distance(p))
    nodes = plants.apply(nodeofaplant, axis=1)

    capacity = plants['Netto-Nennleistung'].groupby((nodes, plants[u'Type'])).sum() / 1e3
    capacity.name = 'Capacity'

    return capacity

cap = backup_capacity_german_grid(network.graph)

cap.describe(),cap.sum(),type(cap)

print(cap[pd.isnull(cap)])

cap.fillna(0.1,inplace=True)


cap.index.levels[1]

m_costs = {"Gas" : 50.,
           "Brown Coal" : 10.,
           "Hard Coal" : 25.,
           "Oil" : 100.,
           "Nuclear" : 8.,
           "Pumped Hydro" : 3.,
           "Storage Hydro" : 3.,
           "Run of River" : 3.,
           "Geothermal" : 26.,
           "Waste" : 6.,
           "Multiple" : 28.,
           "Other" : 32.}

default_cost = 10.

for (bus_name,tech_name) in cap.index:
    print(bus_name,tech_name,cap[(bus_name,tech_name)])
    if tech_name == "Pumped Hydro":
        network.add("StorageUnit",bus_name + " " + tech_name,
                bus=bus_name,p_nom=1000*cap[(bus_name,tech_name)],
                marginal_cost=m_costs.get(tech_name,default_cost),
                source=tech_name,
                max_hours = 6,
                efficiency_store=0.95,
                efficiency_dispatch=0.95)
    else:
        network.add("Generator",bus_name + " " + tech_name,
                bus=bus_name,p_nom=1000*cap[(bus_name,tech_name)],
                marginal_cost=m_costs.get(tech_name,default_cost),
                source=tech_name)   

## Add renewables

import generation.germany as DEgen

reload(DEgen)

generation = DEgen.timeseries_eeg(network.graph)


generation.items

#Kill the Timezone information to avoid pandas bugs
generation.major_axis = generation.major_axis.values

generation.loc[["wind","solar"],network.snapshots,:].sum(axis=2).plot(kind="area")

#make sure the ordering of the minor axis is correc
generation.minor_axis = network.graph.nodes()

network.plot(bus_sizes=1000*generation.windoff.max())

network.plot(bus_sizes=1000*(generation.wind.mean()/generation.wind.max()))

network.plot(bus_sizes=1000*generation.solar.max())

maxes = generation.max(axis=1)


d = {"windoff" : "Wind Offshore",
    "windon" : "Wind Onshore",
    "solar" : "Solar"}

for tech in ["windoff",'windon',"solar"]:
    gens = maxes[tech][maxes[tech] != 0.]
    
    for i in gens.index:
        network.add("Generator","{} {}".format(i,d[tech]),
                    p_nom=gens[i]*1000.,dispatch="variable",
                    bus=i,source=d[tech],
                    p_max_pu=generation[tech].loc[network.snapshots,i]/gens[i])

csv_folder_name = "../../lib/pypsa/examples/opf-scigrid-de/scigrid-with-load-gen-trafos"

time_series = {"loads" : {"p_set" : None},
               "generators" : {"p_max_pu" : lambda g: g.dispatch == "variable"}}


network.export_to_csv_folder(csv_folder_name,time_series,verbose=False)

network.transformers

