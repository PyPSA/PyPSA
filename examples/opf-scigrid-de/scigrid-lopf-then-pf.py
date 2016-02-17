## LOPF then non-linear power flow with SciGRID
#
#This Jupyter Notebook is also available to download: [scigrid-lopf-then-pf.ipynb](http://www.pypsa.org/examples/scigrid-lopf-then-pf.ipynb).
#
#In this example, the dispatch of generators is optimised using the linear OPF, then a non-linear power flow is run on the resulting dispatch.
#
### Data sources
#
#The data is generated in a separate notebook called [add_load_gen_to_scigrid.ipynb](http://www.pypsa.org/examples/add_load_gen_to_scigrid.ipynb).
#
#Grid: from [SciGRID](http://scigrid.de/) which is based on [OpenStreetMap](http://www.openstreetmap.org/).
#
#Load size and location: based on Landkreise GDP and population.
#
#Power plant capacities and locations: BNetzA list.
#
#Wind and solar capacities and locations: EEG Stammdaten.
#
#Wind and solar time series: REatlas, Andresen et al, "Validation of Danish wind time series from a new global renewable energy atlas for energy system analysis," Energy 93 (2015) 1074 - 1088.

#make the code as Python 3 compatible as possible                                                                               \
                                                                                                                                  
from __future__ import print_function, division, absolute_import

import pypsa

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

#%matplotlib inline

csv_folder_name = os.path.dirname(pypsa.__file__) + "/../examples/opf-scigrid-de/scigrid-with-load-gen/"

network = pypsa.Network(csv_folder_name=csv_folder_name)

### Plot the distribution of the load and of generating tech

fig,ax = plt.subplots(1,1)

fig.set_size_inches(6,6)

load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()

network.plot(bus_sizes=0.5*load_distribution,ax=ax,title="Load distribution")

fig.tight_layout()
#fig.savefig('load-distribution.png')

network.generators.groupby("source")["p_nom"].sum()

network.storage_units.groupby("source")["p_nom"].sum()

techs = ["Gas","Brown Coal","Hard Coal","Wind Offshore","Wind Onshore","Solar"]

n_graphs = len(techs)

n_cols = 3

if n_graphs % n_cols == 0:
    n_rows = n_graphs // n_cols
else:
    n_rows = n_graphs // n_cols + 1

    
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)

size = 4

fig.set_size_inches(size*n_cols,size*n_rows)

for i,tech in enumerate(techs):
    i_row = i // n_cols
    i_col = i % n_cols
    
    ax = axes[i_row,i_col]
    
    gens = network.generators[network.generators.source == tech]
    
    gen_distribution = gens.groupby("bus").sum()["p_nom"].reindex(network.buses.index,fill_value=0.)
    
    network.plot(ax=ax,bus_sizes=0.2*gen_distribution)
    
    ax.set_title(tech)
    
    

### Run Linear Optimal Power Flow on the first day of 2011

#There are some infeasibilities at the edge of the network where loads
#are supplied by foreign lines - just add some extra capacity in Germany
for line_name in ["350","583"]:
    network.lines.loc[line_name,"s_nom"] += 500

network.now = network.snapshots[0]

group_size = 4

solver_name = "glpk"

print("Performing linear OPF for one day, {} snapshots at a time:".format(group_size))

network.storage_units.state_of_charge_initial = 0.

for i in range(int(24/group_size)):
    
    #set the initial state of charge based on previous round
    if i>0:
        network.storage_units.state_of_charge_initial = network.storage_units_t.state_of_charge.loc[network.snapshots[group_size*i-1]]

    network.lopf(network.snapshots[group_size*i:group_size*i+group_size],
                 solver_name=solver_name,
                 keep_files=True)

p_by_source = network.generators_t.p.groupby(network.generators.source, axis=1).sum()

p_by_source.drop((p_by_source.max()[p_by_source.max() < 1500.]).index,axis=1,inplace=True)

p_by_source.columns

colors = {"Brown Coal" : "brown",
          "Hard Coal" : "k",
          "Nuclear" : "r",
          "Run of River" : "green",
          "Wind Onshore" : "blue",
          "Solar" : "yellow",
          "Wind Offshore" : "cyan",
          "Waste" : "orange"}

fig,ax = plt.subplots(1,1)

fig.set_size_inches(12,6)

(p_by_source/1e3).plot(kind="area",ax=ax,linewidth=4,colors=[colors[col] for col in p_by_source.columns])


ax.legend(ncol=4)

ax.set_ylabel("GW")

ax.set_xlabel("")

fig.tight_layout()
#fig.savefig("stacked-gen.png")

fig,ax = plt.subplots(1,1)
fig.set_size_inches(12,6)

p_storage = network.storage_units_t.p.sum(axis=1)
state_of_charge = network.storage_units_t.state_of_charge.sum(axis=1)
p_storage.plot(label="Pumped hydro dispatch",ax=ax,linewidth=3)
state_of_charge.plot(label="State of charge",ax=ax,linewidth=3)

ax.legend()
ax.grid()
ax.set_ylabel("MWh")
ax.set_xlabel("")

fig.tight_layout()
fig.savefig("storage-scigrid.png")

print("With the linear load flow, there is the following per unit loading:")
loading = network.lines_t.p0.loc[network.snapshots[4]]/network.lines.s_nom
print(loading.describe())

fig,ax = plt.subplots(1,1)
fig.set_size_inches(6,6)

network.plot(ax=ax,line_colors=abs(loading),line_cmap=plt.cm.jet,title="Line loading")

fig.tight_layout()
#fig.savefig("line-loading.png")

network.buses_t.marginal_price.loc[network.now].describe()

fig,ax = plt.subplots(1,1)
fig.set_size_inches(6,4)


network.plot(ax=ax,line_widths=pd.Series(0.5,network.lines.index))
plt.hexbin(network.buses.x, network.buses.y, 
           gridsize=20,
           C=network.buses_t.marginal_price.loc[network.now],
           cmap=plt.cm.jet)

#for some reason the colorbar only works with graphs plt.plot
#and must be attached plt.colorbar

cb = plt.colorbar()
cb.set_label('Locational Marginal Price (EUR/MWh)') 

fig.tight_layout()
#fig.savefig('lmp.png')

## Check power flow

for bus in network.buses.index:
    bus_sum = network.buses_t.p.loc[network.now,bus]
    lines_sum = network.lines_t.p0.loc[network.now,network.lines.bus0==bus].sum() - network.lines_t.p0.loc[network.now,network.lines.bus1==bus].sum()
    if abs(bus_sum-lines_sum) > 1e-4:
        print(bus,bus_sum,lines_sum)

### Now perform a full Newton-Raphson power flow on the first hour

#For the PF, set the P to the optimised P
network.generators_t.p_set.loc[network.now] = network.generators_t.p.loc[network.now] 


#set all buses to PV, since we don't know what Q set points are
network.generators.control = "PV"

#set slack
#network.generators.loc["1 Coal","control"] = "Slack"


#Need some PQ buses so that Jacobian doesn't break
f = network.generators[network.generators.bus == "492"]
network.generators.loc[f.index,"control"] = "PQ"


print("Performing non-linear PF on results of LOPF:")

network.pf()

print("With the non-linear load flow, there is the following per unit loading:")
print((network.lines_t.p0.loc[network.now]/network.lines.s_nom).describe())

#Get voltage angle differences

df = network.lines.copy()

for b in ["bus0","bus1"]:
    df = pd.merge(df,network.buses_t.v_ang.loc[[network.now]].T,how="left",
         left_on=b,right_index=True)

s = df[str(network.now)+"_x"]- df[str(network.now)+"_y"]

print("The voltage angle differences across the lines have (in degrees):")
print((s*180/np.pi).describe())

#plot the reactive power

fig,ax = plt.subplots(1,1)

fig.set_size_inches(6,6)

q = network.buses_t.q.loc[network.now]

bus_colors = pd.Series("r",network.buses.index)
bus_colors[q< 0.] = "b"


network.plot(bus_sizes=abs(q),ax=ax,bus_colors=bus_colors,title="Reactive power feed-in (red=+ve, blue=-ve)")

fig.tight_layout()
#fig.savefig("reactive-power.png")


