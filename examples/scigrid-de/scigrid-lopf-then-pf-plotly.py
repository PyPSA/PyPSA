## LOPF then non-linear power flow with SciGRID
#
#This Jupyter Notebook is also available to download at: <http://www.pypsa.org/examples/scigrid-lopf-then-pf.ipynb> and can be viewed as an HTML page at: <http://pypsa.org/examples/scigrid-lopf-then-pf.html>.
#
#In this example, the dispatch of generators is optimised using the linear OPF, then a non-linear power flow is run on the resulting dispatch.
#
#The data files for this example are in the examples folder of the github repository: <https://github.com/PyPSA/PyPSA>.
#
### Data sources
#
#The data is generated in a separate notebook at <http://www.pypsa.org/examples/add_load_gen_trafos_to_scigrid.ipynb>.
#
#
#Grid: based on [SciGRID](http://scigrid.de/) Version 0.2 which is based on [OpenStreetMap](http://www.openstreetmap.org/).
#
#Load size and location: based on Landkreise (NUTS 3) GDP and population.
#
#Load time series: from ENTSO-E hourly data, scaled up uniformly by factor 1.12 (a simplification of the methodology in Schumacher, Hirth (2015)).
#
#Conventional power plant capacities and locations: BNetzA list.
#
#Wind and solar capacities and locations: EEG Stammdaten, based on  http://www.energymap.info/download.html, which represents capacities at the end of 2014. Units without PLZ are removed.
#
#Wind and solar time series: REatlas, Andresen et al, "Validation of Danish wind time series from a new global renewable energy atlas for energy system analysis," Energy 93 (2015) 1074 - 1088.
#
#NB:
#
#All times in the dataset are UTC.
#
#Where SciGRID nodes have been split into 220kV and 380kV substations, all load and generation is attached to the 220kV substation.
#
### Warning
#
#This dataset is ONLY intended to demonstrate the capabilities of PyPSA and is NOT (yet) accurate enough to be used for research purposes.
#
#Known problems include:
#
#i) Rough approximations have been made for missing grid data, e.g. 220kV-380kV transformers and connections between close sub-stations missing from OSM.
#
#ii) There appears to be some unexpected congestion in parts of the network, which may mean for example that the load attachment method (by Voronoi cell overlap with Landkreise) isn't working, particularly in regions with a high density of substations.
#
#iii) Attaching power plants to the nearest high voltage substation may not reflect reality.
#
#iv) There is no proper n-1 security in the calculations - this can either be simulated with a blanket e.g. 70% reduction in thermal limits (as done here) or a proper security constrained OPF (see e.g.  <http://www.pypsa.org/examples/scigrid-sclopf.ipynb>).
#
#v) The borders and neighbouring countries are not represented.
#
#vi) Hydroelectric power stations are not modelled accurately.
#
#viii) The marginal costs are illustrative, not accurate.
#
#ix) Only the first day of 2011 is in the github dataset, which is not representative. The full year of 2011 can be downloaded at <http://www.pypsa.org/examples/scigrid-with-load-gen-trafos-2011.zip>.
#
#x) The ENTSO-E total load for Germany may not be scaled correctly; it is scaled up uniformly by factor 1.12 (a simplification of the methodology in Schumacher, Hirth (2015), which suggests monthly factors).
#
#xi) Biomass from the EEG Stammdaten are not read in at the moment.
#
#xii) Power plant start up costs, ramping limits/costs, minimum loading rates are not considered.

import pypsa
import numpy as np
import pandas as pd
import os
import plotly.offline as pltly
import cufflinks as cf

pltly.init_notebook_mode(connected=True)

#You may have to adjust this path to where
#you downloaded the github repository
#https://github.com/PyPSA/PyPSA

csv_folder_name = os.path.dirname(pypsa.__file__) + "/../examples/scigrid-de/scigrid-with-load-gen-trafos/"

network = pypsa.Network(csv_folder_name=csv_folder_name)

### Plot the distribution of the load and of generating tech

load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum().reindex(network.buses.index,fill_value=0.)

fig = dict(data=[],layout=dict(width=700,height=700))

fig = network.iplot(bus_sizes=0.05*load_distribution, fig=fig,
                     bus_text='Load at bus ' + network.buses.index + ': ' + round(load_distribution).values.astype(str) + ' MW',
                     title="Load distribution",
                     line_text='Line ' + network.lines.index)

network.generators.groupby("carrier")["p_nom"].sum()

network.storage_units.groupby("carrier")["p_nom"].sum()

tech = 'Wind Onshore' # in ["Gas","Brown Coal","Hard Coal","Wind Offshore","Wind Onshore","Solar"]

gens = network.generators[network.generators.carrier == tech]
gen_distribution = gens.groupby("bus").sum()["p_nom"].reindex(network.buses.index,fill_value=0.)

#set the figure size first
fig = dict(data=[],layout=dict(width=700,height=700))

fig = network.iplot(bus_sizes=0.05*gen_distribution, fig=fig,
                     bus_text=tech + ' at bus ' + network.buses.index + ': ' + round(gen_distribution).values.astype(str) + ' MW',
                     title=tech + " distribution")

### Run Linear Optimal Power Flow on the first day of 2011

#to approximate n-1 security and allow room for reactive power flows,
#don't allow any line to be loaded above 70% of their thermal rating

contingency_factor = 0.7

network.lines.s_nom = contingency_factor*network.lines.s_nom

#There are some infeasibilities without small extensions
for line_name in ["316","527","602"]:
    network.lines.loc[line_name,"s_nom"] = 1200


#the lines to extend to resolve infeasibilities can
#be found by
#uncommenting the lines below to allow the network to be extended

#network.lines["s_nom_original"] = network.lines.s_nom

#network.lines.s_nom_extendable = True
#network.lines.s_nom_min = network.lines.s_nom

#Assume 450 EUR/MVA/km
#network.lines.capital_cost = 450*network.lines.length

group_size = 4

solver_name = "cbc"

print("Performing linear OPF for one day, {} snapshots at a time:".format(group_size))

network.storage_units.state_of_charge_initial = 0.

for i in range(int(24/group_size)):
    #set the initial state of charge based on previous round
    if i>0:
        network.storage_units.state_of_charge_initial = network.storage_units_t.state_of_charge.loc[network.snapshots[group_size*i-1]]
    network.lopf(network.snapshots[group_size*i:group_size*i+group_size],
                 solver_name=solver_name,
                 keep_files=True)
    network.lines.s_nom = network.lines.s_nom_opt

#if lines are extended, look at which ones are bigger
#network.lines[["s_nom_original","s_nom"]][abs(network.lines.s_nom - contingency_factor*network.lines.s_nom_original) > 1]

p_by_carrier = network.generators_t.p.groupby(network.generators.carrier, axis=1).sum()

p_by_carrier.drop((p_by_carrier.max()[p_by_carrier.max() < 1700.]).index,axis=1,inplace=True)

p_by_carrier.columns

colors = {"Brown Coal" : "brown",
          "Hard Coal" : "black",
          "Nuclear" : "red",
          "Run of River" : "green",
          "Wind Onshore" : "blue",
          "Solar" : "yellow",
          "Wind Offshore" : "cyan",
          "Waste" : "orange",
          "Gas" : "orange"}
#reorder
cols = ["Nuclear","Run of River","Brown Coal","Hard Coal","Gas","Wind Offshore","Wind Onshore","Solar"]
p_by_carrier = p_by_carrier[cols]

#Unfortunately this shows cumulative sums on hover, not the individual contributions
#Blame cufflinks, not us :-)
(p_by_carrier/1e3).iplot(kind="area",fill=True,
                         width=4,
                         yTitle="GW",
                         color=[colors[col] for col in p_by_carrier.columns])

storage = pd.DataFrame(index=network.snapshots)
storage["Pumped hydro dispatch [MW]"] = network.storage_units_t.p.sum(axis=1)
storage["State of charge [MWh]"] = network.storage_units_t.state_of_charge.sum(axis=1)
storage.iplot(width=3)

now = network.snapshots[4]

print("With the linear load flow, there is the following per unit loading:")
loading = network.lines_t.p0.loc[now]/network.lines.s_nom
print(loading.describe())

#a hacked colormap for lines until we can work this out properly
from plotly.colors import label_rgb, find_intermediate_color
def colormap(i):
    return label_rgb([int(n) for n in find_intermediate_color([0,255.,255.],[255.,0.,0],i)])


#set the figure size first
fig = dict(data=[],layout=dict(width=700,height=700))

fig = network.iplot(line_colors=abs(loading).map(colormap),fig=fig,
                    line_text='Line ' + network.lines.index + ' has loading ' + abs(100*loading).round(1).astype(str)+'%')

network.buses_t.marginal_price.loc[now].describe()


#set the figure size first
fig = dict(data=[],layout=dict(width=900,height=700))

fig = network.iplot(bus_sizes=40, bus_colors=network.buses_t.marginal_price.loc[now],
                    bus_text = 'Bus ' + network.buses.index + ': ' + network.buses_t.marginal_price.loc[now].reindex(network.buses.index).round().astype(str) + ' EUR/MWh',
                    fig=fig,
                    bus_colorscale='Jet',
                    bus_colorbar=dict(title='Locational Marginal Price [EUR/MWh]'))

### Look at variable curtailment

carrier = "Wind Onshore"

capacity = network.generators.groupby("carrier").sum().at[carrier,"p_nom"]

p_available = network.generators_t.p_max_pu.multiply(network.generators["p_nom"])

p_available_by_carrier =p_available.groupby(network.generators.carrier, axis=1).sum()

p_curtailed_by_carrier = p_available_by_carrier - p_by_carrier

p_df = pd.DataFrame({carrier + " available" : p_available_by_carrier[carrier],
                     carrier + " dispatched" : p_by_carrier[carrier],
                     carrier + " curtailed" : p_curtailed_by_carrier[carrier]})

p_df[carrier + " capacity"] = capacity

p_df["Wind Onshore curtailed"][p_df["Wind Onshore curtailed"] < 0.] = 0.


(p_df[[carrier + " dispatched",carrier + " curtailed"]]/1e3).iplot(kind="area",
                                                                   fill=True,
                                                                   width=3,
                                                                   yTitle='Power [GW]')

## Check power flow

now = network.snapshots[0]

for bus in network.buses.index:
    bus_sum = network.buses_t.p.loc[now,bus]
    branches_sum = 0
    for comp in ["lines","transformers"]:
        comps = getattr(network,comp)
        comps_t = getattr(network,comp+"_t")
        branches_sum += comps_t.p0.loc[now,comps.bus0==bus].sum() - comps_t.p0.loc[now,comps.bus1==bus].sum()

    if abs(bus_sum-branches_sum) > 1e-4:
        print(bus,bus_sum,branches_sum)

### Now perform a full Newton-Raphson power flow on the first hour

#For the PF, set the P to the optimised P
network.generators_t.p_set = network.generators_t.p_set.reindex(columns=network.generators.index)
network.generators_t.p_set = network.generators_t.p


#set all buses to PV, since we don't know what Q set points are
network.generators.control = "PV"

#set slack
#network.generators.loc["1 Coal","control"] = "Slack"


#Need some PQ buses so that Jacobian doesn't break
f = network.generators[network.generators.bus == "492"]
network.generators.loc[f.index,"control"] = "PQ"


print("Performing non-linear PF on results of LOPF:")

info = network.pf()

#any failed to converge?
(~info.converged).any().any()

print("With the non-linear load flow, there is the following per unit loading\nof the full thermal rating:")
print((network.lines_t.p0.loc[now]/network.lines.s_nom*contingency_factor).describe())

#Get voltage angle differences

df = network.lines.copy()

for b in ["bus0","bus1"]:
    df = pd.merge(df,network.buses_t.v_ang.loc[[now]].T,how="left",
         left_on=b,right_index=True)

s = df[str(now)+"_x"]- df[str(now)+"_y"]

print("The voltage angle differences across the lines have (in degrees):")
print((s*180/np.pi).describe())

#plot the reactive power

q = network.buses_t.q.loc[now]

bus_colors = pd.Series("red",network.buses.index)
bus_colors[q< 0.] = "blue"

fig = dict(data=[],layout=dict(width=700,height=700))

fig=network.iplot(bus_sizes=2e-1*abs(q),bus_colors=bus_colors,
                  bus_text='Bus ' + network.buses.index + ' Q: ' + q.reindex(network.buses.index).round().astype(str) + ' MVAr',
                  title="Reactive power feed-in (red=+ve, blue=-ve)",fig=fig)

network.buses_t.q.loc[now].sum()

network.generators_t.q.loc[now].sum()

