#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 08:44:42 2021

Test script for multi-decade investment in pypsa.
 Structure:

(1) ways to set snapshots/investment periods
(2) build small test network (3 nodes)
(3) run multi-decade a pypsa-eur network

@author: bw0928
"""

import sys
import os
sys.path = [os.pardir] + sys.path
import pypsa
print(pypsa.__file__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml

# Functions / helpers
def get_cap_per_investment_period(n, c):
    """
    returns the installed capacities for each investment period and component
    depending on build year and lifetime

    n: pypsa network
    c: pypsa component (e.g. "Generator")
    cap_per_inv: pd.DataFrame(index=investment_period, columns=components)

    """
    df = n.df(c)
    cap_per_inv = pd.DataFrame(np.repeat([df.loc[:,df.columns.str.contains("_nom_opt")].iloc[:,0]],
                                         len(n.investment_periods), axis=0),
                               index=n.investment_periods, columns=df.index)
    # decomissioned set caps to zero
    decomissioned_i = cap_per_inv.apply(lambda x: (x.index.year>df.loc[x.name, ["build_year", "lifetime"]].sum()-1))
    cap_per_inv[decomissioned_i] = 0
    # before build year set caps to zero
    not_build_i = cap_per_inv.apply(lambda x: x.index.year<df.loc[x.name, "build_year"])
    cap_per_inv[not_build_i] = 0

    return cap_per_inv


def set_new_sns_invp(n, inv_years):
    """
    set new snapshots (sns) for all time varying componentents and
    investment_periods depending on investment years ('inv_years')

    input:
        n: pypsa.Network()
        inv_years: list of investment periods, e.g. [2020, 2030, 2040]

    """

    for component in n.all_components:
        pnl = n.pnl(component)
        attrs = n.components[component]["attrs"]

        for k,default in attrs.default[attrs.varying].iteritems():
            pnl[k] = pd.concat([(pnl[k].rename(index=lambda x: x.replace(year=year), level=1)
                                       .rename(index=lambda x: n.snapshots.get_level_values(level=1)[0].replace(year=year), level=0))
                                for year in inv_years])

    # set new snapshots + investment period
    n.snapshot_weightings = pd.concat([(n.snapshot_weightings.rename(index=lambda x: x.replace(year=year), level=1)
                                       .rename(index=lambda x: n.snapshots.get_level_values(level=1)[0].replace(year=year), level=0))
                                for year in inv_years])
    n.set_snapshots(n.snapshot_weightings.index)
    n.set_investment_periods(n.snapshots)

def get_social_discount(t, r=0.01):
    return (1/(1+r)**t)

def get_investment_weighting(energy_weighting, r=0.01):
    """
    returns cost weightings depending on the the energy_weighting (pd.Series)
    and the social discountrate r
    """
    end = energy_weighting.cumsum()
    start = energy_weighting.cumsum().shift().fillna(0)
    return pd.concat([start,end], axis=1).apply(lambda x: sum([get_social_discount(t,r)
                                                               for t in range(int(x[0]), int(x[1]))]),
                                                axis=1)

# %% (1)  ways to set snapshots / investment periods
# create pypsa network
n = pypsa.Network()

# snapshots are now a pandas MultiIndex
# (first level = investment_period (default: "first"), second level = snapshot)
print(n.snapshots)

# there is a new component "investment_period", representing the timesteps at
# which new investments are made. Analog to snapshots there are investment_period_weightings,
#  but those are split into "energy_weightings" (e.g. used for CO2 emissions) and
#  "investment_weightings" (e.g. used for cost weightings)
print(n.investment_periods)
print(n.investment_period_weightings)


# ## How to set snapshots and investment periods
# First set some parameters
# years of investment
years = [2020, 2030, 2040, 2050]
investment = pd.DatetimeIndex(['{}-01-01 00:00'.format(year) for year in years])
# temporal resolution
freq = "2190"
# snapshots (format -> DatetimeIndex)
snapshots = pd.DatetimeIndex([])
snapshots = snapshots.append([(pd.date_range(start ='{}-01-01 00:00'.format(year),
                                               freq ='{}H'.format(freq),
                                               periods=8760/float(freq))) for year in years])


# you can **set the snapshots**:
# (a) either in the way you are used to as pandas DatetimeIndex,
# investment_period will than be default value ("first")
n.set_snapshots(snapshots)
print(n.snapshots)
# (b) or as a pandas MultiIndex, this will also change the investment_periods
# to the first level of the pd.MultiIndex

investment_helper = investment.union(pd.Index([snapshots[-1] + pd.Timedelta(days=1)]))
map_dict = {investment[period] :
            snapshots[(snapshots>=investment_helper[period]) &
                      (snapshots<investment_helper[period+1])]
            for period in range(len(investment))}

multiindex = pd.MultiIndex.from_tuples([(name, l) for name, levels in
                                        map_dict.items() for l in levels])


n.set_snapshots(multiindex)
print(n.snapshots)
print(n.investment_periods)

# (c) you can also **set the investment_periods** analog to snapshots with a
# **list/single Index** or with a **pd.MultiIndex**, both ways change the snapshots as well, e.g.

n = pypsa.Network()
# set investmet periods with list of strings
n.set_investment_periods(["first", "second"])
print(n.investment_periods)
print(n.snapshots)

# set investment periods with multiindex
n.set_investment_periods(multiindex)
print(n.investment_periods)
print(n.snapshots)


# In this example, we can set the **investment_period_weightings** for the
# **energy_weightings** to the difference between the years. The energy
#  weightings are used to calculate the lifetime (if an assets is still active),
# as well as to weight the CO2 emissions. The **investment_weightings** weight
# the costs of an investment_period, we can include e.g. social discount rate
# r=1% here. One could choose the weighting of the last year of the optimisation
# also to a higher value to weight costs and emission in this period more strongly.

r = 0.01 # social discountrate
# set energy weighting -> last year is weighted by 1
n.investment_period_weightings.loc[:, "energy_weighting"] = n.investment_period_weightings.index.year.to_series().diff().shift(-1).fillna(1).values

# set investment_weighting
n.investment_period_weightings.loc[:, "investment_weighting"] = get_investment_weighting(n.investment_period_weightings["energy_weighting"], r)
print(n.investment_period_weightings)


# still **TODO**: split also the snapshot weightings into two groups


#%% (2) Build small test network

# add three buses
for i in range(3):
    n.add("Bus",
          "bus {}".format(i))


# There are 2 new attribute for the components ("Line", "Link", "Generator", Storage",...) <br>
#     (1) "**build_year**" : time when the asset is build (=first year of operation) <br>
#     (2) "**lifetime**": time of operation (not used to annualise the capital costs) <br>
# - If build_year and lifetime is not specified, it is assumed that the asset can operate in all investment_periods. - If only the build_year and no lifetime is specified, it is assumed the the asset can operate from build_year until the end of the optimisation time range
# - If the lifetime and no build_year is specified, it is assumed that the assets operates from the first timestep until end of lifetime
# - If the investment periods are a pd.DatetimeIndex a build year before the considered time frame is considered. E.g. n.investment_periods = [2020, 2030, 2040] and lifetime of an asset is 15 year, build year is 2010, than the asset can only operate in 2020.

# add three lines in a ring
n.add("Line",
      "line 0->1",
      bus0="bus 0",
      bus1="bus 1",
      x=0.0001,
      s_nom=0,
      #build_year=2030,
      s_nom_extendable=True)

n.add("Line",
      "line 1->2",
      bus0="bus 1",
      bus1="bus 2",
      x=0.0001,
      capital_cost=10,
      build_year=2030,
      s_nom=0,
      s_nom_extendable=True)


n.add("Line",
      "line 2->0",
      bus0="bus 2",
      bus1="bus 0",
      x=0.0001,
      s_nom=0,
      s_nom_extendable=True,
      build_year=2030)


# the function **n.determine_network_topology()** takes now as an optional
# argument the investment_period, be aware that the column values of sub_network are only valid for a certain investment_period. In this example, in the first investment period 2020 "bus 2" would be not connected to "bus 0" and "bus 1"

# works as before -> not considering if the assets are active
n.determine_network_topology()
print(n.buses.sub_network)

# determines network topolgy in first investment period (bus 2 isolated)
n.determine_network_topology(n.investment_periods[0])
print(n.buses.sub_network)

# determines network topology in third investment period (all lines are build)
n.determine_network_topology(n.investment_periods[2])
print(n.buses.sub_network)


n.lines.loc["line 2->0", "build_year"] = 2020


# add some generators
p_nom_max = pd.Series((np.random.uniform() for sn in range(len(n.snapshots))),
                  index=n.snapshots, name="generator ext 2020")

# renewable (can operate 2020, 2030)
n.add("Generator","generator ext 0 2020",
       bus="bus 0",
       p_nom=50,
       build_year=2020,
       lifetime=20,
       marginal_cost=2,
       capital_cost=1,
       p_max_pu=p_nom_max,
       carrier="solar",
       p_nom_extendable=True)

# add an expensive generator (can operate in all investment periods)
n.add("Generator",
      "generator fix expensive 2 2020",
      bus="bus 2",
      p_nom=100,
      build_year=2020,
      lifetime=31,
      carrier="lignite",
      marginal_cost=1000,
      capital_cost=10)

# can operate 2040, 2050
n.add("Generator","generator ext 0 2040",
      bus="bus 0",
      p_nom=50,
      build_year=2040,
      lifetime=11,
      marginal_cost=25,
      capital_cost=10,
      carrier="OCGT",
      p_nom_extendable=True)

# can operate in 2040
n.add("Generator",
      "generator fix 1 2040",
      bus="bus 1",
      p_nom=50,
      build_year=2040,
      lifetime=10,
      carrier="CCGT",
      marginal_cost=20,
      capital_cost=1,
      )


# add StorageUnits
n.add("StorageUnit",
      "storageunit cyclic 2030",
      bus="bus 2",
      p_nom=0,
      # marginal_cost=5,
      capital_cost=0.1,
      build_year=2030,
      lifetime=21,
      # efficiency_dispatch=0.9,
      # efficiency_store=0.99,
      cyclic_state_of_charge=True,
      p_nom_extendable=True,
      max_hours=180
      )

n.add("StorageUnit",
      "storageunit noncyclic 2030",
      bus="bus 2",
      p_nom=0,
      # marginal_cost=5,
      capital_cost=0.1,
      build_year=2030,
      lifetime=21,
      # efficiency_dispatch=0.9,
      # efficiency_store=0.99,
      # cyclic_state_of_charge=True,
      p_nom_extendable=True,
      max_hours=180
      )


# add battery store
n.add("Bus",
      "bus 2 battery")
# #
n.add("Store",
      "store 2 battery 2020",
       bus="bus 2 battery",
      e_cyclic=True,
      e_nom_extendable=True,
      build_year=2020,
      lifetime=20,
      capital_cost=2)

n.add("Link",
      "bus2 battery charger",
      bus0= "bus 2" ,
      bus1= "bus 2" + " battery",
      # efficiency=0.8,
      # capital_cost=2,
      p_nom_extendable=True)

n.add("Link",
      "My bus2 battery discharger",
      bus0="bus 2 battery",
      bus1="bus 2",
       efficiency=0.8,
      # marginal_cost=1,
      p_nom_extendable=True)


# add a Load
load_var =  pd.Series((100*np.random.uniform() for sn in range(len(n.snapshots))),
                  index=n.snapshots, name="load")
load_fix = pd.Series([250 for sn in range(len(n.snapshots))],
                  index=n.snapshots, name="load")

#add a load at bus 2
n.add("Load",
      "load 2",
      bus="bus 2",
      p_set=load_fix)

n.add("Load",
      "load 1",
      bus="bus 1",
      p_set=0.3*load_fix)


#%% ### Solve and results

# Currently, all modifications for multi-decade are only made for **pyomo=False**



n.lopf(snapshots=n.snapshots,solver_name="gurobi",
       pyomo=False)

total = pd.concat([n.generators_t.p, n.storage_units_t.p_dispatch,
           -1 * n.storage_units_t.p_store, n.stores_t.p,
           -1 * n.loads_t.p_set,
           -1 * pd.concat([n.links_t.p0, n.links_t.p1], axis=1).sum(axis=1).rename("Link losses"),
           -1 * pd.concat([n.lines_t.p0, n.lines_t.p1], axis=1).sum(axis=1).rename("Line losses")], axis=1)
total = total.groupby(total.columns, axis=1).sum()
total.plot(kind="bar", stacked=True, grid=True, title="Demand and Generation per snapshot", width=0.8)
plt.ylabel("Demand and Generation")
plt.xlabel("snapshot")
plt.legend(bbox_to_anchor=(1,1))



total.groupby(level=0).sum().plot(kind="bar", stacked=True, width=2000, grid=True,
                                  title="generation and demand per investment period").legend(bbox_to_anchor=(1,1))
plt.ylabel("Demand and Generation")
plt.xlabel("snapshot")


total.groupby(level=0).sum().sum(axis=1)


for component in ["Line", "Generator", "Store", "StorageUnit", "Link"]:
    caps = get_cap_per_investment_period(n, component)
    if caps.empty: continue
    ax=caps.plot(kind="bar", stacked=True, title=component, grid=True, width=2000)
    ticklabels = caps.index.year
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
    plt.ylabel("installed capacity \n [MW]")
    plt.xlabel("investment period")
    plt.legend(bbox_to_anchor=(1,1))


#%%  (3) Test with PyPSA-Eur network

path_eur = "/home/ws/bw0928/Dokumente/pypsa-eur/"
n = pypsa.Network(path_eur + "networks/elec_s_38_ec_lv1.0_.nc")

# if a pypsa network from current version **without** MultiIndex snapshot is
# imported, a warning is given and the investment_period is set to default(["first"])
n.snapshots


# For GlobalConstraint of the technical limit at each node, get the p_nom_max
p_nom_max_limit = n.generators.p_nom_max.groupby([n.generators.carrier, n.generators.bus]).sum()


# Only consider a few snapshots to speed up the calculations
nhours=876
n.set_snapshots(n.snapshots[::nhours])
n.snapshot_weightings.loc[:] = nhours
n.snapshots


# Test if network solves without changing to multi-decade investment
n.lopf(snapshots=n.snapshots,solver_name="gurobi",
       pyomo=False)


# %%
n = pypsa.Network(path_eur + "networks/elec_s_45_ec.nc")
n.set_snapshots(n.snapshots[::nhours])
n.snapshot_weightings.loc[:] = nhours

# create new time-varying data for all investment years
set_new_sns_invp(n, years)
n.generators_t.p_max_pu

n.snapshots


# set investment period weightings
# last year is weighted by 1
n.investment_period_weightings.loc[:, "energy_weighting"] = n.investment_period_weightings.index.year.to_series().diff().shift(-1).fillna(1).values
# set investment_weighting
n.investment_period_weightings.loc[:, "investment_weighting"] = get_investment_weighting(n.investment_period_weightings["energy_weighting"], r)


n.investment_period_weightings


# ### Play around with assumptions:
# 1. conventional phase out  <br>
# 2. renewable generators are build in every investment_period  <br>
# 3. build years for certain AC Lines or DC Links <br>
# 4. global constraints <br>
#     a. carbon budget  <br>
#     b. limit onshore wind and solar capacities at each node for each investment period

# 1) conventional phase out
# conventional lifetime + build year
conventionals = ["lignite", "coal", "oil", "nuclear", "CCGT", "OCGT"]
gens = n.generators[n.generators.carrier.isin(conventionals)].index
n.generators.loc[gens, "build_year"] = 2013
n.generators.loc[gens, "lifetime"] = 20


# 2.) renewable generator assumptions (e.g. can be newly build in each investment
# period, capital costs are decreasing,...)

# renewable
renewables = ["solar", "onwind", "offwind-ac", "offwind-dc"]
gen_names = n.generators[n.generators.carrier.isin(renewables)].index
df = n.generators.loc[gen_names]
p_max_pu = n.generators_t.p_max_pu[gen_names]

# drop old renewable generators
n.generators.drop(gen_names, inplace=True)
n.generators_t.p_max_pu.drop(gen_names, axis=1, inplace=True)

# add new renewable generator for each investment period
counter = 0
for year in n.investment_periods.year:
    n.madd("Generator",
           df.index,
           suffix=" " + str(year),
           bus=df.bus,
           carrier=df.carrier,
           p_nom_extendable=True,
           p_nom_max=df.p_nom_max,
           build_year=year,
           marginal_cost=df.marginal_cost,
           lifetime=15,
           capital_cost=df.capital_cost * 0.9**counter,
           efficiency=df.efficiency * 1.01**counter,
           p_max_pu=p_max_pu)

    counter += 1


n.generators[(n.generators.carrier=="solar") & (n.generators.bus=="DE0 0")]


# 3.) build year / transmission expansion for AC Lines and DC Links

later_lines = n.lines.iloc[::5].index
n.lines.loc[later_lines, "build_year"] = 2030
# later_lines = n.lines.iloc[::7].index
# n.lines.loc[later_lines, "build_year"] = 2040
# n.lines["s_nom_extendable"] = True
n.links["p_nom_extendable"] = True


# 4.) Test global constraints <br>
# a) CO2 constraint: can be specified as a budget,  which limits the CO2
#    emissions over all investment_periods
# b)  or/and implement as a constraint for an investment period, if the
# GlobalConstraint attribute "investment_period" is not specified, the limit
#  applies for each investment period.


# (a) add CO2 Budget constraint
n.add("GlobalConstraint",
      "CO2Budget",
      type="Budget",
      carrier_attribute="co2_emissions", sense="<=",
      constant=1e7)

# add CO2 limit for last investment period
n.add("GlobalConstraint",
      "CO2Limit",
      carrier_attribute="co2_emissions", sense="<=",
      investment_period = n.investment_periods[-1],
      constant=1e2)


# b) NOT WORKING YET add constraint for the technical maximum capacity for a carrier (e.g. "onwind", at one node and for one investment_period)

# global p_nom_max for each carrier + investment_period at each node
p_nom_max_inv_p = pd.DataFrame(np.repeat([p_nom_max_limit.values],
                                         len(n.investment_periods), axis=0),
                               index=n.investment_periods, columns=p_nom_max_limit.index)

#n.add("GlobalConstraint",
#      "TechLimit",
#      carrier_attribute=["onwind", "solar"],
#      sense="<=",
#      type="tech_capacity_expansion_limit",
#      constant=p_nom_max_inv_p[["onwind", "solar"]])


# add constraint for line volumne extension

# # add line volume constraint for first investment period
n.add("GlobalConstraint",
      "lv_limit_first_inv",
      type='transmission_volume_expansion_limit',
      carrier_attribute='AC, DC',
      investment_period=n.investment_periods[0],
      sense="<=",
      constant=2e8)

# for each investment period the line volume extension <= constant
n.add("GlobalConstraint",
      "lv_limit",
      type='transmission_volume_expansion_limit',
      carrier_attribute='AC, DC',
      sense="<=",
      constant=2.5e8)

n.global_constraints


#%% ### solve network

n.lopf(snapshots=n.snapshots,solver_name="gurobi",
       pyomo=False)


# Plotting

# colormap
path_sec = "/home/ws/bw0928/Dokumente/pypsa-eur-sec/"
with open(path_sec + 'config.yaml', encoding='utf8') as f:
    config = yaml.safe_load(f)
color_map = config["plotting"]['tech_colors']
color_map["biomass"] = "olive"
color_map["geothermal"] = "brown"
color_map["AC"] = "purple"
color_map["DC"] = "magenta"
color_map["demand"] = "slategray"
n.lines["carrier"] = "AC"

# plots
for component in ["Line", "Generator", "Store", "StorageUnit", "Link"]:
    caps = get_cap_per_investment_period(n, component)
    if caps.empty: continue
    caps = caps.groupby(n.df(component)["carrier"], axis=1).sum()/1e3
    ticklabels = caps.index.year
    ax=caps.plot(kind="bar", stacked=True, title=component, grid=True,  width=2000,
                 color=[color_map.get(x, '#333333') for x in caps.columns])
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
    plt.ylabel("installed capacity \n [GW]")
    plt.xlabel("investment period")
    plt.legend(bbox_to_anchor=(1,1))

generation = (n.generators_t.p.groupby(n.generators.carrier,axis=1).sum()
              .groupby(level=0).sum())/1e6
demand = (-n.loads_t.p_set.sum(axis=1).groupby(level=0).sum()/1e6)
storage = n.storage_units_t.p.groupby(level=0).sum().groupby(n.storage_units.carrier,axis=1).sum()/1e6
total = pd.concat([demand.rename("demand"), generation, storage],axis=1)
total.plot(kind="bar", stacked=True, title="demand + generation", grid=True,
           width=2000,
                color=[color_map.get(x, '#333333') for x in total.columns]).legend(bbox_to_anchor=(1,1))
plt.ylabel("TWh")
plt.xlabel("investment year")


# In[60]:


# check generation covers demand in each investment period
total.sum(axis=1)


# In[ ]:





# In[ ]:




