#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:25:05 2021

example script to illustrate new features of the multi-decade investment

@author: Lisa
"""
import pypsa
import pandas as pd
import numpy as np


n = pypsa.Network()

# snapshot weightings are now seperated between weightings for the objective
# function and weightings for the single components
# store_weightings: for elapsed time in the filling of Store and StorageUnit components
# generator_weightings: for the contribution of generators to GlobalConstraint components for e.g. CO2 budgets
print(n.snapshot_weightings)

# there is a new component investment period weightings
# split  by weighting for the objective function (objective_weightings)
# and weightings for the elapsed time between the investment periods (time_weightings)
# to calculate CO2 emissions or assets lifetime
# default is an empty pd.DataFrame
print(n.investment_period_weightings)

# default snapshots are still single Index
print(n.snapshots)

# when a pypsa network from an older version is imported, the old snapshot weightings
# are set row-wise and a warning is given
n = pypsa.Network("opf-storage-hvdc/opf-storage-data")
print(n.snapshot_weightings)

# snapshots can be set as a multiindex, this can be done be either
# (a) set the snapshots as before
snapshots_multiindex = pd.MultiIndex.from_product([[2020, 2030], n.snapshots])
n.set_snapshots(snapshots_multiindex)
print(n.snapshots)
print(n.investment_period_weightings)
# (b) or by setting directly the investment periods (input as a list), which will
# reset the snapshots to a MultiIndex as well
n.set_investment_periods([2020, 2030])
print(n.snapshots)
print(n.investment_period_weightings)

# investment_periods have to be integer and increasing, otherwise there will be an
# error message
n.set_investment_periods([2030, 2020])

# new attributes for generators/store/storage/links/lines/transformers used to
#  calculate when an asset is active:
# "build_year": indicating when an asset is built (integer, default: 0)
# "lifetime": lifetime of asset (float, default: inf)
print(n.generators[["build_year", "lifetime"]])

# store/storage unit behaviour
# cyclic (e_cyclic/cyclic_state_of_charge) means now: cyclic within each investment period
# there is a new attribute (bool) called for stores "e_period" and for storage units 'state_of_charge_period'
# default is true, which sets the state of charge at the beginning of each investment period to the same value
print(n.storage_units.state_of_charge_period)
print(n.stores.e_period)


# there is a new attribute "investment_period" for global constraints where the investment period can
# be specified, for which the constraint is valid
# if this attribute is not set it applies for all investment periods
print(n.global_constraints.columns)


# when calling the lopf new attribute "multi_investment_periods" (Bool, default:False)
# only implemented for pyomo = False

# PYOMO = True
# single investment as before if snapshots single Index
# if snapshots are pd.MultiIndex or multi_investment_periods=True returns a not implemented error
n.lopf(pyomo=True, multi_investment_periods=False)
# this returns a not implemented error
n.lopf(pyomo=True, multi_investment_periods=True)

# PYOMO = False
# this returns single optimisation as before, if snapshots are multiindex there
# is a warning that only single investment is assumed
n.lopf(pyomo=False, multi_investment_periods=False)

# this returns multi investment optimisation if the snapshots are pd.MultiIndex
# otherwise returns a typeError
n.lopf(pyomo=False, multi_investment_periods=True)

#%% small test network

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

# create pypsa network
n = pypsa.Network()

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


# to the first level of the pd.MultiIndex
investment_helper = investment.union(pd.Index([snapshots[-1] + pd.Timedelta(days=1)]))
map_dict = {years[period] :
            snapshots[(snapshots>=investment_helper[period]) &
                      (snapshots<investment_helper[period+1])]
            for period in range(len(investment))}

multiindex = pd.MultiIndex.from_tuples([(name, l) for name, levels in
                                        map_dict.items() for l in levels])


n.set_snapshots(multiindex)
print(n.snapshots)


# (c) you can also **set the investment_periods** analog to snapshots with a
# **list/single Index** or with a **pd.MultiIndex**, both ways change the snapshots as well, e.g.


r = 0.01 # social discountrate
# set energy weighting -> last year is weighted by 1
n.investment_period_weightings.loc[:, 'time_weightings'] = n.investment_period_weightings.index.to_series().diff().shift(-1).fillna(1)
# n.investment_period_weightings.loc[:, 'generator_weightings'] = n.investment_period_weightings.index.to_series().diff().shift(-1).fillna(1)
# set investment_weighting
n.investment_period_weightings.loc[:, "objective_weightings"] = get_investment_weighting(n.investment_period_weightings["time_weightings"], r)
print(n.investment_period_weightings)


# still **TODO**: split also the snapshot weightings into two groups


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
# n.determine_network_topology()
# print(n.buses.sub_network)

# determines network topolgy in first investment period (bus 2 isolated)
# n.determine_network_topology(n.investment_period_weightings.index[0])
print(n.buses.sub_network)

# determines network topology in third investment period (all lines are build)
# n.determine_network_topology(n.investment_period_weightings.index[2])
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
# n.add("Generator",
#       "generator fix expensive 2 2020",
#       bus="bus 2",
#       p_nom=100,
#       build_year=2020,
#       lifetime=31,
#       carrier="lignite",
#       marginal_cost=1000,
#       capital_cost=10)

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
# n.add("StorageUnit",
#       "storageunit cyclic 2030",
#       bus="bus 2",
#       p_nom=0,
#       # marginal_cost=5,
#       capital_cost=0.1,
#       build_year=2030,
#       lifetime=21,
#       # efficiency_dispatch=0.9,
#       # efficiency_store=0.99,
#       cyclic_state_of_charge=True,
#       p_nom_extendable=True,
#       max_hours=180
#       )

n.add("StorageUnit",
      "storageunit non-periodic 2030",
      bus="bus 2",
      p_nom=0,
      # marginal_cost=5,
      capital_cost=2,
      build_year=2030,
      lifetime=21,
      # efficiency_dispatch=0.9,
      # efficiency_store=0.99,
      # cyclic_state_of_charge=True,
      p_nom_extendable=False,
      # max_hours=180
      )

n.add("StorageUnit",
      "storageunit periodic 2020",
      bus="bus 2",
      p_nom=0,
      # marginal_cost=5,
      capital_cost=1,
      build_year=2020,
      lifetime=21,
      # efficiency_dispatch=0.9,
      # efficiency_store=0.99,
      # cyclic_state_of_charge=True,
      p_nom_extendable=True,
      # max_hours=180
      )

# n.add("StorageUnit",
#       "storageunit noncyclic 2030",
#       bus="bus 2",
#       p_nom=0,
#       # marginal_cost=5,
#       capital_cost=0.1,
#       build_year=2030,
#       lifetime=21,
#       state_of_charge_period = False,
#       # efficiency_dispatch=0.9,
#       # efficiency_store=0.99,
#       # cyclic_state_of_charge=True,
#       p_nom_extendable=True,
#       max_hours=180
#       )


# add battery store
n.add("Bus",
      "bus 2 battery")
# #
n.add("Store",
      "store 2 battery 2020",
       bus="bus 2 battery",
      # e_cyclic=True,
      e_nom_extendable=True,
      e_inital=20,
      build_year=2020,
      lifetime=20,
      capital_cost=0.1)

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

# currently only for pyomo=False
n.lopf(pyomo=False, multi_investment_periods=True)