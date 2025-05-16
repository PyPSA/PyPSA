# type: ignore
"""
Created on Wed Feb 17 15:25:05 2021.

example script to illustrate new features of the multi-decade investment

@author: Lisa
"""

import numpy as np
import pandas as pd

import pypsa

n = pypsa.Network()

# snapshot weightings are now separated between weightings for
# (i) the objective function (objective)
# (ii) elapsed time in the filling of Store and StorageUnit components (stores)
# (ii) for the contribution of generators to GlobalConstraint components for e.g. CO2 budgets (generators)
print(n.snapshot_weightings)

# there is a new component investment period analog to n.snapshots
n.investment_periods
# along with corresponding weightings split  by weighting
# for the objective function (objective)
# and weightings for the elapsed time between the investment periods (years)
# to calculate CO2 emissions or assets lifetime
# default is an empty pd.DataFrame
print(n.investment_period_weightings)

# default snapshots are still single Index
print(n.snapshots)

# when a pypsa network from an older version is imported, the old snapshot weightings
# are set row-wise and a warning is given
# n = pypsa.examples.storage_hvdc()  -> "HTTPError: Not Found"
n = pypsa.Network("opf-storage-hvdc/opf-storage-data")
print(n.snapshot_weightings)

# snapshots can be set as a multiindex
# a convenience functionality is provided: when setting the investment periods
# directly, the all time-series are repeated for each period.
single_snapshots = n.snapshots
n.investment_periods = [2020, 2030]
print(n.snapshots)
print(n.investment_period_weightings)
assert all(n.snapshots == pd.MultiIndex.from_product([[2020, 2030], single_snapshots]))

# new attributes for generators/store/storage/links/lines/transformers used to
#  calculate when an asset is active:
# "build_year": indicating when an asset is built (integer, default: 0)
# "lifetime": lifetime of asset (float, default: inf)
print(n.generators[["build_year", "lifetime"]])

# store/storage unit behaviour
# cyclic ('e_cyclic_per_period'/'cyclic_state_of_charge_per_period') means now:
# cyclic within each investment period -> default: True
print(n.storage_units["cyclic_state_of_charge_per_period"])
print(n.stores.e_initial_per_period)
# there is a new attribute (bool) called for stores "e_initial_per_period" and
# for storage units 'state_of_charge_initial_per_period'
# default is true, which sets the state of charge at the beginning of each
# investment period to the same value
print(n.storage_units.state_of_charge_initial_per_period)
print(n.stores.e_initial_per_period)


# there is a new attribute "investment_period" for global constraints where the investment period can
# be specified, for which the constraint is valid
# if this attribute is not set it applies for all investment periods
print(n.global_constraints.columns)


# when calling the lopf new attribute "multi_investment_periods" (Bool, default:False)
# only implemented for pyomo = False

# PYOMO = True
# single investment as before if snapshots single Index
# if snapshots are pd.MultiIndex or multi_investment_periods=True returns a not implemented error

try:
    n.lopf(pyomo=True, multi_investment_periods=False)
except Exception as e:
    print(e)

# this returns a not implemented error
try:
    n.lopf(pyomo=True, multi_investment_periods=True)
except Exception as e:
    print(e)

# this returns multi investment optimisation if the snapshots are pd.MultiIndex
# otherwise returns a typeError
n.lopf(pyomo=False, multi_investment_periods=True)

# %% small test network


def get_social_discount(t, r=0.01):
    """Calculate social discount rate."""
    return 1 / (1 + r) ** t


def get_investment_weighting(energy_weighting, r=0.01):
    """
    Return cost weightings.

    Weightings depend on the energy_weighting (pd.Series) and the
    social discountrate r.
    """
    end = energy_weighting.cumsum()
    start = energy_weighting.cumsum().shift().fillna(0)
    return pd.concat([start, end], axis=1).apply(
        lambda x: sum(get_social_discount(t, r) for t in range(int(x[0]), int(x[1]))),
        axis=1,
    )


# create pypsa network
n = pypsa.Network()

# ## How to set snapshots and investment periods
# First set some parameters as years and temporal resolution
years = [2020, 2030, 2040, 2050]
freq = "24"

# init snapshots (format -> DatetimeIndex)
snapshots = pd.DatetimeIndex([])
for year in years:
    period = pd.date_range(
        start=f"{year}-01-01 00:00",
        freq=f"{freq}H",
        periods=8760 / float(freq),
    )
    snapshots = snapshots.append(period)

# convert to multiindex and assign to network
n.snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])
n.investment_periods = years

print(n.snapshots)
print(n.investment_periods)


r = 0.01  # social discountrate
# set energy weighting -> last year is weighted by 10
n.investment_period_weightings.loc[:, "years"] = (
    n.investment_periods.to_series().diff().shift(-1).fillna(10)
)

# set investment_weighting
n.investment_period_weightings.loc[:, "objective"] = get_investment_weighting(
    n.investment_period_weightings["years"], r
)
print(n.investment_period_weightings)

# add three buses
for i in range(3):
    n.add("Bus", f"bus {i}")


# There are 2 new attribute for the components ("Line", "Link", "Generator", Storage", ...) <br>
#     (1) "**build_year**" : time when the asset is build (=first year of operation) <br>
#     (2) "**lifetime**": time of operation (not used to annualise the capital costs) <br>
# - If build_year and lifetime is not specified, it is assumed that the asset can operate in all investment_periods. - If only the build_year and no lifetime is specified, it is assumed the asset can operate from build_year until the end of the optimisation time range
# - If the lifetime and no build_year is specified, it is assumed that the assets operates from the first timestep until end of lifetime
# - If the investment periods are a pd.DatetimeIndex a build year before the considered time frame is considered. E.g. n.investment_periods = [2020, 2030, 2040] and lifetime of an asset is 15 year, build year is 2010, than the asset can only operate in 2020.

# add three lines in a ring
n.add(
    "Line",
    "line 0->1",
    bus0="bus 0",
    bus1="bus 1",
    x=0.0001,
    s_nom=0,
    # build_year=2030,
    s_nom_extendable=True,
)

n.add(
    "Line",
    "line 1->2",
    bus0="bus 1",
    bus1="bus 2",
    x=0.0001,
    capital_cost=10,
    build_year=2030,
    s_nom=0,
    s_nom_extendable=True,
)


n.add(
    "Line",
    "line 2->0",
    bus0="bus 2",
    bus1="bus 0",
    x=0.0001,
    s_nom=0,
    s_nom_extendable=True,
    build_year=2030,
)


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

# Create a random number generator
rng = np.random.default_rng()

# add some generators
p_nom_max = pd.Series(
    (rng.uniform() for _ in range(len(n.snapshots))),
    index=n.snapshots,
    name="generator ext 2020",
)

# renewable (can operate 2020, 2030)
n.add(
    "Generator",
    "generator ext 0 2020",
    bus="bus 0",
    p_nom=50,
    build_year=2020,
    lifetime=20,
    marginal_cost=2,
    capital_cost=1,
    p_max_pu=p_nom_max,
    carrier="solar",
    p_nom_extendable=True,
)

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
n.add(
    "Generator",
    "generator ext 0 2040",
    bus="bus 0",
    p_nom=50,
    build_year=2040,
    lifetime=11,
    marginal_cost=25,
    capital_cost=10,
    carrier="OCGT",
    p_nom_extendable=True,
)

# can operate in 2040
n.add(
    "Generator",
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

n.add(
    "StorageUnit",
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

n.add(
    "StorageUnit",
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
n.add("Bus", "bus 2 battery")
# #
n.add(
    "Store",
    "store 2 battery 2020",
    bus="bus 2 battery",
    # e_cyclic=True,
    e_nom_extendable=True,
    e_initial=20,
    build_year=2020,
    lifetime=20,
    capital_cost=0.1,
)

n.add(
    "Link",
    "bus2 battery charger",
    bus0="bus 2",
    bus1="bus 2" + " battery",
    # efficiency=0.8,
    # capital_cost=2,
    p_nom_extendable=True,
)

n.add(
    "Link",
    "My bus2 battery discharger",
    bus0="bus 2 battery",
    bus1="bus 2",
    efficiency=0.8,
    # marginal_cost=1,
    p_nom_extendable=True,
)


# add a Load
load_var = pd.Series(
    (100 * rng.uniform() for _ in range(len(n.snapshots))),
    index=n.snapshots,
    name="load",
)
load_fix = pd.Series(
    [250 for _ in range(len(n.snapshots))], index=n.snapshots, name="load"
)

# add a load at bus 2
n.add("Load", "load 2", bus="bus 2", p_set=load_fix)

n.add("Load", "load 1", bus="bus 1", p_set=0.3 * load_fix)

# currently only for pyomo=False
n.lopf(pyomo=False, multi_investment_periods=True)
