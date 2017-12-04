## Show classic screening curve analysis for generation investment
#
#Compute the long-term equilibrium power plant investment for a given load duration curve (1000-1000z for z \in [0,1]) and a given set of generator investment options.
#
#Available as a Jupyter notebook at http://www.pypsa.org/examples/generation-investment-screening-curve.ipynb.

import pypsa
import numpy as np
import pandas as pd
#%matplotlib inline

#Generator marginal (m) and capital (c) costs in EUR/MWh - numbers chosen for simple answer
generators = {"coal" : {"m" : 2, "c" : 15},
              "gas" : {"m" : 12, "c": 10},
              "load-shedding" : {"m" : 1012, "c" : 0}}

#Screening curve intersections at 0.01 and 0.5
x = np.linspace(0,1,101)
df = pd.DataFrame({key : pd.Series(item["c"] + x*item["m"],x) for key,item in generators.items()})
df.plot(ylim=[0,50],title="Screening Curve")

n = pypsa.Network()

num_snapshots = 1001

snapshots = np.linspace(0,1,num_snapshots)

n.set_snapshots(snapshots)

n.snapshot_weightings = n.snapshot_weightings/num_snapshots

n.add("Bus",name="bus")

n.add("Load",name="load",bus="bus",
      p_set=1000-1000*snapshots)

for gen in generators:
    n.add("Generator",name=gen,bus="bus",
          p_nom_extendable=True,
          marginal_cost=float(generators[gen]["m"]),
          capital_cost=float(generators[gen]["c"]))

n.loads_t.p_set.plot(title="Load Duration Curve")

n.lopf(solver_name="cbc")

print(n.objective)

#capacity set by total electricity required
#NB: no load shedding since all prices < 1e4
n.generators.p_nom_opt.round(2)

n.buses_t.marginal_price.plot(title="Price Duration Curve")

#The prices correspond either to VOLL (1012) for first 0.01 or the marginal costs (12 for 0.49 and 2 for 0.5)

#EXCEPT for (infinitesimally small) points at the screening curve intersections, which
#correspond to changing the load duration near the intersection, so that capacity changes
#This explains 7 = (12+10 - 15) (replacing coal with gas) and 22 = (12+10) (replacing load-shedding with gas)

#I have no idea what is causing \l = 0; it should be 2.

n.buses_t.marginal_price.round(2).sum(axis=1).value_counts()

n.generators_t.p.plot(ylim=[0,600],title="Generation Dispatch")

#Demonstrate zero-profit condition
print("Total costs:")
print(n.generators.p_nom_opt*n.generators.capital_cost + n.generators_t.p.multiply(n.snapshot_weightings,axis=0).sum()*n.generators.marginal_cost)



print("\nTotal revenue:")
print(n.generators_t.p.multiply(n.snapshot_weightings,axis=0).multiply(n.buses_t.marginal_price["bus"],axis=0).sum())

## Without expansion optimisation
#
#Take the capacities from the above long-term equilibrium, then disallow expansion.
#
#Show that the resulting market prices are identical.
#
#This holds in this example, but does NOT necessarily hold and breaks down in some circumstances (for example, when there is a lot of storage and inter-temporal shifting).

n.generators.p_nom_extendable = False
n.generators.p_nom = n.generators.p_nom_opt

n.lopf(solver_name='glpk')

n.buses_t.marginal_price.plot(title="Price Duration Curve")

n.buses_t.marginal_price.sum(axis=1).value_counts()

#Demonstrate zero-profit condition

#Differences are due to singular times, see above, not a problem

print("Total costs:")
print(n.generators.p_nom*n.generators.capital_cost + n.generators_t.p.multiply(n.snapshot_weightings,axis=0).sum()*n.generators.marginal_cost)



print("Total revenue:")
print(n.generators_t.p.multiply(n.snapshot_weightings,axis=0).multiply(n.buses_t.marginal_price["bus"],axis=0).sum())

