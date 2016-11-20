## Two chained reservoirs with inflow in one supply two electric loads
#
#Two disconnected electrical loads are fed from two reservoirs linked by a river; the first reservoir has inflow from a water basin.

import pypsa
import pandas as pd

from pyomo.environ import Constraint


network = pypsa.Network()

network.set_snapshots(pd.date_range("2016-01-01 00:00","2016-01-01 03:00",freq="H"))

network.add("Bus",
            "0",
            carrier="AC")

network.add("Bus",
            "1",
            carrier="AC")

network.add("Bus",
            "0 reservoir",
            carrier="reservoir")


network.add("Bus",
            "1 reservoir",
            carrier="reservoir")

network.add("Carrier",
            "reservoir")
    
network.add("Carrier",
            "rain")

network.add("Generator",
            "rain",
            bus="0 reservoir",
            carrier="rain",
            p_nom=1000,
            p_max_pu=[0.,0.2,0.7,0.4])

network.add("Load",
            "0 load",
            bus="0",
            p_set=20.)


network.add("Load",
            "1 load",
            bus="1",
            p_set=30.)

network.add("Link",
            "0 turbine",
            bus0="0 reservoir",
            bus1="0",
            efficiency=0.9,
            capital_cost=1000,
            p_nom_extendable=True)

network.add("Link",
            "1 turbine",
            bus0="1 reservoir",
            bus1="1",
            efficiency=0.9,
            capital_cost=1000,
            p_nom_extendable=True)




#The efficiency of a river is the relation between the gravitational potential
#energy of 1 m^3 of water in reservoir 0 relative to its turbine versus the 
#potential energy of 1 m^3 of water in reservoir 1 relative to its turbine

network.add("Link",
            "river",
            bus0="0 reservoir",
            bus1="1 reservoir",
            efficiency=0.5,
            p_nom_extendable=True)    


network.add("Store",
            "0 reservoir",
            bus="0 reservoir",
            e_cyclic=True,
            e_nom_extendable=True) 


network.add("Store",
            "1 reservoir",
            bus="1 reservoir",
            e_cyclic=True,
            e_nom_extendable=True)

network.lopf(network.snapshots)
print("Objective:",network.objective)

print(pd.DataFrame({attr: network.stores_t[attr]["0 reservoir"] for attr in ["p","e"]}))

print(pd.DataFrame({attr: network.stores_t[attr]["1 reservoir"] for attr in ["p","e"]}))

print(network.links_t.p0)

print(network.generators_t.p)

