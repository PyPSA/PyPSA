## Heat demand provided by a wind turbine, heat pump and water tank
#
#The water tank stores hot water with a standing loss.

import pypsa
import pandas as pd

from pyomo.environ import Constraint


network = pypsa.Network()

network.set_snapshots(pd.date_range("2016-01-01 00:00","2016-01-01 03:00",freq="H"))

network.add("Bus",
            "0",
            carrier="AC")

network.add("Bus",
            "0 heat",
            carrier="heat")

network.add("Source",
            "wind")
    
network.add("Source",
            "heat")

network.add("Generator",
            "wind turbine",
            bus="0",
            source="wind",
            dispatch="variable",
            p_nom_extendable=True,
            p_max_pu=[0.,0.2,0.7,0.4],
            capital_cost=500)

network.add("Load",
            "heat demand",
            bus="0 heat",
            p_set=20.)


network.add("Link",
            "heat pump",
            bus0="0",
            bus1="0 heat",
            efficiency=3.,
            capital_cost=1000,
            s_nom_extendable=True)
    
network.add("Store",
            "water tank",
            bus="0 heat",
            e_cyclic=True,
            e_nom_extendable=True,
            source="heat",
            standing_loss=0.01)    

network.lopf(network.snapshots)

print(network.stores_t.loc[["p","e"],:,"water tank"])

print(network.links_t.loc[["p0","p1"],:,"heat pump"])

print(network.stores.loc["water tank"])

print(network.generators.loc["wind turbine"])

