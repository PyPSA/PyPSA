## Heat demand provided by a wind turbine, heat pump and water tank
#
#The water tank stores hot water with a standing loss.

import pypsa
import pandas as pd

network = pypsa.Network()

network.set_snapshots(pd.date_range("2016-01-01 00:00","2016-01-01 03:00",freq="H"))

network.add("Bus",
            "0",
            carrier="AC")

network.add("Bus",
            "0 heat",
            carrier="heat")

network.add("Carrier",
            "wind")
    
network.add("Carrier",
            "heat")

network.add("Generator",
            "wind turbine",
            bus="0",
            carrier="wind",
            p_nom_extendable=True,
            p_max_pu=[0.,0.2,0.7,0.4],
            capital_cost=500)

network.add("Load",
            "heat demand",
            bus="0 heat",
            p_set=20.)

#NB: Heat pump has changing efficiency (properly the Coefficient of Performance, COP)
#due to changing ambient temperature
network.add("Link",
            "heat pump",
            bus0="0",
            bus1="0 heat",
            efficiency=[2.5,3.,3.2,3.],
            capital_cost=1000,
            p_nom_extendable=True)
    
network.add("Store",
            "water tank",
            bus="0 heat",
            e_cyclic=True,
            e_nom_extendable=True,
            standing_loss=0.01)    

network.lopf(network.snapshots)

print(pd.DataFrame({attr: network.stores_t[attr]["water tank"] for attr in ["p","e"]}))

print(pd.DataFrame({attr: network.links_t[attr]["heat pump"] for attr in ["p0","p1"]}))

print(network.stores.loc["water tank"])

print(network.generators.loc["wind turbine"])

