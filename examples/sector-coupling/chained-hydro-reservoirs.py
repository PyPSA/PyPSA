## Two chained reservoirs with inflow in one supply two electric loads
#
#Two disconnected electrical loads are fed from two reservoirs linked by a river; the first reservoir has inflow from rain onto a water basin.
#
#Note that the two reservoirs are tightly coupled, meaning there is no time delay between the first one emptying and the second one filling, as there would be if there were a long stretch of river between the reservoirs. The reservoirs are essentially assumed to be close to each other. A time delay would require a "Link" element between different snapshots, which is not yet supported by PyPSA (but could be enabled by passing network.lopf() an extra_functionality function).

import pypsa
import pandas as pd
import numpy as np

#First tell PyPSA that links will have a 2nd bus by
#overriding the component_attrs. This is needed so that
#water can both go through a turbine AND feed the next reservoir

override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]


network = pypsa.Network(override_component_attrs=override_component_attrs)

network.set_snapshots(pd.date_range("2016-01-01 00:00","2016-01-01 03:00",freq="H"))

network.add("Carrier",
            "reservoir")
    
network.add("Carrier",
            "rain")


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


#The efficiency of a river is the relation between the gravitational potential
#energy of 1 m^3 of water in reservoir 0 relative to its turbine versus the 
#potential energy of 1 m^3 of water in reservoir 1 relative to its turbine

network.add("Link",
            "spillage",
            bus0="0 reservoir",
            bus1="1 reservoir",
            efficiency=0.5,
            p_nom_extendable=True) 


#water from turbine also goes into next reservoir
network.add("Link",
            "0 turbine",
            bus0="0 reservoir",
            bus1="0",
            bus2="1 reservoir",
            efficiency=0.9,
            efficiency2=0.5,
            capital_cost=1000,
            p_nom_extendable=True)

network.add("Link",
            "1 turbine",
            bus0="1 reservoir",
            bus1="1",
            efficiency=0.9,
            capital_cost=1000,
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

print(network.generators_t.p)

print(network.links_t.p0)

print(network.links_t.p1)

print(network.links_t.p2)

print(pd.DataFrame({attr: network.stores_t[attr]["0 reservoir"] for attr in ["p","e"]}))

print(pd.DataFrame({attr: network.stores_t[attr]["1 reservoir"] for attr in ["p","e"]}))

