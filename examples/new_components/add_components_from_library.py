
#Use the component_library to add components.

import component_library

import pandas as pd

n = component_library.Network()

n.set_snapshots(list(range(4)))

places = pd.Index(["Aarhus","Aalborg","Copenhagen"])


n.madd("Bus", places)

n.madd("Load", places, bus=places, p_set=10.5)

n.madd("Bus",places + " heat")

n.madd("Load", places+ " heat", bus=places+ " heat", p_set=4.2)

n.loads_t.p_set["Aalborg heat"] = 3.7

n.madd("Bus",places + " gas")

n.madd("Store",places + " gas",
       bus=places + " gas",
       e_min_pu=-1,
       marginal_cost=50,
       e_nom_extendable=True)

#Some HVDC transmission lines
n.madd("Link",
       [0,1],
       bus0="Aarhus",
       bus1=["Aalborg","Copenhagen"],
       p_nom_extendable=True,
       p_min_pu=-1,
       capital_cost=1e2)

n.add("Generator",
      "wind",
      p_nom_extendable=True,
      capital_cost=1e2,
      bus="Aalborg")

n.madd("CHP",
       places + " CHP",
       bus_source=places + " gas",
       bus_elec=places,
       bus_heat=places + " heat",
       p_nom_extendable=True,
       capital_cost=1.4e4,
       eta_elec=0.468,
       c_v=0.15,
       c_m=0.75,
       p_nom_ratio=1.5)



print("\nn.chps:\n")
print(n.chps)

n.lopf()



print("\nGenerator capacity:\n")
print(n.generators.p_nom_opt)

print("\nLine capacities:\n")
print(n.links.loc[["0","1"],"p_nom_opt"])

print("\nCHP electrical output:\n")
print(n.links.loc[places + " CHP electric","p_nom_opt"]*n.links.loc[places + " CHP electric","efficiency"])
