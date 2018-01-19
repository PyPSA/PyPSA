

#In this example we add a new component type "dynamically" by passing
#the pypsa.Network the required information.

import pypsa, pandas as pd

#create a pandas.DataFrame with the properties of the new components
#the format should be the same as pypsa/components.csv, except add a column "attrs"
new_components = pd.DataFrame(data = [["chps","Combined heat and power plant.","controllable_one_port",None],
                                      ["methanations","Methanation plant.","controllable_one_port",None]],
                              index = ["CHP","Methanation"],
                              columns = ["list_name","description","type","attrs"])
print("\nNew components:\n")
print(new_components)

#now attach pandas.DataFrames for the attributes
#the format should be the same as pypsa/component_attrs/*.csv

chp_attrs = pd.DataFrame(data = [["string","n/a","n/a","Unique name","Input (required)"],
                                 ["string","n/a","n/a","name of bus to which generator is attached","Input (required)"],
                                 ["static or series","MW",0.,"active power set point (for PF)","Input (optional)"]],
                         index = ["name","bus","p_set"],
                         columns = ["type","unit","default","description","status"])

print("\nComponent attributes:\n")
print(chp_attrs)

new_components.at["CHP","attrs"] = chp_attrs
new_components.at["Methanation","attrs"] = chp_attrs

#pass Network the information
n = pypsa.Network(new_components=new_components)

n.set_snapshots(range(4))

n.add("Bus","Aarhus")
n.add("CHP","My CHP",
      bus="Aarhus",
      p_set=list(range(5,1,-1)))

print("\nn.chps:\n")
print(n.chps)

print("\nn.chps_t.p_set:\n")
print(n.chps_t.p_set)
