

#In this example we add a new component type "dynamically" by passing
#the pypsa.Network the required information. Here we add a simple
#"ShadowPrice" component for storing the shadow prices of global
#constraints.

import pypsa, pandas as pd, numpy as np

from pypsa.descriptors import Dict

#take a copy of the components pandas.DataFrame
override_components = pypsa.components.components.copy()

#Pass it the list_name, description and component type.
override_components.loc["ShadowPrice"] = ["shadow_prices","Shadow price for a global constraint.",np.nan]

print("\nNew components table:\n")
print(override_components)

#create a pandas.DataFrame with the properties of the new component attributes.
#the format should be the same as pypsa/component_attrs/*.csv
override_component_attrs = Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["ShadowPrice"] = pd.DataFrame(columns = ["type","unit","default","description","status"])
override_component_attrs["ShadowPrice"].loc["name"] = ["string","n/a","n/a","Unique name","Input (required)"]
override_component_attrs["ShadowPrice"].loc["value"] = ["float","n/a",0.,"shadow value","Output"]

print("\nComponent attributes for ShadowPrice:\n")
print(override_component_attrs["ShadowPrice"])

#pass Network the information
n = pypsa.Network(override_components=override_components,
                  override_component_attrs=override_component_attrs)

n.add("ShadowPrice","line_volume_constraint",value=4567.1)
n.add("ShadowPrice","co2_constraint",value=326.3)

print("\nnetwork.shadow_prices:\n")

print(n.shadow_prices)
