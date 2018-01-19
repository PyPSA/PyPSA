
#Use the component_library to add components.

#NB: This only works with Python 3 because of super() in component_library


import component_library

n = component_library.Network(csv_folder_name="../ac-dc-meshed/ac-dc-data/")

n.add("Bus","Aarhus")

n.add("CHP","My CHP",bus="Aarhus")


print("\nn.chps:\n")
print(n.chps)

print("\nn.chps_t.p_set:\n")
print(n.chps_t.p_set)


n.lopf()
