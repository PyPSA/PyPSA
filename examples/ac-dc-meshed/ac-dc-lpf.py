

#Compute Linear Power Flow for each snapshot for AC-DC network in
#folder ac-dc-data/


# make the code as Python 3 compatible as possible
from __future__ import print_function, division
from __future__ import absolute_import


import pypsa, os

import pandas as pd

import numpy as np

from itertools import chain

network = pypsa.Network()

folder_name = "ac-dc-data"
network.import_from_csv_folder(folder_name)

for snapshot in network.snapshots:
    network.lpf(snapshot)


print("\nSub-Networks:")

for sn in network.sub_networks.obj:
    print(sn,sn.current_type,len(sn.buses()),len(sn.branches()))


print("\nControllable branches:")

for t in chain(network.transport_links.obj,network.converters.obj):
    print(t)

now = network.snapshots[5]


print("\nCheck power balance at each branch:")


for bus in network.buses.obj:
    print("\n"*3+bus.name)
    print("Generators:",sum(bus.generators_t().p.loc[now]))
    print("Loads:",sum(bus.loads_t().p.loc[now]))
    print("Total:",sum(item.p[now]*item.sign for item in chain(bus.generators().obj,bus.loads().obj)))

    print("Branches",sum(b.p0[now] for b in network.branches().obj if b.bus0 == bus.name)+sum(b.p1[now] for b in network.branches().obj if b.bus1 == bus.name))

    print("")

    for b in network.branches().obj:
        if b.bus0 == bus.name:
            print(b,b.p0[now])
        elif b.bus1 == bus.name:
            print(b,b.p1[now])

print(sum(network.generators_t.p.loc[now]))

print(sum(network.loads_t.p.loc[now]))


results_folder_name = os.path.join(folder_name,"results-lpf")

if True:
    network.export_to_csv_folder(results_folder_name,time_series={"generators" : {"p" : None},
                                                            "lines" : {"p0" : None},
                                                            "converters" : {"p0" : None},
                                                             "transport_links" : {"p0" : None}})
