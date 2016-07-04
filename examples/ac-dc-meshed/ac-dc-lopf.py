

# make the code as Python 3 compatible as possible
from __future__ import print_function, division
from __future__ import absolute_import


import pypsa,os

import pandas as pd

import numpy as np

from itertools import chain

network = pypsa.Network()

folder_name = "ac-dc-data"
network.import_from_csv_folder(folder_name)

now = network.snapshots[4]
network.now = now

network.controllable_branches().obj["Converter","Norway Converter"]

list(network.controllable_branches().index)

network.lopf(network.snapshots)

for sn in network.sub_networks.obj:
    print(sn,sn.carrier,len(sn.buses()),len(sn.branches()))


for t in chain(network.transport_links.obj,network.converters.obj):
    print(t)



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

results_folder_name = os.path.join(folder_name,"results-lopf")

if True:
    network.export_to_csv_folder(results_folder_name)
