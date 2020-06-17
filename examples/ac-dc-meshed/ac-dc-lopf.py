import pypsa, os
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

network = pypsa.Network()

folder_name = "ac-dc-data"
network.import_from_csv_folder(folder_name)


network.lopf(network.snapshots)

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.EqualEarth()},
                                   figsize=(5,5))
line_colors = network.lines.bus0.map(network.buses.carrier)\
                     .replace({'AC': 'indianred', 'DC': 'limegreen'})
network.plot(bus_colors='grey', ax=ax,
       margin=.5, line_widths={'Line':2., 'Link':0},
       line_colors=line_colors,
       geomap='10m', title='Mixed AC-DC (red - green) network',
#       flow='mean',
       color_geomap=True)
fig.canvas.draw(); fig.tight_layout()
fig.savefig('ac_dc_meshed.png')


for sn in network.sub_networks.obj:
    print(sn,network.sub_networks.at[sn.name,"carrier"],len(sn.buses()),len(sn.branches()))

print("\nControllable branches:")

print(network.links)

now = network.snapshots[5]

print("\nCheck power balance at each bus:")

for bus in network.buses.index:
    print("\n"*3+bus)
    generators = sum(network.generators_t.p.loc[now,network.generators.bus==bus])
    loads = sum(network.loads_t.p.loc[now,network.loads.bus==bus])
    print("Generators:",generators)
    print("Loads:",loads)
    print("Total:",generators-loads)

    p0 = 0.
    p1 = 0.

    for c in network.iterate_components(network.branch_components):

        bs = (c.df.bus0 == bus)

        if bs.any():
            print(c,"\n",c.pnl.p0.loc[now,bs])
            p0 += c.pnl.p0.loc[now,bs].sum()

        bs = (c.df.bus1 == bus)

        if bs.any():
            print(c,"\n",c.pnl.p1.loc[now,bs])
            p1 += c.pnl.p1.loc[now,bs].sum()

    print("Branches",p0+p1)

    np.testing.assert_allclose(generators-loads+1.,p0+p1+1.)

    print("")

print(sum(network.generators_t.p.loc[now]))

print(sum(network.loads_t.p.loc[now]))

results_folder_name = os.path.join(folder_name,"results-lopf")

if True:
    network.export_to_csv_folder(results_folder_name)

