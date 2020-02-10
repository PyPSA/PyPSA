# -*- coding: utf-8 -*-

# Minimal example of PyPSA power flow with switches
# make the code as Python 3 compatible as possible
from __future__ import print_function, division
import pypsa
import matplotlib.pyplot as plt


def plot_with_switches(network):
    bus_colors = network.buses_t.p.iloc[0]
    bus_colors[bus_colors > 0] = "blue"
    bus_colors[bus_colors != "blue"] = "red"
    network.plot(branch_components=['Line', 'Switch'],
                 line_widths={'Line': 2, 'Switch': 0},  # Note: switch width set to 0
                 line_colors={'Line': 'green', 'Switch': 'gray'},
                 bus_sizes=abs(network.buses_t.p.iloc[0]),
                 bus_colors=bus_colors,
                 geomap=False)
    plt.show()


network = pypsa.Network()
network.set_snapshots(["test"])
# add a bus and a generator
network.add("Bus", "n1", v_nom=20., x=0, y=0)
network.add("Generator", "gen", bus="n1", p_set=100, control="Slack")
# add two buses and lines
network.add("Bus", "n2", v_nom=20., x=1, y=1)
network.add("Bus", "n3", v_nom=20., x=1, y=-1)
network.add("Line", "line1", bus0="n1", bus1="n2", x=0.1, r=0.01, s_nom=50)
network.add("Line", "line2", bus0="n1", bus1="n3", x=0.1, r=0.01, s_nom=50)
# add two buses and switches and connect them with a line
network.add("Bus", "n4", v_nom=20., x=1, y=.9)
network.add("Bus", "n5", v_nom=20., x=1, y=-.9)
network.add("Line", "line3", bus0="n4", bus1="n5", x=0.1, r=0.01, s_nom=50)
network.add("Load", "load1", bus="n4", p_set=50)
network.add("Load", "load2", bus="n3", p_set=50)
network.add_switch("s1", "n2", "n4", 1)  # can (dis)connect line1 and (line3 + load1)
network.add_switch("s2", "n3", "n5", 0)  # can (dis)connect line2 and line3

network.pf()
print(network.buses_t.p)
print(network.lines_t.p0)
plot_with_switches(network)
print("switching, so that load1 is disconnected from line1 and connected to line3")
network.open_switches(["s1"])
network.close_switches(["s2"])
network.pf()
print(network.buses_t.p)
print(network.lines_t.p0)
plot_with_switches(network)
