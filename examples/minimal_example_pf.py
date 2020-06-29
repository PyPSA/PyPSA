## Minimal 3-node example of PyPSA power flow
#
#Available as a Jupyter notebook at <http://www.pypsa.org/examples/minimal_example_pf.ipynb>.


import pypsa

import numpy as np

network = pypsa.Network()

#add three buses
n_buses = 3

for i in range(n_buses):
    network.add("Bus","My bus {}".format(i),
                v_nom=20.)

print(network.buses)

#add three lines in a ring
for i in range(n_buses):
    network.add("Line","My line {}".format(i),
                bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1)%n_buses),
                x=0.1,
                r=0.01)

print(network.lines)

#add a generator at bus 0
network.add("Generator","My gen",
            bus="My bus 0",
            p_set=100,
            control="PQ")

print(network.generators)

print(network.generators.p_set)

#add a load at bus 1
network.add("Load","My load",
            bus="My bus 1",
            p_set=100)

print(network.loads)

print(network.loads.p_set)

network.loads.q_set = 100.

#Do a Newton-Raphson power flow
network.pf()

print(network.lines_t.p0)

print(network.buses_t.v_ang*180/np.pi)

print(network.buses_t.v_mag_pu)

