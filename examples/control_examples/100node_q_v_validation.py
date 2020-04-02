from __future__ import print_function, division
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np
import math

snapshots = pd.date_range(start="2020-01-01 00:00", end="2020-01-01 02:00",  periods=15)
L = pd.Series([-0.00001]*1  + [-0.000002]*1 + [-0.000003]*1 + [-0.000005]*1 +[-0.000007]*2+ [-0.000008]*1 +[-0.000001]*2+ [-0.0000021]*2 + [-0.0000022]*2 + [-0.000023]*2  , snapshots)
L1 = pd.Series([0.0006]*1  + [0.00055]*1 + [0.0005]*1 + [0.00045]*1 +[0.0004]*2+ [0.0001]*1 +[0.00009]*1+ [0.00008]*1+ [0.00006]*1 +[0.00003]*1 + [0.00001]*1 +[0.000007]*1 +[0.000002]*1 + [0.0000001]*1  , snapshots)
Results_v = pd.DataFrame(columns=["V_pu_no_control", "p_set", "p/pmaxdroop"])
Results_q = pd.DataFrame(columns=["V_pu_no_control", "p_set", "p/pmaxdroop"])
# Results.loc[:, 'Time'] = snapshots

network = pypsa.Network()
network.set_snapshots(snapshots)
n_buses = 100
for i in range(n_buses):
    network.add("Bus", "My bus {}".format(i), v_nom=.4)
network.add("Bus", "MV bus", v_nom=20, v_mag_pu_set=1.02)
for i in range(n_buses):
    network.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)

for i in range(n_buses):
    network.add("Load", "My load {}".format(i), bus="My bus {}".format((i+1) % n_buses),
                p_set=L1, damper=0.8, v1=0.89, v2=0.94, v3=0.96, v4=1.02)


network.add("Line", "MV cable", type="NAYY 4x50 SE", bus0="MV bus", bus1="My bus 0", length=0.1)
buses = network.buses
network.add("Transformer", "MV-LV trafo", type="0.63 MVA 10/0.4 kV", bus0="MV bus", bus1="My bus 0")
network.add("Generator", "External Grid", bus="MV bus", control="Slack")
for i in range(n_buses):
    network.add("Generator", "My Gen {}".format(i), bus="My bus {}".format((i+1) % n_buses), control="PQ", p_set=L, 
                sn=0.000025, v1=0.89, v2=0.94, v3=0.96, v4=1.02, damper=1)

def power_flow():
    network.lpf(network.snapshots)
    network.pf(use_seed=True, snapshots=network.snapshots, x_tol_outer=1e-4)

power_flow()
Results_v.loc[:, 'V_pu_no_control'] = network.buses_t.v_mag_pu.loc[:, 'My bus 27'].values
Results_q.loc[:, 'p_set'] = network.generators_t.p_set.loc[:, 'My Gen 26'].values

network.generators.type_of_control_strategy = 'q_v'
power_flow()
for i in range(n_buses):
    Results_v.loc[:, "V_pu_with_control {}".format(i)] = network.buses_t.v_mag_pu.loc[:, "My bus {}".format(i)].values
    Results_q.loc[:, "q_set_controller_output {}".format(i)] = ((network.generators_t.q_set.loc[:, "My Gen {}".format(i)].values)/(np.sqrt((network.generators.loc["My Gen {}".format(i), 'sn'])**2-(Results_q["p_set"])**2)))*100

vbus = network.buses_t.v_mag_pu
loads = network.loads
# # This function (q_v) is only used to get droop curve
def q_v(v_pu_bus):
    v1=network.generators.loc['My Gen 7', 'v1']
    v2=network.generators.loc['My Gen 7', 'v2']
    v3=network.generators.loc['My Gen 7', 'v3']
    v4=network.generators.loc['My Gen 7', 'v4']
    damper = network.generators.loc['My Gen 7', 'damper']

    if v_pu_bus < v1:
        qbyqmax = 100

    elif v_pu_bus >= v1 and v_pu_bus <= v2:
        qbyqmax = np.interp(v_pu_bus, [v1, v2], [100, 0], 100,  0)

    elif v_pu_bus > v2 and v_pu_bus <= v3:
        qbyqmax = 0

    elif v_pu_bus > v3 and v_pu_bus <= v4:
        qbyqmax = np.interp(v_pu_bus, [v3, v4], [0, -100], 0, -100)

    elif v_pu_bus > v4:
        qbyqmax = -100
    
    return -qbyqmax*damper

# # q_set % calculation for droop characteristic
vmag = [0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03]
for index, row in Results_q.iterrows():
    Results_q.loc[index, "p/pmaxdroop"] = q_v(vmag[index])


# # # plotting
vdroop = vmag
qdroop = -Results_q["p/pmaxdroop"]

v_control = Results_v.loc[:, "V_pu_with_control 1" : "V_pu_with_control 99"]
qcontrol = Results_q.loc[:, "q_set_controller_output 0" : "q_set_controller_output 98"]

plt.plot(vmag, qdroop, label="Q(U) droop characteristic", color='r')
plt.scatter(v_control, qcontrol, color="g", label="Q(U) Controller behavior")

plt.legend(loc="best", bbox_to_anchor=(0.4, 0.4))
plt.title("Q(U) Controller Behavior \n  100 node example \n snapshots = 15\n while loop condition = 1e-4\n number of required iterations 52")
plt.xlabel('bus_mag_pu_voltage')
plt.ylabel('Reactive power compensation Q(U) %')
plt.show()






