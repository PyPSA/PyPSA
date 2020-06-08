"""
This example shows validation of reactive power as a function of voltage Q(U)
inverter control strategy against its droop characteristic.
"""
# importing important modules
from __future__ import print_function, division
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
pypsa.pf.logger.setLevel("INFO")

snapshots = pd.date_range(start="2020-01-01 00:00", end="2020-01-01 02:00",
                          periods=15)

# power generation or injection
L = pd.Series([0.00001]*1+[0.00002]*1+[0.00003]*1+[0.00005]*1+[0.00007]*2 +
              [0.00008]*1+[0.0001]*2+[0.00021]*2+[0.00022]*2+[0.00023]*2,
              snapshots)
# load
L1 = pd.Series([0.007]*1+[0.0066]*1+[0.006]*1+[0.0055]*1+[0.005]*2+[0.0045]*1 +
               [0.00499]*1+[0.0047]*1+[0.0043]*1+[0.002]*1+[0.0008]*1 +
               [0.0003]*1+[0.00009]*1+[0.00001]*1, snapshots)

# building empty dataframes for storing results
Results_v = pd.DataFrame(columns=[])
Results_q = pd.DataFrame(columns=[])

# defining network
network = pypsa.Network()
network.set_snapshots(snapshots)
n_buses = 30

for i in range(n_buses):
    network.add("Bus", "My bus {}".format(i), v_nom=.4)
for i in range(n_buses):
    network.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
        (i) % n_buses), control="PQ", p_set=L, s_nom=0.00025, v1=0.89, v2=0.94,
        v3=0.96, v4=1.02)
    network.add("Load", "My load {}".format(i), bus="My bus {}".format(
        (i) % n_buses), s_nom=0.00025, p_set=L1)
    network.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)
# setting control strategy type to Q(U) controller
network.generators.type_of_control_strategy = 'q_v'
network.lpf(network.snapshots)
network.pf(use_seed=True, snapshots=network.snapshots, x_tol_outer=1e-4)

# # saving the necessary results for plotting controller behavior
for i in range(n_buses-1): # slack bus and slack genenerator are excluded
    # P_set values are same for all generators at each snapshot
    Results_q.loc[:, 'p_set'] = network.generators_t.p_set.loc[
        :, 'My Gen 26'].values
    # format(i+1) to exclude slack bus voltage magnitude
    Results_v.loc[:, "V_pu_with_control {}".format(i)
                  ] = network.buses_t.v_mag_pu.loc[
                                              :, "My bus {}".format(i+1)].values
    # forma(i+1) to exclude slack generator
    Results_q.loc[:, "q_set_controller_output {}".format(i)] = ((
        network.generators_t.q_set.loc[:, "My Gen {}".format(i+1)].values)/(
            np.sqrt((network.generators.loc["My Gen {}".format(i+1),'s_nom'])**2-(
                Results_q["p_set"])**2)))*100

# Q(U) controller droop characteristic
def q_v_controller_droop(v_pu_bus):
    # parameters are same for all the generators, here from Gen 7 are used
    v1 = network.generators.loc['My Gen 7', 'v1']
    v2 = network.generators.loc['My Gen 7', 'v2']
    v3 = network.generators.loc['My Gen 7', 'v3']
    v4 = network.generators.loc['My Gen 7', 'v4']
    damper = network.generators.loc['My Gen 7', 'damper']
    # np.select([conditions], [choices]), qbyqmax is chosen based on conditions
    qbyqmax = np.select([(v_pu_bus < v1), (v_pu_bus >= v1) & (
                v_pu_bus <= v2), (v_pu_bus > v2) & (v_pu_bus <= v3), (
                    v_pu_bus > v3) & (v_pu_bus <= v4), (v_pu_bus > v4)
                    ], [100, (100-(100-0)/(v2-v1)*(v_pu_bus-v1)),
                        0, -(100)*(v_pu_bus-v3) / (v4-v3), (-100)])

    return qbyqmax*damper

# power factor optional values to calculate droop characteristic (%)
Results_droop = pd.DataFrame(np.arange(0.88, 1.03, 0.01), columns=['v_mag'])
# calculation of droop characteristic (output)
for index, row in Results_droop.iterrows():
    Results_droop.loc[index, "p/pmaxdroop"] = q_v_controller_droop(
                                              Results_droop.loc[index, 'v_mag'])

''' Plotting '''
# droop characteristic input and output variables
vdroop = Results_droop['v_mag']
qdroop = Results_droop["p/pmaxdroop"]
# controller input and output variables
v_control = Results_v.loc[:, "V_pu_with_control 0":"V_pu_with_control 28"]
qcontrol = Results_q.loc[:, "q_set_controller_output 0":
                          "q_set_controller_output 28"]
# plot droop characteristic
plt.plot(vdroop, qdroop, label="Droop characteristic", color='r')
# plot controller behavior
plt.scatter(v_control, qcontrol, color="g",
            label="Controller behaviour")
# labeling controller parameters
plt.text(network.generators.loc['My Gen 7', 'v1'], 100, "v1= %s" % (
    network.generators.loc['My Gen 7', 'v1']), rotation=0)
plt.text(network.generators.loc['My Gen 7', 'v2'], 0, "v2= %s" % (
    network.generators.loc['My Gen 7', 'v2']), rotation=90)
plt.text(network.generators.loc['My Gen 7', 'v3'], 0, "v3= %s" % (
    network.generators.loc['My Gen 7', 'v3']), rotation=90)
plt.text(network.generators.loc['My Gen 7', 'v4'], -100, "v4= %s" % (
    network.generators.loc['My Gen 7', 'v4']), rotation=90)
# legendsa and titles
plt.legend(loc="best", bbox_to_anchor=(0.4, 0.4))
plt.title("Q(U) Controller Behavior \n  30 node example 15 snapshots\n"
          "while loop condition = 1e-4\n v1,v2,v3,v4 are controller setpoints")
plt.xlabel('voltage_pu')
plt.ylabel('Reactive power compensation Q(U) %')
plt.show()
end = time.time()
print(f"Runtime of the program is {end - start}")
