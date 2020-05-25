from __future__ import print_function, division
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import math
import numpy as np
import time

start = time.time()
'''
This example shows the validation of fixed power factor control strategy
(fixed_cosphi) by plotting the controller output and its droop characteristic
in the same figure.
'''
snapshots = pd.date_range(
    start="2020-01-01 00:00", end="2020-01-01 02:00", periods=15)

# power injection
L = pd.Series([0.0025]*1+[0.00225]*1+[0.002]*1+[0.00175]*1 +
              [0.002]*1+[0.0015]*2+[0.00125]*1+[0.0004]*2 +
              [0.001]*2+[0.00075]*2+[0.005]*1, snapshots)

# building empty dataframes
Results_p = pd.DataFrame(columns=[])
Results_q = pd.DataFrame(columns=[])
# defining the network
network = pypsa.Network()
network.set_snapshots(snapshots)
n_buses = 30
for i in range(n_buses):
    network.add("Bus", "My bus {}".format(i), v_nom=.4)
for i in range(n_buses):
    network.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)
    network.add("Load", "My load {}".format(i), bus="My bus {}".format(
        (i+1) % n_buses))
    network.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
        (i+1) % n_buses), control="PQ", p_set=L, power_factor=0.95)


def power_flow():
    network.lpf(network.snapshots)
    network.pf(use_seed=True, snapshots=network.snapshots, x_tol_outer=1e-4)


# setting control method
network.generators.type_of_control_strategy = 'fixed_cosphi'
power_flow()
v_pu_yes = network.buses_t.v_mag_pu
# saving the necessary data for plotting controller behavior
for i in range(n_buses):
    Results_q.loc[:, "q_set_out_controller {}".format(
        i)] = network.generators_t.q_set.loc[:, "My Gen {}".format(i)].values
    Results_p.loc[:, "inverter_injection(p_set) {}".format(i)] = (
        network.generators_t.p_set.loc[:, "My Gen {}".format(i)])


# controller droop characteristic method simplified
def droop_fixed_cosphi(p_set):
    pf = network.generators.loc['My Gen 7', 'power_factor']
    q_out = p_set*(math.tan((math.acos(pf))))
    return q_out


droop_df = pd.DataFrame(np.arange(0.004, 0.0001, -0.0003), columns=['optional_power_input'])
# calculating droop output (q_out)

for index, row in droop_df.iterrows():
    droop_df.loc[index, "droop_output"] = droop_fixed_cosphi(droop_df.loc[index, 'optional_power_input'])


'''  Plotting  '''
# droop characteristic for x and y axis
droop_input = droop_df['optional_power_input']
droop_output = droop_df['droop_output']

# controller input and output variables
controller_p_set = Results_p.loc[
    :, "inverter_injection(p_set) 1":"inverter_injection(p_set) 29"]
controller_q_set = Results_q.loc[
    :, "q_set_out_controller 1":"q_set_out_controller 29"]

# plot droop characteristic
plt.plot(droop_input, droop_output, label="Q(U) droop characteristic",
          color='r')
# plot controller behavior
plt.scatter(controller_p_set, -controller_q_set,
            color="g", label="fixed_cosphi controller characteristic")
# q_set and p_set are same for all compoenents, here #29 for ticks is chosen
plt.xticks((Results_p['inverter_injection(p_set) 27']), rotation=70)
plt.yticks(-controller_q_set['q_set_out_controller 27'])
plt.title("fixed_cosphi control strategy validation \n  30 node example, \n"
          "snapshots = 15")
plt.xlabel('Real power input')
plt.ylabel('Reactive power compensation (new q_set)')
plt.show()
end = time.time()
print(f"Runtime of the program is {end - start}")
