'''
This example shows validation of fixed power factor (fixed_cosphi) control
strategy against its droop characteristic.
'''
from __future__ import print_function, division
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import math
import numpy as np
import time

start = time.time()
pypsa.pf.logger.setLevel("INFO")

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
        (i) % n_buses))
    network.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
        (i+1) % n_buses), control="PQ", p_set=L, power_factor=0.95)

# setting control method
network.generators.type_of_control_strategy = 'fixed_cosphi'
network.lpf(network.snapshots)
network.pf(use_seed=True, snapshots=network.snapshots, x_tol_outer=1e-4)

# saving the necessary data for plotting controller behavior
for i in range(n_buses-1):
    Results_q.loc[:, "q_set_out_controller {}".format(
        i)] = network.generators_t.q_set.loc[:, "My Gen {}".format(i+1)].values
    Results_p.loc[:, "inverter_injection(p_set) {}".format(i)] = (
        network.generators_t.p_set.loc[:, "My Gen {}".format(i+1)])


# controller droop characteristic method simplified
def fixed_cosphi_controller_droop(p_set):
    pf = network.generators.loc['My Gen 7', 'power_factor']
    q_out = p_set*(math.tan((math.acos(pf))))
    return q_out


droop_df = pd.DataFrame(np.arange(0.004, 0.0001, -0.0003), columns=['optional_power_input'])
# calculating droop output (q_out)

for index, row in droop_df.iterrows():
    droop_df.loc[index, "droop_output"] = fixed_cosphi_controller_droop(
                                   droop_df.loc[index, 'optional_power_input'])

'''  Plotting  '''
# droop characteristic for x and y axis
droop_input = droop_df['optional_power_input']
droop_output = droop_df['droop_output']

# controller input and output variables
controller_p_set = Results_p.loc[
    :, "inverter_injection(p_set) 0":"inverter_injection(p_set) 28"]
controller_q_set = Results_q.loc[
    :, "q_set_out_controller 0":"q_set_out_controller 28"]

# plot droop characteristic
plt.plot(droop_input, droop_output, label="Q(U) droop characteristic", color='r')
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
