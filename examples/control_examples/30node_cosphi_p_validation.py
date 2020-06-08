"""
This example shows validation of power factor as a function of real power control
strategy (cosphi_p) inverter control strategy against its droop characteristic.
"""
# importing important libraries
from __future__ import print_function, division
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np
import time
start = time.time()
pypsa.pf.logger.setLevel("INFO")

snapshots = pd.date_range(
                  start="2020-01-01 00:00", end="2020-01-01 02:00", periods=15)

# power injection for each snapshots
L = pd.Series([0.00025]*1+[0.000225]*1+[0.0002]*1+[0.000175]*1 +
              [0.0002]*1+[0.00015]*2+[0.000125]*1+[0.00004]*2 +
              [0.0001]*2+[0.000075]*2+[0.00005]*1, snapshots)

# building empty dataframes for storing results
Results_pf = pd.DataFrame(columns=[])
Results_injection = pd.DataFrame(columns=[])

network = pypsa.Network()
network.set_snapshots(snapshots)
n_buses = 30
for i in range(n_buses):
    network.add("Bus", "My bus {}".format(i), v_nom=.4)
for i in range(n_buses):
    network.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)
    network.add("Load", "My load {}".format(i), bus="My bus {}".format(
        (i) % n_buses), power_factor=0.95, s_nom=0.00025, v1=0.89, v2=0.94,
                            v3=0.96, v4=1.02, p_set=-L)
    network.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
                    (i+1) % n_buses), control="PQ", p_set=L, power_factor=0.95,
                             s_nom=0.00025, v1=0.89, v2=0.94, v3=0.96, v4=1.02)

# setting control strategy type
network.generators.type_of_control_strategy = 'cosphi_p'
network.lpf(network.snapshots)
network.pf(use_seed=True, snapshots=network.snapshots, x_tol_outer=1e-4)

# saving the necessary results for plotting controller behavior
for i in range(n_buses-1):
    Results_pf.loc[:, "pf_controller_output {}".format(
        i)] = network.generators_t.power_factor.loc[:, "My Gen {}".format(i+1)].values

    Results_injection.loc[:, "inverter_injection(p_set/s_nom) % {}".format(i)] = (
        network.generators_t.p_set.loc[:, "My Gen {}".format(i+1)])/(
            network.generators.loc["My Gen {}".format(i+1), 's_nom'])*100

# cosphi_p controller droop characteristic
def cosphi_p_controller_droop(injection):
    # The parameters for all generators (Gen) are same, here from # 7 are chosen
    set_p1 = network.generators.loc['My Gen 7', 'set_p1']
    set_p2 = network.generators.loc['My Gen 7', 'set_p2']
    power_factor_min = network.generators.loc['My Gen 7', 'power_factor_min']
    
    power_factor = np.select([(injection < set_p1), (
        injection >= set_p1) & (injection <= set_p2), (
            injection > set_p2)], [1 ,(1 - ((1 - power_factor_min) / (
             set_p2 - set_p1) * (injection - set_p1))), power_factor_min])
 
    return power_factor


# inverter injection (p_set/inverter_capacity)%, input for droop calculation
controller_droop_injection_percentage = [10, 20, 30, 40, 50, 55, 60, 65, 70,
                                         75, 80, 85, 90, 95, 100]
# droop output which is power factor(pf)
for i in range(len(snapshots)):
    Results_pf.loc[Results_pf.index[i], 'droop_pf'] = cosphi_p_controller_droop(
        controller_droop_injection_percentage[i])

'''  Plotting  '''
# droop characteristic input and output variables
droop_power_factor = Results_pf['droop_pf']
droop_real_power = controller_droop_injection_percentage

# controller input and output variables
power_factor = Results_pf.loc[
    :, "pf_controller_output 0":"pf_controller_output 28"]
p_set_injection_percentage = Results_injection.loc[
    :, "inverter_injection(p_set/s_nom) % 0":"inverter_injection(p_set/s_nom) % 28"]

# plot droop characteristic
plt.plot(droop_real_power, droop_power_factor,
         label="Q(U) droop characteristic", color='r')
# plot controller behavior
plt.scatter(p_set_injection_percentage, power_factor,
            color="g", label="cosphi_p controller characteristic")
# adding x and y ticks
# p_set_injection_percentage are same for all inverters so we chose #29 here
plt.xticks(p_set_injection_percentage['inverter_injection(p_set/s_nom) % 28'])
plt.yticks(power_factor['pf_controller_output 28'])

plt.title("Cosphi_p control strategy validation \n  30 node example, \n"
          "snapshots = 15")
plt.xlabel('Inverter_injection_percentage %')
plt.ylabel('Power factor')
plt.show()
end = time.time()
print(f"Runtime of the program is {end - start}")