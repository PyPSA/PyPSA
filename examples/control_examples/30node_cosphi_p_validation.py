"""
This example shows validation of power factor as a function of real power
(cosphi_p) inverter control strategy against its droop characteristic.
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
Results_droop = pd.DataFrame(columns=[])

n = pypsa.Network()
n.set_snapshots(snapshots)
n_buses = 30
for i in range(n_buses):
    n.add("Bus", "My bus {}".format(i), v_nom=.4)
for i in range(n_buses):
    n.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
          bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)

    n.add("Load", "My load {}".format(i), bus="My bus {}".format(
        (i) % n_buses), p_set=-L)

    n.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
                    (i+1) % n_buses), control="PQ", p_set=L, power_factor_min=0.9,
                            s_nom=0.00030, v1=0.89, v2=0.94, v3=0.96, v4=1.02,
                            set_p1=40, sep_p2=100, p_ref=0.000250)

# setting control strategy type
n.generators.control_strategy = 'cosphi_p'
n.lpf(n.snapshots)
n.pf(use_seed=True, snapshots=n.snapshots, x_tol_outer=1e-4, inverter_control=True)

# saving the necessary results for plotting controller behavior
Results_power_factor = n.generators_t.power_factor.loc[:, 'My Gen 1':'My Gen 28']
Results_power_injection_percent = n.generators_t.p.loc[:, 'My Gen 1':'My Gen 28'] / (
        n.generators.loc['My Gen 1':'My Gen 28', 'p_ref'])*100

# cosphi_p controller droop characteristic
def cosphi_p_controller_droop(injection):
    # The parameters for all generators (Gen) are same, here from # 7 are chosen
    set_p1 = n.generators.loc['My Gen 7', 'set_p1']
    set_p2 = n.generators.loc['My Gen 7', 'set_p2']
    power_factor_min = n.generators.loc['My Gen 7', 'power_factor_min']

    power_factor = np.select([(injection < set_p1), (
        injection >= set_p1) & (injection <= set_p2), (
            injection > set_p2)], [1, (1 - ((1 - power_factor_min) / (
             set_p2 - set_p1) * (injection - set_p1))), power_factor_min])

    return power_factor


# inverter injection (p_set/inverter_capacity)%, input for droop calculation
controller_droop_injection_percentage = [10, 20, 30, 40, 50, 55, 60, 65, 70,
                                         75, 80, 85, 90, 95, 100]
# droop output which is power factor(pf)
for i in range(len(snapshots)):
    Results_droop.loc[i, 'droop_pf'] = cosphi_p_controller_droop(
        controller_droop_injection_percentage[i])

'''  Plotting  '''
# droop characteristic input and output variables
droop_power_factor = Results_droop['droop_pf']
droop_real_power = controller_droop_injection_percentage

# plot droop characteristic
plt.plot(droop_real_power, Results_droop,
         label="Q(U) droop characteristic", color='r')
# plot controller behavior
plt.scatter(Results_power_injection_percent, Results_power_factor,
            color="g", label="cosphi_p controller characteristic")
# adding x and y ticks
# p_set_injection_percentage are same for all inverters so we chose #7 here
plt.xticks(Results_power_injection_percent['My Gen 7'], rotation=70)
plt.yticks(Results_power_factor['My Gen 7'])

plt.title("Cosphi_p control strategy validation \n  30 node example, \n"
          "snapshots = 15")
plt.xlabel('Inverter__power_injection_percentage %')
plt.ylabel('Power factor (per unit)')
plt.show()
end = time.time()
print(f"Runtime of the program is {end - start}")
