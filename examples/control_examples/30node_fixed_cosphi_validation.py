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
L = pd.Series([0.0001]*1+[0.0003]*1+[0.0005]*1+[0.0008]*1 +
              [0.001]*1+[0.0015]*2+[0.002]*1+[0.0025]*2 +
              [0.003]*2+[0.0035]*2+[0.004]*1, snapshots)

# building empty dataframes
Results_p = pd.DataFrame(columns=[])
Results_q = pd.DataFrame(columns=[])
# defining the n
n = pypsa.Network()
n.set_snapshots(snapshots)
n_buses = 30
for i in range(n_buses):
    n.add("Bus", "My bus {}".format(i), v_nom=.4)
for i in range(n_buses):
    n.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)
    n.add("Load", "My load {}".format(i), bus="My bus {}".format(
        (i) % n_buses))
    n.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
        (i+1) % n_buses), control="PQ", p_set=L, power_factor=0.98)

# setting control method
n.generators.control_strategy = 'fixed_cosphi'
n.lpf(n.snapshots)
n.pf(use_seed=True, snapshots=n.snapshots, x_tol_outer=1e-4, inverter_control=True)

# saving the necessary data for plotting controller behavior

Results_q = n.generators_t.q.loc[:, 'My Gen 1':'My Gen 28']
Results_p = n.generators_t.p.loc[:, 'My Gen 1':'My Gen 28']

# controller droop characteristic method simplified
def fixed_cosphi_controller_droop(p_set):
    pf = n.generators.loc['My Gen 7', 'power_factor']
    q_out = p_set*(math.tan((math.acos(pf))))
    return q_out


droop_df = pd.DataFrame(np.arange(0.004, 0.0001, -0.0003), columns=['optional_power_input'])
# calculating droop output (q_out)

for index, row in droop_df.iterrows():
    droop_df.loc[index, "droop_output"] = fixed_cosphi_controller_droop(
                                   droop_df.loc[index, 'optional_power_input'])

'''  Plotting  '''

# plot droop characteristic
plt.plot(droop_df['optional_power_input'], droop_df['droop_output'], label="Q(U) droop characteristic", color='r')
# plot controller behavior
plt.scatter(Results_p, -Results_q,
            color="g", label="fixed_cosphi controller characteristic")
# q_set and p_set are same for all compoenents, here #29 for ticks is chosen
plt.xticks((Results_p['My Gen 27']), rotation=70)
plt.yticks(-Results_q['My Gen 27'])
plt.title("fixed_cosphi control strategy validation \n  30 node example, \n"
          "snapshots = 15")
plt.xlabel('Real power input (MW)')
plt.ylabel('Reactive power compensation (Mvar)')
plt.show()
end = time.time()
print(f"Runtime of the program is {end - start}")
