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

# power generation or injection to the grid
L = pd.Series([0.000001]*1+[0.000002]*1+[0.000003]*1+[0.00005]*1+[0.00007]*2 +
              [0.00008]*1+[0.0001]*1+[0.00015]*1+[0.00018]*1+[0.0002]*1 +
              [0.00021]*1+[0.00022]*1+[0.00023]*1+[0.00024]*1, snapshots)
# load
L1 = pd.Series([0.00675]*1+[0.0067]*1+[0.0066]*1+[0.0065]*1+[0.0063]*2 +
               [0.0062]*1+[0.0060]*1+[0.0058]*1+[-0.0]*1+[0.0004]*1 +
               [0.0006]*1+[0.003]*1+[0.004]*1+[0.005]*1, snapshots)

# building empty dataframes for storing results
Results_v = pd.DataFrame(columns=[])
Results_q = pd.DataFrame(columns=[])

# defining n
n = pypsa.Network()
n.set_snapshots(snapshots)
n_buses = 30

for i in range(n_buses):
    n.add("Bus", "My bus {}".format(i), v_nom=.4, v_mag_pu_set=1.03)
for i in range(n_buses):
    n.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
        (i) % n_buses), control="PQ", p_set=L, s_nom=0.00025, v1=0.89, v2=0.94,
        v3=0.96, v4=1.02, power_factor=0.9)
    n.add("Load", "My load {}".format(i), bus="My bus {}".format(
        (i) % n_buses), p_set=L1)
    n.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)

# setting control strategy type to Q(U) controller
n.generators.control_strategy = 'q_v'
n.lpf(n.snapshots)
n.pf(use_seed=True, snapshots=n.snapshots, x_tol_outer=1e-6, inverter_control=True)

Results_v = n.buses_t.v_mag_pu.loc[:, 'My bus 1':'My bus 29']

# q_v controller calculates the amount of reactive power internally
q_max = n.generators_t.p.loc[:, 'My Gen 1':'My Gen 29'].mul(
    np.tan((np.arccos(n.generators.loc['My Gen 1':'My Gen 29']['power_factor'],
                      dtype=np.float64)), dtype=np.float64))
# then it also calculates the reactive power capacity of the inverter
q_cap = n.generators.loc['My Gen 1':'My Gen 29']['s_nom']*np.sin(np.arccos(
    n.generators.loc['My Gen 1':'My Gen 29']['power_factor']))
# here controller chooses the needed q based on grid need and availability of q in s_nom
q_allowable = np.where(q_max <= q_cap, q_max, q_cap)  # maximum available q
# calculation of q compensation by q_v controller in percentage
c_q_set_inj_percentage = n.generators_t.q.loc[:, 'My Gen 1':'My Gen 29']/q_allowable*100


# Q(V) controller droop characteristic
def q_v_controller_droop(v_pu_bus):
    # parameters are same for all the generators, here from Gen 7 are used
    v1 = n.generators.loc['My Gen 7', 'v1']
    v2 = n.generators.loc['My Gen 7', 'v2']
    v3 = n.generators.loc['My Gen 7', 'v3']
    v4 = n.generators.loc['My Gen 7', 'v4']
    damper = n.generators.loc['My Gen 7', 'damper']
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
# plot droop characteristic
plt.plot(Results_droop['v_mag'], Results_droop["p/pmaxdroop"],
         label="Droop characteristic", color='r')
# plot controller behavior
plt.scatter(Results_v, c_q_set_inj_percentage, color="g",
            label="Controller behaviour")
# labeling controller parameters
plt.text(n.generators.loc['My Gen 7', 'v1'], 100, "v1= %s" % (
    n.generators.loc['My Gen 7', 'v1']), rotation=0)
plt.text(n.generators.loc['My Gen 7', 'v2'], 0, "v2= %s" % (
    n.generators.loc['My Gen 7', 'v2']), rotation=90)
plt.text(n.generators.loc['My Gen 7', 'v3'], 0, "v3= %s" % (
    n.generators.loc['My Gen 7', 'v3']), rotation=90)
plt.text(n.generators.loc['My Gen 7', 'v4'], -100, "v4= %s" % (
    n.generators.loc['My Gen 7', 'v4']), rotation=90)
# legendsa and titles
plt.legend(loc="best", bbox_to_anchor=(0.4, 0.4))
plt.title("Q(U) Controller Behavior \n  30 node example 15 snapshots\n"
          "while loop condition = 1e-4\n v1,v2,v3,v4 are controller setpoints")
plt.xlabel('voltage_pu')
plt.ylabel('Reactive power compensation Q(U) %')
plt.show()
end = time.time()
print(f"Runtime of the program is {end - start}")
