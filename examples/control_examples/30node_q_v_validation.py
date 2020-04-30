from __future__ import print_function, division
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np

'''
This example is for validation of reactive power compensation control strategy
(Q(v)) by plotting the controller output and its droop characteristic in the
same figure.
'''
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

# building empty dataframes
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
        (i+1) % n_buses), control="PQ", p_set=L, s_nom=0.00025, v1=0.89, v2=0.94,
        v3=0.96, v4=1.02)
    network.add("Load", "My load {}".format(i), bus="My bus {}".format(
        (i+1) % n_buses), s_nom=0.00025, p_set=L1)
    network.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)


def power_flow():
    network.lpf(network.snapshots)
    network.pf(use_seed=True, snapshots=network.snapshots, x_tol_outer=1e-4)

# setting control strategy type
network.generators.type_of_control_strategy = 'q_v'
power_flow()
# saving the necessary data for plotting controller behavior
for i in range(n_buses):
    Results_q.loc[:, 'p_set'] = network.generators_t.p_set.loc[
        :, 'My Gen 26'].values
    Results_v.loc[:, "V_pu_with_control {}".format(i)
                  ] = network.buses_t.v_mag_pu.loc[:, "My bus {}".format(i)
                                                   ].values
    Results_q.loc[:, "q_set_controller_output {}".format(i)] = ((
        network.generators_t.q_set.loc[:, "My Gen {}".format(i)].values)/(
            np.sqrt((network.generators.loc["My Gen {}".format(i), 's_nom'])**2-(
                Results_q["p_set"])**2)))*100


# controller droop characteristic method simplified
def q_v(v_pu_bus):
    '''
    Parameters
    ----------
    v_pu_bus : fload
        voltage of the bus where controller is connected
    v1, v2 : float
        v_pu setpoint for lower curve of the controller.
    v3, v4 : float
        v_pu setpoint for upper curve of the controller.
    damper : float / integer
        Adds damping characteristic to the controller output.

    Returns
    -------
    pf : float
        power factor (pf).
    '''
    # parameters are same for all the generators, thus Gen 7 is chosen here
    v1 = network.generators.loc['My Gen 7', 'v1']
    v2 = network.generators.loc['My Gen 7', 'v2']
    v3 = network.generators.loc['My Gen 7', 'v3']
    v4 = network.generators.loc['My Gen 7', 'v4']
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


# power factor optional values to calculate droop characteristic (%)
vmag = [0.85, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
        1.01, 1.02, 1.05]
# calculation of droop characteristic (output)
for index, row in Results_q.iterrows():
    Results_q.loc[index, "p/pmaxdroop"] = q_v(vmag[index])


''' Plotting '''
# droop characteristic input and output variables
vdroop = vmag
qdroop = -Results_q["p/pmaxdroop"]
# controller input and output variables
v_control = Results_v.loc[:, "V_pu_with_control 1":"V_pu_with_control 29"]
qcontrol = Results_q.loc[:, "q_set_controller_output 0":
                         "q_set_controller_output 28"]
# plot droop characteristic
plt.plot(vmag, qdroop, label="Q(U) droop characteristic", color='r')
# plot controller behavior
plt.scatter(v_control, qcontrol, color="g",
            label="Q(U) Controller characteristic")

plt.legend(loc="best", bbox_to_anchor=(0.4, 0.4))
plt.title("Q(U) Controller Behavior \n  30 node example, \n snapshots = 15\n"
          "while loop condition = 1e-4")
plt.xlabel('voltage_pu')
plt.ylabel('Reactive power compensation Q(U) %')
plt.show()
