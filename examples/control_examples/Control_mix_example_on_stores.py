"""
This example shows the effect of application of each controller and mix of them
on Store component while charging. The effect of each controller and mix of them are
compared in a graph against no control case. The discharging effect can be
also seen by changing the sign of p_set to positive..
"""
# importing important modules
from __future__ import print_function, division
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

power_inj = [0, 0.03, 0.05, 0.07]
Load = [0.15, 0.18, 0.2, 0.25]

# defining network
n = pypsa.Network()

n_buses = 4

for i in range(n_buses):
    n.add("Bus", "My bus {}".format(i), v_nom=.4, v_mag_pu_set=1.02)

for i in range(n_buses):
    n.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
        (i) % n_buses), control="PQ")

    n.add("Store", "My store {}".format(i), bus="My bus {}".format(
        (i) % n_buses), p_set=-power_inj[i], v1=0.89, v2=0.94, v3=0.96, v4=1.02,
        s_nom=0.065, set_p1=50, set_p2=100, power_factor=0.95,
        power_factor_min=0.9, damper=0.8)

    n.add("Load", "My load {}".format(i), bus="My bus {}".format(
        (i) % n_buses), p_set=0)
for i in range(n_buses-1):
    n.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
          bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)


def run_pf():
    n.lpf()
    n.pf(use_seed=True)

# run pf without controller and save the results
run_pf()
Store_Result = pd.DataFrame(index=[n.buses.index], columns=[])
Store_Result['power_inj'] = power_inj
Store_Result['no_control'] = n.buses_t.v_mag_pu.values.T
# now apply reactive power as a function of voltage Q(U) or q_v controller,
# parameters (v1,v2,v3,v4,s_nom,damper) are already set in (n.add('Store', v1, v2 ...))
n.stores.type_of_control_strategy = 'q_v'
run_pf()
Store_Result['q_v_control'] = n.buses_t.v_mag_pu.values.T

# now apply fixed power factor controller (fixed_cosphi), parameters
# (power_factor, damper) are already set in (n.add(Generator...))
n.stores.q_set = 0  # to clean up q_v q_set
n.stores.type_of_control_strategy = 'fixed_cosphi'
run_pf()
Store_Result['fixed_pf_control'] = n.buses_t.v_mag_pu.values.T

# now apply power factor as a function of real power (cosphi_p), parameters
# (set_p1,set_p2,s_nom,damper,power_factor_min) are already set in (n.add('Store'...))
n.stores.q_set = 0  # to clean fixed_cosphi q_set
n.stores.type_of_control_strategy = 'cosphi_p'
run_pf()
Store_Result['cosphi_p_control'] = n.buses_t.v_mag_pu.values.T

# now apply mix of controllers
n.stores.q_set = 0
# q_v controller
n.stores.loc['My store 1', 'type_of_control_strategy'] = 'q_v'
n.stores.loc['My store 1', 'v1'] = 0.89
n.stores.loc['My store 1', 'v2'] = 0.94
n.stores.loc['My store 1', 'v3'] = 0.96
n.stores.loc['My store 1', 'v4'] = 1.01
n.stores.loc['My store 1', 's_nom'] = 0.065
# fixed_cosphi controller
n.stores.loc['My store 2', 'type_of_control_strategy'] = 'fixed_cosphi'
n.stores.loc['My store 2', 'power_factor'] = 0.95
# cosphi_p controller
n.stores.loc['My store 3', 'type_of_control_strategy'] = 'cosphi_p'
n.stores.loc['My store 3', 'set_p1'] = 50
n.stores.loc['My store 3', 'set_p2'] = 100
n.stores.loc['My store 3', 'power_factor_min'] = 0.9
n.stores.loc['My store 3', 's_nom'] = 0.065
run_pf()
Store_Result['mix_controllers'] = n.buses_t.v_mag_pu.values.T

plt.plot(Store_Result['power_inj'], Store_Result['no_control'], linestyle='--',
         label="no_control")
plt.plot(Store_Result['power_inj'], Store_Result['cosphi_p_control'],
         label="cosphi_p")
plt.plot(Store_Result['power_inj'], Store_Result['q_v_control'], label="q_v")
plt.plot(Store_Result['power_inj'], Store_Result['fixed_pf_control'],
         label="fixed_cosphi")
plt.plot(Store_Result['power_inj'], Store_Result['mix_controllers'], label="mix")
plt.axhline(y=1.02, color='y', linestyle='--', label='max_v_mag_pu_limit',
            linewidth=3, alpha=0.5)
plt.axhline(y=0.9, color='r', linestyle='--', label='min_v_mag_pu_limit',
            linewidth=3, alpha=0.5)
plt.xlabel('Active_power_injection (MW)')
plt.ylabel('V_mag_pu (per unit)')
plt.title("Application of controllers and mix of them on Store component")
plt.legend()
plt.show()
