"""
This example shows the effect of application of each controller and mix of them
on Load component. The effect of each controller and mix of them are
compared in a graph against no control case.
"""
# importing important modules
from __future__ import print_function, division
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

Load = [0, 0.03, 0.05, 0.07]

# defining network
n = pypsa.Network()
n_buses = 4
for i in range(n_buses):
    n.add("Bus", "My bus {}".format(i), v_nom=.4, v_mag_pu_set=1.02)

    n.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
        (i) % n_buses), control="PQ")

    n.add("Load", "My load {}".format(i), bus="My bus {}".format(
        (i) % n_buses), p_set=Load[i], v1=0.89, v2=0.94, v3=0.96, v4=1.02,
        s_nom=0.075, set_p1=50, set_p2=100, power_factor=0.95,
        power_factor_min=0.95, damper=0.2, p_ref=0.035)

    n.add("Store", "My test_load {}".format(i), bus="My bus {}".format(
        (i) % n_buses), p_set=-0.07)
for i in range(n_buses-1):
    n.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
          bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)


def run_pf(inverter_control=False):
    n.lpf()
    n.pf(use_seed=True, inverter_control=inverter_control)

# run pf without controller and save the results
run_pf()
Loads_Result = pd.DataFrame(columns=[])
Bus_v_mag_pu = pd.DataFrame(columns=[])
Loads_Result['no_control'] = n.loads_t.p.T['now']
Bus_v_mag_pu['no_control'] = n.buses_t.v_mag_pu.T['now']

# now apply reactive power as a function of voltage Q(U) or q_v controller,
# parameters (v1,v2,v3,v4,s_nom,damper) are already set in (n.add('Load',...))
n.loads.control_strategy = 'q_v'
# run pf and save the results
run_pf(inverter_control=True)
Bus_v_mag_pu['q_v_control'] = n.buses_t.v_mag_pu.T['now']
Loads_Result['q_v_control'] = n.loads_t.p.T['now']

# now apply fixed power factor controller (fixed_cosphi), parameters
# (power_factor, damper) are already set in (n.add(Load...))
n.loads.control_strategy = 'fixed_cosphi'
# run pf and save the results
run_pf(inverter_control=True)
Bus_v_mag_pu['fixed_pf_control'] = n.buses_t.v_mag_pu.T['now']
Loads_Result['fixed_pf_control'] = n.loads_t.p.T['now']

# now apply power factor as a function of real power (cosphi_p), parameters
# (set_p1,set_p2,s_nom,damper,power_factor_min) are already set in (n.add('Load'...))
n.loads.control_strategy = 'cosphi_p'
# run pf and save the results
run_pf(inverter_control=True)
Bus_v_mag_pu['cosphi_p_control'] = n.buses_t.v_mag_pu.T['now']
Loads_Result['cosphi_p_control'] = n.loads_t.p.T['now']

# now apply mix of controllers
# q_v controller
n.loads.loc['My load 1', 'control_strategy'] = 'q_v'
n.loads.loc['My load 1', 'v1'] = 0.89
n.loads.loc['My load 1', 'v2'] = 0.94
n.loads.loc['My load 1', 'v3'] = 0.96
n.loads.loc['My load 1', 'v4'] = 1.01
n.loads.loc['My load 1', 's_nom'] = 0.075
# fixed_cosphi controller
n.loads.loc['My load 2', 'control_strategy'] = 'fixed_cosphi'
n.loads.loc['My load 2', 'power_factor'] = 0.95
# cosphi_p controller
n.loads.loc['My load 3', 'control_strategy'] = 'cosphi_p'
n.loads.loc['My load 3', 'set_p1'] = 50
n.loads.loc['My load 3', 'set_p2'] = 100
n.loads.loc['My load 3', 'power_factor_min'] = 0.9
n.loads.loc['My load 3', 's_nom'] = 0.075
run_pf(inverter_control=True)

Bus_v_mag_pu['mix_controllers'] = n.buses_t.v_mag_pu.T['now']
Loads_Result['mix_controllers'] = n.loads_t.p.T['now']

plt.plot(Loads_Result['no_control'], Bus_v_mag_pu['no_control'], linestyle='--',
         label="no_control")
plt.plot(Loads_Result['cosphi_p_control'], Bus_v_mag_pu['cosphi_p_control'], label="cosphi_p")
plt.plot(Loads_Result['q_v_control'], Bus_v_mag_pu['q_v_control'], label="q_v")
plt.plot(Loads_Result['fixed_pf_control'], Bus_v_mag_pu['fixed_pf_control'], label="fixed_cosphi")
plt.plot(Loads_Result['mix_controllers'], Bus_v_mag_pu['mix_controllers'], label="mix")
plt.axhline(y=1.02, color='y', linestyle='--', label='max_v_mag_pu_limit',
            linewidth=3, alpha=0.5)
plt.axhline(y=0.9, color='r', linestyle='--', label='min_v_mag_pu_limit',
            linewidth=3, alpha=0.5)

plt.xlabel('Active_power_injection (MW)')
plt.ylabel('V_mag_pu (per unit)')
plt.title("Application of controllers and mix of them on Load component")
plt.legend()
plt.show()
