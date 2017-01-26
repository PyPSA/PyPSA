## Example of transformer with non-trivial phase shift and tap ratio
#
#This example is a copy of pandapower's minimal example.

import pypsa
import numpy as np

network = pypsa.Network()

network.add("Bus","MV bus",v_nom=20,v_mag_pu_set=1.02)
network.add("Bus","LV1 bus",v_nom=.4)
network.add("Bus","LV2 bus",v_nom=.4)

network.add("Transformer","MV-LV trafo",type="0.4 MVA 20/0.4 kV",bus0="MV bus",bus1="LV1 bus")

network.add("Line","LV cable",type="NAYY 4x50 SE",bus0="LV1 bus",bus1="LV2 bus",length=0.1)

network.add("Generator","External Grid",bus="MV bus",control="Slack")

network.add("Load","LV load",bus="LV2 bus",p_set=0.1,q_set=0.05)

def run_pf():
    network.lpf()
    network.pf(use_seed=True)
    print("Voltage angles:")
    print(network.buses_t.v_ang*180./np.pi)
    print("Voltage magnitudes:")
    print(network.buses_t.v_mag_pu)

run_pf()

network.transformers.tap_position = 2
run_pf()

network.transformers.tap_position = -2
run_pf()

### Now play with tap changer on LV side

new_trafo_lv_tap = network.transformer_types.loc[["0.4 MVA 20/0.4 kV"]]
new_trafo_lv_tap.index = ["New trafo"]
new_trafo_lv_tap.tap_side = 1
new_trafo_lv_tap

network.transformer_types = network.transformer_types.append(new_trafo_lv_tap)
network.transformers.type = "New trafo"

network.transformers.tap_position = 2
run_pf()

network.transformers.tap_position = -2
run_pf()

### Now make sure that the phase shift is also there in the LOPF

network.generators.p_nom = 1.
network.lines.s_nom = 1.
network.lopf()

print("Voltage angles:")
print(network.buses_t.v_ang*180./np.pi)
print("Voltage magnitudes:")
print(network.buses_t.v_mag_pu)

