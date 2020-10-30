## Copyright 2015-2018 Tom Brown (FIAS), Jonas Hoersch (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Test of transformer modelling against NEPLAN 10.8.2.4(App), with focus to tap changer and off-nominal-tap-ratio
"""

__author__ = "Tobias Dess (elena international)"
__copyright__ = "Copyright 2020 Tobias Dess (elena international) GNU GPL 3"

import pypsa
import pandapower
import numpy as np

def config_base_network():

    network = pypsa.Network()

    network.add("Bus",
                name="Bus0",
                v_nom=20.,
                carrier="AC"
               )

    network.add("Bus",
                name="Bus1",
                v_nom=0.4,
                carrier="AC"
               )

    network.add("Generator", "Gen",
            bus="Bus0",
            p_set=333.,
            control="Slack")

    return network

def config_base_network_pandapower():

    pp_net = pandapower.create_empty_network()

    bus_nr = pandapower.create_bus(pp_net,
                                   vn_kv=20.,    
                                   name="Bus0"
                                   )

    pandapower.create_bus(pp_net,
                vn_kv=0.4,    
                name="Bus1"
               )

    pandapower.create_ext_grid(pp_net, bus=bus_nr) 

    return pp_net

def test_real_transformer_with_off_nominal_turn_ratio_and_switched_position_against_pandapower():

    # Config Pandapower Network
    network = config_base_network()
    network.add("TransformerType",
                name="ideal_transformer_with_tap",
                f_nom=50.,
                s_nom=0.630,
                v_nom_0=22.,
                v_nom_1=0.45,
                vsc=30.,
                vscr=10.,
                pfe=50.,
                i0=10.,
                phase_shift=0,
                tap_side=0,
                tap_neutral=0,
                tap_min=-2,
                tap_max=2,
                tap_step=-2.,
                references="testcase")

    network.add("Transformer",
                name="ideal_transformer_with_tap",
                bus0="Bus0",
                bus1="Bus1",
                model="t",
                s_nom=0.630,
                type="ideal_transformer_with_tap",
                tap_position=2
                )
    network.add("Load", "Load",
                bus="Bus1",
                p_set=0.2,
                q_set=0.2)

    # Config Pandapower Network
    pp_network = config_base_network_pandapower()
    pandapower.create_transformer_from_parameters(pp_network, hv_bus=0, lv_bus=1, sn_mva=0.630, vn_hv_kv=22., vn_lv_kv=0.45,
    vkr_percent=10., vk_percent=30., pfe_kw=50., i0_percent=10., shift_degree=0, tap_side="hv", tap_neutral=0, tap_max=2, tap_min=-2,
    tap_step_percent=-2., tap_pos=2, name="real_transformer_with_tap")
    pandapower.create_load(pp_network, bus=1, p_mw=0.2, q_mvar=0.2, name="Bus1")

    # Run Powerflows
    network.pf()
    pandapower.runpp(pp_network)

    # Compare against Pandapower
    ## compare bus angles
    np.testing.assert_array_almost_equal(network.buses_t.v_ang.loc["now"] * 180 / np.pi, pp_network.res_bus.va_degree, decimal=3)
    ## compare bus voltage magnitudes
    np.testing.assert_array_almost_equal(network.buses_t.v_mag_pu.loc["now"], pp_network.res_bus.vm_pu)
    ## compare bus active power (NB: pandapower uses load signs)
    np.testing.assert_array_almost_equal(network.buses_t.p.loc["now"], -pp_network.res_bus.p_mw)
    ## compare bus active power (NB: pandapower uses load signs)
    np.testing.assert_array_almost_equal(network.buses_t.q.loc["now"], -pp_network.res_bus.q_mvar)

def test_real_transformer_with_off_nominal_turn_ratio_and_switched_position_on_low_voltage_side_against_pandapower():

    # Config PaPSA Network
    network = config_base_network()
    network.add("TransformerType",
                name="ideal_transformer_with_tap",
                f_nom=50.,
                s_nom=0.630,
                v_nom_0=22.,
                v_nom_1=0.45,
                vsc=30.,
                vscr=10.,
                pfe=50.,
                i0=10.,
                phase_shift=0,
                tap_side=1,
                tap_neutral=0,
                tap_min=-2,
                tap_max=2,
                tap_step=-2.,
                references="testcase")

    network.add("Transformer",
                name="ideal_transformer_with_tap",
                bus0="Bus0",
                bus1="Bus1",
                model="t",
                s_nom=0.630,
                type="ideal_transformer_with_tap",
                tap_position=2
                )
    network.add("Load", "Load",
                bus="Bus1",
                p_set=0.2,
                q_set=0.2)

    # Config Pandapower Network
    pp_network = config_base_network_pandapower()
    pandapower.create_transformer_from_parameters(pp_network, hv_bus=0, lv_bus=1, sn_mva=0.630, vn_hv_kv=22., vn_lv_kv=0.45,
    vkr_percent=10., vk_percent=30., pfe_kw=50., i0_percent=10., shift_degree=0, tap_side="lv", tap_neutral=0, tap_max=2, tap_min=-2,
    tap_step_percent=-2., tap_pos=2, name="real_transformer_with_tap")
    pandapower.create_load(pp_network, bus=1, p_mw=0.2, q_mvar=0.2, name="Bus1")

    # Run Powerflows
    network.pf()
    pandapower.runpp(pp_network)

    # Compare against Pandapower
    ## compare bus angles
    np.testing.assert_array_almost_equal(network.buses_t.v_ang.loc["now"] * 180 / np.pi, pp_network.res_bus.va_degree, decimal=3)
    ## compare bus voltage magnitudes
    np.testing.assert_array_almost_equal(network.buses_t.v_mag_pu.loc["now"], pp_network.res_bus.vm_pu)
    ## compare bus active power (NB: pandapower uses load signs)
    np.testing.assert_array_almost_equal(network.buses_t.p.loc["now"], -pp_network.res_bus.p_mw)
    ## compare bus active power (NB: pandapower uses load signs)
    np.testing.assert_array_almost_equal(network.buses_t.q.loc["now"], -pp_network.res_bus.q_mvar)

if __name__ == "__main__":
    test_real_transformer_with_off_nominal_turn_ratio_and_switched_position_against_pandapower()
    test_real_transformer_with_off_nominal_turn_ratio_and_switched_position_on_low_voltage_side_against_pandapower()
