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

def test_ideal_transformer_with_tap():
    """
    Tested against NEPLAN 10.8.2.4(App)
    """
    network = config_base_network()

    network.add("TransformerType",
                 name="ideal_transformer_with_tap",
                 f_nom=50.,
                 s_nom=0.630,
                 v_nom_0=20.,
                 v_nom_1=0.40,
                 vsc=0.01,
                 vscr=0.,
                 pfe=0.,
                 i0=0.,
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
                model="pi",
                s_nom=0.630,
                type="ideal_transformer_with_tap",
                tap_position=2
               )

    network.pf()

    v_mag_pu_expected= np.array([[1., 1.04167]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_mag_pu, v_mag_pu_expected,  decimal=4)

def test_real_transformer_with_tap():
    """
    Tested against NEPLAN 10.8.2.4(App)
    """
    network = config_base_network()

    network.add("TransformerType",
                name="ideal_transformer_with_tap",
                f_nom=50.,
                s_nom=0.630,
                v_nom_0=20.,
                v_nom_1=0.40,
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
                model="pi",
                s_nom=0.630,
                type="ideal_transformer_with_tap",
                tap_position=2
                )
    network.add("Load", "Load",
                bus="Bus1",
                p_set=0.2,
                q_set=0.2)

    network.pf()

    v_mag_pu_expected = np.array([[1., 0.89163]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_mag_pu, v_mag_pu_expected, decimal=4)
    v_ang_expected = np.array([[0., -3.99]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_ang * 180 / np.pi, v_ang_expected, decimal=2)
    p0_expected = np.array([[0.26590]], dtype=float)
    np.testing.assert_array_almost_equal(network.transformers_t.p0, p0_expected, decimal=5)
    q0_expected = np.array([[0.28949]], dtype=float)
    np.testing.assert_array_almost_equal(network.transformers_t.q0, q0_expected, decimal=5)

def test_ideal_transformer_with_off_nominal_turn_ratio():
    """
    Tested against NEPLAN 10.8.2.4(App)
    """
    network = config_base_network()

    network.add("TransformerType",
                 name="ideal_transformer_with_tap",
                 f_nom=50.,
                 s_nom=0.630,
                 v_nom_0=22.,
                 v_nom_1=0.45,
                 vsc=0.01,
                 vscr=0.,
                 pfe=0.,
                 i0=0.,
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
                model="pi",
                s_nom=0.630,
                type="ideal_transformer_with_tap",
                tap_position=0
               )

    network.pf()

    v_mag_pu_expected= np.array([[1., 1.02273]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_mag_pu, v_mag_pu_expected,  decimal=4)

def test_ideal_transformer_with_off_nominal_turn_ratio_and_tap_position():
    """
    Tested against NEPLAN 10.8.2.4(App)
    """
    network = config_base_network()

    network.add("TransformerType",
                 name="ideal_transformer_with_tap",
                 f_nom=50.,
                 s_nom=0.630,
                 v_nom_0=22.,
                 v_nom_1=0.45,
                 vsc=0.01,
                 vscr=0.,
                 pfe=0.,
                 i0=0.,
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
                model="pi",
                s_nom=0.630,
                type="ideal_transformer_with_tap",
                tap_position=2
               )

    network.pf()

    v_mag_pu_expected= np.array([[1., 1.0653]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_mag_pu, v_mag_pu_expected,  decimal=4)

def test_real_transformer_with_off_nominal_turn_ratio():
    """
    Tested against NEPLAN 10.8.2.4(App)
    """
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
                model="pi",
                s_nom=0.630,
                type="ideal_transformer_with_tap",
                tap_position=0
                )
    network.add("Load", "Load",
                bus="Bus1",
                p_set=0.3,
                q_set=0.3)

    network.pf()

    v_mag_pu_expected = np.array([[1., 0.6367]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_mag_pu, v_mag_pu_expected, decimal=4)
    v_ang_expected = np.array([[0., -10.04]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_ang * 180 / np.pi, v_ang_expected, decimal=2)
    p0_expected = np.array([[0.4221]], dtype=float)
    np.testing.assert_array_almost_equal(network.transformers_t.p0, p0_expected, decimal=3)
    q0_expected = np.array([[0.5862]], dtype=float)
    np.testing.assert_array_almost_equal(network.transformers_t.q0, q0_expected, decimal=3)

def test_real_transformer_with_off_nominal_turn_ratio_and_switched_position():
    """
    Tested against NEPLAN 10.8.2.4(App)
    """
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
                model="pi",
                s_nom=0.630,
                type="ideal_transformer_with_tap",
                tap_position=2
                )
    network.add("Load", "Load",
                bus="Bus1",
                p_set=0.2,
                q_set=0.2)

    network.pf()

    v_mag_pu_expected = np.array([[1., 0.87455]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_mag_pu, v_mag_pu_expected, decimal=4)
    v_ang_expected = np.array([[0., -4.91]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_ang * 180 / np.pi, v_ang_expected, decimal=2)
    p0_expected = np.array([[0.26144]], dtype=float)
    np.testing.assert_array_almost_equal(network.transformers_t.p0, p0_expected, decimal=3)
    q0_expected = np.array([[0.29640]], dtype=float)
    np.testing.assert_array_almost_equal(network.transformers_t.q0, q0_expected, decimal=3)

def test_real_transformer_with_off_nominal_turn_ratio_and_switched_position_controlled_on_secondary_side():
    """
    Tested against NEPLAN 10.8.2.4(App)
    """
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
                model="pi",
                s_nom=0.630,
                type="ideal_transformer_with_tap",
                tap_position=2
                )
    network.add("Load", "Load",
                bus="Bus1",
                p_set=0.2,
                q_set=0.2)

    network.pf()

    v_mag_pu_expected = np.array([[1., 0.78756]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_mag_pu, v_mag_pu_expected, decimal=4)
    v_ang_expected = np.array([[0., -5.40]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_ang * 180 / np.pi, v_ang_expected, decimal=2)
    p0_expected = np.array([[0.26074]], dtype=float)
    np.testing.assert_array_almost_equal(network.transformers_t.p0, p0_expected, decimal=3)
    q0_expected = np.array([[0.30174]], dtype=float)
    np.testing.assert_array_almost_equal(network.transformers_t.q0, q0_expected, decimal=3)

def test_real_transformer_with_off_nominal_turn_ratio_and_switched_position_controlled_and_higher_secondary_voltage():
    """
    Tested against NEPLAN 10.8.2.4(App)
    """
    network = pypsa.Network()

    network.add("Bus",
                name="Bus0",
                v_nom=20.,
                carrier="AC"
               )

    network.add("Bus",
                name="Bus1",
                v_nom=380.,
                carrier="AC"
               )

    network.add("Generator", "Gen",
            bus="Bus0",
            p_set=333.,
            control="Slack")

    network.add("TransformerType",
                name="ideal_transformer_with_tap",
                f_nom=50.,
                s_nom=0.630,
                v_nom_0=22.,
                v_nom_1=440.,
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
                model="pi",
                s_nom=0.630,
                type="ideal_transformer_with_tap",
                tap_position=2
                )
    network.add("Load", "Load",
                bus="Bus1",
                p_set=0.2,
                q_set=0.2)

    network.pf()

    v_mag_pu_expected = np.array([[1., 0.90013]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_mag_pu, v_mag_pu_expected, decimal=4)
    v_ang_expected = np.array([[0., -4.91]], dtype=float)
    np.testing.assert_array_almost_equal(network.buses_t.v_ang * 180 / np.pi, v_ang_expected, decimal=2)
    p0_expected = np.array([[0.26144]], dtype=float)
    np.testing.assert_array_almost_equal(network.transformers_t.p0, p0_expected, decimal=3)
    q0_expected = np.array([[0.29640]], dtype=float)
    np.testing.assert_array_almost_equal(network.transformers_t.q0, q0_expected, decimal=3)

if __name__ == "__main__":
    test_ideal_transformer_with_tap()
    test_real_transformer_with_tap()
    test_ideal_transformer_with_off_nominal_turn_ratio()
    test_ideal_transformer_with_off_nominal_turn_ratio_and_tap_position()
    test_real_transformer_with_off_nominal_turn_ratio()
    test_real_transformer_with_off_nominal_turn_ratio_and_switched_position()
    test_real_transformer_with_off_nominal_turn_ratio_and_switched_position_controlled_on_secondary_side()
    test_real_transformer_with_off_nominal_turn_ratio_and_switched_position_controlled_and_higher_secondary_voltage()
