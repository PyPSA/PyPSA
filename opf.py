

## Copyright 2015 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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

"""Python for Power Systems Analysis (PyPSA)

Grid calculation library.
"""


# make the code as Python 3 compatible as possible
from __future__ import print_function, division


__version__ = "0.1"
__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"


def network_lopf(network,subindex=None):
    """Optimal power flow for snapshots in subindex."""

    #calculate B,H or PTDF for each subnetwork.


    #network.model = pyomo.Model()


    ### build optimisation variables: ###

    #for each gen: gen.p_nom and for all t: gen.p[t]

    #for each branch: branch.s_nom

    #for each converter/transport link: branch.p_set[t]

    #for each AC bus: bus.v_ang


    ### build constraints: ###

    ## non-time-dependent contraints: ##

    #for each generator: gen.p_nom <= gen.technical_potential

    #for each branch: branch.s_nom <= branch.technical_potential

    #co2 emissions <= global cap


    ## time-dependent constraints ##

    #for each generator: gen.p[t] <= gen.p_max = gen.p_nom*per_unit_availability[t]

    #for slack buses: bus.v_ang[t] = 0

    #for each bus: sum(bus.gen.p[t]) - sum(bus.load.p_set[t]) - sum(transport_lines/converters) = sub_network.B.dot(bus.v_ang)

    #for each line: line.p1[t] = sub_network.H.dot(bus.v_ang)

    #for each line: line.p1[t] <= line.s_nom


    ### build objective function ###

    #model.objective = sum (line.s_nom*line.capital_cost) + sum(gen.p_nom*gen.capital_cost) + sum_t (gen.p[t]*gen.marginal_cost) w_t



    #return model
