

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

from distutils.version import StrictVersion

import networkx as nx
assert StrictVersion(nx.__version__) >= '1.10', "NetworkX needs to be at least version 1.10"

import numpy as np
import pandas as pd
from collections import OrderedDict
from itertools import chain

from .descriptors import Float, String, OrderedDictDesc, Series, GraphDesc



class Basic(object):
    """Common to every object."""
    name = String()

    def __repr__(self):
        return "%s %s" % (self.__class__.__name__, self.name)


class Common(Basic):
    """Common to all objects inside Network object."""
    network = None

    def __init__(self, network):
        self.network = network
        self.sub_network = None


class Source(Common):
    """Energy source, such as wind, PV or coal."""

    #emissions in tonnes CO2-equivalent per MWh primary energy
    co2_emissions = Float()



class Bus(Common):
    """Electrically fundamental node where x-port objects attach."""

    list_name = "buses"

    v_nom = Float()
    control = String(default="PQ",restricted=["PQ","PV","Slack"])

    #2-d location data (e.g. longitude and latitude)
    x = Float()
    y = Float()

    loads = OrderedDictDesc()
    generators = OrderedDictDesc()
    storage_units = OrderedDictDesc()
    p = Series()
    q = Series()
    v_mag = Series(default=1.)
    v_ang = Series()
    v_set = Series(default=1.)

class SubStation(Common):
    """Placeholder for a group of buses."""

class Region(Common):
    """A group of buses such as a country or county."""


class OnePort(Common):
    """Object which attaches to a single bus (e.g. load or generator)."""

    bus = None
    sign = Float(1)
    p = Series()
    q = Series()


class Generator(OnePort):

    list_name = "generators"

    dispatch = String(default="flexible",restricted=["variable","flexible","inflexible"])

    #i.e. coal, CCGT, onshore wind, PV, CSP,....
    source = String()

    #rated power
    p_nom = Float()

    #switch to allow capacity to be extended
    p_nom_extendable = False

    #technical potential
    p_nom_max = None

    p_nom_min = Float()

    capital_cost = Float()
    marginal_cost = Float()


    #power limits for variable generators, which can change e.g. due
    #to weather conditions; per unit to ease multiplication with
    #p_nom, which may be optimised
    p_max_pu = Series()
    p_min_pu = Series()


    #non-variable power limits for de-rating and minimum limits for
    #flexible generators
    p_max_pu_fixed = Float(1)
    p_min_pu_fixed = Float()


    #operator's intended dispatch
    p_set = Series()
    q_set = Series()


    #ratio between electrical energy and primary energy
    efficiency = Float(1)



class StorageUnit(Generator):

    list_name = "storage_units"

    #units are MWh
    state_of_charge_start = Float()
    state_of_charge = Series(default=np.nan)

    #hours from state of state_of_charge_start to first snapshot optimized
    hours_from_state_of_charge_start = Float(1)

    #maximum capacity in terms of hours at full output capacity p_nom
    max_hours = Float(1)

    #in MW
    inflow = Series()

    efficiency_store = Float(1)

    efficiency_dispatch = Float(1)

    #per hour per unit loss in state of charge
    standing_loss = Float()



class Load(OnePort):
    """PQ load."""

    list_name = "loads"

    #set sign convention for powers opposite to generator
    sign = Float(-1)

    p_set = Series()
    q_set = Series()


class Branch(Common):
    """Object which attaches to two buses (e.g. line or 2-winding transformer)."""

    list_name = "branches"

    bus0 = None
    bus1 = None

    capital_cost = Float()

    s_nom = Float()

    s_nom_extendable = False

    s_nom_max = None
    s_nom_min = Float()

    p0 = Series()
    p1 = Series()

    q0 = Series()
    q1 = Series()


class Line(Branch):
    """Lines include distribution and transmission lines, overhead lines and cables."""

    list_name = "lines"

    #series impedance z = r + jx in Ohm
    r = Float()
    x = Float()

    #shunt reactance y = g + jb in 1/Ohm
    g = Float()
    b = Float()

    length = Float(default=1.0)
    terrain_factor = Float(default=1.0)


class Transformer(Branch):
    """2-winding transformer."""

    list_name = "transformers"

    #per unit with reference to s_nom
    x = Float()


class Converter(Branch):
    """Bus 0 is AC, bus 1 is DC."""

    list_name = "converters"

    p_set = Series()

class TransportLink(Branch):
    """Controllable link between two buses - can be used for a transport
    power flow model OR as a simplified version of point-to-point DC
    connection.
    """

    list_name = "transport_links"

    p_min = Float()
    p_max = Float()
    p_set = Series()


class ThreePort(Common):
    """Placeholder for 3-winding transformers."""

class ThreeTransformer(ThreePort):
    pass

class LineType(Common):
    """Placeholder for future functionality to automatically generate line
    parameters from standard parameters (e.g. r/km)."""


class Network(Basic):

    """Network of buses and branches."""


    #the current scenario/time
    now = "now"

    #a list/index of scenarios/times
    index = [now]

    sub_networks = OrderedDictDesc()

    buses = OrderedDictDesc()

    loads = OrderedDictDesc()
    generators = OrderedDictDesc()
    storage_units = OrderedDictDesc()

    branches = OrderedDictDesc()
    converters = OrderedDictDesc()
    transport_links = OrderedDictDesc()
    lines = OrderedDictDesc()
    transformers = OrderedDictDesc()

    graph = GraphDesc()

    def __init__(self, import_file_name=None, **kwargs):
        if import_file_name is not None:
            self.import_from_csv(import_file_name)
            self.determine_network_topology()

        for key, value in kwargs.iteritems():
            setattr(self, key, value)


    def import_from_csv(self,file_name):
        """E.g. read in CSV or pickle of data."""

    def import_from_pypower(self,file_name):
        """Import network from PyPower .py."""


    def add(self,class_name,name,**kwargs):
        """Add single objects to the network."""

        try:
            cls = globals()[class_name]
        except KeyError:
            print(class_name,"not found")
            return None

        obj = cls(self)

        obj.name = name

        for key,value in kwargs.iteritems():
            setattr(obj,key,value)

        #add to list in network object
        getattr(self,cls.list_name)[obj.name] = obj

        #build branches list
        if isinstance(obj, Branch):
            self.branches[obj.name] = obj

        #add oneports to bus lists
        if "bus" in kwargs:
            getattr(obj.bus,cls.list_name)[obj.name] = obj

        return obj


    def add_from(self,object_list):
        """Add objects from a list."""

    def remove(self,obj):
        """Remove object from network."""

    def build_graph(self):
        """Build networkx graph."""

        self.graph.add_edges_from((branch.bus0, branch.bus1, {"obj" : branch}) for branch in self.branches.itervalues())


    def determine_network_topology(self):
        """Build sub_networks from topology."""

        #remove converters and transport links that could be connected networks
        # of different types or non-synchronous areas

        graph = self.graph.__class__(self.graph)

        graph.remove_edges_from((branch.bus0,branch.bus1)
                                for branch in chain(self.converters.itervalues(),
                                                    self.transport_links.itervalues()))


        #now build connected graphs of same type AC/DC
        sub_graphs = nx.connected_component_subgraphs(graph, copy=False)

        for i, sub_graph in enumerate(sub_graphs):
            #name using i for now
            sub_network = self.add("SubNetwork", i, graph=sub_graph)

            sub_network.buses.update((n.name, n) for n in sub_graph.nodes())
            sub_network.branches.update((branch.name, branch) for (u, v, branch) in sub_graph.edges_iter(data="obj"))

            for bus in sub_network.buses.itervalues():
                bus.sub_network = sub_network

            for branch in sub_network.branches.itervalues():
                branch.sub_network = sub_network



class SubNetwork(Common):
    """Connected network of same current type (AC or DC)."""

    list_name = "sub_networks"

    current_type = String(default="AC",restricted=["AC","DC"])

    frequency = Float(default=50)

    num_phases = Float(default=3)

    base_power = Float(default=1)

    buses = OrderedDictDesc()
    branches = OrderedDictDesc()

    graph = GraphDesc()

    
