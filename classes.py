

## Copyright 2015 Tom Brown (FIAS), Jonas Hoersch (FIAS),...

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
#from __future__ import print_function, division




__version__ = "0.1"
__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"




import networkx as nx

import numpy as np

import pf


#Some descriptors to control variables - idea is to do type checking
#and in future facilitate interface with Database / GUI

class Float(object):
    """A descriptor to manage floats."""

    def __init__(self,default=0.0):
        self.default = default
        #should be changed to a weakly-referenced dictionary
        self.values = dict()

    def __get__(self,obj,cls):
        return self.values.get(obj,self.default)

    def __set__(self,obj,val):
        try:
            self.values[obj] = float(val)
        except:
            print "could not convert",val,"to a float"
            self.val = self.default
            return

class String(object):
    """A descriptor to manage strings."""

    def __init__(self,default="",restricted=None):
        self.default = default
        self.restricted = restricted
        self.values = dict()


    def __get__(self,obj,cls):
        return self.values.get(obj,self.default)

    def __set__(self,obj,val):
        try:
            self.values[obj] = str(val)
        except:
            print "could not convert",val,"to a string"
            return
        if self.restricted is not None:
            if self.values[obj] not in self.restricted:
                print val,"not in list of acceptable entries:",self.restricted



class Basic(object):
    """Common to every object."""
    name = String()

    def __repr__(self):
        return "%s %s" % (self.__class__.__name__, self.name)


class Common(Basic):
    """Common to all objects inside Network object."""

    network = None


class Bus(Common):
    """Electrically fundamental node where x-port objects attach."""

    p = Float()
    q = Float()
    v_nom = Float()
    v_mag = Float()
    v_deg = Float()
    v_set = Float()

    control = String(default="PQ",restricted=["PQ","PV","Slack"])

    #2-d location data (e.g. longitude and latitude)
    x = Float()
    y = Float()

    loads = None

    generators = None

    def __init__(self):
        self.loads = []
        self.generators = []


class SubStation(Common):
    """Placeholder for a group of buses."""

class Region(Common):
    """A group of buses such as a country or county."""


class OnePort(Common):
    """Object which attaches to a single bus (e.g. load or generator)."""
    
    bus = None

    p = Float()
    q = Float()


class Generator(OnePort):

    dispatch = String(default="flexible",restricted=["variable","flexible","inflexible"])

    #i.e. coal, CCGT, onshore wind, PV, CSP,....
    source = String()

    #rated power
    p_nom = Float()
    
    #can change e.g. due to weather conditions
    p_max = Float()

    p_min = Float()

    #operator's intended dispatch
    p_set = Float()
    q_set = Float()

    technical_potential = Float()

    capital_cost = Float()

    marginal_cost = Float()

    def __init__(self):
        pass

class Load(OnePort):
    p_set = Float()
    q_set = Float()


class Branch(Common):
    """Object which attaches to two buses (e.g. line or 2-winding transformer)."""

    bus0 = None
    bus1 = None

    p0 = Float()
    p1 = Float()
    
    q0 = Float()
    q1 = Float()


    capital_cost = Float()


class Line(Branch):
    """Lines include distribution and transmission lines, overhead lines and cables."""

    #series impedance z = r + jx in Ohm
    r = Float()
    x = Float()

    #shunt reactance y = g + jb in 1/Ohm
    g = Float()
    b = Float()

    length = Float(default=1.0)
    terrain_factor = Float(default=1.0)

    s_nom = Float()

class Transformer2W(Branch):
    """2-winding transformer."""

    x_pu = Float()

    s_nom = Float()


class Converter(Branch):
    """Bus 0 is AC, bus 1 is DC."""
    p_set = Float()

    p_nom = Float()

class TransportLink(Branch):
    """Controllable link between two buses - can be used for a transport
    power flow model OR as a simplified version of point-to-point DC
    connection.
    """
    p_set = Float()


class ThreePort(Common):
    """Placeholder for 3-winding transformers."""

class Transformer3W(ThreePort):
    pass

class LineType(Common):
    """Placeholder for future functionality to automatically generate line
    parameters from standard parameters (e.g. r/km)."""



class Network(Basic):

    """Network of buses and branches."""

    #a list of scenarios/times
    index = None

    #the current scenario/time - if None, then default values obj.attr
    #are used; if not None, the values are picked from
    #obj.attr_series[i])
    i = None

    #lists for all the different object types - perhaps not so elegant
    buses = None

    branches = None

    graph = None

    converters = None

    sub_networks = None


    def __init__(self,network_data=None):
        self.buses = []
        self.converters = []
        self.branches = []
        #etc.

        self.loads = []
        self.generators = []


        if network_data is not None:
            self.add_network_data(network_data)
            self.determine_network_topology()


    def add_network_data(self,network_data):
        """E.g. read in CSV or pickle of data."""


    def add(self,class_name,name):
        """Add single objects to the network."""

        try:
            cls = globals()[class_name]
        except:
            print class_name,"not found"
            return None
        
        obj = cls()

        obj.name = name

        obj.network = self

        #add obj to some list or something?

        return obj


    def add_from(self,object_list):
        """Add objects from a list."""




    def build_graph(self):
        """Build networkx graph."""

        #or use Jonas' OrderedGraph here?
        self.graph = nx.Graph()
    
        self.graph.add_edges_from([(branch.bus0,branch.bus1,{"obj" : branch}) for branch in self.branches])


    def determine_network_topology(self):
        """Build sub_networks from topology."""

        #make sure to remove converters at this point

        #now build connected graphs of same type AC/DC

        sub_graphs = nx.connected_component_subgraphs(self.graph,copy=False)

        self.sub_networks = []

        for sub_graph in sub_graphs:
            sub_network = SubNetwork()

            self.sub_networks.append(sub_network)

            sub_network.graph = sub_graph

            sub_network.buses = sub_graph.nodes()

            sub_network.branches = [branch for (u,v,branch) in sub_graph.edges_iter(data="obj")]



    def pf(self):
        """Non-linear power flow."""
        #calls pf on each sub_network separately

    def lpf(self):
        """Linear power flow."""
        #calls lpf on each sub_network separately

    def lpf_batch(self,subindex=None):
        """Batched linear power flow with numpy.dot."""


    def opf(self,subindex=None):
        """Optimal power flow."""



class SubNetwork(Common):
    """Connected network of same current type (AC or DC)."""

    current_type = String(default="AC",restricted=["AC","DC"])

    frequency = Float(default=50)
    
    num_phases = Float(default=3)

    base_power = Float(default=1)

    graph = None

    buses = None

    def pf(self):
        pf.do_pf(self)

    def opf(self):
        pass
