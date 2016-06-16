

## Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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

"""Power system components.
"""


# make the code as Python 3 compatible as possible
from __future__ import print_function, division, absolute_import
from six import iteritems
from six.moves import map


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS), GNU GPL 3"

import networkx as nx

import numpy as np
import pandas as pd
from itertools import chain
from collections import namedtuple
from operator import itemgetter
from distutils.version import StrictVersion

from .descriptors import Float, String, Series, GraphDesc, OrderedGraph, Integer, Boolean, get_simple_descriptors, get_series_descriptors

from .io import export_to_csv_folder, import_from_csv_folder, import_from_pypower_ppc, import_components_from_dataframe

from .pf import network_lpf, sub_network_lpf, network_pf, sub_network_pf, find_bus_controls, find_slack_bus, calculate_Y, calculate_PTDF, calculate_B_H, calculate_dependent_values

from .contingency import calculate_BODF, network_lpf_contingency, network_sclopf


from .opf import network_lopf, network_opf

from .plot import plot

import inspect

import sys


class Basic(object):
    """Common to every object."""

    name = ""


    def __init__(self, name=""):
        self.name = name

    def __repr__(self):
        return "%s %s" % (self.__class__.__name__, self.name)




class Common(Basic):
    """Common to all objects inside Network object."""
    network = None

    def __init__(self, network, name=""):

        Basic.__init__(self, name)

        self.network = network

class Source(Common):
    """Energy source, such as wind, PV or coal."""

    list_name = "sources"

    #emissions in CO2-tonnes-equivalent per MWh primary energy
    #(e.g. gas: 0.2 t_CO2/MWh_thermal)
    co2_emissions = Float(default=0.)


class Bus(Common):
    """Electrically fundamental node where x-port objects attach."""

    list_name = "buses"

    v_nom = Float(default=1.)

    #should be removed and set by program based on generators at bus
    control = String(default="PQ",restricted=["PQ","PV","Slack"])

    #2-d location data (e.g. longitude and latitude; Spatial Reference System Identifier (SRID) set in network.srid)
    x = Float(default=0.)
    y = Float(default=0.)

    p = Series()
    q = Series()
    v_mag_pu = Series(default=1.)
    v_ang = Series()
    v_mag_pu_set = Series(default=1.)
    v_mag_pu_min = Float(default=0.)
    v_mag_pu_max = Float(default=np.nan)

    #optimisation output for power balance constraint at bus
    marginal_price = Series()

    current_type = String(default="AC",restricted=["AC","DC"])

    sub_network = String(default="")


    def generators(self):
        return self.network.generators[self.network.generators.bus == self.name]

    def generators_t(self):
        return self.network.generators_t.loc[:,:,self.network.generators.bus == self.name]

    def loads(self):
        return self.network.loads[self.network.loads.bus == self.name]

    def loads_t(self):
        return self.network.loads_t.loc[:,:,self.network.loads.bus == self.name]

    def storage_units(self):
        return self.network.storage_units[self.network.storage_units.bus == self.name]

    def storage_units_t(self):
        return self.network.storage_units_t.loc[:,:,self.network.storage_units.bus == self.name]


    def shunt_impedances(self):
        return self.network.shunt_impedances[self.network.shunt_impedances.bus == self.name]


class SubStation(Common):
    """Placeholder for a group of buses."""

class Region(Common):
    """A group of buses such as a country or county."""


class OnePort(Common):
    """Object which attaches to a single bus (e.g. load or generator)."""

    bus = String(default="")
    sign = Float(default=1.)
    p = Series()
    q = Series()


class Generator(OnePort):

    list_name = "generators"

    dispatch = String(default="flexible",restricted=["variable","flexible","inflexible"])

    control = String(default="PQ",restricted=["PQ","PV","Slack"])

    #i.e. coal, CCGT, onshore wind, PV, CSP,....
    source = String(default="")

    #rated power
    p_nom = Float(default=0.0)

    #switch to allow capacity to be extended
    p_nom_extendable = Boolean(False)

    #technical potential
    p_nom_max = Float(np.nan)

    p_nom_min = Float(0.0)

    #optimised capacity
    p_nom_opt = Float(0.0)

    capital_cost = Float(0.0)
    marginal_cost = Float(0.0)


    #power limits for variable generators, which can change e.g. due
    #to weather conditions; per unit to ease multiplication with
    #p_nom, which may be optimised
    p_max_pu = Series(default=1.)
    p_min_pu = Series()


    #non-variable power limits for de-rating and minimum limits for
    #flexible generators
    p_max_pu_fixed = Float(1.)
    p_min_pu_fixed = Float(0.)


    #operator's intended dispatch
    p_set = Series()
    q_set = Series()


    #ratio between electrical energy and primary energy
    #(e.g. gas: 0.4 MWh_elec/MWh_thermal)
    efficiency = Float(1.)



class StorageUnit(Generator):

    list_name = "storage_units"

    #units for state of charge are MWh
    state_of_charge_initial = Float(0.)

    #state of charge can be forced to a particular value
    state_of_charge_set = Series(default=np.nan)

    #optimisation results are stored here
    state_of_charge = Series(default=np.nan)

    #switch to disregard state_of_charge_initial; instead soc[-1] =
    #soc[len(snapshots)-1]
    cyclic_state_of_charge = Boolean(False)

    #maximum state of charge capacity in terms of hours at full output capacity p_nom
    max_hours = Float(1)

    #the minimum power dispatch is negative
    p_min_pu_fixed = Float(-1)

    #in MW
    inflow = Series()
    spill = Series()

    efficiency_store = Float(1)

    efficiency_dispatch = Float(1)

    #per hour per unit loss in state of charge
    standing_loss = Float(0.)



class Load(OnePort):
    """PQ load."""

    list_name = "loads"

    #set sign convention for powers opposite to generator
    sign = Float(-1.)

    p_set = Series()
    q_set = Series()

class ShuntImpedance(OnePort):
    """Shunt y = g + jb."""

    list_name = "shunt_impedances"

    #set sign convention so that g>0 withdraws p from bus
    sign = Float(-1.)

    g = Float(0.)
    b = Float(0.)

    g_pu = Float(0.)
    b_pu = Float(0.)


class Branch(Common):
    """Object which attaches to two buses (e.g. line or 2-winding transformer)."""

    list_name = "branches"

    bus0 = String("")
    bus1 = String("")

    capital_cost = Float(0.)

    s_nom = Float(0.)

    s_nom_extendable = Boolean(False)

    s_nom_max = Float(np.nan)
    s_nom_min = Float(0.)

    #optimised capacity
    s_nom_opt = Float(0.)

    #{p,q}i is positive if power is flowing from bus i into the branch
    #so if power flows from bus0 to bus1, p0 is positive, p1 is negative
    p0 = Series()
    p1 = Series()

    q0 = Series()
    q1 = Series()


class Line(Branch):
    """Lines include distribution and transmission lines, overhead lines and cables."""

    list_name = "lines"

    #series impedance z = r + jx in Ohm
    r = Float(0.)
    x = Float(0.)

    #shunt reactance y = g + jb in 1/Ohm
    g = Float(0.)
    b = Float(0.)

    x_pu = Float(0.)
    r_pu = Float(0.)
    g_pu = Float(0.)
    b_pu = Float(0.)

    #voltage angle difference across branches
    v_ang_min = Float(np.nan)
    v_ang_max = Float(np.nan)

    length = Float(default=1.0)
    terrain_factor = Float(default=1.0)


    sub_network = String("")


class Transformer(Branch):
    """2-winding transformer."""

    list_name = "transformers"

    #per unit with reference to s_nom
    x = Float(0.)
    r = Float(0.)
    g = Float(0.)
    b = Float(0.)

    x_pu = Float(0.)
    r_pu = Float(0.)
    g_pu = Float(0.)
    b_pu = Float(0.)

    #voltage angle difference across branches
    v_ang_min = Float(np.nan)
    v_ang_max = Float(np.nan)

    #ratio of per unit voltages
    tap_ratio = Float(1.)

    #in degrees
    phase_shift = Float(0.)

    sub_network = String("")


class Converter(Branch):
    """Converter between AC and DC. Bus 0 is AC, bus 1 is DC."""

    list_name = "converters"

    #limits per unit of s_nom
    p_min_pu = Float(-1.)
    p_max_pu = Float(1.)

    p_set = Series()

class TransportLink(Branch):
    """Controllable link between two buses - can be used for a transport
    power flow model OR as a simplified version of point-to-point DC
    connection.
    """

    list_name = "transport_links"

    #limits per unit of s_nom
    p_min_pu = Float(-1.)
    p_max_pu = Float(1.)

    p_set = Series()


class ThreePort(Common):
    """Placeholder for 3-winding transformers."""

class ThreeTransformer(ThreePort):
    pass

class LineType(Common):
    """Placeholder for future functionality to automatically generate line
    parameters from standard parameters (e.g. r/km)."""

Type = namedtuple("Type", ['typ', 'name', 'df', 'pnl', 'ind'])

class Network(Basic):
    """
    Network container for all buses, one-ports and branches.

    Parameters
    ----------
    csv_folder_name : string
        Name of folder from which to import CSVs of network data.
    kwargs
        Any remaining attributes to set

    Returns
    -------
    None

    Examples
    --------
    >>> nw = pypsa.Network(csv_folder_name=/my/folder,co2_limit=10e6)

    """



    #the current scenario/time
    now = "now"

    #limit of total co2-tonnes-equivalent emissions for period
    co2_limit = None


    #Spatial Reference System Identifier (SRID) for x,y - defaults to longitude and latitude
    srid = 4326

    graph = GraphDesc()

    #methods imported from other sub-modules

    import_from_csv_folder = import_from_csv_folder

    export_to_csv_folder = export_to_csv_folder

    import_from_pypower_ppc = import_from_pypower_ppc

    lpf = network_lpf

    pf = network_pf

    lopf = network_lopf

    opf = network_opf

    plot = plot

    calculate_dependent_values = calculate_dependent_values

    lpf_contingency = network_lpf_contingency

    sclopf = network_sclopf


    def __init__(self, csv_folder_name=None, **kwargs):

        Basic.__init__(self,kwargs.get("name",""))

        #hack so that Series descriptor works when looking for obj.network.snapshots
        self.network = self

        #a list/index of scenarios/times
        self.snapshots = [self.now]

        #corresponds to number of hours represented by each snapshot
        self.snapshot_weightings = pd.Series(index=self.snapshots,data=1.)


        descriptors = [obj
                       for name, obj in inspect.getmembers(sys.modules[__name__])
                       if inspect.isclass(obj) and hasattr(obj,"list_name")]

        #make a dictionary of all simple descriptors
        self.component_simple_descriptors = {obj : get_simple_descriptors(obj)
                                             for obj in descriptors}

        #make a dictionary of all simple descriptors
        self.component_series_descriptors = {obj : get_series_descriptors(obj)
                                             for obj in descriptors}


        self.build_dataframes()



        if csv_folder_name is not None:
            self.import_from_csv_folder(csv_folder_name)

        for key, value in iteritems(kwargs):
            setattr(self, key, value)




    def build_dataframes(self):
        for cls in component_types:
            columns = list((k, v.typ) for k, v in iteritems(self.component_simple_descriptors[cls]))

            #store also the objects themselves
            columns.append(("obj", object))

            #very important! must tell the descriptor what it's name is
            for k,v in iteritems(self.component_simple_descriptors[cls]):
                v.name = k

            for k,v in iteritems(self.component_series_descriptors[cls]):
                v.name = k


            df = pd.DataFrame({k: pd.Series(dtype=d) for k, d in columns},
                              columns=list(map(itemgetter(0), columns)))

            df.index.name = "name"

            setattr(self,cls.list_name,df)

            pnl = pd.Panel(items=self.component_series_descriptors[cls].keys(),
                           major_axis=self.snapshots,
                           minor_axis=getattr(self,cls.list_name).index,
                           #it's currently hard to image non-float series, but this should be generalised
                           dtype=float)

            setattr(self,cls.list_name+"_t",pnl)

    def set_snapshots(self,snapshots):
        """
        Set the snapshots and reindex all time-dependent data.

        This will reindex all pandas.Panels of time-dependent data; NaNs are filled
        with the default value for that quantity.

        Parameters
        ----------
        snapshots : list or pandas.Index
            All time steps.

        Returns
        -------
        None
        """


        self.snapshots = snapshots

        self.snapshot_weightings = self.snapshot_weightings.reindex(self.snapshots,fill_value=1.)
        if isinstance(snapshots, pd.DatetimeIndex) and StrictVersion(pd.__version__) < '0.18.0':
            snapshots = pd.Index(snapshots.values)

        for cls in component_types:
            pnl = getattr(self,cls.list_name+"_t")
            pnl = pnl.reindex(major_axis=self.snapshots)

            for k,v in self.component_series_descriptors[cls].items():
                pnl.loc[k,snapshots,:] = pnl.loc[k,self.snapshots,:].fillna(v.default)

            setattr(self,cls.list_name+"_t",pnl)


    def add(self,class_name,name,**kwargs):
        """
        Add a single component to the network.

        Adds it to component DataFrame and Panel and creates object.

        Parameters
        ----------
        class_name : string
            Component class name
        name : string
            Component name in ["Bus","Generator","Load","StorageUnit","ShuntImpedance","Line","Transformer","Converter","TransportLink"]
        kwargs

        Returns
        -------
        obj : component class
            Object corresponding to component


        Examples
        --------
        >>> network.add("Line","line 12345",x=0.1)

        """

        try:
            cls = globals()[class_name]
        except KeyError:
            print(class_name,"not found")
            return None

        cls_df = getattr(self,cls.list_name)

        if str(name) in cls_df.index:
            print("Failed to add",name,"because there is already an object with this name in",cls.list_name)
            return

        obj = cls(self,str(name))

        #build a single-index df of default values to concatenate with cls_df
        cols = list(self.component_simple_descriptors[cls].keys())
        values = [self.component_simple_descriptors[cls][col].default for col in cols]

        obj_df = pd.DataFrame(data=[values+[obj]],index=[obj.name],columns=cols+["obj"])

        setattr(self,cls.list_name,pd.concat((cls_df,obj_df)))


        #now deal with time-dependent variables
        pnl = getattr(self,cls.list_name+"_t")

        pnl = pnl.reindex(minor_axis=pnl.minor_axis.append(pd.Index([obj.name])))

        setattr(self,cls.list_name+"_t",pnl)

        for k,v in iteritems(self.component_series_descriptors[cls]):
            pnl.loc[k,:,obj.name] = v.default


        for key,value in iteritems(kwargs):
            setattr(obj,key,value)

        return obj


    def add_from(self,object_list):
        """Add objects from a list."""

        raise NotImplementedError("Batch adding of components not implemented yet.")


    def remove(self,class_name,name):
        """
        Remove a single component to the network.

        Removes it from component DataFrame and Panel and deletes object.

        Parameters
        ----------
        class_name : string
            Component class name
        name : string
            Component name

        Examples
        --------
        >>> network.remove("Line","line 12345")

        """

        try:
            cls = globals()[class_name]
        except KeyError:
            print(class_name,"not found")
            return None

        cls_df = getattr(self,cls.list_name)

        obj = cls_df.obj[name]

        cls_df.drop(name,inplace=True)

        pnl = getattr(self,cls.list_name+"_t")

        pnl.drop(name,axis=2,inplace=True)

        del obj

    def copy(self, with_time=True):
        """
        Returns a deep copy of the Network object with all components and
        time dependent data.

        Returns
        --------
        network : pypsa.Network


        Examples
        --------
        >>> network_copy = network.copy()

        """

        network = Network()

        for component in self.iterate_components():
            import_components_from_dataframe(network, component.df, component.name)

        if with_time:
            network.set_snapshots(self.snapshots)
            for component in self.iterate_components():
                setattr(network, component.typ.list_name+"_t", component.pnl.copy())

        #catch all remaining attributes of network
        for attr in ["now","co2_limit","srid"]:
            setattr(network,attr,getattr(self,attr))

        return network


    def branches(self):
        return pd.concat((getattr(self, typ.list_name) for typ in branch_types),
                         keys=[typ.__name__ for typ in branch_types])

    def passive_branches(self):
        return pd.concat((getattr(self, typ.list_name)
                          for typ in passive_branch_types),
                         keys=[typ.__name__ for typ in passive_branch_types])

    def controllable_branches(self):
        return pd.concat((getattr(self, typ.list_name)
                          for typ in controllable_branch_types),
                         keys=[typ.__name__ for typ in controllable_branch_types])


    def build_graph(self):
        """Build networkx graph."""

        #reset the graph
        self.graph = OrderedGraph()

        #add nodes first, in case there are isolated buses not connected with branches
        self.graph.add_nodes_from(self.buses.index)

        #Multigraph uses object itself as key
        self.graph.add_edges_from((branch.bus0, branch.bus1, branch.obj, {})
                                  for t in self.iterate_components(branch_types)
                                  for branch in t.df.itertuples())

    def determine_network_topology(self,skip_pre=False):
        """
        Build sub_networks from topology.

        skip_pre: bool, default False
            Skip the preliminary step of building the graph.
        """

        if not skip_pre:
            self.build_graph()

        #remove converters and transport links that could be connected networks
        # of different types or non-synchronous areas

        graph = self.graph.__class__(self.graph)

        graph.remove_edges_from((branch.bus0, branch.bus1, branch.obj)
                                for t in self.iterate_components(controllable_branch_types)
                                for branch in t.df.itertuples())

        #now build connected graphs of same type AC/DC
        sub_graphs = nx.connected_component_subgraphs(graph, copy=False)

        #remove all old sub_networks
        for sub_network in self.sub_networks.index:
            self.remove("SubNetwork",sub_network)

        for i, sub_graph in enumerate(sub_graphs):
            #name using i for now
            sub_network = self.add("SubNetwork", i, graph=sub_graph)

            #grab data from first bus
            sub_network.current_type = self.buses.loc[next(sub_graph.nodes_iter()), "current_type"]

            for bus in sub_graph.nodes():
                self.buses.loc[bus,"sub_network"] = sub_network.name

            for (u,v,branch) in sub_graph.edges_iter(keys=True):
                branch.sub_network = sub_network.name

    def iterate_components(self, types=None, skip_empty=True):
        if types is None:
            types = component_types
        return (Type(typ=typ, name=typ.__name__,
                     df=getattr(self, typ.list_name),
                     pnl=getattr(self, typ.list_name + '_t'),
                     ind=None)
                for typ in types
                if not skip_empty or len(getattr(self, typ.list_name)) > 0)


class SubNetwork(Common):
    """
    Connected network of same current type (AC or DC).

    Generated by network.determine_network_topology().

    """

    list_name = "sub_networks"

    current_type = String(default="AC",restricted=["AC","DC"])

    slack_bus = String("")

    graph = GraphDesc()


    lpf = sub_network_lpf

    pf = sub_network_pf

    find_bus_controls = find_bus_controls

    find_slack_bus = find_slack_bus

    calculate_Y = calculate_Y

    calculate_PTDF = calculate_PTDF

    calculate_B_H = calculate_B_H

    calculate_BODF = calculate_BODF

    def buses_i(self):
        return self.network.buses.index[self.network.buses.sub_network == self.name]

    def lines_i(self):
        return self.network.lines.index[self.network.lines.sub_network == self.name]

    def transformers_i(self):
        return self.network.transformers.index[self.network.transformers.sub_network == self.name]

    def branches_i(self):
        types = []
        names = []
        for t in self.iterate_components(passive_branch_types):
            types += len(t.ind) * [t.name]
            names += list(t.ind)
        return pd.MultiIndex.from_arrays([types, names], names=('type', 'name'))

    def branches(self):
        branches = self.network.branches()
        return branches[branches.sub_network == self.name]

    def generators_i(self):
        sub_networks = self.network.generators.bus.map(self.network.buses.sub_network)
        return self.network.generators.index[sub_networks == self.name]

    def loads_i(self):
        sub_networks = self.network.loads.bus.map(self.network.buses.sub_network)
        return self.network.loads.index[sub_networks == self.name]

    def shunt_impedances_i(self):
        sub_networks = self.network.shunt_impedances.bus.map(self.network.buses.sub_network)
        return self.network.shunt_impedances.index[sub_networks == self.name]

    def storage_units_i(self):
        sub_networks = self.network.storage_units.bus.map(self.network.buses.sub_network)
        return self.network.storage_units.index[sub_networks == self.name]

    def buses(self):
        return self.network.buses[self.buses_i()]

    def generators(self):
        return self.network.generators.loc[self.generators_i()]

    def loads(self):
        return self.network.loads.loc[self.loads_i()]

    def shunt_impedances(self):
        return self.network.shunt_impedances.loc[self.shunt_impedances_i()]

    def storage_units(self):
        return self.network.storage_units.loc[self.storage_units_i()]

    def iterate_components(self, types=None, skip_empty=True):
        for t in self.network.iterate_components(types=types, skip_empty=False):
            t = Type(*t[:-1], ind=getattr(self, t.typ.list_name + '_i')())
            if not (skip_empty and t.df.empty):
                yield t

passive_one_port_types = {ShuntImpedance}
controllable_one_port_types = {Load, Generator, StorageUnit}
one_port_types = passive_one_port_types|controllable_one_port_types

passive_branch_types = {Line, Transformer}
controllable_branch_types = {Converter, TransportLink}
branch_types = passive_branch_types|controllable_branch_types

component_types = branch_types|one_port_types|{Bus, SubNetwork, Source}
