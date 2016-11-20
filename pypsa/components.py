

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
from __future__ import division, absolute_import
import six
from six import iteritems, itervalues, iterkeys
from six.moves import map
from weakref import ref


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS), GNU GPL 3"

import networkx as nx

import numpy as np
import pandas as pd
import scipy as sp, scipy.sparse
from itertools import chain
from collections import namedtuple
from operator import itemgetter

from distutils.version import StrictVersion, LooseVersion
try:
    _pd_version = StrictVersion(pd.__version__)
except ValueError:
    _pd_version = LooseVersion(pd.__version__)

from .descriptors import (Float, String, Series, Integer, Boolean,
                          get_simple_descriptors,
                          get_series_descriptors, Dict)

from .io import (export_to_csv_folder, import_from_csv_folder,
                 import_from_pypower_ppc, import_components_from_dataframe,
                 import_series_from_dataframe)

from .pf import (network_lpf, sub_network_lpf, network_pf,
                 sub_network_pf, find_bus_controls, find_slack_bus, calculate_Y,
                 calculate_PTDF, calculate_B_H, calculate_dependent_values)

from .contingency import (calculate_BODF, network_lpf_contingency,
                          network_sclopf)


from .opf import network_lopf, network_opf

from .plot import plot

from .graph import graph, incidence_matrix, adjacency_matrix

import inspect

import sys

import logging
logger = logging.getLogger(__name__)




inf = float("inf")

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
        self._network = ref(network)

    @property
    def network(self):
        return self._network()

class Carrier(Common):
    """Energy carrier, such as wind, PV or coal. The Carrier tracks co2
    emissions if there is a network.co2_limit.  Buses have direct
    carriers and Generators indicate their primary energy carriers.
    """

    list_name = "carriers"

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

    p = Series(output=True)
    q = Series(output=True)
    v_mag_pu = Series(default=1., output=True)
    v_ang = Series(output=True)

    v_mag_pu_set = Series(default=1.)

    v_mag_pu_min = Float(default=0.)
    v_mag_pu_max = Float(default=inf)

    #optimisation output for power balance constraint at bus
    marginal_price = Series(output=True)

    #energy carrier could be "AC", "DC", "heat", "gas"
    carrier = String(default="AC")

    sub_network = String(default="")


class SubStation(Common):
    """Placeholder for a group of buses."""

class Region(Common):
    """A group of buses such as a country or county."""


class OnePort(Common):
    """Object which attaches to a single bus (e.g. load or generator)."""

    bus = String(default="")
    sign = Float(default=1.)
    p = Series(output=True)
    q = Series(output=True)


class Generator(OnePort):

    list_name = "generators"

    control = String(default="PQ",restricted=["PQ","PV","Slack"])

    #i.e. coal, CCGT, onshore wind, PV, CSP,....
    carrier = String(default="")

    #rated power
    p_nom = Float(default=0.0)

    #switch to allow capacity to be extended
    p_nom_extendable = Boolean(False)

    #technical potential
    p_nom_max = Float(inf)
    p_nom_min = Float(0.0)

    #optimised capacity
    p_nom_opt = Float(0.0)

    capital_cost = Float(0.0)
    marginal_cost = Float(0.0)


    #power limits for variable generators, which can change e.g. due
    #to weather conditions; per unit to ease multiplication with
    #p_nom, which may be optimised
    p_max_pu = Series(default=1.)
    p_min_pu = Series(default=0.)

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
    state_of_charge = Series(default=np.nan, output=True)

    #switch to disregard state_of_charge_initial; instead soc[-1] =
    #soc[len(snapshots)-1]
    cyclic_state_of_charge = Boolean(False)

    #maximum state of charge capacity in terms of hours at full output capacity p_nom
    max_hours = Float(1)

    #the minimum power dispatch is negative
    p_min_pu = Series(default=-1)

    #in MW
    inflow = Series()
    spill = Series(output=True)

    efficiency_store = Float(1)

    efficiency_dispatch = Float(1)

    #per hour per unit loss in state of charge
    standing_loss = Float(0.)


class Store(Common):
    """Generic store, whose capacity may be optimised."""

    list_name = "stores"

    bus = String(default="")
    sign = Float(default=1.)
    p = Series(output=True)
    q = Series(output=True)
    e = Series(output=True)

    p_set = Series()
    q_set = Series()

    #rated energy capacity
    e_nom = Float(default=0.0)

    #switch to allow energy capacity to be extended
    e_nom_extendable = Boolean(False)

    #technical potential
    e_nom_max = Float(inf)

    e_nom_min = Float(0.0)

    #optimised capacity
    e_nom_opt = Float(0.0)

    e_max_pu = Series(default=1.)
    e_min_pu = Series(default=0.)

    e_cyclic = Boolean(False)

    e_initial = Float(0.)

    #cost of increasing e_nom
    capital_cost = Float(0.0)

    #cost of dispatching from the store
    marginal_cost = Float(0.0)

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

    s_nom_max = Float(inf)
    s_nom_min = Float(0.)

    #optimised capacity
    s_nom_opt = Float(0.)

    #{p,q}i is positive if power is flowing from bus i into the branch
    #so if power flows from bus0 to bus1, p0 is positive, p1 is negative
    p0 = Series(output=True)
    p1 = Series(output=True)

    q0 = Series(output=True)
    q1 = Series(output=True)


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
    v_ang_min = Float(-inf)
    v_ang_max = Float(inf)

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
    v_ang_min = Float(-inf)
    v_ang_max = Float(inf)

    #ratio of per unit voltages
    tap_ratio = Float(1.)

    #in degrees
    phase_shift = Float(0.)

    sub_network = String("")


class Link(Common):
    """Link between two buses with controllable active power - can be used
    for a transport power flow model OR as a simplified version of
    point-to-point DC connection OR as a lossy energy converter.

    NB: for a lossless bi-directional HVDC or transport link, set
    p_min_pu = -1 and efficiency = 1.

    NB: It is assumed that the links neither produce nor consume
    reactive power.

    """

    list_name = "links"

    bus0 = String("")
    bus1 = String("")

    capital_cost = Float(0.)

    #NB: marginal cost only makes sense in OPF if p_max_pu >= 0
    marginal_cost = Float(0.)

    p_nom = Float(0.)

    p_nom_extendable = Boolean(False)

    p_nom_max = Float(inf)
    p_nom_min = Float(0.)

    #optimised capacity
    p_nom_opt = Float(0.)

    #pi is positive if power is flowing from bus i into the branch
    #so if power flows from bus0 to bus1, p0 is positive, p1 is negative
    p0 = Series(output=True)
    p1 = Series(output=True)

    #limits per unit of p_nom
    p_min_pu = Series(default=0.)
    p_max_pu = Series(default=1.)

    efficiency = Float(1.)

    #The set point for p0.
    p_set = Series()

    length = Float(default=1.0)
    terrain_factor = Float(default=1.0)

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

    #methods imported from other sub-modules

    import_from_csv_folder = import_from_csv_folder

    export_to_csv_folder = export_to_csv_folder

    import_from_pypower_ppc = import_from_pypower_ppc

    import_components_from_dataframe = import_components_from_dataframe

    import_series_from_dataframe = import_series_from_dataframe

    lpf = network_lpf

    pf = network_pf

    lopf = network_lopf

    opf = network_opf

    plot = plot

    calculate_dependent_values = calculate_dependent_values

    lpf_contingency = network_lpf_contingency

    sclopf = network_sclopf

    graph = graph

    incidence_matrix = incidence_matrix

    adjacency_matrix = adjacency_matrix

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

        #make a dictionary of all series descriptors
        self.component_series_descriptors = {obj : get_series_descriptors(obj)
                                             for obj in descriptors}


        self._build_dataframes()



        if csv_folder_name is not None:
            self.import_from_csv_folder(csv_folder_name)

        for key, value in iteritems(kwargs):
            setattr(self, key, value)




    def _build_dataframes(self):
        """Function called when network is created to build component pandas.DataFrames."""

        for cls in component_types:
            columns = list((k, v.typ) for k, v in iteritems(self.component_simple_descriptors[cls]))

            #store also the objects themselves
            columns.append(("obj", object))

            #very important! must tell the descriptor what it's name is
            for k,v in iteritems(self.component_simple_descriptors[cls]):
                v.name = k

            for k,v in iteritems(self.component_series_descriptors[cls]):
                v.name = k
                if not v.output:
                    columns.append((k, v.dtype))

            df = pd.DataFrame({k: pd.Series(dtype=d) for k, d in columns},
                              columns=list(map(itemgetter(0), columns)))

            df.index.name = "name"

            setattr(self,cls.list_name,df)

            pnl = Dict({k : pd.DataFrame(index=self.snapshots,
                                         columns=[],
                                         #it's currently hard to imagine non-float series, but this could be generalised
                                         dtype=float)
                        for k, desc in iteritems(self.component_series_descriptors[cls])
                        #if not desc.output
                       })

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
        if isinstance(snapshots, pd.DatetimeIndex) and _pd_version < '0.18.0':
            snapshots = pd.Index(snapshots.values)

        for cls in component_types:
            pnl = getattr(self,cls.list_name+"_t")

            for k,v in self.component_series_descriptors[cls].items():
                pnl[k] = pnl[k].reindex(self.snapshots).fillna(v.default)

    def add(self, class_name, name, **kwargs):
        """
        Add a single component to the network.

        Adds it to component DataFrame and Panel and creates object.

        Parameters
        ----------
        class_name : string
            Component class name in ["Bus","Generator","Load","StorageUnit","Store","ShuntImpedance","Line","Transformer","Link"]
        name : string
            Component name
        kwargs
            Component attributes, e.g. x=0.1, length=123

        Examples
        --------
        >>> network.add("Line", "line 12345", x=0.1)

        """

        try:
            cls = globals()[class_name]
        except KeyError:
            logger.error("Component class {} not found".format(class_name))
            return None

        cls_df = getattr(self, cls.list_name)

        if str(name) in cls_df.index:
            logger.error("Failed to add {} component {} because there is already an object with this name in {}".format(class_name, name, cls.list_name))
            return

        obj = cls(self, str(name))

        simple_descriptors = self.component_simple_descriptors[cls]
        series_descriptors = self.component_series_descriptors[cls]

        #build a single-index df of values to concatenate with cls_df
        cols = list(simple_descriptors)
        values = [(kwargs.pop(col)
                   if col in kwargs
                   else simple_descriptors[col].default)
                  for col in cols]

        obj_df = pd.DataFrame(data=[values+[obj]],index=[obj.name],columns=cols+["obj"])
        new_df = pd.concat((cls_df, obj_df))

        setattr(self, cls.list_name, new_df)

        #now deal with time-dependent variables
        pnl = getattr(self,cls.list_name+"_t")

        for k,v in iteritems(series_descriptors):
            if not v.output:
                new_df.at[obj.name, k] = v.default

        for key, value in iteritems(kwargs):
            setattr(obj, key, value)


    def remove(self, class_name, name):
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
        >>> network.remove("Line","my_line 12345")

        """

        try:
            cls = globals()[class_name]
        except KeyError:
            logger.error("Class {} not found".format(class_name))
            return None

        cls_df = getattr(self, cls.list_name)

        obj = cls_df.obj[name]

        cls_df.drop(name, inplace=True)

        pnl = getattr(self, cls.list_name+"_t")

        for df in itervalues(pnl):
            if name in df:
                df.drop(name, axis=1, inplace=True)

        del obj

    def copy(self, with_time=True):
        """
        Returns a deep copy of the Network object with all components and
        time-dependent data.

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
                pnl = getattr(network, component.typ.list_name+"_t")
                for k in iterkeys(component.pnl):
                    pnl[k] = component.pnl[k].copy()

        #catch all remaining attributes of network
        for attr in ["name", "now", "co2_limit", "srid"]:
            setattr(network,attr,getattr(self,attr))

        return network


    #beware, this turns bools like s_nom_extendable into objects because of
    #presence of links without s_nom_extendable
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

    def determine_network_topology(self):
        """
        Build sub_networks from topology.
        """

        adjacency_matrix = self.adjacency_matrix(passive_branch_types)
        n_components, labels = sp.sparse.csgraph.connected_components(adjacency_matrix, directed=False)

        # remove all old sub_networks
        for sub_network in self.sub_networks.index:
            self.remove("SubNetwork", sub_network)

        for i in np.arange(n_components):
            # index of first bus
            buses_i = (labels == i).nonzero()[0]
            carrier = self.buses.carrier.iat[buses_i[0]]

            if carrier not in ["AC","DC"] and len(buses_i) > 1:
                logger.warning("Warning, sub network {} is not electric but contains multiple buses\n"
                                "and branches. Passive flows are not allowed for non-electric networks!".format(i))

            if (self.buses.carrier.iloc[buses_i] != carrier).any():
                logger.warning("Warning, sub network {} contains buses with mixed carriers! Value counts:\n{}".format(i),
                                self.buses.carrier.iloc[buses_i].value_counts())

            sub_network = self.add("SubNetwork", i, carrier=carrier)

        self.buses.loc[:, "sub_network"] = labels.astype(str)

        for t in self.iterate_components(passive_branch_types):
            t.df["sub_network"] = t.df.bus0.map(self.buses["sub_network"])

    def iterate_components(self, types=None, skip_empty=True):
        if types is None:
            types = component_types
        else:
            from . import components
            types = [getattr(components, t)
                     if isinstance(t, six.string_types)
                     else t
                     for t in types]

        return (Type(typ=typ, name=typ.__name__,
                     df=getattr(self, typ.list_name),
                     pnl=getattr(self, typ.list_name + '_t'),
                     ind=None)
                for typ in types
                if not skip_empty or len(getattr(self, typ.list_name)) > 0)


    def consistency_check(self):
        """
        Checks the network for consistency, including bus definitions and impedances.


        Examples
        --------
        >>> network.consistency_check()

        """


        for t in one_port_types:
            list_name = t.list_name
            df = getattr(self,list_name)
            missing = df.index[pd.isnull(df.bus.map(self.buses.v_nom))]
            if len(missing) > 0:
                logger.warning("The following {} have buses which are not defined:\n{}".format(list_name,missing))

        for t in branch_types:
            list_name = t.list_name
            df = getattr(self,list_name)
            for end in ["0","1"]:
                missing = df.index[pd.isnull(df["bus"+end].map(self.buses.v_nom))]
                if len(missing) > 0:
                    logger.warning("The following {} have bus {} which are not defined:\n{}".format(list_name,end,missing))


        for t in passive_branch_types:
            list_name = t.list_name
            df = getattr(self,list_name)
            for attr in ["x","r"]:
                bad = df.index[df[attr] == 0.]
                if len(bad) > 0:
                    logger.warning("The following {} have zero {}, which could break the linear load flow:\n{}".format(list_name,attr,bad))

            bad = df.index[(df["x"] == 0.) & (df["r"] == 0.)]
            if len(bad) > 0:
                logger.warning("The following {} have zero series impedance, which will break the load flow:\n{}".format(list_name,bad))


        for t in [Transformer]:
            list_name = t.list_name
            df = getattr(self,list_name)

            bad = df.index[df["s_nom"] == 0.]
            if len(bad) > 0:
                logger.warning("The following {} have zero s_nom, which is used to define the impedance and will thus break the load flow:\n{}".format(list_name,bad))


class SubNetwork(Common):
    """
    Connected network of electric buses (AC or DC) with passive flows
    or isolated non-electric buses.

    Generated by network.determine_network_topology().

    """

    list_name = "sub_networks"

    #carrier is inherited from the buses
    carrier = String(default="AC")

    slack_bus = String("")

    lpf = sub_network_lpf

    pf = sub_network_pf

    find_bus_controls = find_bus_controls

    find_slack_bus = find_slack_bus

    calculate_Y = calculate_Y

    calculate_PTDF = calculate_PTDF

    calculate_B_H = calculate_B_H

    calculate_BODF = calculate_BODF

    graph = graph

    incidence_matrix = incidence_matrix

    adjacency_matrix = adjacency_matrix

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
        branches = self.network.passive_branches()
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

    def stores_i(self):
        sub_networks = self.network.stores.bus.map(self.network.buses.sub_network)
        return self.network.stores.index[sub_networks == self.name]


    def buses(self):
        return self.network.buses.loc[self.buses_i()]

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
controllable_one_port_types = {Load, Generator, StorageUnit, Store}
one_port_types = passive_one_port_types|controllable_one_port_types

passive_branch_types = {Line, Transformer}
controllable_branch_types = {Link}
branch_types = passive_branch_types|controllable_branch_types

component_types = branch_types|one_port_types|{Bus, SubNetwork, Carrier}
