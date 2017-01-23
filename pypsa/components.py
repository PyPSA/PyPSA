

## Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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
__copyright__ = "Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS), GNU GPL 3"

import networkx as nx

import numpy as np
import pandas as pd
import scipy as sp, scipy.sparse
from itertools import chain
from collections import namedtuple
from operator import itemgetter
import os


from distutils.version import StrictVersion, LooseVersion
try:
    _pd_version = StrictVersion(pd.__version__)
except ValueError:
    _pd_version = LooseVersion(pd.__version__)

from .descriptors import Dict

from .io import (export_to_csv_folder, import_from_csv_folder,
                 import_from_pypower_ppc, import_components_from_dataframe,
                 import_series_from_dataframe, import_from_pandapower_net)

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



dir_name = os.path.dirname(__file__)

component_attrs_dir_name = "component_attrs"

standard_types_dir_name = "standard_types"


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


Component = namedtuple("Component", ['name', 'list_name', 'attrs', 'df', 'pnl', 'ind'])

class Network(Basic):
    """
    Network container for all buses, one-ports and branches.

    Parameters
    ----------
    csv_folder_name : string
        Name of folder from which to import CSVs of network data.
    name : string, default ""
        Network name.
    ignore_standard_types : boolean, default False
        If True, do not read in PyPSA standard types into standard types DataFrames.
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

    import_from_pandapower_net = import_from_pandapower_net

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

    def __init__(self, csv_folder_name=None, name="", ignore_standard_types=False, **kwargs):

        Basic.__init__(self,name)

        #a list/index of scenarios/times
        self.snapshots = pd.Index([self.now])

        #corresponds to number of hours represented by each snapshot
        self.snapshot_weightings = pd.Series(index=self.snapshots,data=1.)

        components = pd.read_csv(os.path.join(dir_name,
                                              "components.csv"),
                                 index_col=0)

        self.components = Dict(components.T.to_dict())

        for component in self.components.keys():
            file_name = os.path.join(dir_name,
                                     component_attrs_dir_name,
                                     self.components[component]["list_name"] + ".csv")

            attrs = pd.read_csv(file_name,
                                index_col=0)

            attrs['static'] = (attrs['type'] != 'series')
            attrs['varying'] = attrs['type'].isin({'series', 'static or series'})
            attrs['typ'] = attrs['type'].map({'boolean': bool, 'int': int, 'string': str}).fillna(float)

            bool_b = attrs.type == 'boolean'
            attrs.loc[bool_b, 'default'] = attrs.loc[bool_b].isin({True, 'True'})

            #exclude Network because it's not in a DF and has non-typical attributes
            if component != "Network":
                for typ in (str, float, int):
                    attrs.loc[attrs.typ == typ, "default"] = attrs.loc[attrs.typ == typ, "default"].apply(typ)

            attrs.loc[attrs.default == "n/a","default"] = ""

            self.components[component]["attrs"] = attrs

        self._build_dataframes()

        if not ignore_standard_types:
            self.read_in_default_standard_types()


        if csv_folder_name is not None:
            self.import_from_csv_folder(csv_folder_name)


        for key, value in iteritems(kwargs):
            setattr(self, key, value)


    def _build_dataframes(self):
        """Function called when network is created to build component pandas.DataFrames."""

        for component in all_components:

            attrs = self.components[component]["attrs"]

            static_typs = attrs.typ[attrs.static].drop(["name"])

            df = pd.DataFrame({k: pd.Series(dtype=d) for k, d in static_typs.iteritems()},
                              columns=static_typs.index)

            df.index.name = "name"

            setattr(self,self.components[component]["list_name"],df)

            pnl = Dict({k : pd.DataFrame(index=self.snapshots,
                                         columns=[],
                                         #it's currently hard to imagine non-float series, but this could be generalised
                                         dtype=float)
                        for k in attrs.index[attrs.varying]
                       })

            setattr(self,self.components[component]["list_name"]+"_t",pnl)


    def read_in_default_standard_types(self):

        for std_type in standard_types:

            list_name = self.components[std_type]["list_name"]

            file_name = os.path.join(dir_name,
                                     standard_types_dir_name,
                                     list_name + ".csv")

            self.components[std_type]["standard_types"] = pd.read_csv(file_name,
                                                                      index_col=0)

            self.import_components_from_dataframe(self.components[std_type]["standard_types"], std_type)


    def df(self, component_name):
        """
        Return the DataFrame of static components for component_name,
        i.e. network.component_names

        Parameters
        ----------
        component_name : string

        Returns
        -------
        pandas.DataFrame
        """
        return getattr(self, self.components[component_name]["list_name"])


    def pnl(self, component_name):
        """
        Return the dictionary of DataFrames of varying components for component_name,
        i.e. network.component_names_t

        Parameters
        ----------
        component_name : string

        Returns
        -------
        dict of pandas.DataFrame
        """
        return getattr(self, self.components[component_name]["list_name"]+"_t")


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

        self.snapshots = pd.Index(snapshots)

        self.snapshot_weightings = self.snapshot_weightings.reindex(self.snapshots,fill_value=1.)
        if isinstance(snapshots, pd.DatetimeIndex) and _pd_version < '0.18.0':
            snapshots = pd.Index(snapshots.values)

        for component in all_components:
            pnl = self.pnl(component)
            attrs = self.components[component]["attrs"]

            for k,default in attrs.default[attrs.varying].iteritems():
                pnl[k] = pnl[k].reindex(self.snapshots).fillna(default)

        #NB: No need to rebind pnl to self, since haven't changed it

        if self.now not in self.snapshots:
            logger.warning("Attribute network.now is not in newly-defined snapshots. (network.now is only relevant if you call e.g. network.pf() without specifying snapshots.)")



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

        if class_name not in self.components:
            logger.error("Component class {} not found".format(class_name))
            return None

        cls_df = self.df(class_name)
        cls_pnl = self.pnl(class_name)

        name = str(name)

        if name in cls_df.index:
            logger.error("Failed to add {} component {} because there is already an object with this name in {}".format(class_name, name, self.components[class_name]["list_name"]))
            return


        attrs = self.components[class_name]["attrs"]

        static_attrs = attrs[attrs.static].drop("name")

        #This guarantees that the correct attribute type is maintained
        obj_df = pd.DataFrame(data=[static_attrs.default],index=[name],columns=static_attrs.index)
        new_df = cls_df.append(obj_df)

        setattr(self, self.components[class_name]["list_name"], new_df)

        for k,v in iteritems(kwargs):
            if k not in attrs.index:
                logger.warning("{} has no attribute {}, ignoring this passed value.".format(class_name,k))
                continue
            typ = attrs.at[k,"typ"]
            if not attrs.at[k,"varying"]:
                new_df.at[name,k] = typ(v)
            elif attrs.at[k,"static"] and not isinstance(v, (pd.Series, np.ndarray, list)):
                new_df.at[name,k] = typ(v)
            else:
                cls_pnl[k][name] = pd.Series(data=v, index=self.snapshots, dtype=typ)


        for attr in ["bus","bus0","bus1"]:
            if attr in new_df.columns:
                bus_name = new_df.at[name,attr]
                if bus_name not in self.buses.index:
                    logger.warning("The bus name `{}` given for {} of {} `{}` does not appear in network.buses".format(bus_name,attr,class_name,name))


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

        if class_name not in self.components:
            logger.error("Component class {} not found".format(class_name))
            return None

        cls_df = self.df(class_name)

        cls_df.drop(name, inplace=True)

        pnl = self.pnl(class_name)

        for df in itervalues(pnl):
            if name in df:
                df.drop(name, axis=1, inplace=True)


    def copy(self, with_time=True, ignore_standard_types=False):
        """
        Returns a deep copy of the Network object with all components and
        time-dependent data.

        Returns
        --------
        network : pypsa.Network

        Parameters
        ----------
        with_time : boolean, default True
            Copy snapshots and time-varying network.component_names_t data too.
        ignore_standard_types : boolean, default False
            Ignore the PyPSA standard types.

        Examples
        --------
        >>> network_copy = network.copy()

        """

        network = Network(ignore_standard_types=ignore_standard_types)

        for component in self.iterate_components(["Bus", "Carrier"] + sorted(all_components - {"Bus","Carrier"})):
            df = component.df
            #drop the standard types to avoid them being read in twice
            if not ignore_standard_types and component.name in standard_types:
                df = component.df.drop(network.components[component.name]["standard_types"].index)

            import_components_from_dataframe(network, df, component.name)

        if with_time:
            network.set_snapshots(self.snapshots)
            for component in self.iterate_components():
                pnl = getattr(network, component.list_name+"_t")
                for k in iterkeys(component.pnl):
                    pnl[k] = component.pnl[k].copy()

        #catch all remaining attributes of network
        for attr in ["name", "now", "co2_limit", "srid"]:
            setattr(network,attr,getattr(self,attr))

        return network


    #beware, this turns bools like s_nom_extendable into objects because of
    #presence of links without s_nom_extendable
    def branches(self):
        return pd.concat((self.df(c) for c in branch_components),
                         keys=branch_components)

    def passive_branches(self):
        return pd.concat((self.df(c) for c in passive_branch_components),
                         keys=passive_branch_components)

    def controllable_branches(self):
        return pd.concat((self.df(c) for c in controllable_branch_components),
                         keys=controllable_branch_components)

    def determine_network_topology(self):
        """
        Build sub_networks from topology.
        """

        adjacency_matrix = self.adjacency_matrix(passive_branch_components)
        n_components, labels = sp.sparse.csgraph.connected_components(adjacency_matrix, directed=False)

        # remove all old sub_networks
        for sub_network in self.sub_networks.index:
            obj = self.sub_networks.at[sub_network,"obj"]
            self.remove("SubNetwork", sub_network)
            del obj

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

            self.add("SubNetwork", i, carrier=carrier)

        #add objects
        self.sub_networks["obj"] = [SubNetwork(self, name) for name in self.sub_networks.index]

        self.buses.loc[:, "sub_network"] = labels.astype(str)

        for c in self.iterate_components(passive_branch_components):
            c.df["sub_network"] = c.df.bus0.map(self.buses["sub_network"])

    def iterate_components(self, components=None, skip_empty=True):
        if components is None:
            components = all_components

        return (Component(name=c,
                          list_name=self.components[c]["list_name"],
                          attrs=self.components[c]["attrs"],
                          df=self.df(c),
                          pnl=self.pnl(c),
                          ind=None)
                for c in components
                if not (skip_empty and self.df(c).empty))


    def consistency_check(self):
        """
        Checks the network for consistency, including bus definitions and impedances.
        Prints warnings if anything is potentially inconsistent.

        Examples
        --------
        >>> network.consistency_check()

        """


        for c in self.iterate_components(one_port_components):
            missing = c.df.index[pd.isnull(c.df.bus.map(self.buses.v_nom))]
            if len(missing) > 0:
                logger.warning("The following {} have buses which are not defined:\n{}".format(c.list_name,missing))

        for c in self.iterate_components(branch_components):
            for end in ["0","1"]:
                missing = c.df.index[pd.isnull(c.df["bus"+end].map(self.buses.v_nom))]
                if len(missing) > 0:
                    logger.warning("The following {} have bus {} which are not defined:\n{}".format(c.list_name,end,missing))


        for c in self.iterate_components(passive_branch_components):
            for attr in ["x","r"]:
                bad = c.df.index[c.df[attr] == 0.]
                if len(bad) > 0:
                    logger.warning("The following {} have zero {}, which could break the linear load flow:\n{}".format(c.list_name,attr,bad))

            bad = c.df.index[(c.df["x"] == 0.) & (c.df["r"] == 0.)]
            if len(bad) > 0:
                logger.warning("The following {} have zero series impedance, which will break the load flow:\n{}".format(c.list_name,bad))


        for c in self.iterate_components({"Transformer"}):
            bad = c.df.index[c.df["s_nom"] == 0.]
            if len(bad) > 0:
                logger.warning("The following {} have zero s_nom, which is used to define the impedance and will thus break the load flow:\n{}".format(c.list_name,bad))


        for c in self.iterate_components(all_components):
            for attr in c.attrs.index[c.attrs.varying & c.attrs.static]:
                attr_df = c.pnl[attr]

                diff = attr_df.columns.difference(c.df.index)
                if len(diff) > 0:
                    logger.warning("The following {} have time series defined for attribute {} in network.{}_t, but are not defined in network.{}:\n{}".format(c.list_name,attr,c.list_name,c.list_name,diff))

                diff = self.snapshots.difference(attr_df.index)
                if len(diff) > 0:
                    logger.warning("In the time-dependent Dataframe for attribute {} of network.{}_t the following snapshots are missing:\n{}".format(attr,c.list_name,diff))

                diff = attr_df.index.difference(self.snapshots)
                if len(diff) > 0:
                    logger.warning("In the time-dependent Dataframe for attribute {} of network.{}_t the following snapshots are defined which are not in network.snapshots:\n{}".format(attr,c.list_name,diff))



	#check all dtypes of component attributes

        #this isn't strictly necessary (except for str)
        #since e.g. float == np.dtype("float64") is True
        #but we do it for easy reading of errors
        np_dtypes = {str : np.dtype("object"),
                     float : np.dtype("float64"),
                     int : np.dtype("int64"),
                     bool : np.dtype("bool")}

        for c in self.iterate_components():

            #first check static attributes

            types_soll = c.attrs["typ"][c.attrs["static"]].drop("name")

            dtypes_soll = types_soll.replace(np_dtypes)

            unmatched = (c.df.dtypes[dtypes_soll.index] != dtypes_soll)

            if unmatched.any():
                logger.warning("The following attributes of the dataframe {} have the wrong dtype:\n{}\nThey are:\n{}\nbut should be:\n{}".format(c.list_name,
                                                                                                                                         unmatched.index[unmatched],
                                                                                                                                         c.df.dtypes[unmatched],
                                                                                                                                         dtypes_soll[unmatched]))

            #now check varying attributes

            types_soll = c.attrs["typ"][c.attrs["varying"]]

            dtypes_soll = types_soll.replace(np_dtypes)

            for attr, typ in dtypes_soll.iteritems():
                if c.pnl[attr].empty:
                    continue

                unmatched = (c.pnl[attr].dtypes != typ)

                if unmatched.any():
                    logger.warning("The following columns of time-varying attribute {} in {}_t have the wrong dtype:\n{}\nThey are:\n{}\nbut should be:\n{}".format(attr,c.list_name,
                                                                                                                                  unmatched.index[unmatched],
                                                                                                                                  c.pnl[attr].dtypes[unmatched],
                                                                                                                                  typ))



class SubNetwork(Common):
    """
    Connected network of electric buses (AC or DC) with passive flows
    or isolated non-electric buses.

    Generated by network.determine_network_topology().

    """

    list_name = "sub_networks"

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
        for c in self.iterate_components(passive_branch_components):
            types += len(c.ind) * [c.name]
            names += list(c.ind)
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

    def iterate_components(self, components=None, skip_empty=True):
        for c in self.network.iterate_components(components=components, skip_empty=False):
            c = Component(*c[:-1], ind=getattr(self, c.list_name + '_i')())
            if not (skip_empty and len(c.ind) == 0):
                yield c


standard_types = {"LineType", "TransformerType"}

passive_one_port_components = {"ShuntImpedance"}
controllable_one_port_components = {"Load", "Generator", "StorageUnit", "Store"}
one_port_components = passive_one_port_components|controllable_one_port_components

passive_branch_components = {"Line", "Transformer"}
controllable_branch_components = {"Link"}
branch_components = passive_branch_components|controllable_branch_components

#i.e. everything except "Network"
all_components = branch_components|one_port_components|standard_types|{"Bus", "SubNetwork", "Carrier"}
