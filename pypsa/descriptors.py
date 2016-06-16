

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

"""Descriptors for component attributes.
"""


# make the code as Python 3 compatible as possible
from __future__ import print_function, division
from __future__ import absolute_import
from six import iteritems


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"





#weak references are necessary to make sure the key-value pair are
#destroyed if the key object goes out of scope
from weakref import WeakKeyDictionary

from collections import OrderedDict

import networkx as nx

import pandas as pd

import inspect

from distutils.version import StrictVersion, LooseVersion
try:
    _nx_version = StrictVersion(nx.__version__)
except ValueError:
    _nx_version = LooseVersion(nx.__version__)

if _nx_version >= '1.12':
    class OrderedGraph(nx.MultiGraph):
        node_dict_factory = OrderedDict
        adjlist_dict_factory = OrderedDict
elif _nx_version >= '1.10':
    class OrderedGraph(nx.MultiGraph):
        node_dict_factory = OrderedDict
        adjlist_dict_factory = OrderedDict

        def __init__(self, data=None, **attr):
            self.node_dict_factory = ndf = self.node_dict_factory
            self.adjlist_dict_factory = self.adjlist_dict_factory
            self.edge_attr_dict_factory = self.edge_attr_dict_factory

            self.graph = {}   # dictionary for graph attributes
            self.node = ndf()  # empty node attribute dict
            self.adj = ndf()  # empty adjacency dict
            # attempt to load graph with data
            if data is not None:
                if isinstance(data, OrderedGraph):
                    try:
                        nx.convert.from_dict_of_dicts(
                            data.adj,
                            create_using=self,
                            multigraph_input=data.is_multigraph()
                        )
                        self.graph = data.graph.copy()
                        self.node.update((n,d.copy()) for n,d in data.node.items())
                    except:
                        raise nx.NetworkXError("Input is not a correct NetworkX graph.")
                else:
                    nx.convert.to_networkx_graph(data, create_using=self)
else:
    raise ImportError("NetworkX version {} is too old. At least 1.10 is needed.".format(nx.__version__))

#Some descriptors to control variables - idea is to do type checking
#and in future facilitate interface with Database / GUI

class SimpleDescriptor(object):
    #the name is set by Network.__init__
    name = None

    def __init__(self, default, restricted=None):
        self.default = default
        self.restricted = restricted

    def __get__(self, obj, cls):
        return getattr(obj.network, obj.__class__.list_name).at[obj.name, self.name]

    def __set__(self, obj, val):
        try:
            cast_value = self.typ(val)
            getattr(obj.network, obj.__class__.list_name).loc[obj.name, self.name] = cast_value
        except:
            print("could not convert '{}' to a '{}'".format(val, self.typ))

        if self.restricted is not None and cast_value not in self.restricted:
            print("'{}' not in list of acceptable entries: {}".format(cast_value, self.restricted))

class Float(SimpleDescriptor):
    """A descriptor to manage floats."""
    typ = float

class Integer(SimpleDescriptor):
    """A descriptor to manage integers."""
    typ = int

class Boolean(SimpleDescriptor):
    """A descriptor to manage booleans."""
    typ = bool

class String(SimpleDescriptor):
    """A descriptor to manage strings."""
    typ = str

class GraphDesc(object):

    typ = OrderedGraph

    #the name is set by Network.__init__
    name = None

    def __init__(self):
        self.values = WeakKeyDictionary()

    def __get__(self,obj,cls):
        try:
            return self.values[obj]
        except KeyError:
            graph = self.typ()
            self.values[obj] = graph
            return graph

    def __set__(self,obj,val):
        if not isinstance(val, self.typ):
            raise AttributeError("val must be an OrderedGraph")
        else:
            self.values[obj] = val

class Series(object):
    """A descriptor to manage series."""

    typ = pd.Series

    #the name is set by Network.__init__
    name = None

    def __init__(self, dtype=float, default=0.0):
        self.dtype = dtype
        self.default = default

    def __get__(self, obj, cls):
        return getattr(obj.network, obj.__class__.list_name+"_t").loc[self.name,:,obj.name]

    def __set__(self, obj, val):
        #following should work for ints, floats, numpy ints/floats, series and numpy arrays of right size
        try:
            getattr(obj.network,obj.__class__.list_name+"_t").loc[self.name,:,obj.name] = self.typ(data=val, index=obj.network.snapshots, dtype=self.dtype)
        except AttributeError:
            print("count not assign",val,"to series")




def get_descriptors(cls,allowed_descriptors=[]):
    d = OrderedDict()

    mro = list(inspect.getmro(cls))

    #make sure get closest descriptor in inheritance tree
    mro.reverse()

    for kls in mro:
        for k,v in iteritems(vars(kls)):
            if type(v) in allowed_descriptors:
                d[k] = v
    return d


simple_descriptors = [Integer, Float, String, Boolean]


def get_simple_descriptors(cls):
    return get_descriptors(cls,simple_descriptors)

def get_series_descriptors(cls):
    return get_descriptors(cls,[Series])
