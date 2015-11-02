

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





#weak references are necessary to make sure the key-value pair are
#destroyed if the key object goes out of scope
from weakref import WeakKeyDictionary

from collections import OrderedDict

from distutils.version import StrictVersion
 
import networkx as nx
assert StrictVersion(nx.__version__) >= '1.10', "NetworkX needs to be at least version 1.10"

import pandas as pd





class OrderedGraph(nx.Graph):
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict



#Some descriptors to control variables - idea is to do type checking
#and in future facilitate interface with Database / GUI

class Float(object):
    """A descriptor to manage floats."""

    def __init__(self,default=0.0):
        self.default = default
        self.values = WeakKeyDictionary()

    def __get__(self,obj,cls):
        return self.values.get(obj,self.default)

    def __set__(self,obj,val):
        try:
            self.values[obj] = float(val)
        except:
            print("could not convert",val,"to a float")
            self.val = self.default
            return

class OrderedDictDesc(object):
    def __init__(self):
        self.values = WeakKeyDictionary()

    def __get__(self,obj,cls):
        try:
            return self.values[obj]
        except KeyError:
            ordereddict = OrderedDict()
            self.values[obj] = ordereddict
            return ordereddict

    def __set__(self,obj,val):
        if not isinstance(val, OrderedDict):
            raise AttributeError("val must be an OrderedDict")
        else:
            self.values[obj] = val

class GraphDesc(object):
    def __init__(self):
        self.values = WeakKeyDictionary()

    def __get__(self,obj,cls):
        try:
            return self.values[obj]
        except KeyError:
            graph = OrderedGraph()
            self.values[obj] = graph
            return graph

    def __set__(self,obj,val):
        if not isinstance(val, nx.Graph):
            raise AttributeError("val must be an nx.Graph")
        else:
            self.values[obj] = val

class Series(object):
    """A descriptor to manage series."""

    def __init__(self, dtype=float, default=0.0):
        self.dtype = dtype
        self.default = default
        self.values = WeakKeyDictionary()

    def __get__(self, obj, cls):
        try:
            return self.values[obj]
        except KeyError:
            series = pd.Series(index=obj.network.index, data=self.default, dtype=self.dtype)
            self.values[obj] = series
            return series

    def __set__(self,obj,val):
        try:
            self.values[obj] = val.reindex(obj.network.index)
        except AttributeError:
            print("could not reindex to the network index")

class String(object):
    """A descriptor to manage strings."""

    def __init__(self,default="",restricted=None):
        self.default = default
        self.restricted = restricted
        self.values = WeakKeyDictionary()

    def __get__(self,obj,cls):
        return self.values.get(obj,self.default)

    def __set__(self,obj,val):
        try:
            self.values[obj] = str(val)
        except:
            print("could not convert",val,"to a string")
            return
        if self.restricted is not None:
            if self.values[obj] not in self.restricted:
                print(val,"not in list of acceptable entries:",self.restricted)

