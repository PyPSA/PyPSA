

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

