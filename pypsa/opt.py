

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

"""Tools to override slow Pyomo problem building.
"""

# make the code as Python 3 compatible as possible
from __future__ import print_function, division
from __future__ import absolute_import
from six.moves import range


from pyomo.environ import Constraint

import pyomo


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"




def l_constraint(model,name,constraints,*args):
    """A replacement for pyomo's Constraint that quickly builds linear
constraints.

Instead of

model.constraint_name = Constraint(index1,index2,...,rule=f)

call instead

l_constraint(model,constraint_name,const_dict,index1,index2,...)

where

const_dict is a dictionary of constraints of the form

const_dict[i] = [[(coeff1,var1),(coeff2,var2),...],sense,constant_term]

sense is one of "==","<=",">=".

I.e. variable coefficients are stored as a list of tuples.

    """

    setattr(model,name,Constraint(*args,noruleinit=True))
    v = getattr(model,name)
    for i in v._index:
        c = constraints[i]
        v._data[i] = pyomo.core.base.constraint._GeneralConstraintData(None,v)
        v._data[i]._body = pyomo.core.base.expr_coopr3._SumExpression()
        v._data[i]._body._args = [item[1] for item in c[0]]
        v._data[i]._body._coef = [item[0] for item in c[0]]
        v._data[i]._body._const = 0.
        if c[1] == "==":
            v._data[i]._equality = True
            v._data[i]._lower = pyomo.core.base.numvalue.NumericConstant(c[2])
            v._data[i]._upper = pyomo.core.base.numvalue.NumericConstant(c[2])
        elif c[1] == "<=":
            v._data[i]._equality = False
            v._data[i]._lower = None
            v._data[i]._upper = pyomo.core.base.numvalue.NumericConstant(c[2])
        elif c[1] == ">=":
            v._data[i]._equality = False
            v._data[i]._lower = pyomo.core.base.numvalue.NumericConstant(c[2])
            v._data[i]._upper = None
