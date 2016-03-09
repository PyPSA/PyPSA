

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


from pyomo.environ import Constraint, Objective

import pyomo


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"


class LExpression(object):
    """Affine expression of optimisation variables.

    Affine expression of the form:

    constant + coeff1*var1 + coeff2*var2 + ....

    Parameters
    ----------
    variables : list of tuples of coefficients and variables
        e.g. [(coeff1,var1),(coeff2,var2),...]
    constant : float

    """

    def __init__(self,variables=None,constant=0.):

        if variables is None:
            self.variables = []
        else:
            self.variables = variables

        self.constant = constant

    def __repr__(self):
        return "{} + {}".format(self.variables, self.constant)



class LConstraint(object):
    """Constraint of optimisation variables.

    Linear constraint of the form:

    coeff1*var1 + coeff2*var2 + .... sense constant

    Parameters
    ----------
    variables : list of tuples of coefficients and variables
        e.g. [(coeff1,var1),(coeff2,var2),...]
    sense : string
        Must be one of "==","<=",">="
    constant : float

    """

    def __init__(self,variables=None,sense="==",constant=0.):

        if variables is None:
            self.variables = []
        else:
            self.variables = variables

        self.sense = sense
        self.constant = constant

    def __repr__(self):
        return "{} + {}".format(self.variables, self.constant)


def l_constraint(model,name,constraints,*args):
    """A replacement for pyomo's Constraint that quickly builds linear
    constraints.

    Instead of

    model.name = Constraint(index1,index2,...,rule=f)

    call instead

    l_constraint(model,name,constraints,index1,index2,...)

    where constraints is a dictionary of constraints of the form:

    constraints[i] = [[(coeff1,var1),(coeff2,var2),...],sense,constant_term]

    i.e. the first argument is a list of tuples with the variables and their
    coefficients, the second argument is the sense string (must be one of
    "==","<=",">=") and the third argument is the constant term (a float).

    Variables may be repeated with different coefficients, which pyomo
    will sum up.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
    name : string
        Name of constraints to be constructed
    constraints : dict
        A dictionary of constraints (see format above)
    *args :
        Indices of the constraints

    """

    setattr(model,name,Constraint(*args,noruleinit=True))
    v = getattr(model,name)
    for i in v._index:
        c = constraints[i]
        if type(c) is LConstraint:
            variables = c.variables
            sense = c.sense
            constant = c.constant
        else:
            variables = c[0]
            sense = c[1]
            constant = c[2]

        v._data[i] = pyomo.core.base.constraint._GeneralConstraintData(None,v)
        v._data[i]._body = pyomo.core.base.expr_coopr3._SumExpression()
        v._data[i]._body._args = [item[1] for item in variables]
        v._data[i]._body._coef = [item[0] for item in variables]
        v._data[i]._body._const = 0.
        if sense == "==":
            v._data[i]._equality = True
            v._data[i]._lower = pyomo.core.base.numvalue.NumericConstant(constant)
            v._data[i]._upper = pyomo.core.base.numvalue.NumericConstant(constant)
        elif sense == "<=":
            v._data[i]._equality = False
            v._data[i]._lower = None
            v._data[i]._upper = pyomo.core.base.numvalue.NumericConstant(constant)
        elif sense == ">=":
            v._data[i]._equality = False
            v._data[i]._lower = pyomo.core.base.numvalue.NumericConstant(constant)
            v._data[i]._upper = None


def l_objective(model,linear_part,constant=0.):
    """
    A replacement for pyomo's Objective that quickly builds linear
    objectives.

    Instead of

    model.objective = Objective(expr=sum(vars[i]*coeffs[i] for i in index)+constant)

    call instead

    l_objective(model,[(vars[i],coeffs[i]) for i in index],constant)


    Variables may be repeated with different coefficients, which pyomo
    will sum up.


    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
    linear_part : list of 2-tuples
    constant : float

    """

    #initialise with a dummy
    model.objective = Objective(expr = 0.)

    model.objective._expr = pyomo.core.base.expr_coopr3._SumExpression()
    model.objective._expr._args = [item[1] for item in linear_part]
    model.objective._expr._coef = [item[0] for item in linear_part]
    model.objective._expr._const = constant
