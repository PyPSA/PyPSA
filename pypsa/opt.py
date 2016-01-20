

#Tools to override slow Pyomo problem building



from pyomo.environ import Constraint

import pyomo


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
