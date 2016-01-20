

#Variable and Constraints to override slow Pyomo problem building



from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, Constraint, Reals, Suffix

from pyomo.opt import SolverFactory

import sys

import pyomo



#This class is never actually used - minimal speed improvement
class LVar(object):
    """A replacement for pyomo's Var. Instead of
model.var_name = Var(index1,index2,within=domain,bounds=f) call instead
LVar(model,var_name,index1,index2,within=domain,ub=ud,lb=ld) where ud and ld are dictionaries."""



    def __init__(self,model,name,*args, **kwd):
        domain = kwd.pop('within', Reals)
        domain = kwd.pop('domain', domain)
        bounds = kwd.pop('bounds', None)
        bs = {}
        bs["ub"] = kwd.pop('ub', None)
        bs["lb"] = kwd.pop('lb', None)
        setattr(model,name,Var(*args,within=domain))
        v = getattr(model,name)
        for b in ["ub","lb"]:
            if bs[b] is not None:
                for i in v._index:
                    setattr(v._data[i],"_"+b,bs[b][i])
        if bounds is not None:
            for i in v._index:
                setattr(v._data[i],"_lb",bounds[i][0])
                setattr(v._data[i],"_ub",bounds[i][1])


#This on the other hand is faster
class LConstraint(object):
    """A replacement for pyomo's Constraint. Instead of model.const_name =
Constraint(index1,index2,...,rule=f) call instead
LConstraint(model,var_name,cd,index1,index2,...) where cd is a dictionary
of constraints of the form cd[i] =
[[(coeff1,var1),(coeff2,var2),...],"==",const].

I.e. variable coefficients are stored as a list of tuples.

    """



    def __init__(self,model,name,constraints,*args):

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
