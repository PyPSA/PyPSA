
## Copyright 2015-2021 PyPSA Developers

## You can find the list of PyPSA Developers at
## https://pypsa.readthedocs.io/en/latest/developers.html

## PyPSA is released under the open source MIT License, see
## https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt

"""
Tools for fast Pyomo linear problem building.

Essentially this library replaces Pyomo expressions with more strict
objects with a pre-defined affine structure.
"""

__author__ = "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
__copyright__ = ("Copyright 2015-2021 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
                 "MIT License")

import logging
logger = logging.getLogger(__name__)


from pyomo.environ import Constraint, Objective, Var, ComponentUID, minimize
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.logical_expr import inequality
from pyomo.core.base.constraint import _GeneralConstraintData

import pyomo
from contextlib import contextmanager
from six.moves import cPickle as pickle
import gc, os, tempfile


# =============================================================================
# Tools for solving with pyomo
# =============================================================================

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


    def __mul__(self,constant):
        try:
            constant = float(constant)
        except:
            logger.error("Can only multiply an LExpression with a float!")
            return None
        return LExpression([(constant*item[0],item[1]) for item in self.variables],
                           constant*self.constant)

    def __rmul__(self,constant):
        return self.__mul__(constant)

    def __add__(self,other):
        if isinstance(other, LExpression):
            return LExpression(self.variables + other.variables,self.constant+other.constant)
        else:
            try:
                constant = float(other)
            except:
                logger.error("Can only add an LExpression to another LExpression or a constant!")
                return None
            return LExpression(self.variables[:],self.constant+constant)


    def __radd__(self,other):
        return self.__add__(other)

    def __pos__(self):
        return self

    def __neg__(self):
        return -1*self

class LConstraint(object):
    """Constraint of optimisation variables.

    Linear constraint of the form:

    lhs sense rhs

    Parameters
    ----------
    lhs : LExpression
    sense : string
    rhs : LExpression

    """

    def __init__(self,lhs=None,sense="==",rhs=None):

        if lhs is None:
            self.lhs = LExpression()
        else:
            self.lhs = lhs

        self.sense = sense

        if rhs is None:
            self.rhs = LExpression()
        else:
            self.rhs = rhs

    def __repr__(self):
        return "{} {} {}".format(self.lhs, self.sense, self.rhs)

def _build_sum_expression(variables, constant=0.):
    expr = LinearExpression()
    expr.linear_vars = [item[1] for item in variables]
    expr.linear_coefs = [item[0] for item in variables]
    expr.constant = constant
    return expr

def l_constraint(model,name,constraints,*args):
    """A replacement for pyomo's Constraint that quickly builds linear
    constraints.

    Instead of

    model.name = Constraint(index1,index2,...,rule=f)

    call instead

    l_constraint(model,name,constraints,index1,index2,...)

    where constraints is a dictionary of constraints of the form:

    constraints[i] = LConstraint object

    OR using the soon-to-be-deprecated list format:

    constraints[i] = [[(coeff1,var1),(coeff2,var2),...],sense,constant_term]

    i.e. the first argument is a list of tuples with the variables and their
    coefficients, the second argument is the sense string (must be one of
    "==","<=",">=","><") and the third argument is the constant term
    (a float). The sense "><" allows lower and upper bounds and requires
    `constant_term` to be a 2-tuple.

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
        if isinstance(c, LConstraint):
            variables = c.lhs.variables + [(-item[0],item[1]) for item in c.rhs.variables]
            sense = c.sense
            constant = c.rhs.constant - c.lhs.constant
        else:
            variables = c[0]
            sense = c[1]
            constant = c[2]

        sum_expr = _build_sum_expression(variables)
        
        if sense == "==":
            constr_expr = sum_expr == constant
        elif sense == "<=":
            constr_expr = sum_expr <= constant
        elif sense == ">=":
            constr_expr = sum_expr >= constant
        elif sense == "><":
            lo, hi = constant
            constr_expr = inequality(lo, sum_expr, hi)
        else:
            raise KeyError('`sense` must be one of "==","<=",">=","><"; got: {}'.format(sense))

        v._data[i] = _GeneralConstraintData(constr_expr, v)


def l_objective(model,objective=None, sense=minimize):
    """
    A replacement for pyomo's Objective that quickly builds linear
    objectives.

    Instead of

    model.objective = Objective(expr=sum(vars[i]*coeffs[i] for i in index)+constant)

    call instead

    l_objective(model,objective,sense)

    where objective is an LExpression.

    Variables may be repeated with different coefficients, which pyomo
    will sum up.


    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
    objective : LExpression
    sense : minimize / maximize

    """

    if objective is None:
        objective = LExpression()

    #initialise with a dummy
    model.objective = Objective(expr = 0., sense=sense)
    model.objective._expr = _build_sum_expression(objective.variables, constant=objective.constant)

def free_pyomo_initializers(obj):
    obj.construct()
    if isinstance(obj, Var):
        attrs = ('_bounds_init_rule', '_bounds_init_value',
                 '_domain_init_rule', '_domain_init_value',
                 '_value_init_rule', '_value_init_value')
    elif isinstance(obj, Constraint):
        attrs = ('rule', '_init_expr')
    else:
        raise NotImplementedError

    for attr in attrs:
        if hasattr(obj, attr):
            setattr(obj, attr, None)

@contextmanager
def empty_model(model):
    logger.debug("Storing pyomo model to disk")
    rules = {}
    for obj in model.component_objects(ctype=Constraint):
        if obj.rule is not None:
            rules[obj.name] = obj.rule
            obj.rule = None

    bounds = {}
    for obj in model.component_objects(ctype=Var):
        if obj._bounds_init_rule is not None:
            bounds[obj.name] = obj._bounds_init_rule
            obj._bounds_init_rule = None

    fd, fn = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(model.__getstate__(), f, -1)

    model.__dict__.clear()
    logger.debug("Stored pyomo model to disk")

    gc.collect()
    yield

    logger.debug("Reloading pyomo model")
    with open(fn, 'rb') as f:
        state = pickle.load(f)
    os.remove(fn)
    model.__setstate__(state)

    for n, rule in rules.items():
        getattr(model, n).rule = rule

    for n, bound in bounds.items():
        getattr(model, n)._bounds_init_rule = bound

    logger.debug("Reloaded pyomo model")

@contextmanager
def empty_network(network):
    logger.debug("Storing pypsa timeseries to disk")

    panels = {}
    for c in network.all_components:
        attr = network.components[c]["list_name"] + "_t"
        panels[attr] = getattr(network, attr)
        setattr(network, attr, None)

    fd, fn = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(panels, f, -1)

    del panels

    gc.collect()
    yield

    logger.debug("Reloading pypsa timeseries from disk")
    with open(fn, 'rb') as f:
        panels = pickle.load(f)
    os.remove(fn)
    for attr, pnl in panels.items():
        setattr(network, attr, pnl)

def patch_optsolver_free_model_before_solving(opt, model):
    orig_apply_solver = opt._apply_solver
    def wrapper():
        with empty_model(model):
            return orig_apply_solver()
    opt._apply_solver = wrapper

def patch_optsolver_record_memusage_before_solving(opt, network):
    try:
        import resource

        orig_apply_solver = opt._apply_solver
        def wrapper():
            network.max_memusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return orig_apply_solver()
        opt._apply_solver = wrapper
        return True
    except ImportError:
        logger.debug("Unable to measure memory usage, since the resource library is missing")
        return False


# =============================================================================
# Helpers for opf_lowmemory
# =============================================================================
