#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Copyright 2015-2021 PyPSA Developers

## You can find the list of PyPSA Developers at
## https://pypsa.readthedocs.io/en/latest/developers.html

## PyPSA is released under the open source MIT License, see
## https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt

"""
Tools for fast Linear Problem file writing. This module contains

- io functions for writing out variables, constraints and objective
  into a lp file.
- functions to create lp format based linear expression
- solver functions which read the lp file, run the problem and return the
  solution

This module supports the linear optimal power flow calculation without using
pyomo (see module linopt.py)
"""

__author__ = "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
__copyright__ = ("Copyright 2015-2021 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
                 "MIT License")


from .descriptors import Dict
import pandas as pd
import os
import logging, re, io, subprocess
import numpy as np
from pandas import IndexSlice as idx
from importlib.util import find_spec
from distutils.version import LooseVersion

logger = logging.getLogger(__name__)

# =============================================================================
# Front end functions
# =============================================================================

def define_variables(n, lower, upper, name, attr='', axes=None, spec='', mask=None):
    """
    Defines variable(s) for pypsa-network with given lower bound(s) and upper
    bound(s). The variables are stored in the network object under n.vars with
    key of the variable name. If multiple variables are defined at ones, at
    least one of lower and upper has to be an array (including pandas) of
    shape > (1,) or axes have to define the dimensions of the variables.

    Parameters
    ----------
    n : pypsa.Network
    lower : pd.Series/pd.DataFrame/np.array/str/float
        lower bound(s) for the variable(s)
    upper : pd.Series/pd.DataFrame/np.array/str/float
        upper bound(s) for the variable(s)
    name : str
        general name of the variable (or component which the variable is
        referring to). The variable will then be stored under:

            * n.vars[name].pnl if the variable is two-dimensional
            * n.vars[name].df if the variable is one-dimensional

        but can easily be accessed with :func:`get_var(n, name, attr)`
    attr : str default ''
        Specifying name of the variable, defines under which name the variable(s)
        are stored in n.vars[name].pnl if two-dimensional or in n.vars[name].df
        if one-dimensional
    axes : pd.Index or tuple of pd.Index objects, default None
        Specifies the axes and therefore the shape of the variables if bounds
        are single strings or floats. This is helpful when multiple variables
        have the same upper and lower bound.
    mask: pd.DataFrame/np.array
        Boolean mask with False values for variables which are skipped.
        The shape of the mask has to match the shape the added variables.


    Example
    --------

    Let's say we want to define a demand-side-managed load at each bus of
    network n, which has a minimum of 0 and a maximum of 10. We then define
    lower bound (lb) and upper bound (ub) and pass it to define_variables

    >>> from pypsa.linopt import define_variables, get_var
    >>> lb = pd.DataFrame(0, index=n.snapshots, columns=n.buses.index)
    >>> ub = pd.DataFrame(10, index=n.snapshots, columns=n.buses.index)
    >>> define_variables(n, lb, ub, 'DSM', 'variableload')

    Now the variables can be accessed by :func:`pypsa.linopt.get_var` using

    >>> variables = get_var(n, 'DSM', 'variableload')

    Note that this is usefull for the `extra_functionality` argument.
    """
    var = write_bound(n, lower, upper, axes, mask)
    set_varref(n, var, name, attr, spec=spec)
    return var


def define_binaries(n, axes, name, attr='',  spec='', mask=None):
    """
    Defines binary-variable(s) for pypsa-network. The variables are stored
    in the network object under n.vars with key of the variable name.
    For each entry for the pd.Series of pd.DataFrame spanned by the axes
    argument the function defines a binary.

    Parameters
    ----------
    n : pypsa.Network
    axes : pd.Index or tuple of pd.Index objects
        Specifies the axes and therefore the shape of the variables.
    name : str
        general name of the variable (or component which the variable is
        referring to). The variable will then be stored under:

            * n.vars[name].pnl if the variable is two-dimensional
            * n.vars[name].df if the variable is one-dimensional

    attr : str default ''
        Specifying name of the variable, defines under which name the variable(s)
        are stored in n.vars[name].pnl if two-dimensional or in n.vars[name].df
        if one-dimensional
    mask: pd.DataFrame/np.array
        Boolean mask with False values for variables which are skipped.
        The shape of the mask has to match the shape given by axes.

    See also
    ---------
    define_variables

    """
    var = write_binary(n, axes)
    set_varref(n, var, name, attr, spec=spec)
    return var


def define_constraints(n, lhs, sense, rhs, name, attr='', axes=None, spec='',
                       mask=None):
    """
    Defines constraint(s) for pypsa-network with given left hand side (lhs),
    sense and right hand side (rhs). The constraints are stored in the network
    object under n.cons with key of the constraint name. If multiple constraints
    are defined at ones, only using np.arrays, then the axes argument can be used
    for defining the axes for the constraints (this is especially recommended
    for time-dependent constraints). If one of lhs, sense and rhs is a
    pd.Series/pd.DataFrame the axes argument is not necessary.

    Parameters
    ----------
    n: pypsa.Network
    lhs: pd.Series/pd.DataFrame/np.array/str/float
        left hand side of the constraint(s), created with
        :func:`pypsa.linot.linexpr`.
    sense: pd.Series/pd.DataFrame/np.array/str/float
        sense(s) of the constraint(s)
    rhs: pd.Series/pd.DataFrame/np.array/str/float
        right hand side of the constraint(s), must only contain pure constants,
        no variables
    name: str
        general name of the constraint (or component which the constraint is
        referring to). The constraint will then be stored under:

            * n.cons[name].pnl if the constraint is two-dimensional
            * n.cons[name].df if the constraint is one-dimensional
    attr: str default ''
        Specifying name of the constraint, defines under which name the
        constraint(s) are stored in n.cons[name].pnl if two-dimensional or in
        n.cons[name].df if one-dimensional
    axes: pd.Index or tuple of pd.Index objects, default None
        Specifies the axes if all of lhs, sense and rhs are np.arrays or single
        strings or floats.
    mask: pd.DataFrame/np.array
        Boolean mask with False values for constraints which are skipped.
        The shape of the mask has to match the shape of the array that come out
        when combining lhs, sense and rhs.


    Example
    --------

    Let's say we want to constraint all gas generators to a maximum of 100 MWh
    during the first 10 snapshots. We then firstly get all operational variables
    for this subset and constraint there sum to less equal 100.

    >>> from pypsa.linopt import get_var, linexpr, define_constraints
    >>> gas_i = n.generators.query('carrier == "Natural Gas"').index
    >>> gas_vars = get_var(n, 'Generator', 'p').loc[n.snapshots[:10], gas_i]
    >>> lhs = linexpr((1, gas_vars)).sum().sum()
    >>> define_(n, lhs, '<=', 100, 'Generator', 'gas_power_limit')

    Now the constraint references can be accessed by
    :func:`pypsa.linopt.get_con` using

    >>> cons = get_var(n, 'Generator', 'gas_power_limit')

    Under the hood they are stored in n.cons.Generator.pnl.gas_power_limit.
    For retrieving their shadow prices add the general name of the constraint
    to the keep_shadowprices argument.

    Note that this is useful for the `extra_functionality` argument.

    """
    con = write_constraint(n, lhs, sense, rhs, axes, mask)
    set_conref(n, con, name, attr, spec=spec)
    return con

# =============================================================================
# writing functions
# =============================================================================

def _get_handlers(axes, *maybearrays):
    axes = [axes] if isinstance(axes, pd.Index) else axes
    if axes is None:
        axes, shape = broadcasted_axes(*maybearrays)
    else:
        shape = tuple(map(len, axes))
    size = np.prod(shape)
    return axes, shape, size


def write_bound(n, lower, upper, axes=None, mask=None):
    """
    Writer function for writing out multiple variables at a time. If lower and
    upper are floats it demands to give pass axes, a tuple of (index, columns)
    or (index), for creating the variable of same upper and lower bounds.
    Return a series or frame with variable references.
    """
    axes, shape, size = _get_handlers(axes, lower, upper)
    if not size: return pd.Series(dtype=float)
    n._xCounter += size
    variables = np.arange(n._xCounter - size, n._xCounter).reshape(shape)
    lower, upper = _str_array(lower), _str_array(upper)
    exprs = lower + ' <= x' + _str_array(variables, True) + ' <= '+ upper + '\n'
    if mask is not None:
        exprs = np.where(mask, exprs, '')
        variables = np.where(mask, variables, -1)
    n.bounds_f.write(join_exprs(exprs))
    return to_pandas(variables, *axes)

def write_constraint(n, lhs, sense, rhs, axes=None, mask=None):
    """
    Writer function for writing out multiple constraints to the corresponding
    constraints file. If lower and upper are numpy.ndarrays it axes must not be
    None but a tuple of (index, columns) or (index).
    Return a series or frame with constraint references.
    """
    axes, shape, size = _get_handlers(axes, lhs, sense, rhs)
    if not size: return pd.Series()
    n._cCounter += size
    cons = np.arange(n._cCounter - size, n._cCounter).reshape(shape)
    if isinstance(sense, str):
        sense = '=' if sense == '==' else sense
    lhs, sense, rhs = _str_array(lhs), _str_array(sense), _str_array(rhs)
    exprs = 'c' + _str_array(cons, True) + ':\n' + lhs + sense + ' ' + rhs + '\n\n'
    if mask is not None:
        exprs = np.where(mask, exprs, '')
        cons = np.where(mask, cons, -1)
    n.constraints_f.write(join_exprs(exprs))
    return to_pandas(cons, *axes)

def write_binary(n, axes, mask=None):
    """
    Writer function for writing out multiple binary-variables at a time.
    According to the axes it writes out binaries for each entry the pd.Series
    or pd.DataFrame spanned by axes. Returns a series or frame with variable
    references.
    """
    axes, shape, size = _get_handlers(axes)
    n._xCounter += size
    variables = np.arange(n._xCounter - size, n._xCounter).reshape(shape)
    exprs = 'x' + _str_array(variables, True) + '\n'
    if mask is not None:
        exprs = np.where(mask, exprs, '')
        variables = np.where(mask, variables, -1)
    n.binaries_f.write(join_exprs(exprs))
    return to_pandas(variables, *axes)


def write_objective(n, terms):
    """
    Writer function for writing out one or multiple objective terms.

    Parameters
    ----------
    n : pypsa.Network
    terms : str/numpy.array/pandas.Series/pandas.DataFrame
        String or array of strings which represent new objective terms, built
        with :func:`linexpr`

    """
    n.objective_f.write(join_exprs(terms))


# =============================================================================
# helpers, helper functions
# =============================================================================

def broadcasted_axes(*dfs):
    """
    Helper function which, from a collection of arrays, series, frames and other
    values, retrieves the axes of series and frames which result from
    broadcasting operations. It checks whether index and columns of given
    series and frames, repespectively, are aligned. Using this function allows
    to subsequently use pure numpy operations and keep the axes in the
    background.

    """
    axes = []
    shape = (1,)

    if set(map(type, dfs)) == {tuple}:
        dfs = sum(dfs, ())

    for df in dfs:
        shape = max(shape, np.asarray(df).shape)
        if isinstance(df, (pd.Series, pd.DataFrame)):
            if len(axes):
                assert (axes[-1] == df.axes[-1]).all(), ('Series or DataFrames '
                       'are not aligned. Please make sure that all indexes and '
                       'columns of Series and DataFrames going into the linear '
                       'expression are equally sorted.')
            axes = df.axes if len(df.axes) > len(axes) else axes
    return axes, shape


def align_with_static_component(n, c, attr):
    """
    Alignment of time-dependent variables with static components. If c is a
    pypsa.component name, it will sort the columns of the variable according
    to the static component.
    """
    if c in n.all_components and (c, attr) in n.variables.index:
        if not n.variables.pnl[c, attr]: return
        if len(n.vars[c].pnl[attr].columns) != len(n.df(c).index): return
        n.vars[c].pnl[attr] = n.vars[c].pnl[attr].reindex(columns=n.df(c).index)


def linexpr(*tuples, as_pandas=True, return_axes=False):
    """
    Elementwise concatenation of tuples in the form (coefficient, variables).
    Coefficient and variables can be arrays, series or frames. Per default
    returns a pandas.Series or pandas.DataFrame of strings. If return_axes
    is set to True the return value is split into values and axes, where values
    are the numpy.array and axes a tuple containing index and column if present.

    Parameters
    ----------
    tuples: tuple of tuples
        Each tuple must of the form (coeff, var), where

        * coeff is a numerical  value, or a numerical array, series, frame
        * var is a str or a array, series, frame of variable strings
    as_pandas : bool, default True
        Whether to return to resulting array as a series, if 1-dimensional, or
        a frame, if 2-dimensional. Supersedes return_axes argument.
    return_axes: Boolean, default False
        Whether to return index and column (if existent)

    Example
    -------
    Initialize coefficients and variables

    >>> coeff1 = 1
    >>> var1 = pd.Series(['a1', 'a2', 'a3'])
    >>> coeff2 = pd.Series([-0.5, -0.3, -1])
    >>> var2 = pd.Series(['b1', 'b2', 'b3'])

    Create the linear expression strings

    >>> linexpr((coeff1, var1), (coeff2, var2))
    0    +1.0 a1 -0.5 b1
    1    +1.0 a2 -0.3 b2
    2    +1.0 a3 -1.0 b3
    dtype: object

    For a further step the resulting frame can be used as the lhs of
    :func:`pypsa.linopt.define_constraints`

    For retrieving only the values:

    >>> linexpr((coeff1, var1), (coeff2, var2), as_pandas=False)
    array(['+1.0 a1 -0.5 b1', '+1.0 a2 -0.3 b2', '+1.0 a3 -1.0 b3'], dtype=object)

    """
    axes, shape = broadcasted_axes(*tuples)
    expr = np.repeat('', np.prod(shape)).reshape(shape).astype(object)
    if np.prod(shape):
        for coeff, var in tuples:
            expr = expr + _str_array(coeff) + ' x' + _str_array(var, True) + '\n'
            if isinstance(expr, np.ndarray):
                isna = np.isnan(coeff) | np.isnan(var) | (var == -1)
                expr = np.where(isna, '', expr)
    if return_axes:
        return (expr, *axes)
    if as_pandas:
        return to_pandas(expr, *axes)
    return expr


def to_pandas(array, *axes):
    """
    Convert a numpy array to pandas.Series if 1-dimensional or to a
    pandas.DataFrame if 2-dimensional. Provide index and columns if needed
    """
    return pd.Series(array, *axes) if array.ndim == 1 else pd.DataFrame(array, *axes)

_to_float_str = lambda f: '%+f'%f
_v_to_float_str = np.vectorize(_to_float_str, otypes=[object])

_to_int_str = lambda d: '%d'%d
_v_to_int_str = np.vectorize(_to_int_str, otypes=[object])

def _str_array(array, integer_string=False):
    if isinstance(array, (float, int)):
        if integer_string:
            return _to_int_str(array)
        return _to_float_str(array)
    array = np.asarray(array)
    if array.dtype.type == np.str_:
        array = np.asarray(array, dtype=object)
    if array.dtype < str and array.size:
        if integer_string:
            array = np.nan_to_num(array, False, -1)
            return _v_to_int_str(array)
        return _v_to_float_str(array)
    else:
        return array

def join_exprs(df):
    """
    Helper function to join arrays, series or frames of strings together.

    """
    return ''.join(np.asarray(df).flatten())

# =============================================================================
#  references to vars and cons, rewrite this part to not store every reference
# =============================================================================

def _add_reference(ref_dict, df, attr, pnl=True):
    if pnl:
        if attr in ref_dict.pnl:
            ref_dict.pnl[attr][df.columns] = df
        else:
            ref_dict.pnl[attr] = df
    else:
        if attr in ref_dict.df:
            ref_dict.df = pd.concat([ref_dict.df, df.to_frame(attr)])
        else:
            ref_dict.df[attr] = df


def set_varref(n, variables, c, attr, spec=''):
    """
    Sets variable references to the network.
    One-dimensional variable references will be collected at `n.vars[c].df`,
    two-dimensional varaibles in `n.vars[c].pnl`. For example:

    * nominal capacity variables for generators are stored in
      `n.vars.Generator.df.p_nom`
    * operational variables for generators are stored in
      `n.vars.Generator.pnl.p`
    """
    if not variables.empty:
        pnl = variables.ndim == 2
        if c not in n.variables.index:
            n.vars[c] = Dict(df=pd.DataFrame(), pnl=Dict())
        if ((c, attr) in n.variables.index) and (spec != ''):
            n.variables.at[idx[c, attr], 'specification'] += ', ' + spec
        else:
            n.variables.loc[idx[c, attr], :] = [pnl, spec]
        _add_reference(n.vars[c], variables, attr, pnl=pnl)

def set_conref(n, constraints, c, attr, spec=''):
    """
    Sets constraint references to the network.
    One-dimensional constraint references will be collected at `n.cons[c].df`,
    two-dimensional in `n.cons[c].pnl`
    For example:

    * constraints for nominal capacity variables for generators are stored in
      `n.cons.Generator.df.mu_upper`
    * operational capacity limits for generators are stored in
      `n.cons.Generator.pnl.mu_upper`
    """
    if not constraints.empty:
        pnl = constraints.ndim == 2
        if c not in n.constraints.index:
            n.cons[c] = Dict(df=pd.DataFrame(), pnl=Dict())
        if ((c, attr) in n.constraints.index) and (spec != ''):
            n.constraints.at[idx[c, attr], 'specification'] += ', ' + spec
        else:
            n.constraints.loc[idx[c, attr], :] = [pnl, spec]
        _add_reference(n.cons[c], constraints, attr, pnl=pnl)

def get_var(n, c, attr, pop=False):
    """
    Retrieves variable references for a given static or time-depending
    attribute of a given component. The function looks into n.variables to
    detect whether the variable is a time-dependent or static.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        component name to which the constraint belongs
    attr: str
        attribute name of the constraints

    Example
    -------
    >>> get_var(n, 'Generator', 'p')

    """
    vvars = n.vars[c].pnl if n.variables.pnl[c, attr] else n.vars[c].df
    return vvars.pop(attr) if pop else vvars[attr]


def get_con(n, c, attr, pop=False):
    """
    Retrieves constraint references for a given static or time-depending
    attribute of a give component.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        component name to which the constraint belongs
    attr: str
        attribute name of the constraints

    Example
    -------
    get_con(n, 'Generator', 'mu_upper')
    """
    cons = n.cons[c].pnl if n.constraints.pnl[c, attr] else n.cons[c].df
    return cons.pop(attr) if pop else cons[attr]


def get_sol(n, name, attr=''):
    """
    Retrieves solution for a given variable. Note that a lookup of all stored
    solutions is given in n.solutions.


    Parameters
    ----------
    n : pypsa.Network
    c : str
        general variable name (or component name if variable is attached to a
        component)
    attr: str
        attribute name of the variable

    Example
    -------
    get_dual(n, 'Generator', 'mu_upper')
    """
    pnl = n.solutions.at[(name, attr), 'pnl']
    if n.solutions.at[(name, attr), 'in_comp']:
        return n.pnl(name)[attr] if pnl else n.df(name)[attr + '_opt']
    else:
        return n.sols[name].pnl[attr] if pnl else n.sols[name].df[attr]


def get_dual(n, name, attr=''):
    """
    Retrieves shadow price for a given constraint. Note that for retrieving
    shadow prices of a custom constraint, its name has to be passed to
    `keep_references` in the lopf, or `keep_references` has to be set to True.
    Note that a lookup of all stored shadow prices is given in n.dualvalues.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        constraint name to which the constraint belongs
    attr: str
        attribute name of the constraints

    Example
    -------
    get_dual(n, 'Generator', 'mu_upper')
    """
    pnl = n.dualvalues.at[(name, attr), 'pnl']
    if n.dualvalues.at[(name, attr), 'in_comp']:
        return n.pnl(name)[attr] if pnl else n.df(name)[attr]
    else:
        return n.duals[name].pnl[attr] if pnl else n.duals[name].df[attr]


# =============================================================================
# solvers
# =============================================================================

def set_int_index(ser):
    ser.index = ser.index.str[1:].astype(int)
    return ser

def run_and_read_cbc(n, problem_fn, solution_fn, solver_logfile,
                     solver_options, warmstart=None, store_basis=True):
    """
    Solving function. Reads the linear problem file and passes it to the cbc
    solver. If the solution is sucessful it returns variable solutions and
    constraint dual values.

    For more information on the solver options, run 'cbc' in your shell
    """
    #printingOptions is about what goes in solution file
    command = f"cbc -printingOptions all -import {problem_fn} "
    if warmstart:
        command += f'-basisI {warmstart} '
    if (solver_options is not None) and (solver_options != {}):
        command += solver_options
    command += f"-solve -solu {solution_fn} "
    if store_basis:
        n.basis_fn = solution_fn.replace('.sol', '.bas')
        command += f'-basisO {n.basis_fn} '

    if not os.path.exists(solution_fn):
        os.mknod(solution_fn)

    log = open(solver_logfile, 'w') if solver_logfile is not None else subprocess.PIPE
    result = subprocess.Popen(command.split(' '), stdout=log)
    result.wait()

    with open(solution_fn, "r") as f:
        data = f.readline()


    if data.startswith("Optimal - objective value"):
        status = "ok"
        termination_condition = "optimal"
        objective = float(data[len("Optimal - objective value "):])
    elif "Infeasible" in data:
        status = "warning"
        termination_condition = "infeasible"
    else:
        status = 'warning'
        termination_condition = "other"

    if termination_condition != "optimal":
        return status, termination_condition, None, None, None

    f = open(solution_fn,"rb")
    trimed_sol_fn = re.sub(rb'\*\*\s+', b'', f.read())
    f.close()

    sol = pd.read_csv(io.BytesIO(trimed_sol_fn), header=None, skiprows=[0],
                      sep=r'\s+', usecols=[1,2,3], index_col=0)
    variables_b = sol.index.str[0] == 'x'
    variables_sol = sol[variables_b][2].pipe(set_int_index)
    constraints_dual = sol[~variables_b][3].pipe(set_int_index)
    return (status, termination_condition, variables_sol,
            constraints_dual, objective)


def run_and_read_glpk(n, problem_fn, solution_fn, solver_logfile,
                     solver_options, warmstart=None, store_basis=True):
    """
    Solving function. Reads the linear problem file and passes it to the glpk
    solver. If the solution is sucessful it returns variable solutions and
    constraint dual values.

    For more information on the glpk solver options:
    https://kam.mff.cuni.cz/~elias/glpk.pdf
    """
    # TODO use --nopresol argument for non-optimal solution output
    command = (f"glpsol --lp {problem_fn} --output {solution_fn}")
    if solver_logfile is not None:
        command += f' --log {solver_logfile}'
    if warmstart:
        command += f' --ini {warmstart}'
    if store_basis:
        n.basis_fn = solution_fn.replace('.sol', '.bas')
        command += f' -w {n.basis_fn}'
    if (solver_options is not None) and (solver_options != {}):
        command += solver_options

    result = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE)
    result.wait()

    f = open(solution_fn)
    def read_until_break(f):
        linebreak = False
        while not linebreak:
            line = f.readline()
            linebreak = line == '\n'
            yield line

    info = io.StringIO(''.join(read_until_break(f))[:-2])
    info = pd.read_csv(info, sep=':',  index_col=0, header=None)[1]
    termination_condition = info.Status.lower().strip()
    objective = float(re.sub(r'[^0-9\.\+\-e]+', '', info.Objective))

    if termination_condition in ["optimal","integer optimal"]:
        status = "ok"
        termination_condition = "optimal"
    elif termination_condition == "undefined":
        status = "warning"
        termination_condition = "infeasible"
    else:
        status = "warning"

    if termination_condition != 'optimal':
        return status, termination_condition, None, None, None

    duals = io.StringIO(''.join(read_until_break(f))[:-2])
    duals = pd.read_fwf(duals)[1:].set_index('Row name')
    if 'Marginal' in duals:
        constraints_dual = pd.to_numeric(duals['Marginal'], 'coerce')\
                              .fillna(0).pipe(set_int_index)
    else:
        logger.warning("Shadow prices of MILP couldn't be parsed")
        constraints_dual = pd.Series(index=duals.index, dtype=float)

    sol = io.StringIO(''.join(read_until_break(f))[:-2])
    variables_sol = (pd.read_fwf(sol)[1:].set_index('Column name')
                     ['Activity'].astype(float).pipe(set_int_index))
    f.close()

    return (status, termination_condition, variables_sol,
            constraints_dual, objective)


def run_and_read_cplex(n, problem_fn, solution_fn, solver_logfile,
                        solver_options, warmstart=None, store_basis=True):
    """
    Solving function. Reads the linear problem file and passes it to the cplex
    solver. If the solution is sucessful it returns variable solutions and
    constraint dual values. Cplex must be installed for using this function

    """
    if find_spec('cplex') is None:
        raise ModuleNotFoundError("Optional dependency 'cplex' not found."
           "Install via 'conda install -c ibmdecisionoptimization cplex' "
           "or 'pip install cplex'")
    import cplex
    _version = LooseVersion(cplex.__version__)
    m = cplex.Cplex()
    if solver_logfile is not None:
        if _version >= "12.10":
            log_file_or_path = open(solver_logfile, "w")
        else:
            log_file_or_path = solver_logfile
        m.set_log_stream(log_file_or_path)
    if solver_options is not None:
        for key, value in solver_options.items():
            param = m.parameters
            for key_layer in key.split("."):
                param = getattr(param, key_layer)
            param.set(value)
    m.read(problem_fn)
    if warmstart:
        m.start.read_basis(warmstart)
    m.solve()
    is_lp = m.problem_type[m.get_problem_type()] == 'LP'
    if solver_logfile is not None:
        if isinstance(log_file_or_path, io.IOBase):
            log_file_or_path.close()

    termination_condition = m.solution.get_status_string()
    if 'optimal' in termination_condition:
        status = 'ok'
        termination_condition = 'optimal'
    else:
        status = 'warning'

    if (status == 'ok') and store_basis and is_lp:
        n.basis_fn = solution_fn.replace('.sol', '.bas')
        try:
            m.solution.basis.write(n.basis_fn)
        except cplex.exceptions.errors.CplexSolverError:
            logger.info('No model basis stored')
            del n.basis_fn

    objective = m.solution.get_objective_value()
    variables_sol = pd.Series(m.solution.get_values(), m.variables.get_names())\
                      .pipe(set_int_index)
    if is_lp:
        constraints_dual = pd.Series(m.solution.get_dual_values(),
                         m.linear_constraints.get_names()).pipe(set_int_index)
    else:
        logger.warning("Shadow prices of MILP couldn't be parsed")
        constraints_dual = pd.Series(index=m.linear_constraints.get_names())\
                             .pipe(set_int_index)
    del m
    return (status, termination_condition, variables_sol, constraints_dual,
            objective)


def run_and_read_gurobi(n, problem_fn, solution_fn, solver_logfile,
                        solver_options, warmstart=None, store_basis=True):
    """
    Solving function. Reads the linear problem file and passes it to the gurobi
    solver. If the solution is sucessful it returns variable solutions and
    constraint dual values. Gurobipy must be installed for using this function

    For more information on solver options:
    https://www.gurobi.com/documentation/{gurobi_verion}/refman/parameter_descriptions.html
    """
    if find_spec('gurobipy') is None:
        raise ModuleNotFoundError("Optional dependency 'gurobipy' not found. "
           "Install via 'conda install -c gurobi gurobi'  or follow the "
           "instructions on the documentation page "
           "https://www.gurobi.com/documentation/")
    import gurobipy
    # disable logging for this part, as gurobi output is doubled otherwise
    logging.disable(50)

    m = gurobipy.read(problem_fn)
    if solver_options is not None:
        for key, value in solver_options.items():
            m.setParam(key, value)
    if solver_logfile is not None:
        m.setParam("logfile", solver_logfile)

    if warmstart:
        m.read(warmstart)
    m.optimize()
    logging.disable(1)

    if store_basis:
        n.basis_fn = solution_fn.replace('.sol', '.bas')
        try:
            m.write(n.basis_fn)
        except gurobipy.GurobiError:
            logger.info('No model basis stored')
            del n.basis_fn

    Status = gurobipy.GRB.Status
    statusmap = {getattr(Status, s) : s.lower() for s in Status.__dir__()
                                                if not s.startswith('_')}
    termination_condition = statusmap[m.status]

    if termination_condition == "optimal":
        status = 'ok'
    elif termination_condition == 'suboptimal':
        status = 'warning'
    elif termination_condition == "infeasible":
        status = 'warning'
    elif termination_condition == "inf_or_unbd":
        status = 'warning'
        termination_condition = "infeasible or unbounded"
    else:
        status = "warning"

    if termination_condition not in ["optimal","suboptimal"]:
        return status, termination_condition, None, None, None

    variables_sol = pd.Series({v.VarName: v.x for v
                               in m.getVars()}).pipe(set_int_index)
    try:
        constraints_dual = pd.Series({c.ConstrName: c.Pi for c in
                                      m.getConstrs()}).pipe(set_int_index)
    except AttributeError:
        logger.warning("Shadow prices of MILP couldn't be parsed")
        constraints_dual = pd.Series(index=[c.ConstrName for c in m.getConstrs()])
    objective = m.ObjVal
    del m
    return (status, termination_condition, variables_sol,
            constraints_dual, objective)


def run_and_read_xpress(n, problem_fn, solution_fn, solver_logfile,
                        solver_options, keep_files, warmstart=None,
                        store_basis=True):
    """
    Solving function. Reads the linear problem file and passes it to
    the Xpress solver. If the solution is successful it returns
    variable solutions and constraint dual values. The xpress module
    must be installed for using this function.

    For more information on solver options:
    https://www.fico.com/fico-xpress-optimization/docs/latest/solver/GUID-ACD7E60C-7852-36B7-A78A-CED0EA291CDD.html
    """

    import xpress

    m = xpress.problem()

    m.read(problem_fn)
    m.setControl(solver_options)

    if solver_logfile is not None:
        m.setlogfile(solver_logfile)

    if warmstart:
        m.readbasis(warmstart)

    m.solve()

    if store_basis:
        n.basis_fn = solution_fn.replace('.sol', '.bas')
        try:
            m.writebasis(n.basis_fn)
        except:
            logger.info('No model basis stored')
            del n.basis_fn

    termination_condition = m.getProbStatusString()

    if termination_condition == 'mip_optimal' or \
       termination_condition == 'lp_optimal':
        status = 'ok'
        termination_condition = 'optimal'
    elif termination_condition == 'mip_unbounded' or \
         termination_condition == 'mip_infeasible' or \
         termination_condition == 'lp_unbounded' or \
         termination_condition == 'lp_infeasible':
        status = 'infeasible or unbounded'
    else:
        status = 'warning'

    if termination_condition not in ["optimal"]:
        return status, termination_condition, None, None, None

    var = [str(v) for v in m.getVariable()]
    variables_sol = pd.Series(m.getSolution(var), index=var).pipe(set_int_index)

    try:
        dual = [str(d) for d in m.getConstraint()]
        constraints_dual = pd.Series(m.getDual(dual), index=dual).pipe(set_int_index)
    except xpress.SolverError:
        logger.warning("Shadow prices of MILP couldn't be parsed")
        constraints_dual = pd.Series(index=dual).pipe(set_int_index)

    objective = m.getObjVal()

    del m

    return (status, termination_condition, variables_sol,
            constraints_dual, objective)
