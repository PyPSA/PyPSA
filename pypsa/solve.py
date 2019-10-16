## Copyright 2019 Tom Brown (KIT), Fabian Hofmann (FIAS)

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

"""nomopyomo: build optimisation problems from PyPSA networks without
Pyomo. nomopyomo = no more Pyomo."""

from .descriptors import get_switchable_as_dense as get_as_dense
from .pf import _as_snapshots

import pandas as pd
from pandas import IndexSlice as idx
import numpy as np

import gc, string, random, time, os, re, subprocess, io

import logging
logger = logging.getLogger(__name__)


lookup = pd.read_csv(os.path.join(os.path.dirname(__file__), 'variables.csv'),
                     index_col=['component', 'variable'])
nominals = lookup.query('nominal').reset_index(level='variable').variable

# =============================================================================
# writing functions
# =============================================================================

xCounter = 0
cCounter = 0
def reset_counter():
    global xCounter, cCounter
    xCounter, cCounter = 0, 0


def write_bound(n, lower, upper, axes=None):
    """
    Writer function for writing out mutliple variables at a time. If lower and
    upper are floats it demands to give pass axes, a tuple of (index, columns)
    or (index), for creating the variable of same upper and lower bounds.
    Return a series or frame with variable references.
    """
    axes = [axes] if isinstance(axes, pd.Index) else axes
    if axes is None:
        axes, shape = broadcasted_axes(lower, upper)
    else:
        shape = tuple(map(len, axes))
    ser_or_frame = pd.DataFrame if len(shape) > 1 else pd.Series
    length = np.prod(shape)
    global xCounter
    xCounter += length
    variables = np.array([f'x{x}' for x in range(xCounter - length, xCounter)],
                          dtype=object).reshape(shape)
    lower, upper = _str_array(lower), _str_array(upper)
    for s in (lower + ' <= '+ variables + ' <= '+ upper + '\n').flatten():
        n.bounds_f.write(s)
    return ser_or_frame(variables, *axes)

def write_constraint(n, lhs, sense, rhs, axes=None):
    """
    Writer function for writing out mutliple constraints to the corresponding
    constraints file. If lower and upper are numpy.ndarrays it axes must not be
    None but a tuple of (index, columns) or (index).
    Return a series or frame with constraint references.
    """
    axes = [axes] if isinstance(axes, pd.Index) else axes
    if axes is None:
        axes, shape = broadcasted_axes(lhs, rhs)
    else:
        shape = tuple(map(len, axes))
    ser_or_frame = pd.DataFrame if len(shape) > 1 else pd.Series
    length = np.prod(shape)
    global cCounter
    cCounter += length
    cons = np.array([f'c{x}' for x in range(cCounter - length, cCounter)],
                            dtype=object).reshape(shape)
    if isinstance(sense, str):
        sense = '=' if sense == '==' else sense
    lhs, sense, rhs = _str_array(lhs), _str_array(sense), _str_array(rhs)
    for c in (cons + ':\n' + lhs + sense + '\n' + rhs + '\n\n').flatten():
        n.constraints_f.write(c)
    return ser_or_frame(cons, *axes)


# =============================================================================
# helpers, helper functions
# =============================================================================

var_ref_suffix = '_varref' # after solving replace with '_opt'
con_ref_suffix = '_conref' # after solving replace with ''

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
    shape = ()
    for df in dfs:
        if isinstance(df, (pd.Series, pd.DataFrame)):
            if len(axes):
                assert (axes[-1] == df.axes[-1]).all(), ('Series or DataFrames '
                       'are not aligned')
            axes = df.axes if len(df.axes) > len(axes) else axes
            shape = tuple(map(len, axes))
    return axes, shape


def linexpr(*tuples, return_axes=False):
    """
    Elementwise concatenation of tuples in the form (coefficient, variables).
    Coefficient and variables can be arrays, series or frames. Returns
    a np.ndarray of strings. If return_axes is set to True and a pd.Series or
    pd.DataFrame was past, the corresponding index (and column if existent) is
    returned additionaly.

    Parameters
    ----------
    tulples: tuple of tuples
        Each tuple must of the form (coeff, var), where
            * coeff is a numerical  value, or a numeical array, series, frame
            * var is a str or a array, series, frame of variable strings
    return_axes: Boolean, default False
        Whether to return index and column (if existent)

    Example
    -------
    >>> coeff1 = 1
    >>> var1 = pd.Series(['a1', 'a2', 'a3'])
    >>> coeff2 = pd.Series([-0.5, -0.3, -1])
    >>> var2 = pd.Series(['b1', 'b2', 'b3'])

    >>> linexpr((coeff1, var1), (coeff2, var2))
    array(['+1.0 a1\n-0.5 b1\n', '+1.0 a2\n-0.3 b2\n', '+1.0 a3\n-1.0 b3\n'],
      dtype=object)


    For turning the result into a series or frame again:
    >>> pd.Series(*linexpr((coeff1, var1), (coeff2, var2), return_axes=True))
    0    +1.0 a1\n-0.5 b1\n
    1    +1.0 a2\n-0.3 b2\n
    2    +1.0 a3\n-1.0 b3\n
    dtype: object

    This can also be applied to DataFrames, using
    pd.DataFrame(*linexpr(..., return_axes=True)).
    """
    axes, shape = broadcasted_axes(*sum(tuples, ()))
    expr = np.repeat('', np.prod(shape)).reshape(shape).astype(object)
    if np.prod(shape):
        for coeff, var in tuples:
            expr += _str_array(coeff) + _str_array(var) + '\n'
    if return_axes:
        return (expr, *axes)
    return expr


def _str_array(array):
    if isinstance(array, (float, int)):
        array = f'+{float(array)} ' if array >= 0 else f'{float(array)} '
    elif isinstance(array, (pd.Series, pd.DataFrame)):
        array = array.values
    if isinstance(array, np.ndarray):
        if not (array.dtype == object) and array.size:
            signs = pd.Series(array) if array.ndim == 1 else pd.DataFrame(array)
            signs = (signs.pipe(np.sign)
                     .replace([0, 1, -1], ['+', '+', '-']).values)
            array = signs + abs(array).astype(str) + ' '
    return array


def join_exprs(df):
    """
    Helper function to join arrays, series or frames of stings together.
    """
    return ''.join(np.asarray(df).flatten())

def expand_series(ser, columns):
    """
    Helper function to fastly expand a series to a dataframe with according
    column axis and every single column being the equal to the given series.
    """
    return ser.to_frame(columns[0]).reindex(columns=columns).ffill(axis=1)

# =============================================================================
#  'getter' functions
# =============================================================================

def get_extendable_i(n, c):
    """
    Getter function. Get the index of extendable elements of a given component.
    """
    return n.df(c)[lambda ds:
        ds[nominals[c] + '_extendable']].index

def get_non_extendable_i(n, c):
    """
    Getter function. Get the index of non-extendable elements of a given
    component.
    """
    return n.df(c)[lambda ds:
            ~ds[nominals[c] + '_extendable']].index

def get_bounds_pu(n, c, sns, index=slice(None), attr=None):
    """
    Getter function to retrieve the per unit bounds of a given compoent for
    given snapshots and possible subset of elements (e.g. non-extendables).
    Depending on the attr you can further specify the bounds of the variable
    you are looking at, e.g. p_store for storage units.

    Parameters
    ----------
    n : pypsa.Network
    c : string
        Component name, e.g. "Generator", "Line".
    sns : pandas.Index/pandas.DateTimeIndex
        set of snapshots for the bounds
    index : pd.Index, default None
        Subset of the component elements. If None (default) bounds of all
        elements are returned.
    attr : string, default None
        attribute name for the bounds, e.g. "p", "s", "p_store"

    """
    min_pu_str = nominals[c].replace('nom', 'min_pu')
    max_pu_str = nominals[c].replace('nom', 'max_pu')

    max_pu = get_as_dense(n, c, max_pu_str, sns)
    if c in n.passive_branch_components:
        min_pu = - max_pu
    elif c == 'StorageUnit':
        min_pu = pd.DataFrame(0, max_pu.index, max_pu.columns)
        if attr == 'p_store':
            max_pu = - get_as_dense(n, c, min_pu_str, sns)
        if attr == 'state_of_charge':
            max_pu = expand_series(n.df(c).max_hours, sns).T
            min_pu = pd.DataFrame(0, *max_pu.axes)
    else:
        min_pu = get_as_dense(n, c, min_pu_str, sns)
    return min_pu[index], max_pu[index]


# =============================================================================
#  references to vars and cons, rewrite this part to not store every reference
# =============================================================================

def _add_reference(n, df, c, attr, suffix, pnl=True):
    attr_name = attr + suffix
    if pnl:
        if attr_name in n.pnl(c):
            n.pnl(c)[attr_name][df.columns] = df
        else:
            n.pnl(c)[attr_name] = df
        if n.pnl(c)[attr_name].shape[1] == n.df(c).shape[0]:
            n.pnl(c)[attr_name] = n.pnl(c)[attr_name].reindex(columns=n.df(c).index)
    else:
        n.df(c).loc[df.index, attr_name] = df

def set_varref(n, variables, c, attr, pnl=True, spec=''):
    """
    Sets variable references to the network.
    If pnl is False it stores a series of variable names in the static
    dataframe of the given component. The columns name is then given by the
    attribute name attr and the globally define var_ref_suffix.
    If pnl is True if stores the given frame of references in the component
    dict of time-depending quantities, e.g. network.generators_t .
    """
    if not variables.empty:
        if ((c, attr) in n.variables.index) and (spec != ''):
            n.variables.at[idx[c, attr], 'specification'] += ', ' + spec
        else:
            n.variables.loc[idx[c, attr], :] = [pnl, spec]
        _add_reference(n, variables, c, attr, var_ref_suffix, pnl=pnl)

def set_conref(n, constraints, c, attr, pnl=True, spec=''):
    """
    Sets constraint references to the network.
    If pnl is False it stores a series of constraints names in the static
    dataframe of the given component. The columns name is then given by the
    attribute name attr and the globally define con_ref_suffix.
    If pnl is True if stores the given frame of references in the component
    dict of time-depending quantities, e.g. network.generators_t .
    """
    if not constraints.empty:
        if ((c, attr) in n.constraints.index) and (spec != ''):
            n.constraints.at[idx[c, attr], 'specification'] += ', ' + spec
        else:
            n.constraints.loc[idx[c, attr], :] = [pnl, spec]
        _add_reference(n, constraints, c, attr, con_ref_suffix, pnl=pnl)


def get_var(n, c, attr, pop=False):
    '''
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
    get_var(n, 'Generator', 'p')

    '''
    if n.variables.at[idx[c, attr], 'pnl']:
        if pop:
            return n.pnl(c).pop(attr + var_ref_suffix)
        return n.pnl(c)[attr + var_ref_suffix]
    else:
        if pop:
            return n.df(c).pop(attr + var_ref_suffix)
        return n.df(c)[attr + var_ref_suffix]


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
    if n.constraints.at[idx[c, attr], 'pnl']:
        if pop:
            return n.pnl(c).pop(attr + con_ref_suffix)
        return n.pnl(c)[attr + con_ref_suffix]
    else:
        if pop:
            return n.df(c).pop(attr + con_ref_suffix)
        return n.df(c)[attr + con_ref_suffix]


# =============================================================================
# solvers
# =============================================================================

def run_and_read_cbc(n, problem_fn, solution_fn, solver_logfile,
                     solver_options, keep_files, warmstart=None,
                     store_basis=True):
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

    if solver_logfile is None:
        os.system(command)
    else:
        result = subprocess.run(command.split(' '), stdout=subprocess.PIPE)
        print(result.stdout.decode('utf-8'), file=open(solver_logfile, 'w'))

    f = open(solution_fn,"r")
    data = f.readline()
    f.close()

    if data.startswith("Optimal - objective value"):
        status = "optimal"
        termination_condition = status
        objective = float(data[len("Optimal - objective value "):])
    elif "Infeasible" in data:
        termination_condition = "infeasible"
    else:
        termination_condition = "other"

    if termination_condition != "optimal":
        return status, termination_condition, None, None, None

    sol = pd.read_csv(solution_fn, header=None, skiprows=[0],
                      sep=r'\s+', usecols=[1,2,3], index_col=0)
    variables_b = sol.index.str[0] == 'x'
    variables_sol = sol[variables_b][2]
    constraints_dual = sol[~variables_b][3]

    if not keep_files:
       os.system("rm "+ problem_fn)
       os.system("rm "+ solution_fn)

    return (status, termination_condition, variables_sol,
            constraints_dual, objective)


def run_and_read_glpk(n, problem_fn, solution_fn, solver_logfile,
                     solver_options, keep_files, warmstart=None,
                     store_basis=True):
    # for solver_options lookup https://kam.mff.cuni.cz/~elias/glpk.pdf
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

    os.system(command)

    data = open(solution_fn)
    info = ''
    linebreak = False
    while not linebreak:
        line = data.readline()
        linebreak = line == '\n'
        info += line
    info = pd.read_csv(io.StringIO(info), sep=':',  index_col=0, header=None)[1]
    status = info.Status.lower().strip()
    objective = float(re.sub('[^0-9]+', '', info.Objective))
    termination_condition = status

    if termination_condition != "optimal":
        return status, termination_condition, None, None, None

    sol = pd.read_fwf(data).set_index('Row name')
    variables_b = sol.index.str[0] == 'x'
    variables_sol = sol[variables_b]['Activity'].astype(float)
    sol = sol[~variables_b]
    constraints_b = sol.index.str[0] == 'c'
    constraints_dual = (pd.to_numeric(sol[constraints_b]['Marginal'], 'coerce')
                        .fillna(0))

    if not keep_files:
       os.system("rm "+ problem_fn)
       os.system("rm "+ solution_fn)

    return (status, termination_condition, variables_sol,
            constraints_dual, objective)


def run_and_read_gurobi(n, problem_fn, solution_fn, solver_logfile,
                        solver_options, keep_files, warmstart=None,
                        store_basis=True):
    import gurobipy
    # for solver options see
    # https://www.gurobi.com/documentation/8.1/refman/parameter_descriptions.html
    if (solver_logfile is not None) and (solver_options is not None):
        solver_options["logfile"] = solver_logfile

    # disable logging for this part, as gurobi output is doubled otherwise
    logging.disable(50)
    m = gurobipy.read(problem_fn)
    if solver_options is not None:
        for key, value in solver_options.items():
            m.setParam(key, value)
    if warmstart:
        m.read(warmstart)
    m.optimize()
    logging.disable(1)

    if store_basis:
        n.basis_fn = solution_fn.replace('.sol', '.bas')
        try:
            m.write(n.basis_fn)
        except gurobipy.GurobiError:
            logging.info('No model basis stored')
            del n.basis_fn

    if not keep_files:
        os.system("rm "+ problem_fn)

    Status = gurobipy.GRB.Status
    statusmap = {getattr(Status, s) : s.lower() for s in Status.__dir__()
                                                if not s.startswith('_')}
    status = statusmap[m.status]
    termination_condition = status
    if termination_condition != "optimal":
        return status, termination_condition, None, None, None

    variables_sol = pd.Series({v.VarName: v.x for v in m.getVars()})
    constraints_dual = pd.Series({c.ConstrName: c.Pi for c in m.getConstrs()})
    termination_condition = status
    objective = m.ObjVal
    del m
    return (status, termination_condition, variables_sol,
            constraints_dual, objective)



# =============================================================================
# Setting up the problem
# =============================================================================

def define_nominal_for_extendable_variables(n, c, attr):
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    lower = n.df(c)[attr+'_min'][ext_i]
    upper = n.df(c)[attr+'_max'][ext_i]
    variables = write_bound(n, lower, upper)
    set_varref(n, variables, c, attr, pnl=False)


def define_dispatch_for_extendable_variables(n, sns, c, attr):
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    variables = write_bound(n, -np.inf, np.inf, axes=[sns, ext_i])
    set_varref(n, variables, c, attr, pnl=True, spec='extendables')


def define_dispatch_for_non_extendable_variables(n, sns, c, attr):
    fix_i = get_non_extendable_i(n, c)
    if fix_i.empty: return
    nominal_fix = n.df(c)[nominals.at[c]][fix_i]
    min_pu, max_pu = get_bounds_pu(n, c, sns, fix_i, attr)
    lower = min_pu.mul(nominal_fix)
    upper = max_pu.mul(nominal_fix)
    variables = write_bound(n, lower, upper)
    set_varref(n, variables, c, attr, pnl=True, spec='nonextendables')


def define_dispatch_for_extendable_constraints(n, sns, c, attr):
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    min_pu, max_pu = get_bounds_pu(n, c, sns, ext_i, attr)
    operational_ext_v = get_var(n, c, attr)[ext_i]
    nominal_v = get_var(n, c, nominals.at[c])[ext_i]
    rhs = 0

    lhs, *axes = linexpr((max_pu, nominal_v), (-1, operational_ext_v),
                         return_axes=True)
    constraints = write_constraint(n, lhs, '>=', rhs, axes)
    set_conref(n, constraints, c, 'mu_upper', pnl=True, spec=attr)

    lhs, *axes = linexpr((min_pu, nominal_v), (-1, operational_ext_v),
                         return_axes=True)
    constraints = write_constraint(n, lhs, '<=', rhs, axes)
    set_conref(n, constraints, c, 'mu_lower', pnl=True, spec=attr)


def define_fixed_variariable_constraints(n, sns, c, attr, pnl=True):
    if pnl:
        if attr + '_set' not in n.pnl(c): return
        fix = n.pnl(c)[attr + '_set'].unstack().dropna()
        if fix.empty: return
        lhs = linexpr((1, get_var(n, c, attr).unstack()[fix.index]))
        constraints = write_constraint(n, lhs, '=', fix).unstack().T
    else:
        if attr + '_set' not in n.df(c): return
        fix = n.df(c)[attr + '_set'].dropna()
        if fix.empty: return
        lhs = linexpr((1, get_var(n, c, attr)[fix.index]))
        constraints = write_constraint(n, lhs, '=', fix)
    set_conref(n, constraints, c, f'mu_{attr}_set', pnl)


def define_ramp_limit_constraints(n, sns):
    c = 'Generator'
    rup_i = n.df(c).query('ramp_limit_up == ramp_limit_up').index
    rdown_i = n.df(c).query('ramp_limit_down == ramp_limit_down').index
    if rup_i.empty & rdown_i.empty:
        return
    p = get_var(n, c, 'p').loc[sns[1:]]
    p_prev = get_var(n, c, 'p').shift(1).loc[sns[1:]]

    #fix up
    gens_i = rup_i & get_non_extendable_i(n, c)
    lhs = pd.DataFrame(*linexpr((1, p[gens_i]), (-1, p_prev[gens_i]),
                                return_axes=True))
    rhs = n.df(c).loc[gens_i].eval('ramp_limit_up * p_nom')
    constraints = write_constraint(n, lhs, '<=', rhs)
    set_conref(n, constraints, c, 'mu_ramp_limit_up', spec='nonextendables')

    #ext up
    gens_i = rup_i & get_extendable_i(n, c)
    limit_pu = n.df(c)['ramp_limit_up'][gens_i]
    p_nom = get_var(n, c, 'p_nom')[gens_i]
    lhs = pd.DataFrame(*linexpr((1, p[gens_i]), (-1, p_prev[gens_i]),
                                (-limit_pu, p_nom), return_axes=True))
    constraints = write_constraint(n, lhs, '<=', 0)
    set_conref(n, constraints, c, 'mu_ramp_limit_up', spec='extendables')

    #fix down
    gens_i = rdown_i & get_non_extendable_i(n, c)
    lhs = pd.DataFrame(*linexpr((1, p[gens_i]), (-1, p_prev[gens_i]),
                                return_axes=True))
    rhs = n.df(c).loc[gens_i].eval('-1 * ramp_limit_down * p_nom')
    constraints = write_constraint(n, lhs, '>=', rhs)
    set_conref(n, constraints, c, 'mu_ramp_limit_down', spec='nonextendables')

    #ext down
    gens_i = rdown_i & get_extendable_i(n, c)
    limit_pu = n.df(c)['ramp_limit_down'][gens_i]
    p_nom = get_var(n, c, 'p_nom')[gens_i]
    lhs = pd.DataFrame(*linexpr((1, p[gens_i]), (-1, p_prev[gens_i]),
                                (limit_pu, p_nom), return_axes=True))
    constraints = write_constraint(n, lhs, '>=', 0)
    set_conref(n, constraints, c, 'mu_ramp_limit_down', spec='extendables')


def define_nodal_balance_constraints(n, sns):

    def bus_injection(c, attr, groupcol='bus', sign=1):
        #additional sign only necessary for branches in reverse direction
        if 'sign' in n.df(c):
            sign = sign * n.df(c).sign
        vals = linexpr((sign, get_var(n, c, attr)), return_axes=True)
        return pd.DataFrame(*vals).rename(columns=n.df(c)[groupcol])

    # one might reduce this a bit by using n.branches and lookup
    args = [['Generator', 'p'], ['Store', 'p'], ['StorageUnit', 'p_dispatch'],
            ['StorageUnit', 'p_store', 'bus', -1], ['Line', 's', 'bus0', -1],
            ['Line', 's', 'bus1', 1], ['Transformer', 's', 'bus0', -1],
            ['Transformer', 's', 'bus1', 1], ['Link', 'p', 'bus0', -1],
            ['Link', 'p', 'bus1', n.links.efficiency]]
    args = [arg for arg in args if not n.df(arg[0]).empty]

    lhs = (pd.concat([bus_injection(*args) for args in args], axis=1)
           .groupby(axis=1, level=0)
           .agg(lambda x: ''.join(x.values))
           .reindex(columns=n.buses.index))
    sense = '='
    rhs = ((- n.loads_t.p_set * n.loads.sign)
           .groupby(n.loads.bus, axis=1).sum()
           .reindex(columns=n.buses.index, fill_value=0))
    constraints = write_constraint(n, lhs, sense, rhs)
    set_conref(n, constraints, 'Bus', 'nodal_balance')


def define_kirchhoff_constraints(n):
    weightings = n.lines.x_pu_eff.where(n.lines.carrier == 'AC', n.lines.r_pu_eff)

    def cycle_flow(ds):
        ds = ds[lambda ds: ds!=0.].dropna()
        vals = linexpr((ds, get_var(n, 'Line', 's')[ds.index])) + '\n'
        return vals.sum(1)

    constraints = []
    for sub in n.sub_networks.obj:
        C = pd.DataFrame(sub.C.todense(), index=sub.lines_i())
        if C.empty:
            continue
        C_weighted = 1e5 * C.mul(weightings[sub.lines_i()], axis=0)
        con = write_constraint(n, C_weighted.apply(cycle_flow), '=', 0)
        constraints.append(con)
    constraints = pd.concat(constraints, axis=1, ignore_index=True)
    set_conref(n, constraints, 'Line', 'kirchhoff_voltage')


def define_storage_unit_constraints(n, sns):
    sus_i = n.storage_units.index
    if sus_i.empty: return
    c = 'StorageUnit'
    #spillage
    upper = get_as_dense(n, c, 'inflow').loc[:, lambda df: df.max() > 0]
    spill = write_bound(n, 0, upper)
    set_varref(n, spill, 'StorageUnit', 'spill')

    #soc constraint previous_soc + p_store - p_dispatch + inflow - spill == soc
    eh = expand_series(n.snapshot_weightings, sus_i) #elapsed hours

    eff_stand = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)
    eff_dispatch = expand_series(n.df(c).efficiency_dispatch, sns).T
    eff_store = expand_series(n.df(c).efficiency_store, sns).T

    soc = get_var(n, c, 'state_of_charge')
    cyclic_i = n.df(c).query('cyclic_state_of_charge').index
    noncyclic_i = n.df(c).query('~cyclic_state_of_charge').index

    prev_soc_cyclic = soc.shift().fillna(soc.loc[sns[-1]])

    coeff_var = [(-1, soc),
                 (-1/eff_dispatch * eh, get_var(n, c, 'p_dispatch')),
                 (eff_store * eh, get_var(n, c, 'p_store'))]
    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    def masked_term(coeff, var, cols):
        return pd.DataFrame(*linexpr((coeff[cols], var[cols]), return_axes=True))\
                 .reindex(index=axes[0], columns=axes[1], fill_value='').values

    lhs += masked_term(-eh, get_var(n, c, 'spill'), spill.columns)
    lhs += masked_term(eff_stand, prev_soc_cyclic, cyclic_i)
    lhs += masked_term(eff_stand.loc[sns[1:]], soc.shift().loc[sns[1:]], noncyclic_i)

    rhs = -get_as_dense(n, c, 'inflow').mul(eh)
    rhs.loc[sns[0], noncyclic_i] -= n.df(c).state_of_charge_initial[noncyclic_i]

    constraints = write_constraint(n, lhs, '==', rhs)
    set_conref(n, constraints, c, 'soc')


def define_store_constraints(n, sns):
    stores_i = n.stores.index
    if stores_i.empty: return
    c = 'Store'
    variables = write_bound(n, -np.inf, np.inf, axes=[sns, stores_i])
    set_varref(n, variables, c, 'p')

    #previous_e - p == e
    eh = expand_series(n.snapshot_weightings, stores_i)  #elapsed hours
    eff_stand = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)

    e = get_var(n, c, 'e')
    cyclic_i = n.df(c).query('e_cyclic').index
    noncyclic_i = n.df(c).query('~e_cyclic').index

    previous_e_cyclic = e.shift().fillna(e.loc[sns[-1]])

    coeff_var = [(-eh, get_var(n, c, 'p')), (-1, e)]

    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    def masked_term(coeff, var, cols):
        return pd.DataFrame(*linexpr((coeff[cols], var[cols]), return_axes=True))\
                 .reindex(index=axes[0], columns=axes[1], fill_value='').values

    lhs += masked_term(eff_stand, previous_e_cyclic, cyclic_i)
    lhs += masked_term(eff_stand.loc[sns[1:]], e.shift().loc[sns[1:]], noncyclic_i)

    rhs = pd.DataFrame(0, sns, stores_i)
    rhs.loc[sns[0], noncyclic_i] -= n.df(c)['e_initial'][noncyclic_i]

    constraints = write_constraint(n, lhs, '==', rhs)
    set_conref(n, constraints, c, 'soc')


def define_global_constraints(n, sns):
    glcs = n.global_constraints.query('type == "primary_energy"')
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f'{carattr} != 0')[carattr]
        if emissions.empty: continue
        gens = n.generators.query('carrier in @emissions.index')
        em_pu = gens.carrier.map(emissions)/gens.efficiency
        em_pu = n.snapshot_weightings.to_frame() @ em_pu.to_frame('weightings').T
        vals = linexpr((em_pu, get_var(n, 'Generator', 'p')[gens.index]))
        lhs = join_exprs(vals)
        rhs = glc.constant

        #storage units
        sus = n.storage_units.query('carrier in @emissions.index and '
                                    'not cyclic_state_of_charge')
        sus_i = sus.index
        if not sus.empty:
            vals = linexpr((-sus.carrier.map(emissions),
                get_var(n, 'StorageUnit', 'state_of_charge').loc[sns[-1], sus_i]))
            lhs = lhs + '\n' + join_exprs(vals)
            rhs -= sus.carrier.map(emissions) @ sus.state_of_charge_initial

        #stores
        n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query('carrier in @emissions.index and not e_cyclic')
        if not stores.empty:
            vals = linexpr((-stores.carrier.map(n.emissions),
                        get_var(n, 'Store', 'e').loc[sns[-1], stores.index]))
            lhs = lhs + '\n' + join_exprs(vals)
            rhs -= stores.carrier.map(emissions) @ stores.state_of_charge_initial


        con = write_constraint(n, lhs, glc.sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', False, name)

    #expansion limits
    glcs = n.global_constraints.query('type == '
                                      '"transmission_volume_expansion_limit"')
    substr = lambda s: re.sub('[\[\]\(\)]', '', s)
    for name, glc in glcs.iterrows():
        carattr = [substr(c.strip()) for c in glc.carrier_attribute.split(',')]
        lines_ext_i = n.lines.query(f'carrier in @carattr '
                                    'and s_nom_extendable').index
        links_ext_i = n.links.query(f'carrier in @carattr '
                                    'and p_nom_extendable').index
        linevars = linexpr((n.lines.length[lines_ext_i],
                          get_var(n, 'Line', 's_nom')[lines_ext_i]))
        linkvars = linexpr((n.links.length[links_ext_i],
                          get_var(n, 'Link', 'p_nom')[links_ext_i]))
        lhs = join_exprs(linevars) + '\n' + join_exprs(linkvars)
        sense = glc.sense
        rhs = glc.constant
        con = write_constraint(n, lhs, sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', False, name)

    #expansion cost limits
    glcs = n.global_constraints.query('type == '
                                      '"transmission_expansion_cost_limit"')
    for name, glc in glcs.iterrows():
        carattr = [substr(c.strip()) for c in glc.carrier_attribute.split(',')]
        lines_ext_i = n.lines.query(f'carrier in @carattr '
                                    'and s_nom_extendable').index
        links_ext_i = n.links.query(f'carrier in @carattr '
                                    'and p_nom_extendable').index
        linevars = linexpr((n.lines.capital_cost[lines_ext_i],
                        get_var(n, 'Line', 's_nom')[lines_ext_i]))
        linkvars = linexpr((n.links.capital_cost[links_ext_i],
                        get_var(n, 'Link', 'p_nom')[links_ext_i]))
        lhs = join_exprs(linevars) + '\n' + join_exprs(linkvars)
        sense = glc.sense
        rhs = glc.constant
        con = write_constraint(n, lhs, sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', False, name)


def define_objective(n):
    for c, attr in lookup.query('marginal_cost').index:
        cost = (get_as_dense(n, c, 'marginal_cost')
                .loc[:, lambda ds: (ds != 0).all()]
                .mul(n.snapshot_weightings, axis=0))
        if cost.empty: continue
        terms = linexpr((cost, get_var(n, c, attr)[cost.columns]))
        for t in terms.flatten():
            n.objective_f.write(t)
    #investment
    for c, attr in nominals.items():
        cost = n.df(c)['capital_cost'][get_extendable_i(n, c)]
        if cost.empty: continue
        terms = linexpr((cost, get_var(n, c, attr)[cost.index])) + '\n'
        for t in terms.flatten():
            n.objective_f.write(t)



def prepare_lopf(n, snapshots=None, keep_files=False,
                 extra_functionality=None):
    reset_counter()

    #used in kirchhoff and globals
    n.lines['carrier'] = n.lines.bus0.map(n.buses.carrier)

    cols = ['component', 'name', 'pnl', 'specification']
    n.variables = pd.DataFrame(columns=cols).set_index(cols[:2])
    n.constraints = pd.DataFrame(columns=cols).set_index(cols[:2])

    snapshots = n.snapshots if snapshots is None else snapshots
    start = time.time()
    def time_info(message):
        logger.info(f'{message} {round(time.time()-start, 2)}s')

    n.identifier = ''.join(random.choice(string.ascii_lowercase)
                        for i in range(8))
    objective_fn = f"/tmp/objective-{n.identifier}.txt"
    constraints_fn = f"/tmp/constraints-{n.identifier}.txt"
    bounds_fn = f"/tmp/bounds-{n.identifier}.txt"
    n.problem_fn = f"/tmp/test-{n.identifier}.lp"

    n.objective_f = open(objective_fn, mode='w')
    n.constraints_f = open(constraints_fn, mode='w')
    n.bounds_f = open(bounds_fn, mode='w')

    n.objective_f.write('\* LOPF *\n\nmin\nobj:\n')
    n.constraints_f.write("\n\ns.t.\n\n")
    n.bounds_f.write("\nbounds\n")


    for c, attr in lookup.query('nominal and not handle_separately').index:
        define_nominal_for_extendable_variables(n, c, attr)
        define_fixed_variariable_constraints(n, snapshots, c, attr, pnl=False)
    for c, attr in lookup.query('not nominal and not handle_separately').index:
        define_dispatch_for_non_extendable_variables(n, snapshots, c, attr)
        define_dispatch_for_extendable_variables(n, snapshots, c, attr)
        define_dispatch_for_extendable_constraints(n, snapshots, c, attr)
        define_fixed_variariable_constraints(n, snapshots, c, attr)

    define_ramp_limit_constraints(n, snapshots)
    define_storage_unit_constraints(n, snapshots)
    define_store_constraints(n, snapshots)
    define_kirchhoff_constraints(n)
    define_nodal_balance_constraints(n, snapshots)
    define_global_constraints(n, snapshots)
    define_objective(n)

    if extra_functionality is not None:
        extra_functionality(n, snapshots)

    n.objective_f.close()
    n.constraints_f.close()
    n.bounds_f.write("end\n")
    n.bounds_f.close()

    del n.objective_f
    del n.constraints_f
    del n.bounds_f

    os.system(f"cat {objective_fn} {constraints_fn} {bounds_fn} "
              f"> {n.problem_fn}")

    time_info('Total preparation time:')

    if not keep_files:
        for fn in [objective_fn, constraints_fn, bounds_fn]:
            os.system("rm "+ fn)


def assign_solution(n, sns, variables_sol, constraints_dual,
                    extra_postprocessing, keep_references=False):
    pop = not keep_references
    #solutions
    def map_solution(c, attr, pnl):
        if pnl:
            variables = get_var(n, c, attr, pop=pop)
            if variables.empty: return
            values = variables.stack().map(variables_sol).unstack()
            if c in n.passive_branch_components:
                n.pnl(c)['p0'] = values
                n.pnl(c)['p1'] = - values
            elif c == 'Link':
                n.pnl(c)['p0'] = values
                n.pnl(c)['p1'] = - values * n.df(c).efficiency
            else:
                n.pnl(c)[attr] = values
        elif not get_extendable_i(n, c).empty:
            n.df(c)[attr+'_opt'] = get_var(n, c, attr, pop=pop)\
                                    .map(variables_sol).fillna(n.df(c)[attr])
        else:
            n.df(c)[attr+'_opt'] = n.df(c)[attr]

    for (c, attr), pnl in n.variables.pnl.items():
        map_solution(c, attr, pnl)

    if not n.df('StorageUnit').empty:
        c = 'StorageUnit'
        n.pnl(c)['p'] = n.pnl(c)['p_dispatch'] - n.pnl(c)['p_store']

    #duals
    def map_dual(c, attr, pnl):
        if pnl:
            n.pnl(c)[attr] = (get_con(n, c, attr, pop=pop).stack()
                                      .map(-constraints_dual).unstack())
        else:
            n.df(c)[attr] = get_con(n, c, attr, pop=pop).map(-constraints_dual)

    for (c, attr), pnl in n.constraints.pnl.items():
        map_dual(c, attr, pnl)

    #load
    n.loads_t.p = n.loads_t.p_set

    #injection, why does it include injection in hvdc 'network'
    ca = [('Generator', 'p', 'bus' ), ('Store', 'p', 'bus'),
          ('Load', 'p', 'bus'), ('StorageUnit', 'p', 'bus'),
          ('Link', 'p0', 'bus0'), ('Link', 'p1', 'bus1')]
    sign = lambda c: n.df(c).sign if 'sign' in n.df(c) else -1 #sign for 'Link'
    n.buses_t.p = pd.concat(
            [n.pnl(c)[attr].mul(sign(c)).rename(columns=n.df(c)[group])
             for c, attr, group in ca], axis=1).groupby(level=0, axis=1).sum()

    def v_ang_for_(sub):
        buses_i = sub.buses_o
        if len(buses_i) == 1: return
        sub.calculate_B_H(skip_pre=True)
        if len(sub.buses_i()) == 1: return
        Z = pd.DataFrame(np.linalg.pinv((sub.B).todense()), buses_i, buses_i)
        Z -= Z[sub.slack_bus]
        return n.buses_t.p[buses_i] @ Z
    n.buses_t.v_ang = (pd.concat(
                       [v_ang_for_(sub) for sub in n.sub_networks.obj], axis=1)
                      .reindex(columns=n.buses.index, fill_value=0))




def network_lopf(n, snapshots=None, solver_name="cbc",
         solver_logfile=None, extra_functionality=None,
         extra_postprocessing=None, formulation="kirchhoff",
         keep_references=False, keep_files=False, solver_options={},
         warmstart=False, store_basis=True):
    """
    Linear optimal power flow for a group of snapshots.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    solver_name : string
        Must be a solver name that pyomo recognises and that is
        installed, e.g. "glpk", "gurobi"
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    extra_functionality : callable function
        This function must take two arguments
        `extra_functionality(network,snapshots)` and is called after
        the model building is complete, but before it is sent to the
        solver. It allows the user to
        add/change constraints and add/change the objective function.
    solver_logfile : None|string
        If not None, sets the logfile option of the solver.
    solver_options : dictionary
        A dictionary with additional options that get passed to the solver.
        (e.g. {'threads':2} tells gurobi to use only 2 cpus)
    keep_files : bool, default False
        Keep the files that pyomo constructs from OPF problem
        construction, e.g. .lp file - useful for debugging
    formulation : string
        Formulation of the linear power flow equations to use; only "kirchhoff"
        is currently supported
    extra_postprocessing : callable function
        This function must take three arguments
        `extra_postprocessing(network,snapshots,duals)` and is called after
        the model has solved and the results are extracted. It allows the user to
        extract further information about the solution, such as additional
        shadow prices.

    Returns
    -------
    None
    """
    supported_solvers = ["cbc", "gurobi", 'glpk', 'scs']
    if solver_name not in supported_solvers:
        raise NotImplementedError(f"Solver {solver_name} not in "
                                  f"supported solvers: {supported_solvers}")

    if formulation != "kirchhoff":
        raise NotImplementedError("Only the kirchhoff formulation is supported")

    #disable logging because multiple slack bus calculations, keep output clean
    snapshots = _as_snapshots(n, snapshots)
    n.calculate_dependent_values()
    n.determine_network_topology()

    if solver_logfile is None:
        solver_logfile = "test.log"

    logger.info("Prepare linear problem")
    prepare_lopf(n, snapshots, keep_files, extra_functionality)
    gc.collect()
    solution_fn = "/tmp/test-{}.sol".format(n.identifier)

    if warmstart == True:
        warmstart = n.basis_fn
        logger.info("Solve linear problem using warmstart")
    else:
        logger.info("Solve linear problem")

    solve = eval(f'run_and_read_{solver_name}')
    res = solve(n, n.problem_fn, solution_fn, solver_logfile,
                solver_options, keep_files, warmstart, store_basis)
    status, termination_condition, variables_sol, constraints_dual, obj = res
    del n.problem_fn

    if termination_condition != "optimal":
        return status,termination_condition

    #adjust objective value
    for c, attr in nominals.items():
        obj -= n.df(c)[attr] @ n.df(c).capital_cost
    n.objective = obj
    gc.collect()
    assign_solution(n, snapshots, variables_sol, constraints_dual,
                    extra_postprocessing, keep_references=keep_references)
    gc.collect()

    return status,termination_condition


def ilopf(n, snapshots=None, msq_threshold=0.05, min_iterations=1,
          max_iterations=100, **kwargs):
    '''
    Iterative linear optimization updating the line parameters for passive
    AC and DC lines. This is helpful when line expansion is enabled. After each
    sucessful solving, line impedances and line resistance are recalculated
    based on the optimization result. If warmstart is possible, it uses the
    result from the previous iteration to fasten the optimization.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    msq_threshold: float, default 0.05
        Maximal mean square difference between optimized line capacity of
        the current and the previous iteration. As soon as this threshold is
        undercut, and the number of iterations is bigger than 'min_iterations'
        the iterative optimization stops
    min_iterations : integer, default 1
        Minimal number of iteration to run regardless whether the msq_threshold
        is already undercut
    max_iterations : integer, default 100
        Maximal numbder of iterations to run regardless whether msq_threshold
        is already undercut
    **kwargs
        Keyword arguments of the lopf function which runs at each iteration

    '''

    ext_i = get_extendable_i(n, 'Line')
    typed_i = n.lines.query('type != ""').index
    ext_untyped_i = ext_i.difference(typed_i)
    ext_typed_i = ext_i & typed_i
    base_s_nom = (np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) *
                  n.lines.bus0.map(n.buses.v_nom))
    n.lines.loc[ext_typed_i, 'num_parallel'] = (n.lines.s_nom/base_s_nom)[ext_typed_i]

    def update_line_params(n, s_nom_prev):
        factor = n.lines.s_nom_opt / s_nom_prev
        for attr, carrier in (('x', 'AC'), ('r', 'DC')):
            ln_i = (n.lines.query('carrier == @carrier').index & ext_untyped_i)
            n.lines.loc[ln_i, attr] /= factor[ln_i]
        ln_i = ext_i & typed_i
        n.lines.loc[ln_i, 'num_parallel'] = (n.lines.s_nom_opt/base_s_nom)[ln_i]

    def msq_diff(n, s_nom_prev):
        lines_err = np.sqrt((s_nom_prev - n.lines.s_nom_opt).pow(2).mean()) / \
                        n.lines['s_nom_opt'].mean()
        logger.info(f"Mean square difference after iteration {iteration} is "
                    f"{lines_err}")
        return lines_err

    iteration = 0
    diff = msq_threshold
    while diff >= msq_threshold or iteration < min_iterations:
        if iteration >= max_iterations:
            logger.info(f'Iteration {iteration} beyond max_iterations '
                        f'{max_iterations}. Stopping ...')
            break

        s_nom_prev = n.lines.s_nom_opt if iteration else n.lines.s_nom
        kwargs['warmstart'] = bool(iteration and ('basis_fn' in n.__dir__()))
        network_lopf(n, snapshots, **kwargs)
        update_line_params(n, s_nom_prev)
        diff = msq_diff(n, s_nom_prev)
        iteration += 1



# =============================================================================
# test/double-check constraints
# =============================================================================


def describe_storage_unit_contraints(n):
    """
    Checks whether all storage units are balanced over time. This function
    requires the network to contain the separate variables p_store and
    p_dispatch, since they cannot be reconstructed from p. The latter results
    from times tau where p_store(tau) > 0 **and** p_dispatch(tau) > 0, which
    is allowed (even though not economic). Therefor p_store is necessarily
    equal to negative entries of p, vice versa for p_dispatch.
    """
    sus = n.storage_units
    sus_i = sus.index
    if sus_i.empty: return
    sns = n.snapshots
    c = 'StorageUnit'
    pnl = n.pnl(c)

    description = {}

    eh = expand_series(n.snapshot_weightings, sus_i)
    stand_eff = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)
    dispatch_eff = expand_series(n.df(c).efficiency_dispatch, sns).T
    store_eff = expand_series(n.df(c).efficiency_store, sns).T
    inflow = get_as_dense(n, c, 'inflow') * eh
    spill = eh[pnl.spill.columns] * pnl.spill

    description['Spillage Limit'] = pd.Series({'min':
                                (inflow[spill.columns] - spill).min().min()})

    if 'p_store' in pnl:
        soc = pnl.state_of_charge

        store = store_eff * eh * pnl.p_store#.clip(upper=0)
        dispatch = 1/dispatch_eff * eh * pnl.p_dispatch#(lower=0)
        start = soc.iloc[-1].where(sus.cyclic_state_of_charge,
                                   sus.state_of_charge_initial)
        previous_soc = stand_eff * soc.shift().fillna(start)


        reconstructed = (previous_soc.add(store, fill_value=0)
                        .add(inflow, fill_value=0)
                        .add(-dispatch, fill_value=0)
                        .add(-spill, fill_value=0))
        description['SOC Balance StorageUnit'] = ((reconstructed - soc)
                                                  .unstack().describe())
    else:
        logging.info('Storage Unit SOC balance not reconstructable as no '
                     'p_store and p_dispatch in n.storage_units_t.')
    return pd.concat(description, axis=1, sort=False)


def describe_nodal_balance_constraint(n):
    """
    Helper function to double check whether network flow is balanced
    """
    network_injection = pd.concat(
            [n.pnl(c)[f'p{inout}'].rename(columns=n.df(c)[f'bus{inout}'])
            for inout in (0, 1) for c in ('Line', 'Transformer')], axis=1)\
            .groupby(level=0, axis=1).sum()
    return (n.buses_t.p - network_injection).unstack().describe()\
            .to_frame('Nodal Balance Constr.')

def describe_upper_dispatch_constraints(n):
    '''
    Recalculates the minimum gap between operational status and nominal capacity
    '''
    description = {}
    key = ' Upper Limit'
    for c, attr in nominals.items():
        dispatch_attr = 'p0' if c in ['Line', 'Transformer', 'Link'] else attr[0]
        description[c + key] = pd.Series({'min':
                               (n.df(c)[attr + '_opt'] *
                               get_as_dense(n, c, attr[0] + '_max_pu') -
                               n.pnl(c)[dispatch_attr]).min().min()})
    return pd.concat(description, axis=1)


def describe_lower_dispatch_constraints(n):
    description = {}
    key = ' Lower Limit'
    for c, attr in nominals.items():
        if c in ['Line', 'Transformer', 'Link']:
            dispatch_attr = 'p0'
            description[c] = pd.Series({'min':
                              (n.df(c)[attr + '_opt'] *
                              get_as_dense(n, c, attr[0] + '_max_pu') +
                              n.pnl(c)[dispatch_attr]).min().min()})
        else:
            dispatch_attr = attr[0]
            description[c + key] = pd.Series({'min':
                                   (-n.df(c)[attr + '_opt'] *
                                   get_as_dense(n, c, attr[0] + '_min_pu') +
                                   n.pnl(c)[dispatch_attr]).min().min()})
    return pd.concat(description, axis=1)


def describe_store_contraints(n):
    """
    Checks whether all stores are balanced over time.
    """
    stores = n.stores
    stores_i = stores.index
    if stores_i.empty: return
    sns = n.snapshots
    c = 'Store'
    pnl = n.pnl(c)

    eh = expand_series(n.snapshot_weightings, stores_i)
    stand_eff = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)

    start = pnl.e.iloc[-1].where(stores.e_cyclic, stores.e_initial)
    previous_e = stand_eff * pnl.e.shift().fillna(start)

    return (previous_e - pnl.p - pnl.e).unstack().describe()\
            .to_frame('SOC Balance Store')


def describe_cycle_constraints(n):
    weightings = n.lines.x_pu_eff.where(n.lines.carrier == 'AC', n.lines.r_pu_eff)

    def cycle_flow(sub):
        C = pd.DataFrame(sub.C.todense(), index=sub.lines_i())
        if C.empty:
            return None
        C_weighted = 1e5 * C.mul(weightings[sub.lines_i()], axis=0)
        return C_weighted.apply(lambda ds: ds @ n.lines_t.p0[ds.index].T)

    return pd.concat([cycle_flow(sub) for sub in n.sub_networks.obj], axis=0)\
             .unstack().describe().to_frame('Cycle Constr.')



def constraint_stats(n, round_digit=1e-30):
    """
    Post-optimization function to recalculate gap statistics of different
    constraints. For inequality constraints only the minimum of lhs - rhs, with
    lhs >= rhs is returned.
    """
    return pd.concat([describe_cycle_constraints(n),
                      describe_store_contraints(n),
                      describe_storage_unit_contraints(n),
                      describe_nodal_balance_constraint(n),
                      describe_lower_dispatch_constraints(n),
                      describe_upper_dispatch_constraints(n)],
                   axis=1, sort=False)

def check_constraints(n, tol=1e-3):
    """
    Post-optimization test function to double-check most of the lopf
    constraints. For relevant equaility constraints, it test whether the
    deviation between lhs and rhs is below the given tolerance. For inequality
    constraints, it test whether the inequality is violated with a higher
    value then the tolerance.

    Parameters
    ----------
    n : pypsa.Network
    tol : float
        Gap tolerance

    Returns AssertionError if tolerance is exceeded.

    """
    stats = constraint_stats(n).rename(index=str.title)
    condition = stats.T[['Min', 'Max']].query('Min < -@tol | Max > @tol').T
    assert condition.empty, (f'The following constraint(s) are exceeding the '
                             f'given tolerance of {tol}: \n{condition}')




