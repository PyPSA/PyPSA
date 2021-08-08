
## Copyright 2015-2021 PyPSA Developers

## You can find the list of PyPSA Developers at
## https://pypsa.readthedocs.io/en/latest/developers.html

## PyPSA is released under the open source MIT License, see
## https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt

"""Descriptors for component attributes.
"""

__author__ = "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
__copyright__ = ("Copyright 2015-2021 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
                 "MIT License")

from collections import OrderedDict
from itertools import repeat

import networkx as nx
import pandas as pd
import numpy as np
import re

import logging
logger = logging.getLogger(__name__)


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

if _nx_version >= '2.0':
    def degree(G):
        return G.degree()
else:
    def degree(G):
        return G.degree_iter()

class Dict(dict):
    """
    Dict is a subclass of dict, which allows you to get AND SET
    items in the dict using the attribute syntax!

    Stripped down from addict https://github.com/mewwts/addict/ .
    """

    def __setattr__(self, name, value):
        """
        setattr is called when the syntax a.b = 2 is used to set a value.
        """
        if hasattr(Dict, name):
            raise AttributeError("'Dict' object attribute "
                                 "'{0}' is read-only".format(name))
        else:
            self[name] = value

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError as e:
            raise AttributeError(e.args[0])

    def __delattr__(self, name):
        """
        Is invoked when del some_addict.b is called.
        """
        del self[name]

    _re_pattern = re.compile('[a-zA-Z_][a-zA-Z0-9_]*')

    def __dir__(self):
        """
        Return a list of object attributes.

        This includes key names of any dict entries, filtered to the
        subset of valid attribute names (e.g. alphanumeric strings
        beginning with a letter or underscore).  Also includes
        attributes of parent dict class.
        """
        dict_keys = []
        for k in self.keys():
            if isinstance(k, str):
                m = self._re_pattern.match(k)
                if m:
                    dict_keys.append(m.string)

        obj_attrs = list(dir(Dict))

        return dict_keys + obj_attrs


def get_switchable_as_dense(network, component, attr, snapshots=None, inds=None):
    """
    Return a Dataframe for a time-varying component attribute with values for all
    non-time-varying components filled in with the default values for the
    attribute.

    Parameters
    ----------
    network : pypsa.Network
    component : string
        Component object name, e.g. 'Generator' or 'Link'
    attr : string
        Attribute name
    snapshots : pandas.Index
        Restrict to these snapshots rather than network.snapshots.
    inds : pandas.Index
        Restrict to these components rather than network.components.index

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> get_switchable_as_dense(network, 'Generator', 'p_max_pu')

"""

    df = network.df(component)
    pnl = network.pnl(component)

    index = df.index

    varying_i = pnl[attr].columns
    fixed_i = df.index.difference(varying_i)

    if inds is not None:
        index = index.intersection(inds)
        varying_i = varying_i.intersection(inds)
        fixed_i = fixed_i.intersection(inds)
    if snapshots is None:
        snapshots = network.snapshots
    return (pd.concat([
        pd.DataFrame(np.repeat([df.loc[fixed_i, attr].values], len(snapshots), axis=0),
                     index=snapshots, columns=fixed_i),
        pnl[attr].loc[snapshots, varying_i]
    ], axis=1, sort=False).reindex(columns=index))

def get_switchable_as_iter(network, component, attr, snapshots, inds=None):
    """
    Return an iterator over snapshots for a time-varying component
    attribute with values for all non-time-varying components filled
    in with the default values for the attribute.

    Parameters
    ----------
    network : pypsa.Network
    component : string
        Component object name, e.g. 'Generator' or 'Link'
    attr : string
        Attribute name
    snapshots : pandas.Index
        Restrict to these snapshots rather than network.snapshots.
    inds : pandas.Index
        Restrict to these items rather than all of network.{generators,..}.index

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> get_switchable_as_iter(network, 'Generator', 'p_max_pu', snapshots)

"""

    df = network.df(component)
    pnl = network.pnl(component)

    index = df.index
    varying_i = pnl[attr].columns
    fixed_i = df.index.difference(varying_i)

    if inds is not None:
        inds = pd.Index(inds)
        index = inds.intersection(index)
        varying_i = inds.intersection(varying_i)
        fixed_i = inds.intersection(fixed_i)

    # Short-circuit only fixed
    if len(varying_i) == 0:
        return repeat(df.loc[fixed_i, attr], len(snapshots))

    def is_same_indices(i1, i2): return len(i1) == len(i2) and (i1 == i2).all()
    if is_same_indices(fixed_i.append(varying_i), index):
        def reindex_maybe(s): return s
    else:
        def reindex_maybe(s): return s.reindex(index)

    return (
        reindex_maybe(df.loc[fixed_i, attr].append(pnl[attr].loc[sn, varying_i]))
        for sn in snapshots
    )


def allocate_series_dataframes(network, series):
    """
    Populate time-varying outputs with default values.

    Parameters
    ----------
    network : pypsa.Network
    series : dict
        Dictionary of components and their attributes to populate (see example)

    Returns
    -------
    None

    Examples
    --------
    >>> allocate_series_dataframes(network, {'Generator': ['p'],
                                             'Load': ['p']})

"""

    for component, attributes in series.items():

        df = network.df(component)
        pnl = network.pnl(component)

        for attr in attributes:
            pnl[attr] = pnl[attr].reindex(columns=df.index,
                                          fill_value=network.components[component]["attrs"].at[attr,"default"])

def free_output_series_dataframes(network, components=None):
    if components is None:
        components = network.all_components

    for component in components:
        attrs = network.components[component]['attrs']
        pnl = network.pnl(component)

        for attr in attrs.index[attrs['varying'] & (attrs['status'] == 'Output')]:
            pnl[attr] = pd.DataFrame(index=network.snapshots, columns=[])

def zsum(s, *args, **kwargs):
    """
    pandas 0.21.0 changes sum() behavior so that the result of applying sum
    over an empty DataFrame is NaN.

    Meant to be set as pd.Series.zsum = zsum.
    """
    return 0 if s.empty else s.sum(*args, **kwargs)

#Perhaps this should rather go into components.py
nominal_attrs = {'Generator': 'p_nom',
                 'Line': 's_nom',
                 'Transformer': 's_nom',
                 'Link': 'p_nom',
                 'Store': 'e_nom',
                 'StorageUnit': 'p_nom'}

def expand_series(ser, columns):
    """
    Helper function to quickly expand a series to a dataframe with according
    column axis and every single column being the equal to the given series.
    """
    return ser.to_frame(columns[0]).reindex(columns=columns).ffill(axis=1)


def get_extendable_i(n, c):
    """
    Getter function. Get the index of extendable elements of a given component.
    """
    return n.df(c)[lambda ds: ds[nominal_attrs[c] + '_extendable']].index

def get_non_extendable_i(n, c):
    """
    Getter function. Get the index of non-extendable elements of a given
    component.
    """
    return n.df(c)[lambda ds: ~ds[nominal_attrs[c] + '_extendable']].index


def get_active_assets(n, c, investment_period):
    """
    Getter function. Get True values for elements of component c which are active
    at a given investment period. These are calculated from lifetime and the
    build year.
    """
    if investment_period not in n.investment_periods:
        raise ValueError("Investment period not in `network.investment_periods`")
    return n.df(c).eval("build_year <= @investment_period < build_year + lifetime")


def get_activity_mask(n, c, sns=None):
    """
    Getter function. Get a boolean array with True values for elements of
    component c which are active at a specific snapshot. If the network is
    in multi_investment_period mode (given by n._multi_invest),
    these are calculated from lifetime and the build year. Otherwise all
    values are set to True.
    """
    if sns is None:
        sns = n.snapshots
    if getattr(n, '_multi_invest', False):
        _ = {period: get_active_assets(n, c, period) for period in n.investment_periods}
        return pd.concat(_, axis=1).T.reindex(n.snapshots, level=0).loc[sns]
    else:
        return pd.DataFrame(True, sns, n.df(c).index)


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
    min_pu_str = nominal_attrs[c].replace('nom', 'min_pu')
    max_pu_str = nominal_attrs[c].replace('nom', 'max_pu')

    max_pu = get_switchable_as_dense(n, c, max_pu_str, sns)
    if c in n.passive_branch_components:
        min_pu = - max_pu
    elif c == 'StorageUnit':
        min_pu = pd.DataFrame(0, max_pu.index, max_pu.columns)
        if attr == 'p_store':
            max_pu = - get_switchable_as_dense(n, c, min_pu_str, sns)
        if attr == 'state_of_charge':
            max_pu = expand_series(n.df(c).max_hours, sns).T
            min_pu = pd.DataFrame(0, *max_pu.axes)
    else:
        min_pu = get_switchable_as_dense(n, c, min_pu_str, sns)

    return min_pu[index], max_pu[index]

def additional_linkports(n):
    return [i[3:] for i in n.links.columns if i.startswith('bus')
            and i not in ['bus0', 'bus1']]
