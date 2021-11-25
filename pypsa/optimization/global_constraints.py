#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:02:55 2021

@author: fabian
"""

# def define_nominal_constraints_per_bus_carrier(n, sns):
#     for carrier in n.carriers.index:
#         for bound, sense in [("max", "<="), ("min", ">=")]:

#             col = f'nom_{bound}_{carrier}'
#             if col not in n.buses.columns: continue
#             rhs = n.buses[col].dropna()
#             lhs = pd.Series('', rhs.index)

#             for c, attr in nominal_attrs.items():
#                 if c not in n.one_port_components: continue
#                 attr = nominal_attrs[c]
#                 if (c, attr) not in n.variables.index: continue
#                 nominals = get_var(n, c, attr)[n.df(c).carrier == carrier]
#                 if nominals.empty: continue
#                 per_bus = linexpr((1, nominals)).groupby(n.df(c).bus).sum(**agg_group_kwargs)
#                 lhs += per_bus.reindex(lhs.index, fill_value='')

#             if bound == 'max':
#                 lhs = lhs[lhs != '']
#                 rhs = rhs.reindex(lhs.index)
#             else:
#                 assert (lhs != '').all(), (
#                     f'No extendable components of carrier {carrier} on bus '
#                     f'{list(lhs[lhs == ""].index)}')
#             n.model.add_constraints(n, lhs, sense, rhs, 'Bus', 'mu_' + col)
