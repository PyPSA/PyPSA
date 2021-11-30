#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 12:56:06 2021

@author: fabian
"""
import os

from linopy import Model, LinearExpression
import pandas as pd

from ..descriptors import nominal_attrs, get_switchable_as_dense as get_as_dense
from ..pf import _as_snapshots

from .common import reindex

from .variables import (
    define_nominal_variables,
    define_operational_variables,
    define_status_variables,
    define_spillage_variables,
)
from .constraints import (
    define_operational_constraints_for_non_extendables,
    define_operational_constraints_for_committables,
    define_operational_constraints_for_extendables,
    define_nominal_constraints_for_extendables,
    define_fixed_operation_constraints,
    define_fixed_nominal_constraints,
    define_ramp_limit_constraints,
    define_nodal_balance_constraints,
    define_kirchhoff_constraints,
    define_storage_unit_constraints,
    define_store_constraints,
)

from .global_constraints import (
    define_growth_limit,
    define_nominal_constraints_per_bus_carrier,
)

# from .constraints import

lookup = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "variables.csv"),
    index_col=["component", "variable"],
)


def sanity_check(n):
    for c in n.components:
        intersection = n.get_committable(c).intersection(n.get_extendable_i(c))
        if not intersection.empty:
            raise ValueError(
                "Assets can only be commitable or extendable. Found "
                f"assets in component {c} which are both:"
                f"\n\n\t{', '.join(intersection)}"
            )


def define_objective(n, sns):
    """
    Defines and writes out the objective function

    """
    m = n.model

    if n._multi_invest:
        periods = sns.unique("period")
        period_weighting = n.investment_period_weightings.objective[periods]

    # constant for already done investment
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = n.get_extendable_i(c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: n.get_active_assets(c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        constant += cost @ n.df(c)[attr][ext_i]

    if constant != 0:
        object_const = m.add_variables(constant, constant, name="objective_constant")
        m.objective = m.objective - 1 * object_const
        n.objective_constant = constant

    # marginal cost
    weighting = n.snapshot_weightings.objective
    if n._multi_invest:
        weighting = weighting.mul(period_weighting, level=0).loc[sns]
    else:
        weighting = weighting.loc[sns]

    for c, attr in lookup.query("marginal_cost").index:
        cost = (
            get_as_dense(n, c, "marginal_cost", sns)
            .loc[:, lambda ds: (ds != 0).all()]
            .mul(weighting, axis=0)
        )
        if cost.empty:
            continue
        operation = m[f"{c}-{attr}"].sel({"snapshot": sns, c: cost.columns})
        m.objective = m.objective + (operation * cost).sum()

    # investment
    for c, attr in nominal_attrs.items():
        ext_i = n.get_extendable_i(c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: n.get_active_assets(c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        caps = m[f"{c}-{attr}"]
        m.objective = m.objective + (caps * cost).sum()


def create_model(n, snapshots=None, multi_investment_periods=False, **kwargs):
    """
    Sets up the linear problem and writes it out to a lp file.

    Returns
    -------
    Tuple (fdp, problem_fn) indicating the file descriptor and the file name of
    the lp file

    """
    kwargs.setdefault("force_dim_names", True)
    n.model = Model(**kwargs)
    sns = _as_snapshots(n, snapshots)
    n._multi_invest = int(multi_investment_periods)

    # Define variables
    for c, attr in lookup.query("nominal").index:
        define_nominal_variables(n, c, attr)

    for c, attr in lookup.query("not nominal and not handle_separately").index:
        define_operational_variables(n, sns, c, attr)
        define_status_variables(n, sns, c)

    define_spillage_variables(n, sns)
    define_operational_variables(n, sns, "Store", "p")

    # Define constraints
    for c, attr in lookup.query("nominal").index:
        define_nominal_constraints_for_extendables(n, c, attr)
        define_fixed_nominal_constraints(n, c, attr)

    for c, attr in lookup.query("not nominal and not handle_separately").index:
        define_operational_constraints_for_non_extendables(n, sns, c, attr)
        define_operational_constraints_for_extendables(n, sns, c, attr)
        define_operational_constraints_for_committables(n, sns, c)
        define_ramp_limit_constraints(n, sns, c)
        define_fixed_operation_constraints(n, sns, c, attr)

    define_nodal_balance_constraints(n, sns)
    define_kirchhoff_constraints(n, sns)
    define_storage_unit_constraints(n, sns)
    define_store_constraints(n, sns)

    # Define global constraints
    define_nominal_constraints_per_bus_carrier(n, sns)
    define_growth_limit(n, sns)
    # define_global_constraints(n, sns)

    define_objective(n, sns)

    return n.model
