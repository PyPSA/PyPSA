# -*- coding: utf-8 -*-

"""
Statistics Accessor.
"""


from weakref import ref

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2022 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import os
import sys

import numpy as np
import pandas as pd


def check_if_optimised(func, *args, **kwargs):
    def wrapper(self):
        if not hasattr(self._parent, "objective") or self._parent.objective == np.nan:
            print("Network not optimised yet or optimisation failed")
        else:
            return func(self, *args, **kwargs)

    return wrapper


class StatisticsAccessor:
    def __init__(self, network):
        self._parent = network

    def __call__(self, *args, **kwargs):
        return "TBD"

    @check_if_optimised
    def calculate_capex(self, components=None):
        if components == None:
            components = ["Generator", "Store", "StorageUnit", "Line", "Link"]
        eval_mapper = {
            "Generator": "p",
            "Store": "e",
            "StorageUnit": "p",
            "Line": "s",
            "Link": "p",
        }
        n = self._parent
        capex = pd.DataFrame()
        for component in components:
            mapper = eval_mapper[component]
            expression = "capital_cost*(" + mapper + "_nom_opt-" + mapper + "_nom)"
            df = n.df(component).eval(expression).groupby(n.df(component).carrier).sum()
            index = pd.MultiIndex.from_product(
                [[component], df.index], names=["Component", "Carrier"]
            )
            df = pd.DataFrame(df, columns=["Capital Cost"]).set_index(index)
            capex = pd.concat([capex, df])
        return capex

    @check_if_optimised
    def calculate_opex(self, components=None):
        if components == None:
            components = ["Generator", "Store", "StorageUnit", "Link"]
        eval_mapper = {
            "Generator": "p",
            "Store": "e",
            "StorageUnit": "p",
            "Link": "p0",
        }
        n = self._parent
        opex = pd.DataFrame()
        for component in components:
            mapper = eval_mapper[component]
            df = (
                np.abs(n.pnl(component)[mapper])
                .sum()
                .mul(n.df(component).marginal_cost)
                .groupby(n.df(component).carrier)
                .sum()
            )
            index = pd.MultiIndex.from_product(
                [[component], df.index], names=["Component", "Carrier"]
            )
            df = pd.DataFrame(df, columns=["Marginal Cost"]).set_index(index)
            opex = pd.concat([opex, df])
        return opex

    @check_if_optimised
    def curtailment(self):
        n = self._parent
        renewables = n.meta["renewable"].keys()
        renewable_generators = n.generators[n.generators.carrier.isin(renewables)]
        curtailment = (
            n.generators_t.p_max_pu[renewable_generators.index].mul(
                renewable_generators.p_nom_opt
            )
            - n.generators_t.p[renewable_generators.index]
        )
        curtailment = (
            curtailment.groupby(by=renewable_generators.carrier, axis=1).sum().sum()
        )
        return curtailment
