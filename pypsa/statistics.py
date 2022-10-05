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


def check_if_optimised(func):
    def wrapper(self, *args, **kwargs):
        if not hasattr(self._parent, "objective") or self._parent.objective == np.nan:
            print("Network not optimised yet or optimisation failed")
        else:
            return func(self, *args, **kwargs)

    return wrapper


def get_multiindex(df, component):
    index = pd.MultiIndex.from_product(
        [[component], df.index], names=["Component", "Carrier"]
    )
    return index


eval_mapper_static = {
    "Generator": "p",
    "Store": "e",
    "StorageUnit": "p",
    "Line": "s",
    "Link": "p",
}
eval_mapper_dynamic = {
    "Generator": "p",
    "Store": "e",
    "StorageUnit": "p",
    "Link": "p0",
}


class StatisticsAccessor:
    def __init__(self, network):
        self._parent = network

    def __call__(self, *args, **kwargs):
        method_list = [
            method for method in dir(self) if method.startswith("_") is False
        ]
        print("The following methods can be called with statistics:\n")
        print(*method_list, sep="\n")

    @check_if_optimised
    def calculate_capex(self, components=None):
        if components == None:
            components = ["Generator", "Store", "StorageUnit", "Line", "Link"]

        n = self._parent
        capex = pd.DataFrame()
        for component in components:
            mapper = eval_mapper_static[component]
            expression = "capital_cost*(" + mapper + "_nom_opt-" + mapper + "_nom)"
            df = n.df(component).eval(expression).groupby(n.df(component).carrier).sum()
            index = get_multiindex(df, component)
            df = pd.DataFrame(df, columns=["Capital Cost"]).set_index(index)
            capex = pd.concat([capex, df])
        return capex

    @check_if_optimised
    def calculate_opex(self, components=None):
        if components == None:
            components = ["Generator", "Store", "StorageUnit", "Link"]
        n = self._parent
        opex = pd.DataFrame()
        for component in components:
            mapper = eval_mapper_dynamic[component]
            marginal_cost = (
                n.pnl(component)["marginal_cost"]
                if not n.pnl(component)["marginal_cost"].empty
                else n.df(component).marginal_cost
            )
            df = (
                np.abs(n.pnl(component)[mapper])
                .mul(marginal_cost)
                .sum()
                .groupby(n.df(component).carrier)
                .sum()
            )
            index = get_multiindex(df, component)
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

    @check_if_optimised
    def congestion_rent(self):
        n = self._parent
        congestion_rent = (n.lines_t.mu_lower + n.lines_t.mu_upper).mul(n.lines_t.p0)
        congestion_rent = np.abs(congestion_rent).sum()
        return congestion_rent

    @check_if_optimised
    def p_nom_opt(self, components=None):
        if components == None:
            components = ["Generator", "Store", "StorageUnit", "Line", "Link"]
        n = self._parent
        p_nom_opt = pd.DataFrame()
        for component in components:
            mapper = eval_mapper_static[component]
            expression = mapper + "_nom_opt"
            df = n.df(component).eval(expression).groupby(n.df(component).carrier).sum()
            index = get_multiindex(df, component)
            df = pd.DataFrame(df, columns=["Optimized Capacity"]).set_index(index)
            p_nom_opt = pd.concat([p_nom_opt, df])
        return p_nom_opt

    @check_if_optimised
    def generation(self, components=None):
        if components == None:
            components = ["Generator", "Store", "StorageUnit"]
        n = self._parent
        generation = pd.DataFrame()
        for component in components:
            mapper = eval_mapper_dynamic[component]
            df = n.pnl(component)[mapper].clip(lower=0)
            df = df.sum().groupby(n.df(component).carrier).sum()
            index = get_multiindex(df, component)
            df = pd.DataFrame(df, columns=["Generation"]).set_index(index)
            generation = pd.concat([generation, df])
        return generation

    @check_if_optimised
    def revenue(self, components=None):
        if components == None:
            components = ["Generator", "Store", "StorageUnit"]
        n = self._parent
        revenue = pd.DataFrame()
        revenue.attrs["unit"] = "billion €"
        nodal_prices = n.buses_t.marginal_price
        for component in components:
            mapper = eval_mapper_dynamic[component]
            columns = pd.Series(
                n.pnl(component)[mapper].columns.map(n.df(component).bus),
                index=n.pnl(component)[mapper].columns,
            )
            df = (
                n.pnl(component)[mapper]
                .clip(lower=0)
                .mul(nodal_prices.loc[:, columns].set_axis(columns.index, axis=1))
            )
            df = df.sum().groupby(n.df(component).carrier).sum()
            index = get_multiindex(df, component)
            df = pd.DataFrame(df, columns=["Revenue"]).set_index(index)
            revenue = pd.concat([revenue, df])
        return revenue

    @check_if_optimised
    def market_value(self, components=None):
        total_revenue = self.revenue(components=components).values.sum()
        total_generation = self.generation(components=components).values.sum()
        market_value = total_revenue / total_generation
        return market_value
