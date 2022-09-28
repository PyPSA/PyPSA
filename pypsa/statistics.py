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
        if not hasattr(self._parent, "objective") or self._parent.objective==np.nan:
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
    def calculate_capex(self):
        n = self._parent
        capex=n.generators.eval("capital_cost*p_nom_opt").sum()
        return capex

