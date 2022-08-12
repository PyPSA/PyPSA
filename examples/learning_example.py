#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:11:20 2022
Minimal example for endogeneous learning
@author: lisa
"""

import pypsa
import pandas as pd

n = pypsa.Network()
# Set snapshots to the year 2022
snapshots = pd.date_range('2025-01-01', '2026-01-01', freq='1H', inclusive='left')
n.set_snapshots(snapshots)
# add AC bus
n.add("Bus",
  "AC",
  )
# add AC demand
n.add("Load",
      "AC",
      bus="AC",
      p_set=100,
      sign=-1)

n.investment_periods = [2025, 2035, 2045]

for year in n.investment_periods:
    # add generator
    n.add("Generator",
          "wind-{}".format(year),
          carrier="wind",
          bus="AC",
          build_year=year,
          lifetime=10,
          p_nom_extendable=True)


n.add("Carrier",
      "wind",
      learning_rate=0.1,
      global_capacity=100, # how much capacitiy is installed today
      max_capacity=1000, # maximum capacity, end point of linearisation of the cost curve
      initial_cost=100, # today's investment costs
      global_factor=0.5,  # if global factor=1 -> local learning else share of global capacity
      )

n.lopf(learning=True, pyomo=False, multi_investment_periods=True, time_delay=True,
       solver_name="gurobi", keep_references=True)
