<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Overview

PyPSA can optimise the following types of problems:

1. **Economic Dispatch (ED)** market model with unit commitment and storage operation with perfect foresight or rolling horizon, coupled across different energy carriers (electricity, heat, hydrogen, etc.) and with conversion between them ([:material-notebook: Example](../../examples/simple-electricity-market-examples.ipynb)),

2. **Linear Optimal Power Flow (LOPF)** with network constraints for Kirchhoff's Voltage Law (KVL) and Kirchhoff's Current Law (KCL) ([:material-notebook: Example](../../examples/scigrid-lopf-then-pf.ipynb)),

3. **Security-Constrained Linear Optimal Power Flow (SCLOPF)** for network contingency analysis ([:material-notebook: Example](../../examples/scigrid-sclopf.ipynb)),

4. **Capacity Expansion Planning (CEP)** with single or multiple investment periods and system-wide constraints ([:material-notebook: Example](../../examples/capacity-expansion-planning-single-node.ipynb)),

5. **Stochastic Optimisation (SO)** in form of a two-stage stochastic program with investments as first-stage decisions and dispatch as recourse decisions across weighted scenarios of uncertain input parameters  ([:material-notebook: Example](../../examples/stochastic-optimization.ipynb)), and

6. **Modelling-to-Generate-Alternatives (MGA)** for near-optimal space exploration ([:material-notebook: Example](../../examples/mga.ipynb)).

These problems build on each other, e.g., capacity expansion planning models include economic dispatch and linear optimal power flow constraints. Thereby, the dispatch of generation, conversion and storage technologies, as well as the capacities of generation, storage, conversion and transmission infrastructure are co-optimised. In any case, the objective is to minimize the total system cost for the snapshots selected.

The kind of optimisation problem is determined by the parameters provided for the network components (e.g. whether components are extendable or committable). Depending on the data input, the optimisation is then formulated as a **linear program (LP)**, **quadratic program (QP)** or **mixed-integer linear program (MILP)**. Most variables are continuous, but unit commitment constraints and block-sized investments can be modelled with binary variables. Quadratic terms are added by quadratic marginal dispatch costs.

To solve a network with a solver of your choice, run

``` py
n.optimize(solver_name="highs", solver_options={"solver": "ipm"})
```

where `solver_name` is a string and `solver_options` is a dictionary of solver-specific flags to pass to the solver. See [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__] for details.

A call to this function will formulate the optimisation problem with the [Linopy](https://linopy.readthedocs.io) library, solve it by interfacing with the solver, and store the results in the network object `n`.

!!! note "Problem Extensions"

    While most types of optimisation problems are dynamically formulated by [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__] depending on data inputs, some features require different optimization functions to be called. For instance, to optimise dispatch in a sequential rolling horizon with myopic operational foresight,
    the function [`n.optimize.optimize_with_rolling_horizon()`][pypsa.optimization.OptimizationAccessor.optimize_with_rolling_horizon] can be used.
