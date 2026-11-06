<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Overview

PyPSA can optimise the following types of problems:

1. **Economic Dispatch (ED)** market model with unit commitment, maintenance scheduling, and storage operation with perfect foresight or rolling horizon, coupled across different energy carriers (electricity, heat, hydrogen, etc.) and with conversion between them ([:material-notebook: Example](../../examples/simple-electricity-market-examples.ipynb)),

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

!!! note "Rolling horizon with `method="restrict"`"

    By default (`method="rebuild"`), the rolling horizon builds and solves a new model for every window. With `method="restrict"`, the model is built once for all snapshots and every window is cut from its matrices with linopy's `Model.restrict`, which saves the repeated model building on large networks. All variables of earlier windows are fixed at their optimized values, which carries over storage levels and the dispatch for ramp limits. Capacities of extendable components have no snapshot dimension and are re-optimized in every window, as with `method="rebuild"`, so `p_nom_opt` and the like are those of the last window. Each window is solved from scratch without a warm start, and the solution of all windows is written to the network once at the end.

    ``` py
    n.optimize.optimize_with_rolling_horizon(horizon=48, overlap=8, method="restrict")
    ```

    The method requires linopy >= 1.0.0 and solvers with linopy's direct API. Until linopy 1.0.0 is released, install linopy from its master branch. It supports problems without multi-period investment, also with scenarios, extendable components and committable components, both linearized and with integer variables. Since all unit commitment constraints look backward in time, the commitment status, start-ups and ramps of earlier windows are carried over through the fixed variables. It raises a `NotImplementedError` for `multi_investment_periods=True`, global constraints, volume limits (`e_sum_min`, `e_sum_max`), cyclic stores or storage units, risk preferences and custom constraints that couple a window to later snapshots, and a `RuntimeError` as soon as the solver returns no solution for a window.

    A window is accepted when the solver returns a solution, e.g. a MIP incumbent at a time limit, with a warning if it is not optimal. In mixed-integer windows, integer values are rounded before they are fixed for later windows, and the duals are taken from a re-solve of the window as an LP with its integer variables fixed.

    `log_fn` writes the solver log of all windows to one file. `window_callback(n, w, sns, i)` is called with the linopy window `w` of the `i`-th window over snapshots `sns`, once before its solve (`w.status == "initialized"`) and once after, before its status is checked. Use it e.g. to log the status, write the window with `w.to_file()`, compute an IIS with `w.compute_infeasibilities()` or leave rows out of a window with `w.advance(drop=...)`.

    Compared to `method="rebuild"`:

    - `n.objective` is the objective of the whole horizon, including the capital costs of the last window's capacities, instead of the objective of the last window.
    - The standing loss of stores and storage units is applied to the storage level carried over into a window, as in a full-horizon optimization. `method="rebuild"` carries the level over without loss.
