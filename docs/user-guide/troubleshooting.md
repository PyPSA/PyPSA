<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Troubleshooting

## Minimum Working Examples

For your own debugging and when asking for help, it is very useful to create a
Minimum Working Example (MWE). An MWE is a small example that reproduces the
problem you are experiencing. It should be as small as possible, but still
reproduce the problem. Often, when creating a MWE, you will find the problem
yourself, and if not it will be much easier for others to help you.

For details on what a MWE is and how to create one, see this blog on how [Craft Minimal Bug Reports](https://matthewrocklin.com/minimal-bug-reports).

## Library dependencies

If you are experiencing problems with PyPSA or with the importing of the
libraries on which PyPSA depends, please first check that you are working with
the latest versions of all packages. See [Upgrading PyPSA](installation.md#upgrading).

## Consistency check

A consistency check can be performed using the function
[`n.consistency_check()`][pypsa.Network.consistency_check], which can point to
potential issues in the network.

## Optimisation convergence & infeasibility

If your [`n.optimize()`]() is not converging,
here are some suggestions to try out:

* Very small non-zero values, for example in `n.generators_t.p_max_pu` can
  confuse the solver. Consider e.g. removing values smaller than 0.001 with
  [`pandas.DataFrame.clip`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.clip.html).

* Open source solvers like HiGHS can struggle with large problems; consider
  switching to a commercial solver like Gurobi or Xpress. Alternatively, scale
  down the model size to make it easier to debug the problem (e.g. reducing
  spatial or temporal resolution while keeping the same structure).

* Use the interior point or barrier method, and stop it from crossing over to
  the simplex algorithm once it is close to the solution. This will provide a
  good approximate solution. Also set a random seed for reproducibility. Note
  that solver parameters may differ between solvers and have varying effect on
  different types of problems.

=== "HiGHS"

    ``` py
    n.optimize(solver_name='highs', solver="ipm", run_crossover="off", random_seed=123)

=== "SCIP"

    ``` py
    n.optimize(solver_name='scip', solver_options={"lp/initalgorithm": "b"})  
    ```

=== "Gurobi"

    ``` py
    n.optimize(solver_name='gurobi', method=2, crossover=0, Seed=123)  
    ```

=== "CPLEX"

    ``` py
    n.optimize(solver_name='cplex', lpmethod=4, solutiontype=2)  
    ```

=== "COPT"

    ``` py
    n.optimize(solver_name='copt', LpMethod=2, Crossover=0)  
    ```

=== "Xpress"

    ``` py
    n.optimize(solver_name='xpress', LPFLAGS=4, CROSSOVER=0, BARALG=2)  
    ```

* Your problem may be infeasible, i.e. there is no solution that satisfies all
  constraints. If you are using Gurobi, you can check which constraints cause an
  infeasibility by adding the keyword argument `compute_infeasibilities` to
  [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__] to compute an [Irreducible
  Inconsistent Subset
  (IIS)](https://support.gurobi.com/hc/en-us/articles/360029969391-How-do-I-determine-why-my-model-is-infeasible):

  ``` py
  n.optimize(solver_name='gurobi', compute_infeasibilities=True)  
  ```

* Add a load shedding generator with high marginal cost to all buses, which can
  be used to shed load that cannot be met. This will often allow the
  optimisation to find a solution even if the original problem is infeasible.
  Then, based on where the load shedding generator is used, you can identify
  which constraints are causing the infeasibility.

  ``` py
  n.add(
       "Generator",
       n.buses.index,
       suffix="load-shedding",
       bus=n.buses.index,
       marginal_cost=10_000, # high marginal cost
       p_nom=1e9, # non-binding capacity
       carrier="load_shedding",
  )
  ```

## Power flow convergence

If your [`n.pf()`][pypsa.Network.pf] is not converging there are two possible reasons:

* The problem you have defined is not solvable (e.g. because in
  reality you would have a voltage collapse).

* The problem is solvable, but there are numerical instabilities in the solving
  algorithm (e.g. Newton-Raphson is known not to converge even for
  ill-conditioned solvable problems; or the flat solution PyPSA uses as an
  initial guess is too far from the correction solution because of transformer
  phase-shifts)

There are some steps you can take to distinguish these two cases:

* Check the units you have used to define the problem are correct. If your units
  are out by a factor 1000 (e.g. using kW instead of MW) do not be surprised if
  your problem is no longer solvable.

* Check with a linear power flow [`n.lpf()`][pypsa.Network.lpf] that all voltage
  angles differences across branches are less than 40 degrees. You can do this with the following code:

  ``` py
  >>> import pandas as pd
  >>> import numpy as np
  >>> now = n.snapshots[0]  #
  >>> angle_diff = pd.Series(
  ...     n.buses_t.v_ang.loc[now,n.lines.bus0].values -
  ...     n.buses_t.v_ang.loc[now,n.lines.bus1].values,
  ...     index=n.lines.index
  ... )  
  >>> (angle_diff * 180 / np.pi).describe()  #D doctest: +SKIP
  ```

* You can seed the non-linear power flow initial guess with the
  voltage angles from the linear power flow. This is advisable if you
  have transformers with phase shifts in the network, which lead to
  solutions far away from the flat initial guess of all voltage angles
  being zero. To seed the problem activate the `use_seed` switch:

  ``` py
  n.lpf()
  n.pf(use_seed=True)
  ```

* Reduce all power values `p_set` and `q_set` of generators and
  loads to a fraction, e.g. 10%, solve the load flow and use it as a
  seed for the power at 20%, iteratively up to 100%.  

## Reporting bugs and issues

See the [Support](support.md) page for how to report bugs and issues.

