<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Numerical Scaling

Energy system models combine quantities that span many orders of magnitude.
Capacities in MW sit next to capital costs in EUR/MW, and a CO2 budget of
millions of tonnes next to emission factors below one tonne per MWh. The
resulting constraint matrix, bounds, right-hand sides and objective have
coefficient ranges that solvers handle poorly. Solver tolerances are absolute,
so quantities of very different magnitude are checked against the same
threshold, and a wide range leaves the smallest of them below the level the
solver can distinguish from zero [^gurobi]. Solver vendors therefore recommend
keeping matrix coefficients within six orders of magnitude, ideally in
$[10^{-3}, 10^{6}]$, and objective coefficients, bounds and right-hand sides
below about $10^{6}$ [^gurobi], or as close to order unity as the model allows
[^highs].

Every solver applies its own internal scaling, but this operates on the matrix
alone and cannot see the physical units behind a column or row [^highs].
Rescaling the model in its physical units before export, for instance from MW
to GW and from EUR to MEUR, compresses the ranges of coefficients, bounds and
objective at once.

PyPSA can perform this rescaling on the built linopy model before it is handed
to the solver. Factors are powers of two, so the rescaling is exact in
floating-point arithmetic and introduces no rounding error [^highs]. Nothing in
the network data is touched. linopy maps solution, duals and objective value
back to the original units, so results are identical in form to an unscaled
solve. The `scaling` argument of `n.optimize` accepts three kinds of values,
described below. The default comes from `pypsa.options.params.optimize.scaling`
(see [Options](../options.md)).

## Automatic scaling

``` py
n.optimize(scaling=True)
```

`scaling=True` chooses one factor for energy quantities and one for cost
quantities. A small integer program selects them from the coefficient ranges
of the built model so that the spread of all magnitudes is minimised within
the target window.

Each constraint row is divided by the factor of its variables. Matrix
coefficients therefore keep their values and only bounds, right-hand sides and
the objective move. This is a pure change of units, equivalent to converting
the inputs from MW to GW and from EUR to MEUR, which makes it safe as a
default. Rows are not tuned individually, since in benchmarks on PyPSA-Eur
networks per-row factors helped some solver algorithms and hurt others. They
remain available through the manual mode below.

## Manual scaling

A dict gives the factors explicitly. It never runs the tuner, and values are
used verbatim, they do not need to be powers of two.

``` py
n.optimize(
    scaling={
        "energy": 1e3,
        "cost": 1e6,
        "constraint_factors": {"GlobalConstraint-co2_limit": 1e-6},
    }
)
```

- `energy` and `cost` are the unit factors. A missing key defaults to 1.
- `constraint_factors` maps linopy constraint group names to row factors.
  Listed rows get exactly that factor, unlisted rows are rescaled by the factor
  of their variables as in automatic mode. Unknown names raise an error.

Constraint group names are the ones shown by `n.model.constraints`. Constraints
added through `extra_functionality` can be listed as well.

## Inspecting and editing the automatic choice

[`n.optimize.tune_scaling()`][pypsa.optimization.OptimizationAccessor.tune_scaling]
runs the tuner on the built model and returns its choice in the manual dict
form. `scaling=True` is equivalent to passing that dict unchanged.

``` py
n.optimize.create_model()
factors = n.optimize.tune_scaling()
factors["constraint_factors"]["GlobalConstraint-co2_limit"] *= 4
n.optimize(scaling=factors)
```

With `constraint_factors=True` the tuner also chooses one factor per
constraint group. This is the per-row mode discussed above, useful for
experiments with a specific solver algorithm.

``` py
factors = n.optimize.tune_scaling(constraint_factors=True)
```

## Limitations

Scaling is skipped with a warning when the model has indicator constraints,
frozen constraints or a quadratic objective. Integer and binary variables are
never scaled.

The objective constant for existing capacity is a large additive term that can
dominate relative solver tolerances. Consider `include_objective_constant=False`
alongside scaling, see [Options](../options.md).

[^gurobi]: Gurobi Optimization. Guidelines for Numerical Issues, section
    [Tolerances and User-Scaling](https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/tolerances_scaling.html).
[^highs]: HiGHS. [Numerical considerations](https://ergo-code.github.io/HiGHS/stable/guide/numerics/).
