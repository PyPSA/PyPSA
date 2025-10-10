<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Custom Constraints

Custom constraints allow users to tailor optimization problems to specific
requirements or scenarios. Users can model more complex limits and interactions
that are not captured by the default optimization formulations provided by
[`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__]. To build custom constraints, users can access, modify and
amend the [Linopy](https://linopy.readthedocs.io) model instance associated with
a network object, `n.model`. 

Some key functions used in the code for working with custom constraints include:

* [`n.optimize.create_model()`][pypsa.optimization.OptimizationAccessor.create_model]: Creates Linopy model instance for the network, `n.model`.
* [`n.model.variables`](https://linopy.readthedocs.io/en/latest/generated/linopy.model.Model.html): Accesses the decision variables.
* [`n.model.add_variables()`](https://linopy.readthedocs.io/en/latest/creating-variables.html): Adds decision variables.
* [`n.model.add_constraints()`](https://linopy.readthedocs.io/en/latest/creating-constraints.html): Adds custom constraints.
* [`n.model.add_objective(overwrite=True)`](https://linopy.readthedocs.io/en/latest/generated/linopy.model.Model.add_objective.html): Overwrites the objective function.
* [`n.optimize.solve_model()`][pypsa.optimization.OptimizationAccessor.solve_model]: Solves the current model instance and writes the solution into `n`.

!!! note "Understanding the `linopy` library"

    Before using custom constraints, ensure that you have a good understanding of the [Linopy](https://linopy.readthedocs.io/en/latest/index.html) library and its functionalities, as it is the underlying optimization framework used by PyPSA for creating and solving optimization problems. Checkout its [documentation](https://linopy.readthedocs.io/en/latest/index.html).

A typical workflow starts with creating a Linopy model instance for a network
using
[`n.optimize.create_model()`][pypsa.optimization.OptimizationAccessor.create_model].
This model instance contains all variables, constraints, and the objective
function of the optimization problem. 

``` py
>>> m = n.optimize.create_model()
```

This will create a Linopy model instance `m` for the network `n` and is also
accessible using the [`n.model`][pypsa.Network.model] attribute. By accessing
the model instance, users can directly access, add, remove, or modify variables,
constraints, and the objective as needed.

To get a first overview of the variables and constraints in the model, call

``` py
>>> m  # doctest: +ELLIPSIS
Linopy LP model
===============
<BLANKLINE>
Variables:
----------
 * Generator-p_nom (name)
 * ...
<BLANKLINE>
Constraints:
------------
 * Generator-ext-p_nom-lower (name)
 * ...
<BLANKLINE>
Status:
-------
initialized
```

Specific variables can be accessed using `m.variables`, which provides a
dictionary-like structure containing the variables associated with each
component. For example, the following call retrieves generator active power variables:

``` py
>>> gen_p = m.variables["Generator-p"]
```

This will return a `linopy.Variable`, and array of variables with generators and
snapshots as dimensions. The `linopy.Variable` type is closely related to
`xarray.DataArray` and `pandas.DataFrame`, and can be used in similar ways.

To create custom constraints, sets of variables are first combined into
`linopy.LinearExpression` objects with coefficients and operations (e.g.
addition, subtraction, multiplication, division) that represent the relationship
between variables involved in the constraint.

``` py
>>> 2 * m.variables["Generator-p"] + 0.5 * m.variables["Link-p"]  # doctest: +ELLIPSIS
LinearExpression [snapshot: 10, name: 10]:
------------------------------------------
[2015-01-01 00:00:00, Bremen Converter]: +0.5 Link-p[2015-01-01 00:00:00, Bremen Converter]
...
```

The constraint can then be created using standard Python operators like `==`,
`>=`, and `<=` and right-hand side constants. For example, a constraint that
forces the total generation at a bus to be at least 80% of the total demand,
would be written as follows:

``` py
>>> bus = n.generators.bus.to_xarray()
>>> total_generation = gen_p.groupby(bus).sum().sum("snapshot")
>>> total_demand = n.loads_t.p_set.sum().sum()
>>> constraint_expression = total_generation >= 0.8 * total_demand
```

After defining the constraint expression, it is added to the Linopy model instance using the
`m.add_constraints()` function, providing a name for the constraint to
facilitate further modifications or inspection:

``` py
>>> m.add_constraints(constraint_expression, name="Bus-minimum_generation_share")  # doctest: +ELLIPSIS
Constraint `Bus-minimum_generation_share` [bus: 3]:
---------------------------------------------------
[Frankfurt]: +1 Generator-p[2015-01-01 00:00:00, Frankfurt Wind] + ... ≥ 26038.102467283523
```

Once the custom constraints is registered, calling
[`n.optimize.solve_model()`][pypsa.optimization.OptimizationAccessor.solve_model]
solves the model including any modifications after
[`n.optimize.create_model()`][pypsa.optimization.OptimizationAccessor.create_model]
and writes the solution.

``` py
>>> n.optimize.solve_model()
('warning', 'infeasible')
```

Generally, optimised values for custom variables are not written back to the network object `n`. They must be retrieved seperately from the Linopy model instance `n.model`. For example, if you created a custom variable `custom_var`, you can access its optimised values as follows:

``` py
>>> custom_var_values = n.model.variables["custom_var"].solution  # doctest: +SKIP
```

<!-- However, if you follow the naming convention `{component}-{variable}`, where `component` is the name of the component (e.g., "Generator") and `variable` is the name of the variable (e.g., "custom_var"),
the optimised values will be stored for the network component (e.g. `n.generators_t.custom_var`). -->

!!! note "Alternative approach using `n.optimize(extra_functionality=...)`"

    The workflow described above is the recommended way to add custom constraints to a PyPSA network. It allows for direct access to the Linopy model instance and provides flexibility in defining and modifying constraints.  However, if you prefer a more integrated approach, you can use the `extra_functionality` argument in the [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__] function. This allows you to pass a function that will be executed after the model is created and before it is solved, enabling you to add custom constraints or modify the model as needed:

    ``` py
    >>> def custom_constraints(n: pypsa.Network, sns: pd.Index) -> None:
    ...     m = n.model
    ...     # Define and add custom constraints here
    ...     ...
    >>> n.optimize(extra_functionality=custom_constraints)  
    ```

!!! warning "Persistence of Linopy model instances"

    The Linopy model instance, `n.model`, is not retained when exporting the network to files. It is only available in memory during the current session. If you need to retain the model instance beyond
    the current session, use Linopy functionality to save it separately with [`n.model.to_netcdf()`](https://linopy.readthedocs.io/en/latest/generated/linopy.model.Model.to_netcdf.html#linopy.model.Model.to_netcdf). That means, any custom constraints added to the model will not be saved when exporting the network to files.
