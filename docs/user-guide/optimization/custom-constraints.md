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
 * Generator-p (name, scenario, snapshot)
 * ...
<BLANKLINE>
Constraints:
------------
 * Bus-nodal_balance (name, scenario, snapshot)
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

Operands with different labels along a shared dimension (here `name`: generators
and links) need explicit alignment. Under linopy's legacy semantics, the default
before linopy 1.0, a plain `expr1 + expr2` still works and silently combines the
two label sets with an outer join. Under the v1 semantics, the only ones from
linopy 1.0, the same `+` raises a `ValueError`, so the intended
alignment is stated with the `join` argument on `.add()`/`.sub()` or
`linopy.merge`:

``` py
>>> (2 * m.variables["Generator-p"]).add(0.5 * m.variables["Link-p"], join="outer")  # doctest: +ELLIPSIS
LinearExpression (sparse) [name: 10, scenario: 1, snapshot: 10]:
----------------------------------------------------------------
[Bremen Converter, base, 2015-01-01 00:00:00]: +0.5 Link-p[Bremen Converter, base, 2015-01-01 00:00:00]
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
Constraint `Bus-minimum_generation_share` [bus: 3, scenario: 1]:
----------------------------------------------------------------
[Frankfurt, base]: +1 Generator-p[Frankfurt Wind, base, 2015-01-01 00:00:00] + ... ≥ 26038.102467283523
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

## Overriding the formulation

[`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__] builds its
model from the specification shown in
[Formulation (generated)](formulation/index.md). The argument `overrides` lays
changes over it before the model is built, without writing linopy code. Each
override is a YAML file, YAML text or a dictionary, written as a
[mathspec](https://github.com/energy-models/mathspec) patch:

* an entry it states in full is added;
* a field it states on an existing entry replaces that field;
* an entry set to `null` is removed.

Overrides apply in turn, so a later one wins. They name entries by their keys
in the specification, as `Generator_ext_p_upper`, not by the names of the
built model, as `Generator-ext-p-upper`. An added entry keeps its key as its
name in `n.model`. Added entries read their data from the network:

* a parameter `<Component>_<attr>` reads the column `<attr>` of the
  component's static data;
* a relation `<Component>_<column>` maps each unit to the label in the column
  `<column>`. The column must be the same in every scenario. Where a cell is
  blank, the unit maps to none; declare such a relation `missing: absent`;
* a dimension takes the labels its relations map units to.

For example, a combined heat and power plant that must cover a heat demand, or
a plant bound by a fuel supply contract, has to run a minimum number of
full-load hours. This file adds the parameter `Generator_full_load_hours_min`
and a constraint on the energy each extendable generator produces:

``` yaml title="full_load_hours.yaml"
parameters:
  Generator_full_load_hours_min:
    description: least full-load hours over the horizon
    dims: [Generator]
constraints:
  Generator_full_load_hours:
    description: an extendable generator runs at least its full-load hours
    dims: [scenario, Generator]
    where: Generator_full_load_hours_min > 0 AND Generator_p_nom_extendable
    expression: >-
      sum(Generator_p * snapshot_weighting_generators, over=snapshot)
      >= Generator_full_load_hours_min * Generator_p_nom_ext
```

The parameter reads the column `full_load_hours_min`; a generator with `0` gets
no row:

``` py
gas = n.c.generators.static.carrier == "gas"
n.c.generators.static["full_load_hours_min"] = 0.0
n.c.generators.static.loc[gas, "full_load_hours_min"] = 4000
n.optimize(overrides=["full_load_hours.yaml"])
```

A relation groups units across rows. In a battery, the charger and the
discharger often share one inverter, so they must have the same rating. This
file adds the dimension `battery`, two relations that map links to it and the
constraint between them:

``` yaml title="battery_inverter.yaml"
dimensions:
  battery:
    description: batteries, each with a charger and a discharger
relations:
  Link_battery_charger: {key: Link, values: battery, missing: absent}
  Link_battery_discharger: {key: Link, values: battery, missing: absent}
constraints:
  Link_battery_inverter:
    description: the charger and the discharger of a battery have the same rating
    dims: [scenario, battery]
    expression: >-
      sum(Link_p_nom_effective, over=Link, by=Link_battery_charger[battery])
      == sum(Link_p_nom_effective, over=Link, by=Link_battery_discharger[battery])
```

The relations read the columns `battery_charger` and `battery_discharger`:

``` py
n.add("Link", "north charger", bus0="north", bus1="north battery",
      p_nom_extendable=True, battery_charger="north")
n.add("Link", "north discharger", bus0="north battery", bus1="north",
      p_nom_extendable=True, battery_discharger="north")
n.optimize(overrides=["battery_inverter.yaml"])
```

`Link_p_nom_effective` is the chosen rating of an extendable link and the given
rating of a fixed link, so the constraint also holds when one side is fixed.
A side whose link is not built, as an inactive one, sums to zero and so holds
the other side at zero.

Removing an entry takes one line each. A quick screening run may ignore the
ramp limits of generators:

``` py
n.optimize(overrides=[{
    "constraints": {
        "Generator_p_ramp_limit_up": None,
        "Generator_p_ramp_limit_down": None,
    }
}])
```

Committable extendable generators state their ramp limits in the
`Generator_p_ramp_limit_*_bigM` entries instead.

PyPSA checks the specification with its overrides once per process, so a
repeated build with the same overrides skips the check.

## Evaluating expressions

The specification also holds `expressions`: named terms that its constraints
and objective read, as `total_cost` or `Bus_injection`.
[Formulation (generated)](formulation/index.md) lists them. After
`n.optimize()`,
[`n.optimize.evaluate()`][pypsa.optimization.OptimizationAccessor.evaluate]
evaluates one over the solution:

``` py
n.optimize()
n.optimize.evaluate("total_cost")                 # a float
n.optimize.evaluate("Generator_p_nom_effective")  # a Series by generator
n.optimize.evaluate("Bus_injection")              # a DataFrame of snapshots by bus
```

An expression over `snapshot` has the snapshots as rows and its other
dimensions as columns. In a network with scenarios, `scenario` comes first, as
in `n.c.generators.dynamic.p`.

An override can add an expression. It can read variables, parameters and
duals. This one gives what each generator earns at the price of its bus, as
`n.statistics.revenue()` does:

``` yaml title="revenue.yaml"
expressions:
  Generator_revenue:
    description: what a generator earns at the price of its bus
    dims: [scenario, Generator]
    expression: >-
      sum(snapshot_weighting_objective * Generator_p
      * at(dual(Bus_nodal_balance), by=Generator_bus[bus]), over=snapshot)
```

``` py
n.optimize(overrides=["revenue.yaml"])
n.optimize.evaluate("Generator_revenue")
```

`evaluate` reads the model of the last optimization, not the network, and
covers only its snapshots. After a rolling horizon, these are the snapshots of
the last window. A copy or a saved network keeps no model, so optimize it
again.

## Handling snapshots in multi-investment optimizations

In multi-period networks, `n.snapshots` is a `pandas.MultiIndex`. linopy's v1
semantics do not allow a `MultiIndex` on a model dimension, so under v1 PyPSA
builds `snapshot` as a flat dimension labelled by the `(period, timestep)`
tuples, with `period` and `timestep` as auxiliary coordinates. Auxiliary
coordinates are extra labels on a dimension that do not index it but still work
for selecting and grouping, e.g. `expr.groupby("period")`. `n.snapshots` stays
a `MultiIndex` and results are unchanged.

Migrating existing `extra_functionality` to the flat representation:

* `n.c.<component>.da.<attr>` instead of `n.get_switchable_as_dense(...).loc[sns]`. The arrays are already on the model's labels; select the snapshots you need with `.sel(snapshot=sns)`.
* `expr.groupby("period").sum()` instead of slicing `sns` by `sns.unique("period")` — the auxiliary coordinate takes the role of the `MultiIndex` level.
* For data that has to stay in pandas, the flat labels *are* the `(period, timestep)` tuples of `n.snapshots`. The `sns` argument of `extra_functionality` carries the model's labels, so `net_sns = pd.MultiIndex.from_tuples(sns, names=n.snapshots.names)` recovers the network's labelling and `df.loc[net_sns].set_axis(sns)` relabels a snapshot-indexed frame onto the model's.

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
