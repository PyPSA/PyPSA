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

A `Path`, or a string that names an existing file, is read as a file; any other
string is read as YAML text.

Overrides apply in turn, so a later one wins. They name entries by their keys
in the specification, as `Generator_ext_p_upper`, not by the names of the
built model, as `Generator-ext-p-upper`; see
[How keys are named](#how-keys-are-named). An added entry keeps its key as its
name in `n.model`. Added entries read their data from the network:

* a parameter `<Component>_<attr>` reads the column `<attr>` of the
  component's static data. Where the column is absent, it takes the value
  its `missing:` declares;
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

### How keys are named

An override names each entry by its key. A key that the specification lacks
fails the build, and the error proposes the closest key. A key is unique within
its kind: `Generator_p_set` is a parameter and also a constraint.

1. **Prefix.** The key of a component's entry starts with the class name of the
   component and `_`, as `Generator_p`, `StorageUnit_state_of_charge` or
   `Bus_nodal_balance`. A system-wide entry has no component prefix, as
   `snapshot_weighting_objective`, `total_cost` or `Kirchhoff_Voltage_Law`.
   Some groups of system-wide entries share a prefix that is no component:
   `CVaR_`, `FlowBased_`, `Cycle_` and `Outage_`.
2. **Parameters.** A parameter `<Component>_<attr>` reads the attribute `<attr>`
   of the component, static or varying, as `Generator_p_max_pu`,
   `Link_marginal_cost` or `Store_e_nom`. PyPSA prepares some of these values
   first. `Generator_capital_cost` is the cost per period, also from
   `overnight_cost`. `Generator_p_nom_max` is the tightest bound of the
   scenarios. The derived `Generator_activity_mask` also applies the build year
   and the lifetime to `active`, per snapshot. Other parameters
   have no attribute, and PyPSA derives them, as `Generator_big_m`, `Line_BODF`
   or `snapshot_weighting_generators`.
3. **Variables and constraints.** Take the name in `n.model` and replace each
   `-` with `_`: `Generator-p` is `Generator_p`, `Generator-fix-p-upper` is
   `Generator_fix_p_upper`, `Line-ext-s_nom-lower` is `Line_ext_s_nom_lower`.
   These are the names of earlier PyPSA versions. The rule works in this
   direction only, since the names in `n.model` mix `-` and `_`. The
   exceptions:
    * The capacity variables add `_ext`: `Generator-p_nom` is
      `Generator_p_nom_ext`, `Store-e_nom` is `Store_e_nom_ext`.
      `Generator_p_nom` is the parameter.
    * `GlobalConstraint_upper`, `GlobalConstraint_lower` and
      `GlobalConstraint_fix` hold all global constraints of one sense. `n.model`
      holds one constraint per global constraint, named after its label, as
      `GlobalConstraint-co2_limit`.
    * With `transmission_losses` in the mode `secants`, `Line_loss_tangents_pos`
      builds `Line-loss_secants-pos`. The same holds for `_neg` and for
      transformers.
    * A piecewise curve, as `Generator_marginal_cost_lp`, builds variables and
      constraints that have no key, as `Generator-marginal_cost_piecewise_delta`.
    * The variable `objective_constant` is not part of the specification.

    An entry that an override adds keeps its key as its name in `n.model`.
4. **Dimensions.** The dimension of a component's units is its class name, as
   `Generator` or `StorageUnit`. All other dimensions are in lower case, as
   `snapshot`, `period`, `scenario`, `bus`, `carrier` or `port`. An override
   uses these names, also where `n.model` names a dimension otherwise. In
   `n.model`, the dimension of units is `name` where an entry has one only. On
   a constraint, `bus` and `global_constraint` also become `name`.
   `breakpoint` becomes `_breakpoint`, and `Carrier-growth_limit` names its
   dimensions `Carrier` and `periods`.

!!! warning "Which keys are stable"

    Overrides are new. Until the first release with overrides, a key can
    still change. From that release until PyPSA 2.0, a stable key keeps its
    name and its meaning. If a stable key must change before 2.0, the old key
    stays valid and gives a deprecation warning. A test locks every key, so
    no key changes by accident.

    The stable keys are:

    * the keys of variables, constraints and dimensions, and the objective,
    * the keys of parameters that carry the name of an attribute (rule 2),
    * the keys of the expressions `total_cost` and `<Component>_<nom>_effective`,
      as `Link_p_nom_effective` or `Store_e_nom_effective`.

    All other keys are internal helpers: they can change in any release
    without notice.

    A stable key does not mean stable math. A release can change the math of
    an entry, for example to correct an error. A
    [stored override](#storing-overrides-on-the-network) of such an entry then
    fails the build with a clear error, and you must store it again.

| Kind | Example key | Stable |
| --- | --- | --- |
| variable | `Generator_p` | yes |
| constraint | `Generator_fix_p_upper` | yes |
| dimension | `snapshot` | yes |
| objective | `objective` | yes |
| parameter with the name of an attribute | `Generator_p_max_pu` | yes |
| parameter that PyPSA derives | `Generator_big_m` | no |
| expression | `total_cost`, `Link_p_nom_effective` | yes |
| other expression | `Generator_opex` | no |
| condition | `Generator_ext` | no |
| relation | `Generator_bus` | no |
| piecewise | `Generator_marginal_cost_lp` | no |
| macro | `Generator_operating_cost` | no |
| assumption | `Generator_maintenance_duration_positive` | no |

The objective has no inner key: an override names the section itself, as
`{"objective": {"expression": "my_cost"}}`.

To find a key:

* Look it up in [Formulation (generated)](formulation/index.md). It has a
  page for the system and one per component, with each entry under its key
  as heading. The tab
  "YAML" shows the name in `n.model` as `name:`. The table "Symbols" lists
  the parameters and relations that the page reads.
* Read the YAML files in `pypsa/optimization/spec/`. The trait files write
  keys with placeholders, as `COMPONENT_ext_NOM_upper`; each trait page
  lists the names that each component gives them. The entries for
  `linearized_unit_commitment` are in `linearized_uc/` and on no page.
* Take a name from `n.model` and apply rule 3.
* List the keys of one kind that your network builds, as
  `list(n.optimize.spec().constraints)`; see
  [Inspecting the effective specification](#inspecting-the-effective-specification).

### Storing overrides on the network

The argument `overrides` applies to one call. To make an override part of the
network, store it with
[`n.optimize.overrides`][pypsa.optimization.overrides.Overrides]. Every later
build applies it, and copies, slices, merges, spatial clustering and saved
files keep it, so a loaded network builds the model its author built:

``` py
n.optimize.overrides.register("full_load_hours.yaml")  # stored as "full_load_hours"
n.optimize.overrides.register(
    {"constraints": {"Generator_p_ramp_limit_up": None}}, name="no_ramp"
)
n.optimize.overrides             # Overrides ['full_load_hours', 'no_ramp']
n.optimize.overrides["no_ramp"]  # the stored patch
n.optimize.overrides.unregister("no_ramp")
n.optimize.overrides.clear()     # unregisters all
n.optimize()
```

`n.optimize.overrides` reads as a mapping of each name to its patch. A file is
stored under its stem; text and a dictionary take a `name`, an identifier.
`register` refuses a name that is stored; with `overwrite=True` it replaces
the override in its place. The network keeps the patch by its content, and a saved network
holds it under the meta key `"_pypsa"`, so the patch holds JSON values only: no
NumPy numbers, dates, tuples or keys that are not text. A build lays the layers over the specification in this order, and a later
layer wins:

1. the specification, patched for `linearized_unit_commitment`;
2. the `piecewise_options`;
3. the [composite types](../optimization/composites.md);
4. the stored overrides, in the order they were registered;
5. the `overrides` of the call.

A stored override may edit, remove and add entries, as a per-call one may.
`register` checks it laid over the default specification, the composite types
and the stored overrides, without linearised unit commitment and piecewise
options. An override that holds only with these options stays per-call.
`unregister` refuses an override that later ones build on, and names them.
`unregister` and `clear` also drop overrides that no longer build, as after a
PyPSA upgrade. A
composite type cannot declare an entry that a stored override adds.

A loaded network reproduces the stored math, but not the options of the call:
pass `linearized_unit_commitment` and `piecewise_options` again (see
[#2146](https://github.com/PyPSA/PyPSA/issues/2146)). Each stored override
records which entries it edits, with a fingerprint of their content, and which
it adds. If a later PyPSA release removes or changes an entry it edits, or adds
one it adds, the build fails instead of changing the problem silently; the
same holds where the options of a build remove or add such entries. After a
release, check the override and register it again with `overwrite=True`; for the options
of a build, pass it per call. The check does not cover entries an override
only reads, such as a variable its own constraint uses.

To add a group of components with the math that ties them as often as needed,
define a [composite](../optimization/composites.md).

### Inspecting the effective specification

[`n.optimize.spec()`][pypsa.optimization.OptimizationAccessor.spec] returns
the specification that `n.optimize()` builds the model from: the default
specification with all layers above laid over it. It takes the arguments of
`n.optimize()` that change the specification: `overrides`,
`linearized_unit_commitment` and `piecewise_options`. Use it to check an
override before you solve:

``` py
s = n.optimize.spec(overrides=["full_load_hours.yaml"])
list(s.constraints)                         # the keys of one kind
s.constraints["Generator_full_load_hours"]  # one entry, as the build reads it
print(s.to_yaml())                          # the whole specification
```

Each kind, as `variables`, `parameters`, `constraints` or `expressions`, maps
keys to entries. `pypsa.optimization.spec.spec()` returns the default
specification, without the composite types and stored overrides of a network.

## Reading expressions

The specification also holds `expressions`: named terms that its constraints
and objective read, as `total_cost` or `Generator_p_nom_effective`.
[Formulation (generated)](formulation/index.md) lists them. Only the keys
`total_cost` and `<Component>_<nom>_effective` are stable; the other keys can
change in any release (see [How keys are named](#how-keys-are-named)). With
`include_expressions=True`, `n.optimize()` adds each one to
`n.model.expressions`, and `.solution` gives its value after the solve:

``` py
n.optimize(include_expressions=True)
n.model.expressions["total_cost"].solution
n.model.expressions["Generator_p_nom_effective"].solution
```

An expression has the same snapshots as the variables. Its one dimension of
units, as `Generator`, becomes `name`, as in the variables; its other
dimensions keep their names from the specification, as `bus` or `period`. One
over the units of two components keeps both names, as `Generator` and `Link`.
An expression added by
`overrides` is built too. An expression that reads only data, or reads a dual
as `dual(Bus_nodal_balance)`, is not built. The option is off by default,
since the expressions add to the time and memory of the build.

## Handling snapshots in multi-investment optimizations

In multi-period networks, `n.snapshots` is a `pandas.MultiIndex`. linopy's v1
semantics do not allow a `MultiIndex` on a model dimension, so PyPSA
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
