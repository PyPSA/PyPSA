<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Changing the Formulation

To change the math of `n.optimize()`, edit the YAML files in
`pypsa/optimization/spec/`. You need Python only for derived data and for
checks of the input. Each change fails the lock test on purpose until you
record it again. For other math in one study, a user needs no code change but
an [override](../user-guide/optimization/custom-constraints.md#overriding-the-formulation)
or a [composite](../user-guide/optimization/composites.md).

## How a build works

| Step | File |
| --- | --- |
| Each component class lists its traits in `_formulation`. | `pypsa/components/_types/<component>.py` |
| Check the network data. | `spec/checks.py` |
| Join `system.yaml`, each trait file renamed for each component and the component's own file. Take out `name:` and `scaling:`. | `spec/_render.py` |
| Lay `linearized_uc/`, piecewise options, composites and overrides over the result. Cache it. | `spec/layers.py` |
| Derive each parameter: with its `@source` function, else from the column `<attr>` for the key `<Component>_<attr>`. | `spec/sources.py`, `spec/_<trait>.py` |
| linopy builds the model. Give each entry its PyPSA name, as `Generator-p`. | `spec/build.py`, `spec/naming.py` |
| Apply the scaling rules. | `spec/scaling.py` |

A trait file writes its keys over placeholders: `COMPONENT`, `UNIT`, `NOM`,
`POWER`, `LEVEL`, `CYCLIC` and `COEF`. Thus `Generator_p_ramp_limit_up` is
`COMPONENT_p_ramp_limit_up` in `traits/ramping.yaml`. To find a key, search
for its end, as `grep -rn "_p_ramp_limit_up:" pypsa/optimization/spec/`. An
entry that two or more components share goes in a trait. An entry of one
component goes in its own file, as `components/storage_unit.yaml`.

## Recipes

### Change the math of an entry

1. Find the file that holds the key.
2. Edit its `expression:` or `where:`. For example:

    ``` diff
       COMPONENT_p_ramp_limit_up:
         ...
    -    expression: COMPONENT_p - COMPONENT_previous_p <= COMPONENT_ramp_up_allowance
    +    expression: COMPONENT_p - COMPONENT_previous_p <= 0.9 * COMPONENT_ramp_up_allowance
    ```

3. If `linearized_unit_commitment=True` must change too, edit the file at the
   same path under `linearized_uc/`. A file there is a patch: each field it
   states replaces that field of the base entry.
4. Build a small network with `n.optimize.create_model()`, print the entry,
   as `n.model.constraints["Generator-p-ramp_limit_up"]`, and go on with
   [After a change](#after-a-change).

### Add an attribute and a constraint that uses it

1. Add the attribute to `pypsa/data/component_attrs/<component>.csv` of each
   component that declares the trait. If you add it to some only, give the
   parameter a `missing:` value, as `missing: .inf`.

    ``` text
    full_load_hours_max,float,h,inf,"The most full-load hours over all snapshots of the optimization.",Input (optional)
    ```

2. Declare the parameter and the constraint in the trait file. A parameter
   `COMPONENT_<attr>` reads the column `<attr>`. The `name:` of the constraint
   must give its key when you replace each `-` with `_`. A test checks this.

    ``` yaml title="traits/dispatch.yaml"
    parameters:
      COMPONENT_full_load_hours_max:
        description: most full-load hours over the horizon; infinity means no row
        dims: [scenario, UNIT]
    constraints:
      COMPONENT_full_load_hours_max:
        name: COMPONENT-full_load_hours_max
        description: the energy over the horizon is at most the full-load hours times the nominal power
        dims: [scenario, UNIT]
        where: COMPONENT_full_load_hours_max < inf AND count(COMPONENT_activity_mask, over=snapshot) > 0
        expression: >-
          sum(COMPONENT_p * energy_limit_weight, over=snapshot)
          <= COMPONENT_full_load_hours_max * COMPONENT_NOM_effective
    ```

3. If the value is not the column itself, register a function with `@source`
   in the module of the trait, as `spec/_storage.py` does. Import a new module
   in `spec/__init__.py`. Without the import, the build reads the column and
   gives no error.
4. Build a small network that sets the attribute, print
   `n.model.constraints["Generator-full_load_hours_max"]`, and go on with
   [After a change](#after-a-change).

### Add a variable

1. Declare it under `variables:`. Give it a `name:` of the form
   `<Component>-<attr>`. With this name, PyPSA writes the solution to
   `n.c.<component>.dynamic.<attr>`, as `n.c.generators.dynamic.p_curtailed`.
   An entry without `name:` keeps its key as its name in `n.model`. PyPSA then
   does not write its solution to the network, and `scaling: bus` fails.
2. Set `scaling:` to `bus` for a power or energy at a bus, to `objective` for
   a cost and to `none` for a binary or a share. Without `scaling:`, a
   continuous variable of a component at a bus takes `bus`, any other `none`.
   See [Numerical Scaling](../user-guide/optimization/numerical-scaling.md).
3. Give the constraints that use it other keys, as `Generator_p_available`.
   A variable and a constraint with the same key build, but the docs pages
   fail.
4. Build a small network, print `n.model.variables["Generator-p_curtailed"]`,
   and go on with [After a change](#after-a-change).

``` yaml title="components/generator.yaml"
variables:
  Generator_p_curtailed:
    name: Generator-p_curtailed
    scaling: bus
    description: the available power a generator does not produce
    dims: [scenario, snapshot, Generator]
    where: Generator_activity_mask
    bounds: {lower: 0}
```

### Give a component a trait, or add a trait

1. Add the name of the trait to `_formulation` of each component class that
   gets it, as `"full_load_hours"` in `Generator` and `Link`. A trait file
   that no component lists is not built.
2. For a new trait, add `traits/<trait>.yaml`. Under `given:`, declare what
   it reads from other files. Under `dimensions:`, declare the dimensions it
   uses. A trait must serve two or more components; the tests require this.

    ``` yaml title="traits/full_load_hours.yaml"
    given:
      parameters:
        COMPONENT_activity_mask: {dims: [snapshot, UNIT], dtype: bool}
      variables:
        COMPONENT_p: {dims: [scenario, snapshot, UNIT]}
      expressions:
        COMPONENT_NOM_effective: {dims: [scenario, UNIT]}
        energy_limit_weight: {dims: [snapshot]}
    dimensions:
      scenario: {}
      snapshot: {dtype: str, ordered: true}
      UNIT: {}
    # parameters and constraints as in the recipe above
    ```

3. Build a small network. If a name is "not found", add the trait that
   declares it to the component, or declare it in the component's file. Then
   go on with [After a change](#after-a-change).

### Refuse input that the formulation does not support

1. If you can state the rule with the parameters of the spec, add an
   assumption to the YAML file. The docs pages show it next to the math.

    ``` yaml title="traits/maintenance.yaml"
    assumptions:
      COMPONENT_maintenance_duration_positive:
        holds: COMPONENT_maintenance_duration > 0
        where: COMPONENT_maintainable
        description: an event lasts longer than zero
    ```

    Input that breaks it fails the build with a `SpecDataError` that names
    the units.

2. For a warning, a deprecation or a rule on data that the spec does not read,
   add a function to `spec/checks.py` and call it from `check()`:

    ``` py
    def _check_ramp_limits(n: Network) -> None:
        """Raise if a ramp limit is negative."""
        for c in formulated(n, "ramping"):
            bad = _flagged(c.da.ramp_limit_up < 0)
            if bad:
                msg = f"{c.name} {bad} have a negative `ramp_limit_up`."
                raise ValueError(msg)
    ```

`n.consistency_check()` in `pypsa/consistency.py` runs before both, so its
message comes first where it refuses the same input.

## After a change

- **The lock fails on purpose.** `test/test_optimization_spec_lock.py` holds
  a hash of each key in `test/data/equivalence/spec_keys.json` and a digest of
  22 built models in `model_digests.json`. It prints each difference:

    ``` text
    changed default > constraints > Generator_p_ramp_limit_up: c4b6f90090018ccd (lock) -> 10245e77629715fd (now)
    changed maintenance_modules_ramps > constraints > Link-p-ramp_limit_up > rhs: 150.0 (lock) -> 135.0 (now)
    ```

    Check that each line is a change you want. Then record both files again
    and commit them. This takes about one minute:

    ``` bash
    uv run --frozen --no-sync python test/test_optimization_spec_lock.py
    ```

- **The equivalence test** `test/test_optimization_spec_equivalence.py`
  solves 24 networks and compares each objective with the value in
  `objectives.json`, recorded on `master`. A refactor must pass it. A change
  that changes an optimum fails it on purpose. Record the values again with
  `python test/test_optimization_spec_equivalence.py <commit>` and give the
  reason in the pull request.
- **The structure tests** in `test/test_optimization_spec.py` check the names,
  the trait files and the docs pages.
- **The math pages** under
  [Formulation (generated)](../user-guide/optimization/formulation/index.md)
  are rendered from the YAML when the docs build. A guide page shows an
  entry with a marker in an HTML comment, as `md:spec-table Store_cycles_max`
  in `docs/user-guide/optimization/storage.md`. `symbols.yaml` can give a new
  name a LaTeX symbol.
- **The release note** in `docs/release-notes.md` names each new attribute or
  entry. Mark a change of results as **Behavior change**. End each item with
  the `md:pr` marker of the pull request, as the other items do.

## Rules

- [How keys are named](../user-guide/optimization/custom-constraints.md#how-keys-are-named)
  states which keys are stable until PyPSA 2.0 and which are internal.
- A change of an entry, other than of its `description:`, breaks each
  [stored override](../user-guide/optimization/custom-constraints.md#storing-overrides-on-the-network)
  that edits the entry. The build fails until the user registers the
  override again.
- A rename of a key breaks the overrides and guide-page markers that name it.
  The docs promise that a renamed stable key stays valid with a deprecation
  warning. PyPSA has no code for this yet, so add it before you rename a
  stable key.

## Where to look when it fails

| Message | Meaning |
| --- | --- |
| `PyPSA has no data for the parameter 'Link_full_load_hours_max': Link has no attribute 'full_load_hours_max'.` | A component of the trait lacks the column. Add it to its CSV file or give the parameter a `missing:` value. |
| `Constraint 'COMPONENT_full_load_hours_max': 'energy_limit_weight' not found.` | A trait reads a name that its `given:` does not declare. |
| `Named expression 'Store_ramp_down_allowance', case 'committed': 'Store_committable' not found.` | A component got a trait, but lacks an entry that the trait reads. |
| `KeyError: 'LEVEL'` | A trait uses a placeholder that a component of the trait has no value for. |
| `` … declares ['Generator_e_sum_max'] in `parameters`, which another file declares too. `` | Two files declare one key. Keep one. |
| `'Generator_p_curtailed' is declared twice, as constraint and as variable. Rename one.` | The docs pages refuse a variable and a constraint with the same key. |
| `The stored override 'ramp' changes ['constraints.Generator_p_ramp_limit_up'], which PyPSA now defines differently.` | You changed an entry that a stored override edits. See [Rules](#rules). |
