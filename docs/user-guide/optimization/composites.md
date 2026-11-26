<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Composites

A composite is a group of components that model one technology, together with
the math that ties them. A battery, for example, is a bus, a charger, a
discharger and a store, where the charger and the discharger share one
inverter. A composite type is defined once and added as often as needed. Each
instance is nothing but its member components, so statistics, plots, copies and
saved files work as before.

## Defining a type

A definition is a YAML file, YAML text or a dictionary. It has a `name` and
four sections:

* `parameters`: what `n.add` takes, each with its default. `null` makes a
  parameter required. Each parameter is read by a template or by the math,
  not by both.
* `components`: the members per component class, each a template of
  attributes. A value `$<parameter>` takes the parameter, a value `@<member>`
  the name of another member of the same instance. Quote it in YAML, as
  `"@dc"`.
* `math`: the `variables`, `expressions` and `constraints` of the type, in the
  language of the [specification](formulation/index.md).
* `description`: what an instance is.

``` yaml title="battery.yaml"
name: battery
description: a battery whose charger and discharger share one inverter
parameters:
  bus: null
  efficiency: 0.95
  max_hours: 8
  capital_cost_power: 0
  capital_cost_energy: 0
components:
  Bus:
    dc: {carrier: battery}
  Link:
    charger: {bus0: $bus, bus1: "@dc", efficiency: $efficiency, p_nom_extendable: true, capital_cost: $capital_cost_power}
    discharger: {bus0: "@dc", bus1: $bus, efficiency: $efficiency, p_nom_extendable: true}
  Store:
    store: {bus: "@dc", e_nom_extendable: true, e_cyclic: true, capital_cost: $capital_cost_energy}
math:
  constraints:
    battery_inverter:
      description: the charger and the discharger share one rating
      dims: [scenario, battery]
      expression: >-
        at(Link_p_nom_effective, by=battery_members[charger])
        == at(Link_p_nom_effective, by=battery_members[discharger])
    battery_duration:
      description: the store holds at most `max_hours` of discharge
      dims: [scenario, battery]
      expression: >-
        at(Store_e_nom_effective, by=battery_members[store])
        <= battery_max_hours * at(Link_p_nom_effective, by=battery_members[discharger])
```

The math may use every entry of the formulation. PyPSA declares three more
for each type:

* the dimension `battery`, over the instances;
* the relation `battery_members`, which maps each instance to its members, one
  column per member. `at(Link_p_nom_effective, by=battery_members[charger])`
  reads the rating of each instance's charger;
* a parameter `battery_<parameter>` for each parameter the math reads, as
  `battery_max_hours`. Its default sets its dtype: `true` or `false` makes it
  a bool, a text a str, a number a float. So it cannot be required.

Where the math needs an attribute a template sets, it reads it from the
member, as `at(Link_efficiency, by=battery_members[charger])`.

`Link_p_nom_effective` is the chosen rating of an extendable link and the given
rating of a fixed one, so the math holds when a member is fixed.

## Adding instances

``` py
n.composites.register("battery.yaml")
n.add("battery", ["north", "south"], bus=["north", "south"], max_hours=[4, 8])
n.optimize()
```

`n.composites.battery.add(...)` does the same as `n.add("battery", ...)`, and
takes a `suffix` alike. A parameter is a scalar for all instances, a list or a
Series over the instances, or, for a time-varying attribute of a member, a
DataFrame with the instances as columns or a Series over the snapshots for all
of them. If `add` fails, it adds nothing. A member is named `<instance> <member>`, as `north charger`, and
carries three columns: `composite` with its instance, `composite_type` and
`composite_member`. The parameters of the math take one value per instance.
They are stored in the columns `<type>_<parameter>` of the first member listed,
here the bus `dc`, as `n.c.buses.static["battery_max_hours"]`. In a stochastic
network, that column may differ by scenario.

`n.remove("battery", "north")` removes every member of an instance.

## Reading instances

``` py
n.composites.battery.members   # the members of each instance
n.composites.battery.static    # the parameters of each instance, read from its members
n.statistics.energy_balance(groupby="composite")
```

## Rules

* The definition is stored in `n.meta["_pypsa"]`, so a copy, a slice `n[buses]`, a
  merge and a saved network keep its composites. `n.merge` refuses two
  different types of one name. Clustering does not keep it; the build then
  refuses the tagged members until the type is registered again.
* `register` checks the math against the default formulation, the other types and the
  [stored overrides](custom-constraints.md#storing-overrides-on-the-network),
  so a wrong name fails there, not at the build. A type cannot declare an
  entry that a stored override adds.
* `n.composites.remove("battery")` unregisters a type once no instance or stored
  override uses it, also where its math no longer builds.
* `n.add` takes no `overwrite` for a composite, and refuses an instance whose
  member names exist.
* Every key of the math starts with `<type>_`. The math adds to the
  formulation; it does not change or remove an entry. Use
  [overrides](custom-constraints.md#overriding-the-formulation) for that.
* An instance is built in full or not at all: deactivate or remove all of its
  members, not some.
