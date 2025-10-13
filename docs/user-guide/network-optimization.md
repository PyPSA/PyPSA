<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Optimization

!!! info "See Also"

    Please also refer the [optimization section](optimization/overview.md) above, which offers more information on the mathematical formulations PyPSA uses when running [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__].

PyPSA uses [Linopy](https://linopy.readthedocs.io/en/latest/) as the optimization backend. Linopy is a stand-alone package and works similar to [Pyomo](https://www.pyomo.org/), but without the memory overhead and much faster. However, it has a reduced set of optimization problem types it can support (LP, MILP, QP). The core optimization functions can be called with [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__]. This call unifies creating the problem, solving it and retrieving the optimized results.
The accessor [n.optimize][pypsa.optimization.OptimizationAccessor] additionally offers a range of different functionalities. 

## Initialize Network

Initially, let us consider one of the canonical PyPSA examples and solve it with [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__].

``` py
>>> import pypsa
>>> n = pypsa.examples.ac_dc_meshed()
```

In order to make the network a bit more interesting, we modify its data by setting gas generators to be non-extendable,

``` py
>>> n.generators.loc[n.generators.carrier == "gas", "p_nom_extendable"] = False
```

... adding ramp limits,

``` py
>>> n.generators.loc[n.generators.carrier == "gas", "ramp_limit_down"] = 0.2
>>> n.generators.loc[n.generators.carrier == "gas", "ramp_limit_up"] = 0.2
```

... adding additional storage units (cyclic and non-cyclic) and fixing the state of charge of one storage unit,

``` py
>>> n.add(
...     "StorageUnit",
...     "su",
...     bus="Manchester",
...     marginal_cost=10,
...     inflow=50,
...     p_nom_extendable=True,
...     capital_cost=10,
...     p_nom=2000,
...     efficiency_dispatch=0.5,
...     cyclic_state_of_charge=True,
...     state_of_charge_initial=1000,
... )
>>> n.add(
...     "StorageUnit",
...     "su2",
...     bus="Manchester",
...     marginal_cost=10,
...     p_nom_extendable=True,
...     capital_cost=50,
...     p_nom=2000,
...     efficiency_dispatch=0.5,
...     carrier="gas",
...     cyclic_state_of_charge=False,
...     state_of_charge_initial=1000,
... )
>>> n.storage_units_t.state_of_charge_set.loc[n.snapshots[7], "su"] = 100
```

...and adding an additional store.

``` py
>>> n.add("Bus", "storebus", carrier="hydro", x=-5, y=55)
>>> n.add(
...     "Link",
...     ["battery_power", "battery_discharge"],
...     "",
...     bus0=["Manchester", "storebus"],
...     bus1=["storebus", "Manchester"],
...     p_nom=100,
...     efficiency=0.9,
...     p_nom_extendable=True,
...     p_nom_max=1000,
... )
>>> n.add(
...     "Store",
...     ["store"],
...     bus="storebus",
...     e_nom=2000,
...     e_nom_extendable=True,
...     marginal_cost=10,
...     capital_cost=10,
...     e_nom_max=5000,
...     e_initial=100,
...     e_cyclic=True,
... )
```

## Run Optimization

Now, let's solve the network.

``` py
>>> n.optimize()
('ok', 'optimal')
```

We now have a model instance attached to our network object. It is a container of all variables, constraints and the objective function. It can be modified by directly adding or deleting variables or constraints or changing the objective function.

``` py
>>> n.model  # doctest: +ELLIPSIS
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
ok
```

Results are written to the network components' data fields, for instance:

``` py
n.generators_t.p
n.stores.e_nom_opt
n.buses_t.marginal_price
```

## Modify Model & Re-Optimize

The function call to [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__] already solves the model directly after creating it.
To create the model instance without solving it yet, you can use the following command:

``` py
>>> n.optimize.create_model()  # doctest: +ELLIPSIS
Linopy LP model
===============
<BLANKLINE>
Variables:
----------
...
<BLANKLINE>
Constraints:
------------
...
<BLANKLINE>
Status:
-------
initialized
```

With the access to the model instance we gain a lot of flexibility. Let's say, for example, we want to remove the Kirchhoff Voltage Law constraint, thus converting the model to a transport model. This can be done via

``` py
>>> n.model.constraints.remove("Kirchhoff-Voltage-Law")
```

Now, we can solve the altered model and write the solution back to the network. Here again, we use the [`n.optimize`][pypsa.optimization.OptimizationAccessor] accessor:

``` py
>>> n.optimize.solve_model()
('ok', 'optimal')
```

Here, we followed the recommended way to create and solve models with custom alterations:

1. **Create the model instance** - `n.optimize.create_model()`
2. **Modify the model to your needs** - modify `n.model`
3. **Solve and write back solution** - `n.optimize.solve_model()`

It is also possible to pass modifications as an `extra_functionality` argument to [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__]:

``` py
import pandas as pd

def remove_kvl(n: pypsa.Network, sns: pd.Index) -> None:
    n.model.constraints.remove("Kirchhoff-Voltage-Law")

n.optimize(extra_functionality=remove_kvl)
```

## Custom Constraints

!!! info "See Also"

    Please also refer to the user guide section on [custom constraints](optimization/custom-constraints.md).

In the following, we present a selection of examples for additional constraints. Note, the dual values of the additional constraints will not be stored in in the network object `n`, but just under `n.model` which is accessible only in the current session.

Again, we **first build** the optimization model, then **add our constraints** and finally **solve the network**:

``` py
>>> m = n.optimize.create_model()  # the return value is the model, let's use it directly!
```

### Minimum for state of charge

Assume we want to set a minimum state of charge of 50 MWh for our storage unit. This is done by: 

``` py
>>> sus = m.variables["StorageUnit-state_of_charge"]
>>> m.add_constraints(sus >= 50, name="StorageUnit-minimum_soc")  # doctest: +ELLIPSIS
Constraint `StorageUnit-minimum_soc` [snapshot: ..., name: ...]:
-------------------------------------------------------------
...
```

The return value of the `m.add_constraints()` function is an array containing constraint labels, which can be accessed through `m.constraints`.

``` py
>>> m.constraints["StorageUnit-minimum_soc"]  # doctest: +SKIP
```

and inspected via its attributes like `lhs`, `sign` and `rhs`, e.g.

``` py
>>> m.constraints["StorageUnit-minimum_soc"].rhs  # doctest: +SKIP
```

### Fix the ratio between incoming and outgoing capacity of the `Store`

The battery in our system is modelled with two links and a store. We should make sure that its charging and discharging capacities, i.e. the links representing the inverter, are coupled. 

``` py
>>> capacity = m.variables["Link-p_nom"]
>>> eff = n.links.at["battery_power", "efficiency"]
>>> lhs = capacity.loc["battery_power"] - eff * capacity.loc["battery_discharge"]
>>> m.add_constraints(lhs == 0, name="Link-battery_fix_ratio")  # doctest: +SKIP
```

### Every bus must in total produce the 20% of the total demand

For this, we use the `linopy` function `groupby_sum` which follows the pattern from `pandas` or `xarray` `groupby` functions.

``` py
>>> total_demand = n.loads_t.p_set.sum().sum()
>>> buses = n.generators.bus.to_xarray()
>>> prod_per_bus = m.variables["Generator-p"].groupby(buses).sum().sum("snapshot")
>>> m.add_constraints(prod_per_bus >= total_demand / 5, name="Bus-minimum_production_share")  # doctest: +SKIP
```

``` py
>>> con = prod_per_bus >= total_demand / 5
>>> con  # doctest: +SKIP
```

Now, let's solve the network again:

``` py
>>> n.optimize.solve_model()
('ok', 'optimal')
```

## Analysing Constraints

Let's see if the optimized system adheres to our custom constraints. Let's first look at `n.constraints` which summarises the constraints of the model:

``` py
>>> n.model.constraints  # doctest: +SKIP
```

The last three entries show our constraints. Let's check whether our two custom constraints are fulfilled:

``` py
>>> n.links.loc[["battery_power", "battery_discharge"], ["p_nom_opt"]]  # doctest: +SKIP
```

``` py
>>> n.storage_units_t.state_of_charge  # doctest: +SKIP
```

``` py
>>> n.generators_t.p.T.groupby(n.generators.bus).sum().sum() / n.loads_t.p.sum().sum()  # doctest: +SKIP
```

Looks good! Now, let's see which dual values were parsed. For that, we have a look into `n.model.dual`:

``` py
>>> n.model.dual  # doctest: +SKIP
```

``` py
>>> n.model.dual["StorageUnit-minimum_soc"]  # doctest: +SKIP
```

``` py
>>> n.model.dual["Link-battery_fix_ratio"]  # doctest: +SKIP
```

``` py
>>> n.model.dual["Bus-minimum_production_share"]  # doctest: +SKIP
```

These are the basic functionalities of the [`n.optimize`][pypsa.optimization.OptimizationAccessor] accessor. There are many more functions like extended problem formulations for security constraint optimization, iterative transmission expansion optimization, and rolling-horizon optimization alongside several helper functions (fixing optimized capacities, adding load shedding). Check them out in the [:octicons-code-16: API reference](../api/networks/optimize.md).
