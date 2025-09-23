# Optimization with Linopy

PyPSA uses Linopy as the optimization backend. Linopy is a stand-alone package and works similar to Pyomo, but without the memory overhead and much faster. For additional information on the Linopy package, have a look at the [documentation](https://linopy.readthedocs.io/en/latest/).

## Let's get started

Now, we demonstrate the behaviour of the optimization with linopy. The core functions for the optimization can be called via the [pypsa.Network.optimize](https://pypsa.readthedocs.io/en/latest/api/optimization.html) accessor. The accessor is used for creating, solving, modifying the optimization problem. Further, it supports to run different optimization formulations and provides helper functions. 

At first, we run the ordinary linearized optimal power flow (LOPF). We then extend the formulation by some additional constraints.

```python
import pypsa
```

```python
n = pypsa.examples.ac_dc_meshed(from_master=True)
```

In order to make the network a bit more interesting, we modify its data: We set gas generators to non-extendable,

```python
n.generators.loc[n.generators.carrier == "gas", "p_nom_extendable"] = False
```

... add ramp limits,

```python
n.generators.loc[n.generators.carrier == "gas", "ramp_limit_down"] = 0.2
n.generators.loc[n.generators.carrier == "gas", "ramp_limit_up"] = 0.2
```

... add additional storage units (cyclic and non-cyclic) and fix one state_of_charge,

```python
n.add(
    "StorageUnit",
    "su",
    bus="Manchester",
    marginal_cost=10,
    inflow=50,
    p_nom_extendable=True,
    capital_cost=10,
    p_nom=2000,
    efficiency_dispatch=0.5,
    cyclic_state_of_charge=True,
    state_of_charge_initial=1000,
)

n.add(
    "StorageUnit",
    "su2",
    bus="Manchester",
    marginal_cost=10,
    p_nom_extendable=True,
    capital_cost=50,
    p_nom=2000,
    efficiency_dispatch=0.5,
    carrier="gas",
    cyclic_state_of_charge=False,
    state_of_charge_initial=1000,
)

n.storage_units_t.state_of_charge_set.loc[n.snapshots[7], "su"] = 100
```

...and add an additional store.

```python
n.add("Bus", "storebus", carrier="hydro", x=-5, y=55)
n.add(
    "Link",
    ["battery_power", "battery_discharge"],
    "",
    bus0=["Manchester", "storebus"],
    bus1=["storebus", "Manchester"],
    p_nom=100,
    efficiency=0.9,
    p_nom_extendable=True,
    p_nom_max=1000,
)
n.add(
    "Store",
    ["store"],
    bus="storebus",
    e_nom=2000,
    e_nom_extendable=True,
    marginal_cost=10,
    capital_cost=10,
    e_nom_max=5000,
    e_initial=100,
    e_cyclic=True,
);
```

## Run Optimization

The optimization based on linopy mimics the well-known [`n.lopf`](https://pypsa.readthedocs.io/en/v0.28.0/api_reference.html#pypsa.Network.lopf) optimization. We run it by calling the `optimize` accessor.

```python
n.optimize()
```

We now have a model instance attached to our network. It is a container of all variables, constraints and the objective function. You can modify this as much as you please, by directly adding or deleting variables or constraints etc.

```python
n.model
```

## Modify model, optimize and feed back to network 

When you have a fresh network and you just want to create the model instance, run

```python
n.optimize.create_model()
```

Through the model instance we gain a lot of flexibility. Let's say for example we want to remove the Kirchhoff Voltage Law constraint, thus convert the model to a transport model. This can be done via

```python
n.model.constraints.remove("Kirchhoff-Voltage-Law")
```

Now, we want to optimize the altered model and feed to solution back to the network. Here again, we use the `optimize` accessor.

```python
n.optimize.solve_model()
```

Here, we followed the recommended way to run altered models:

1. **Create the model instance** - `n.optimize.create_model()`
2. **Modify the model to your needs**
3. **Solve and feed back** - `n.optimize.solve_model()`


For compatibility reasons the `optimize` function, also allows passing a `extra_funcionality` argument, as we know it from the `lopf` function. The above behaviour with use of the extra functionality is obtained through

```python
def remove_kvl(n, sns):
    print("KVL removed!")
    n.model.constraints.remove("Kirchhoff-Voltage-Law")


n.optimize(extra_functionality=remove_kvl)
```

## Additional constraints

In the following, we exemplarily present a set of additional constraints. Note, the dual values of the additional constraints won't be stored in default data fields in the `PyPSA` network. But in any case, they are stored in the `linopy.Model`. 

Again, we **first build** the optimization model, **add our constraints** and finally **solve the network**. For the first step, we use again our accessor `optimize` to access the function `create_model`. This returns the `linopy` model that we can modify.

```python
m = n.optimize.create_model()  # the return value is the model, let's use it directly!
```

1. **Minimum for state of charge**

Assume we want to set a minimum state of charge of 50 MWh in our storage unit. This is done by: 

```python
sus = m.variables["StorageUnit-state_of_charge"]
m.add_constraints(sus >= 50, name="StorageUnit-minimum_soc")
```

The return value of the `add_constraints` function is a array with the labels of the constraints. You can access the constraint now through: 

```python
m.constraints["StorageUnit-minimum_soc"]
```

and inspects its attributes like `lhs`, `sign` and `rhs`, e.g.

```python
m.constraints["StorageUnit-minimum_soc"].rhs
```

2. **Fix the ratio between ingoing and outgoing capacity of the Store**

The battery in our system is modelled with two links and a store. We should make sure that its charging and discharging capacities, meaning their links, are somehow coupled. 

```python
capacity = m.variables["Link-p_nom"]
eff = n.links.at["battery_power", "efficiency"]
lhs = capacity.loc["battery_power"] - eff * capacity.loc["battery_discharge"]
m.add_constraints(lhs == 0, name="Link-battery_fix_ratio")
```

3. **Every bus must in total produce the 20% of the total demand**

For this, we use the linopy function `groupby_sum` which follows the pattern from `pandas`/`xarray`'s `groupby` function.

```python
total_demand = n.loads_t.p_set.sum().sum()
buses = n.generators.bus.to_xarray()
prod_per_bus = m.variables["Generator-p"].groupby(buses).sum().sum("snapshot")
m.add_constraints(prod_per_bus >= total_demand / 5, name="Bus-minimum_production_share")
```

```python
con = prod_per_bus >= total_demand / 5
```

```python
con
```

... and now let's solve the network again. 

```python
n.optimize.solve_model()
```

## Analysing the constraints

Let's see if the system got our own constraints. We look at `n.constraints` which combines summarises constraints going into the linear problem

```python
n.model.constraints
```

The last three entries show our constraints. Let's check whether out two custom constraint are fulfilled:

```python
n.links.loc[["battery_power", "battery_discharge"], ["p_nom_opt"]]
```

```python
n.storage_units_t.state_of_charge
```

```python
n.generators_t.p.groupby(n.generators.bus, axis=1).sum().sum() / n.loads_t.p.sum().sum()
```

Looks good! Now, let's see which dual values were parsed. Therefore we have a look into `n.model.dual` 

```python
n.model.dual
```

```python
n.model.dual["StorageUnit-minimum_soc"]
```

```python
n.model.dual["Link-battery_fix_ratio"]
```

```python
n.model.dual["Bus-minimum_production_share"]
```

These are the basic functionalities of the `optimize` accessor. There are many more functions like abstract optimziation formulations (security constraint optimization, iterative transmission expansion optimization, etc.) or helper functions (fixing optimized capacities, adding load shedding). Try them out if you want!

```python
print("\n".join([func for func in n.optimize.__dir__() if not func.startswith("_")])) 