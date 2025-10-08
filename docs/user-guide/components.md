# Components object

TODO: Update section

The [`pypsa.Components`][] are the store for all component specific data. While the [`pypsa.Network`][] bundles together functionality across components, the Components class is the interface for all data and processing for a specific component type. The two data stores are [`c.static`][pypsa.Components.static] and [`c.dynamic`][pypsa.Components.dynamic] (with an additional `snapshots` dimension).

``` py
>>> import pypsa
>>> n = pypsa.Network()
>>> n.components.generators
Empty 'Generator' Components
```

!!! tip

    A short name, such as `c`, is recommended since it is used frequently to access the stored data, properties and methods.

TODO: Update section

## Components Store

Components can be accessed in any Network via the Components store [`n.components`][pypsa.Network.components]. The Components store is a dict-like object that contains all components of the network.

!!! info

    There is also an alias [`n.c`][pypsa.Network.c] for [`n.components`][pypsa.Network.components].

Access a single component:
``` py
>>> c = n.components.generators
>>> c = n.components["Generator"] # also subscriptable
>>> c
Empty 'Generator' Components
```

Access a list of components:
``` py
>>> comps = n.components["Generator", "Bus"]  # doctest: +SKIP
>>> comps  # doctest: +SKIP
[Empty 'Generator' Components, Empty 'Bus' Components]
```

Loop through all components:
``` py
>>> for c in n.components:
...     break
```
!!! info

    Even if assigned to a variable, the components are not copied. They are still attached to the network. If you change any data of the components object, the changes will be reflected in the network as well. There is no need to re-assign the components object to the network.

## New Components Class API
PyPSA components have been introduced in version [`v0.33`](../release-notes.md#v0.33.0). Prior to that components data was only available in the two data stores `n.generators` and `n.generators_t`, directly attached to the network and not coupled together. For backward compatibility, the same `pandas`-based structure is still used, but `n.generators` actually refers to [`n.components.generators.static`][pypsa.Components.static].

The current components API is therefore a bit confusing:

- `n.generators` -> reference to [`n.components.generators.static`][pypsa.Components.static]
- `n.generator_t` -> reference to [`n.components.generators.dynamic`][pypsa.Components.dynamic]

To get access to the full functionality of [`pypsa.Components`][] you need to use the long namespace or assign them to a variable. E.g. to get the list of components which support unit commitment:

``` py
>>> committables = n.components["Generator"].committables
>>> committables
Index([], dtype='int64', name='name')
>>> c = n.components["Generator"]
>>> c.committables
Index([], dtype='int64', name='name')
```

### New API

PyPSA `v1.0` now allows you to opt in to the new Components API. This simply changes the reference to:

| Namespace | Current API | Opt-in API |
|-----------|--------------|------------|
| `n.generators` | [`n.components.generators.static`][pypsa.Components.static] | [`n.components.generators`][pypsa.components.Components] |
| `n.generators_t` | [`n.components.generators.dynamic`][pypsa.Components.dynamic] | Deprecated |

With the new API, the usage would be much more intuitive, the actual relationship between static and dynamic data is clear and the vast amount of Components class features are faster to access with full support for auto-completion in IDEs. Using them can remove a lot of boilerplate code, which is otherwise needed.

On the downside is that accessing the main static dataframe of components data is a bit more cumbersome (e.g. `n.generators.static` instead of `n.generators`). But assigning the dataframe to an variable is still possible and we are currently discussing another alias.

And secondly, the existing code base needs to be refactored. 

### Migrating to new API
To make the migration as easy as possible, and to also allow step-by-step migration, the <!-- md:guide options.md --> module is used.

As a quick example, let's take a snippet of the [Three-Node Capacity Expansion Example](../examples/3-node-cem.ipynb).

``` py
# Interconnector Capacities
>>> n.links.query("carrier == 'HVDC'").p_nom_opt.round(2)
Series([], Name: p_nom_opt, dtype: float64)

# Interconnector Flows
>>> n.links_t.p0.loc[:, n.links.carrier == "HVDC"].rolling("7d").mean()  # doctest: +SKIP
```

To switch to the new API, we just need to set the package option `api.new_components_api` to `True`.

``` py
pypsa.options.api.new_components_api = True

# Interconnector Capacities
n.links.static.query("carrier == 'HVDC'").p_nom_opt.round(2)

# Interconnector Flows
n.links.dynamic.p0.loc[:, n.links.static.carrier == "HVDC"].rolling("7d").mean()
```

#### Step-by-step migration
To migrate a script step by step, we can just set the option back again.

``` py
pypsa.options.api.new_components_api = True

# Interconnector Capacities
n.links.static.query("carrier == 'HVDC'").p_nom_opt.round(2)

# Switch back to old API
pypsa.options.api.new_components_api = False

# Interconnector Flows
n.links_t.p0.loc[:, n.links.carrier == "HVDC"].rolling("7d").mean()
```

Another way is to use the [`option_context`][pypsa.option_context] context manager to temporarily switch the API.

``` py
with pypsa.options_context(api.new_components_api=True):
    # Interconnector Capacities
    n.links.static.query("carrier == 'HVDC'").p_nom_opt.round(2)

# Interconnector Flows
n.links_t.p0.loc[:, n.links.carrier == "HVDC"].rolling("7d").mean()
```

PyPSA `v1.0` will support full functionality for both APIs. The example above just shows how to immediately translate between the two APIs. It often also makes sense to assign the components object to a variable, which is then used in the rest of the script.

### Default in PyPSA v2.0

With Version `v2.0` of PyPSA, we plan to enable the new API by default with an opt-out option to still support old implementations. But already the following versions will add more functionality to the Components class.

!!! info

    We are happy to receive feedback on this planned change of the API. Please open an issue on [GitHub](https://github.com/PyPSA/PyPSA/issues) or join the shared [Discord server](https://discord.gg/AnuJBk23FU).