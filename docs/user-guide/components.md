<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Components Object

[`pypsa.Components`][] are the store for all component specific data. While a [`pypsa.Network`][] bundles together functionality across components, the `Components` class is the interface for all data and processing for a specific component type. 

``` py
>>> import pypsa
>>> n = pypsa.examples.ac_dc_meshed()
>>> n.components.generators  # doctest: +ELLIPSIS
'Generator' Components
----------------------
...
```

!!! tip

    A short name, such as `c`, is recommended since it is used frequently to access the stored data, properties and methods.

## Components Store

Components can be accessed in any `Network` via the `Components` store [`n.components`][pypsa.Network.components]. The `Components` store is a dict-like object that contains all components of the network.

!!! info

    There is also an alias [`n.c`][pypsa.Network.c] for [`n.components`][pypsa.Network.components].

Access a single component:
``` py
>>> c = n.components.generators
>>> c  # doctest: +ELLIPSIS
'Generator' Components
----------------------
...
>>> c = n.components["Generator"] # also subscriptable
>>> c  # doctest: +ELLIPSIS
'Generator' Components
----------------------
...
```

Access a list of components:
``` py
>>> comps = n.components["Generator", "Bus"]  # doctest: +SKIP
>>> comps  # doctest: +SKIP
[Empty 'Generator' Components, Empty 'Bus' Components]
```

Loop through all components:
``` py
>>> for comp in n.components:
...     break
```
!!! info

    Even if assigned to a variable, the components are not copied. They are still attached to the network. If you change any data of the components object, the changes will be reflected in the network as well. There is no need to re-assign the components object to the network.

## Stored Data
All components data is stored in the two stores [`c.static`][pypsa.Components.static] and [`c.dynamic`][pypsa.Components.dynamic]:

- [`c.static`][pypsa.Components.static] contains all static data of the components, i.e. data that does not change over time. It is a simple `pandas.DataFrame` with the component names as index and all attributes as columns. E.g. for generators this includes `bus`, `carrier`, `p_nom`, etc.
``` py
>>> c = n.components.generators  # Set c to generators for examples
>>> c.static # doctest: +ELLIPSIS
                        bus control  ...
name                                 ...
Manchester Wind  Manchester      PQ  ...
Manchester Gas   Manchester      PQ  ...
...
```
- [`c.dynamic`][pypsa.Components.dynamic] contains all time-varying data of the component. It is a dict-like object that contains a `pandas.DataFrame` for each time-varying attribute. E.g. `p` or `p_max_pu`.
``` py
>>> c.dynamic.p_max_pu # doctest: +ELLIPSIS
name                 Manchester Wind  Frankfurt Wind  Norway Wind
snapshot
2015-01-01 00:00:00         0.930020        0.559078     0.974583
2015-01-01 01:00:00         0.485748        0.752910     0.481290
...
```

Stochastic Networks (see <!-- md:guide optimization/stochastic.md -->) and multi-period networks (see <!-- md:guide design.md -->) use also use the same structure, but add additional dimensions to the dataframes.

!!! info

    `c.static` and `c.dynamic` are exactly the same than `n.generators` and `n.generators_t`, which is still the default way to access the data and used in most of this documentation and any examples previous to version `v1.0`. See [below](#new-components-class-api) on the alternative way to access components data.


## Features
[`pypsa.Components`][] have been introduced as experimental in <!-- md:badge-version v0.33.0 --> and are now released as stable in <!-- md:badge-version v1.0.0 -->. They do not change anything in the underlying data structure of static and dynamic data stored in `pandas` DataFrames.

However, they add a lot of additional functionality that would otherwise have to be reimplemented on the underlying `pandas` DataFrames repeatedly. Using them reduces the need for boilerplate code, while still allowing users to continue using only the underlying DataFrames directly.

For a list of features, checkout the API documentation. [pypsa.Components][] lists all functionality which is available for all component types, while each type has its own class listing type-sensitive features. E.g. [pypsa.components.Generators][].

!!! info

    More features will be added in future releases. If you have any suggestions or requests, please open an issue on [GitHub](https://github.com/PyPSA/PyPSA/issues).   

## New Components Class API
Prior to version  <!-- md:badge-version v0.33.0 --> components data was only available in the two data stores `n.generators` and `n.generators_t`, which were directly attached to the network and not linked. With <!-- md:badge-version v1.0.0 --> they are still available in the same way, but now also via the newly introduced class. A new **optional** breaking API is introduced to make the usage of components more intuitive.

The current components API works as follows:

- `n.generators` -> reference to [`n.components.generators.static`][pypsa.Components.static]
- `n.generator_t` -> reference to [`n.components.generators.dynamic`][pypsa.Components.dynamic]

To get access to the full functionality of [pypsa.Components][] the long namespace must be used or assigned to a variable. For example, to rename a component across all dataframes ([pypsa.Components.rename_component_names][]) or add components with attribute type hints ([pypsa.components.Generators.add][]), the following code must be run:

``` py
>>> n.components["Generator"].rename_component_names(bus1="bus_renamed")  # doctest: +SKIP
>>> c = n.components["Generator"]
>>> c.add(name="New Gen", bus="bus_renamed", p_nom=100, carrier="wind")  # doctest: +SKIP
```

### Opt-in to new API

Therefore, PyPSA `v1.0` now allows you to opt in to an alternative, new Components API, which changes the reference to:

| Namespace | Current API | Opt-in API |
|-----------|--------------|------------|
| `n.generators` | [`n.components.generators.static`][pypsa.Components.static] | [`n.components.generators`][pypsa.Components] |
| `n.generators_t` | [`n.components.generators.dynamic`][pypsa.Components.dynamic] | Deprecated |

With the new API, working with components should become more intuitive. The relationship between static and dynamic data is clearer and the numerous features of the `Components` class are faster to access with full support for auto-completion in IDEs when adding components with `n.generators.add(...)`.

One downside is that accessing the main static dataframe with `n.generators.static` instead of `n.generators` changes and is a bit longer. However,using a variable (e.g. `c`) as reference is still possible. Additionally, using the new API requires changes in existing code bases.

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

# Now you can also use the full functionality of the Components class, for example:
n.links.additional_ports
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

!!! info

    With `v2.0` of PyPSA, there are ongoing discussions to enable the new API by default with an opt-out option to still support old implementations. 

    We are happy to receive feedback on this planned change of the API. Please open an issue on [GitHub](https://github.com/PyPSA/PyPSA/issues) or join the shared [Discord server](https://discord.gg/AnuJBk23FU).