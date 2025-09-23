# Components class

Version `0.33` of PyPSA introduces a structural refactoring of how component data is stored and accessed. The new structure adds an extra layer to move all component-specific data from the networks class to a new component class. With version `0.33`, most of these changes will be unnoticeable to the user. 

But this makes it easy to add new features. Below are some simple examples to show which other features could be added in the future. If you have any ideas, wishes, feedback or suggestions, please let us know via https://github.com/PyPSA/PyPSA/issues.

Note that this is experimental. Features will be added and the newly introduced API may change. You can still use PyPSA as usual. Major API changes will not be introduced before version `1.0`. While the `Components` class does not introduce any breaking changes, **the `ComponentsStore` leads to slightly different behavior for `n.components`**. See the explanation below.

Also, while all classes and methods have docstrings, there is no dedicated documentation yet.

## General

```python
import pypsa

n = pypsa.examples.scigrid_de()
```

### Components class

So far, components data was directly attached to the network object (e.g. `n.generators`, `n.generators_t` etc.). While you still can access the data there, both actually sit now in a new `Components` class:

```python
c = n.components.generators  # also via alias n.c.generators
c
```

The datasets for static and dynamic data can be accessed via the class now, but also still via the old network properties:

```python
c.static.head()
```

```python
c.dynamic.keys()
```

```python
# Both ways refer to the same DataFrame/ Dict Container of
print(c.static is n.generators)
print(c.dynamic is n.generators_t)
```

### Components Store
There have been some major changes to `n.components`, which is now the basic store for all components. Before version `0.33`, `n.components` was a dictionary containing only the default component data, and no static or dynamic data. Now it contains both (as described above), while still allowing access to the default data:

```python
print(f"List name: '{n.components['Generator'].list_name}'")
print(f"Description: '{n.components['Generator'].description}'")
```

But the **iteration behaviour is different**. While a dictionary only returns the keys, the new `ComponentsStore' object returns the components themselves (similar to a list). This leads to a break when using it:

```python
for comp in n.components:
    break
```

and `__contains__`/ x in n.components is not supported anymore:

```python
try:
    "x" in n.components
except Exception as e:
    print(f"{type(e).__name__}: {e}")
```

## Examples

```python
c = n.components.generators
```

### Simple alias properties

```python
# Basic component information
print(f"Component name: '{c.name}'")
print(f"Component list name: '{c.list_name}'")
print(f"Component type: '{c.type}'")
```

```python
# Quick access to attribute units
c.units.head()
```

```python
# Get ports of component (e.g. for multiport components)
n.c.links.ports
```

```python
# Check if component is attached to network
if c.attached:
    print(f"{c} is attached to {c.n}")
```

### Rename components and propagate new names through network

```python
# Old names
print(f"Old bus names: {', '.join(c.static.head(2).index)}")
```

```python
# Rename "1 Gas" component
c = n.components.buses
rename_map = {"1": "Super Bus"}
c.rename_component_names(**rename_map)
```

```python
# New names
print(f"New bus names: {', '.join(c.static.head(2).index)}")
```

```python
# Changes in other components of network
n.c.generators.static.head(2)
```

### Calculate line length from attached buses

```python
c = n.c.lines
c.calculate_line_length()
```

Those are just a couple of simple examples. Many other features could be added. If you have any ideas, wishes, feedback or suggestions, please let us know via the https://github.com/PyPSA/PyPSA/issues. 