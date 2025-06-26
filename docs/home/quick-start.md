# Quickstart

For installation instructions see [installation](installation.md).

In principle, PyPSA networks can be created with

``` python

import pypsa

network = pypsa.Network()
```

Components like buses can be added with [`n.add`][pypsa.Network.add].

``` python

#add three buses
n_buses = 3

for i in range(n_buses):
    network.add("Bus", "My bus {}".format(i),  v_nom=20.)
```

The same counts for lines, generators and loads. See the list of all components in the [:material-bookshelf: user guide section](/user-guide/components.md).

``` python

#add three lines in a ring
for i in range(n_buses):
    network.add("Line", "My line {}".format(i),
    bus0="My bus {}".format(i),
    bus1="My bus {}".format((i+1)%3),
    x=0.1,
    r=0.01)

#add a generator at bus 0
network.add("Generator", "My gen",
bus="My bus 0",
p_set=100,
control="PQ")


#add a load at bus 1
network.add("Load", "My load",
bus="My bus 1",
p_set=100,
q_set=100)

```

Note that fixed values of active and reactive power are set with `p_set` and `q_set` respectively. After initializing, there are plenty of options for what you can do with your network. The examples section gives a broad overview.

## Basic Examples

Two more basic examples are given in the following notebooks:

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Minimal three node network**

    ---

    Here, we are going to create a network with three nodes, three lines and one generator. We then solve the non-linear power flow using a Newton-Raphson. 
    
    [:material-notebook: Go to example](../examples/minimal_example_pf.ipynb)

-   :material-view-list:{ .lg .middle } **Meshed AC-DC example**

    ---

    This example has a 3-node AC network coupled via AC-DC converters to a 3-node DC network. There is also a single point-to-point DC using the Link component.

    [:material-notebook: Go to example](../examples/ac-dc-lopf.ipynb)

</div>


Find many more extensive examples in the [examples](examples.md) section. The [user guide](user-guide.md) section contains detailed information on architecture, components and utilities.

