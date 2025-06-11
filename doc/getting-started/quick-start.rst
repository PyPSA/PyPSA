###########################
Quick Start
###########################

For installation instructions see :doc:`/getting-started/installation`.

In principle, PyPSA networks can be assigned with

.. code:: python

    import pypsa
    import numpy as np

    network = pypsa.Network()

Components like buses can be added with :py:meth:`pypsa.Network.add`.

.. code:: python

    #add three buses
    n_buses = 3

    for i in range(n_buses):
        network.add("Bus", "My bus {}".format(i),  v_nom=20.)

The same counts for lines, generators and loads, see the list of all components :doc:`/user-guide/components`.

.. code:: python

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


Note that fixed values of active and reactive power are set with ``p_set`` and ``q_set`` respectively. After initializing, there are plenty of options for what you can do with your network. The examples section gives a broad overview.

Basic Examples
~~~~~~~~~~~~~~~

Two more basic examples are given in the following notebooks:

.. toctree::
   :maxdepth: 1

   ../examples/minimal_example_pf.ipynb
   ../examples/ac-dc-lopf.ipynb

Find many more extensive examples in the :doc:`Examples </examples-index/lopf>` section. Also
have a look on the :doc:`User Guide </user-guide/design>` section. 
