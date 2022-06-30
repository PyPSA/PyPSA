###########################
Quick Start
###########################

For installation instructions see :doc:`installation`.

See also the :doc:`examples-basic` examples with executable Jupyter notebooks.

In principle, PyPSA networks can be assigned with

.. code:: python

    import pypsa
    import numpy as np

    network = pypsa.Network()

Components like buses can be added with :py:meth:`pypsa.Network.add` or :py:meth:`pypsa.Network.madd`

.. code:: python

    #add three buses
    n_buses = 3

    for i in range(n_buses):
        network.add("Bus", "My bus {}".format(i),  v_nom=20.)

The same counts for lines, generators and loads, see the list of all components :doc:`components`.

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


Note that fixed values of active and reactive power are set with ``p_set`` and ``q_set`` respectively. After initializing, there are plenty of options what you can do with your network. The examples section gives a broad overview.
