###########################
Quick Start
###########################

See also the existing :doc:`examples` and the example Jupyter
notebooks at `http://www.pypsa.org/examples/
<http://www.pypsa.org/examples/>`_.


Installation
============

For full installation instructions see :doc:`installation`.

If you have the Python package installer ``pip`` then just run::

    pip install pypsa



Build a minimal network for power flow
======================================

This example is downloadable at `http://www.pypsa.org/examples/
<http://www.pypsa.org/examples/>`_.

.. code:: python

    import pypsa

    import numpy as np

    network = pypsa.Network()

    #add three buses
    for i in range(3):
        network.add("Bus","My bus {}".format(i))

    print(network.buses)

    #add three lines in a ring
    for i in range(3):
        network.add("Line","My line {}".format(i),
	            bus0="My bus {}".format(i),
		    bus1="My bus {}".format((i+1)%3),
		    x=0.0001)

    print(network.lines)

    #add a generator at bus 0
    network.add("Generator","My gen",
                bus="My bus 0",
		p_set=100)

    print(network.generators)

    print(network.generators_t.p_set)

    #add a load at bus 1
    network.add("Load","My load",
                bus="My bus 1",
		p_set=100)

    print(network.loads)

    print(network.loads_t.p_set)

    #Do a Newton-Raphson power flow
    network.pf()

    print(network.lines_t.p0)

    print(network.buses_t.v_ang*180/np.pi)



Build a minimal network for optimal power flow
==============================================


This example is downloadable at `http://www.pypsa.org/examples/
<http://www.pypsa.org/examples/>`_.


.. code:: python

	import pypsa

	import numpy as np

	network = pypsa.Network()

	#add three buses
	for i in range(3):
	    network.add("Bus","My bus {}".format(i))

	print(network.buses)

	#add three lines in a ring
	for i in range(3):
	    network.add("Line","My line {}".format(i),
		        bus0="My bus {}".format(i),
		        bus1="My bus {}".format((i+1)%3),
		        x=0.0001,
		        s_nom=60)

	print(network.lines)

	#add a generator at bus 0
	network.add("Generator","My gen 0",
		    bus="My bus 0",
		    p_nom=100,
		    marginal_cost=50)

	#add a generator at bus 1
	network.add("Generator","My gen 1",
		    bus="My bus 1",
		    p_nom=100,
		    marginal_cost=25)

	print(network.generators)

	print(network.generators_t.p_set)

	#add a load at bus 2
	network.add("Load","My load",
		    bus="My bus 2",
		    p_set=100)

	print(network.loads)

	print(network.loads_t.p_set)

	#Do a linear OPF
	network.lopf()

	print(network.generators_t.p)

	print(network.lines_t.p0)

	print(network.buses_t.v_ang*180/np.pi)



Use an existing example
=======================

The examples are NOT bundled with the ``pip`` package but can be
downloaded from the `PyPSA github repository
<https://github.com/PyPSA/PyPSA>`_ or as Jupyter notebooks from the
`PyPSA website <http://www.pypsa.org/examples/>`_.


.. code:: python

    import pypsa

    #adjust the path to pypsa examples directory
    network = pypsa.Network(csv_folder_name="path/to/pypsa/examples/ac-dc-meshed/ac-dc-data")

    #set to your favourite solver
    solver_name = "glpk"

    network.lopf(snapshots=network.snapshots,solver_name=solver_name)


    print(network.generators.p_nom_opt)

    print(network.generators_t.p)

    print(network.storage_units.p_nom_opt)

    print(network.storage_units_t.p)

    print(network.lines.s_nom_opt)

    print(network.lines_t.p0)
