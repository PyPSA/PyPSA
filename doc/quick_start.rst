###########################
Quick Start
###########################

See also the :doc:`examples`.



.. code:: python

    import pypsa

    #adjust the path to pypsa
    network = pypsa.Network(csv_folder_name="path/to/pypsa/examples/ac-dc-meshed/ac-dc-data")

    #set to your favourite solver
    solver_name = "glpk"

    network.lopf(snapshots=network.snapshots,solver_name=solver_name)


    print(network.generators.p_nom)

    print(network.generators.p)

    print(network.storage_units.p_nom)

    print(network.storage_units.p)

    print(network.lines.s_nom)

    print(network.lines.p0)
