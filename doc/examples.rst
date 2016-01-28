################
 Examples
################


See ``pypsa/examples/`` and ``pypsa/test/`` for usage cases.



AC-DC meshed example
====================

See ``pypsa/examples/ac-dc-meshed/``.

This computes an example with three synchronous AC areas:

* The UK (Manchester, London, Norwich)
* Germany (Frankfurt and Bremen)
* Norway (a single node)

London and Bremen are connected by a point-to-point HVDC Transport
Link.


Norwich, Bremen and Norway are connected by a three-node, three-line
DC network.


The example scripts do LOPF and LPF.

SciGRID LOPF and LPF
====================

See ``pypsa/examples/opf-scigrid-de``.

The script ``add_load_gen_to_scigrid.py`` takes the `SciGRID
<http://scigrid.de/>`_ network for Germany, attaches load and
conventional generation data and exports the resulting network.

The script ``scigrid-lopf-then-pf.py`` then performs a linear OPF on
the network for several hours, expanding any lines where capacity is
needed.

Then the resulting optimised dispatch is run through a full non-linear
Newton-Raphson power flow.


Storage and HVDC OPF example
============================

See ``pypsa/examples/opf-storage-hvdc/``

System capacity optimisation with storage, AC and DC.
