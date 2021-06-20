################
 Examples
################


There are `extensive examples <http://www.pypsa.org/examples/>`_
available as Jupyter notebooks.
They are also available as Python scripts in ``pypsa/examples/`` and ``pypsa/test/``
in the `PyPSA github repository <https://github.com/PyPSA/PyPSA>`_.


SciGRID Germany LOPF, LPF and SCLOPF
====================================

See ``pypsa/examples/scigrid-de``.

The script ``add_load_gen_to_scigrid.py`` takes the `SciGRID
<http://scigrid.de/>`_ network for Germany, attaches load and
conventional generation data and exports the resulting network.

The script ``scigrid-lopf-then-pf.py`` then performs a linear OPF on
the network for several hours, expanding any lines where capacity is
needed.

Then the resulting optimised dispatch is run through a full non-linear
Newton-Raphson power flow.



AC-DC meshed example
====================

See ``pypsa/examples/ac-dc-meshed/``.

This computes an example with three synchronous AC areas:

* The UK (Manchester, London, Norwich)
* Germany (Frankfurt and Bremen)
* Norway (a single node)

London and Bremen are connected by a point-to-point HVDC Link.


Norwich, Bremen and Norway are connected by a three-node, three-line
DC network.


The example scripts do LOPF and LPF.

Storage and HVDC OPF example
============================

See ``pypsa/examples/opf-storage-hvdc/``

System capacity optimisation with storage, AC and DC.



Example of linear optimal power flow with coupling to the heating sector
========================================================================


See ``pypsa/examples/coupling-with-heating/`` and
`<http://www.pypsa.org/examples/lopf-with-heating.html>`_.


In this example three locations are optimised, each with an electric
bus and a heating bus and corresponding loads. At each location the
electric and heating buses are connected with heat pumps; heat can
also be supplied to the heat bus with a boiler. The electric buses are
connected with transmission lines and there are electrical generators
at two of the nodes.
