######################
 Plotting Networks
######################

PyPSA has several functions available for plotting networks with
different colors/widths/labels on buses and branches and
geographic features in the background.


Static plotting with matplotlib
===============================

Static plots of networks can be created that use the library
`matplotlib <https://matplotlib.org/>`_.

To plot a network with ``matplotlib``, run
``n.plot()``, see :py:meth:`pypsa.Network.plot` for details.

See also the `SciGRID matplotlib example
<https://pypsa.readthedocs.io/en/latest/examples/scigrid-lopf-then-pf.html>`_ and the `Flow plotting matplotlib example
<https://pypsa.readthedocs.io/en/latest/examples/flow-plot.html>`_.


Interactive plotting with plotly
================================

Interactive plots of networks can be created using `plotly
<https://plot.ly/python/>`_.
To plot a network with ``plotly``, run
``n.iplot()``, see :py:meth:`pypsa.Network.iplot` for details.
