######################
 Plotting Networks
######################

PyPSA has several functions available for plotting networks with
different colors/widths/labels on buses and branches.


Static plotting with matplotlib
===============================

Static plots of networks can be created that use the library
`matplotlib <https://matplotlib.org/>`_.  This is meant for use with
`Jupyter notebooks <https://jupyter.org/>`_, but can also be used to
generate image files.
To plot a network with matplotlib run
``network.plot()``, see :py:meth:`pypsa.Network.plot` for details.

See also the `SciGRID matplotlib example
<https://pypsa.readthedocs.io/en/latest/examples/scigrid-lopf-then-pf.html>`_ and the `Flow plotting matplotlib example
<https://pypsa.readthedocs.io/en/latest/examples/flow-plot.html>`_.


Interactive plotting with plotly
================================

Interactive plots of networks can be created that use the `d3js
<https://d3js.org/>`_-based library `plotly
<https://plot.ly/python/>`_ (this uses JavaScript and SVGs). This is
meant for use with `Jupyter notebooks <https://jupyter.org/>`_.
To plot a network with plotly run
``network.iplot()``, see :py:meth:`pypsa.Network.iplot` for details.
