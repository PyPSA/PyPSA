######################
 Plotting Networks
######################

See the module ``pypsa.plot``.

PyPSA has several functions available for plotting networks with
different colors/widths/labels on buses and branches.


Static plotting with matplotlib
===============================

Static plots of networks can be created that use the library
`matplotlib <https://matplotlib.org/>`_.  This is meant for use with
`Jupyter notebooks <https://jupyter.org/>`_, but can also be used to
generate image files.
To plot a network with matplotlib run
``network.plot()``.

.. automethod:: pypsa.Network.plot

See also the `SciGRID matplotlib example
<https://pypsa.org/examples/scigrid-lopf-then-pf.html>`_.


Interactive plotting with plotly
================================

Interactive plots of networks can be created that use the `d3js
<https://d3js.org/>`_-based library `plotly
<https://plot.ly/python/>`_ (this uses JavaScript and SVGs). This is
meant for use with `Jupyter notebooks <https://jupyter.org/>`_.
To plot a network with plotly run
``network.iplot()``.

.. automethod:: pypsa.Network.iplot

See also the `SciGRID plotly example
<https://pypsa.org/examples/scigrid-lopf-then-pf-plotly.html>`_.

