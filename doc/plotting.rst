######################
 Plotting Networks
######################


See the module ``pypsa.plot``.

PyPSA has several functions available for plotting networks with
different colors/widths/labels on buses and branches.


Interactive plotting with plotly
================================

Interactive plots of networks can be created that use the `d3js
<https://d3js.org/>`_-based library `plotly
<https://plot.ly/python/>`_ (this uses JavaScript and SVGs). This is
meant for use with `Jupyter notebooks <https://jupyter.org/>`_.

Call ``network.iplot()``; see the doc string for more instructions and
the `SciGRID plotly example
<https://pypsa.org/examples/scigrid-lopf-then-pf-plotly.html>`_.


Static plotting with matplotlib
===============================

Static plots of networks can be created that use the library
`matplotlib <https://matplotlib.org/>`_.  This is meant for use with
`Jupyter notebooks <https://jupyter.org/>`_, but can also be used to
generate image files.

Call ``network.plot()``; see the doc string for more instructions and
the `SciGRID matplotlib example
<https://pypsa.org/examples/scigrid-lopf-then-pf.html>`_.
