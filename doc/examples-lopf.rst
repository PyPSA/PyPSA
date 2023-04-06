#########################
Power System Optimization
#########################


This section contains examples of network optimization with PyPSA. PyPSA allows you to optimize networks in three ways: with `Linopy <https://linopy.readthedocs.io>`_, with `Pyomo <http://www.pyomo.org/>`_, and based on `custom code <https://github.com/PyPSA/PyPSA/blob/v0.22.1/pypsa/linopt.py>`_. The new default optimization framework is based on Linopy. In earlier versions, PyPSA used the ``pyomo`` interface and an in-house implementation, both available through the ``n.lopf()`` function. The previous options are still available but will be deprecated in the future. The following examples all use the Linopy-based implementation.


.. toctree::
    :maxdepth: 1

    examples/optimization-with-linopy.ipynb
    examples/capacity-constraint-per-bus.ipynb
    examples/multi-investment-optimisation.ipynb
    examples/chained-hydro-reservoirs.ipynb
    examples/generation-investment-screening-curve.ipynb
    examples/scigrid-lopf-then-pf.ipynb
    examples/scigrid-redispatch.ipynb
    examples/scigrid-sclopf.ipynb
    examples/simple-electricity-market-examples.ipynb
    examples/transformer_example.ipynb
    examples/unit-commitment.ipynb
    examples/optimization-with-linopy-migrate-extra-functionalities.ipynb
