#########################
Power System Optimization
#########################


This section provides examples related to network optimization with PyPSA. Note that PyPSA allows for optimising networks in three ways, first using the ``pyomo`` interface and second using the in-home implementation (which is way faster and memory-efficient). The third one is the new Linopy based implementation, which was inspired by the in-house optimization code. We are *slowly* moving towards using the linopy implementation only.

.. toctree::
    :maxdepth: 1
    :caption: No Pyomo

    examples/lopf_with_pyomo_False.ipynb
    examples/capacity-constraint-per-bus.ipynb
    examples/multi-investment-optimisation.ipynb

.. toctree::
    :maxdepth: 1
    :caption: Pyomo

    examples/chained-hydro-reservoirs.ipynb
    examples/generation-investment-screening-curve.ipynb
    examples/scigrid-lopf-then-pf.ipynb
    examples/scigrid-redispatch.ipynb
    examples/scigrid-sclopf.ipynb
    examples/simple-electricity-market-examples.ipynb
    examples/transformer_example.ipynb
    examples/unit-commitment.ipynb

.. toctree::
    :maxdepth: 1
    :caption: Linopy

    examples/optimization-with-linopy.ipynb
    examples/optimization-with-linopy-migrate-extra-functionalities.ipynb
