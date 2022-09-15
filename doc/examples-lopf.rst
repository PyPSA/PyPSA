#########################
Power System Optimization
#########################


This section provides examples related to network optimization with PyPSA. Note that PyPSA allows for optimising networks in two ways, first using the ``pyomo`` interface and second using the in-home implementation (which is way faster and memory-efficient). Some newer examples use the in-house implementation while others are based in the pyomo package.


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
