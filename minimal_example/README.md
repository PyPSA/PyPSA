# PyPSA_Lines_Marginal

This project demonstrates how marginal costs for transmission lines and transformers can be modeled and analyzed in PyPSA using a minimal example.

## What was changed in PyPSA

- The PyPSA source code was extended to support a `marginal_cost` attribute for lines and transformers.
- When adding a line or transformer, you can specify a `marginal_cost` (scalar or time series). This cost is considered in the network optimization and can influence dispatch decisions.

## What is done in the minimal example

- The script `minimal_example.py` sets up a small network with two generators, each connected to the load via its own transformer and line.
- Marginal costs for lines and transformers are varied over time for each path.
- The network is solved for 7 timesteps, and the results are visualized.

## How to read the results

- The first plot shows the generator dispatch over time, with colored regions indicating different cost regimes (normal operation, line cost increase, transformer cost increase, etc.).
- The second plot shows the marginal costs for each path (generator + line for Gen_A, generator + transformer for Gen_B), as well as the individual marginal costs for the line and transformer.
- You can observe how the dispatch switches between generators depending on the marginal costs for lines and transformers.

## How to activate and interpret marginal costs

- Marginal costs are activated by setting the `marginal_cost` parameter when adding lines or transformers, e.g.:
  ```python
  network.add("Line", "A1_C", bus0="A1", bus1="C", s_nom=100, marginal_cost=[0, 3, 6, 6, 6, 0, 0])
  network.add("Transformer", "Trafo_B1", bus0="B", bus1="B1", s_nom=100, marginal_cost=[0, 0, 0, 3, 6, 6, 0])
  ```
- Use a list or pandas Series to specify time-dependent marginal costs.
- The optimizer will choose the cheapest path (generator + line/transformer) for each timestep, and the plots will show when and why the dispatch switches.

Refer to `minimal_example.py` for a complete demonstration.