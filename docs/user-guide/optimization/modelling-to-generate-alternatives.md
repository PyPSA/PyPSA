<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Modelling to Generate Alternatives (MGA)

## Searching for alternatives with given cost slack

The function [`n.optimize.optimize_mga`][pypsa.optimization.OptimizationAccessor.optimize_mga] runs modelling-to-generate-alternatives (MGA) on network to find near-optimal solutions. This is a technique where a solved network is re-optimized with an alternative objective function, while adding a global constraint limiting the total system cost.

For example, the alternative objective function may represent total installed renewable capacity; minimizing this subject to a total system cost constraint could lead to a slightly more expensive system containing fewer renewables but more storage or backup capacity.

In `optimize_mga`, the bound on total system cost is specified as a relative increase over the cost-optimum; the relative increase is called _slack_. Denote the slack as $\varepsilon$, as is common in the literature, and let $c$ be the total system cost objective function and $c^*$ its optimal value. Then the near-optimality constraint added in `optimize_mga` is $c \leq (1 + \varepsilon) \cdot c^*$. Typical values found in the literature range from about 2% to 20%, with 5% being a common starting value for exploration.

Numerous research articles cover the theory and application of MGA in the context of energy systems modelling; many of them have used PyPSA under the hood. See for instance [Brown et al. (2021)](https://doi.org/10.1016/j.epsr.2020.106690), [Grochowicz et al. (2023)](https://doi.org/10.1016/j.eneco.2022.106496) and [Lau et al. (2024)](https://doi.org/10.1088/2753-3751/ad7d10) (the latter containing a brief literature review) for an introduction to the topic.

## Exploring trade-offs in the near-optimal space

Often, it is useful to explore the trade-offs between several alternative objectives using near-optimal techniques. While this can be done using the `optimize_mga` function, described above, PyPSA includes several convenient functions facilitating such exploration.

To start with, [`n.optimize.optimize_mga_in_direction`][pypsa.optimization.OptimizationAccessor.optimize_mga_in_direction] is similar to `optimize_mga`, but instead of a single alternative objective (the `weights` argument in `optimize_mga`), this function takes a dictionary of any number of alternative objectives (the `dimensions` argument) and a `direction` vector (given as a dictionary with the same keys as `dimensions`) telling PyPSA how to balance the objectives against each other.

For example, supposing you pass as `dimensions` two objectives $\{\text{foo}: c_1, \text{bar}: c_2\}$ and as `direction` the vector $\{\text{foo}: 1, \text{bar}: -1\}$, the `optimize_mga_in_direction` will maximize $c_1$ and minimize $c_2$ jointly. As before, these optimizations are subject to a bound on total system cost, specified by the user as a relative cost slack $\varepsilon$.

The alternative objectives $c_1, c_2, \dots$ effectively specify a coordinates in a low-dimensional projection of the full feasible space of the linear program defined by a network. By solving the network in many different directions in this low-dimensional space (under a cost bound), it is possible to approximate the geometry of the near-optimal space in this projection. For example, supposing that $c_1$ and $c_2$ represent total installation of wind and solar generation, respectively, the near-optimal space projected to $c_1,c_2$-coordinates represents all near-optimal and feasible combinations of wind and solar installation.

As running `optimize_mga_in_direction` in multiple different directions is a common operation, PyPSA provides a separate function, [`n.optimize.optimize_mga_in_multiple_directions`][pypsa.optimization.OptimizationAccessor.optimize_mga_in_multiple_directions], which takes a list (or pandas `DataFrame`) of directions, and solves for them in parallel.

Moreover, three functions ([`pypsa.optimization.mga.generate_directions_random`][pypsa.optimization.mga.generate_directions_random], [`pypsa.optimization.mga.generate_directions_evenly_spaced`][pypsa.optimization.mga.generate_directions_evenly_spaced], [`pypsa.optimization.mga.generate_directions_halton`][pypsa.optimization.mga.generate_directions_halton]) are provided which generate sets of directions in the format expected by `optimize_mga_in_multiple_directions`. Of course, you can also provide your own directions.

The function [`n.optimize.optimize_mga`][pypsa.optimization.OptimizationAccessor.optimize_mga] runs modelling-to-generate-alternatives (MGA) on network to find near-optimal solutions.


## Examples


<div class="grid cards" markdown>


-   :material-notebook:{ .lg .middle } **Modelling-to-Generate Alternatives**

    ---

    Explores near-optimal solution diversity by generating alternative system
    designs with similar costs.

    [:octicons-arrow-right-24: Go to example](../../examples/mga.ipynb)

-   :material-notebook:{ .lg .middle } **Exploring Near-Optimal Spaces**

    ---

    Explores near-optimal space to understand flexibility in investment
    decisions while maintaining cost-effectiveness.

    [:octicons-arrow-right-24: Go to example](../../examples/near-opt-space.ipynb)

</div>