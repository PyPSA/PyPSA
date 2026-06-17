<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Piecewise linearisation

If you have non-linear relationships between two decision variables, you may want to represent them as a [piecewise linear function](https://en.wikipedia.org/wiki/Piecewise_linear_function).
Common non-linear relationships found in energy systems include:

- **marginal cost curves**: Where the marginal cost of generating a unit of energy changes as the quantity of energy generated changes.
- **capital cost curves**: Where the capital cost of investing in the next unit of nominal capacity changes as the quantity nominal capacity changes.
- **part-load efficiency curves**: Where the efficiency of energy generation / conversion changes as the load rate ($\frac{\text{generation}{\text{nominal capacity}}}$) changes.

!!! info "See Also"

    - [:material-notebook: PyPSA piecewise example](../../examples/piecewise-constraints.ipynb)
    - [Linopy piecewise documentation](https://linopy.readthedocs.io/en/latest/piecewise-linear-constraints.html)

## Defining piecewise data

A limited set of component attributes can be defined using piecewise curves:

{{ read_csv('../../../pypsa/data/piecewise.csv') }}

For each, they can be defined when using [pypsa.Network.add][] by either providing an appropriately formatted dictionary or [pandas.DataFrame][].
If a dictionary, keys should be x-values and values should be y-values at each of the _breakpoints_ of the curve being described.
If a DataFrame, the index should be breakpoint number, the columns should be the x- and y-axis names (e.g. `p_pu` and `marginal_cost`), and the values should be the x-/y-axis values at each breakpoint.

!!! question "What is a _breakpoint_?"

    A breakpoint is a point along the piecewise curve where two straight lines of different gradients meet.
    We define breakpoints in terms of integer values (breakpoint 0, 1, 2, etc.).

## Inspecting piecewise data

Piecewise data is stored in a component `piecewise` dictionary-like object, much like `dynamic` data.
For instance, to access marginal_cost piecewise data for generators, you can call `n.c.generators.piecewise.marginal_cost`.
The returned [pandas.DataFrame][] will be of the same form as when [defining an input attribute as a DataFrame](#defining-piecewise-data), with an added column level with the names of the components for which you've defined piecewise data.

## Piecewise formulations

When applying the piecewise constraints to the optimisation problem, each will be formulated by [linopy][] in the [format deemed to be "cheapest" for solving the problem](https://linopy.readthedocs.io/en/latest/piecewise-linear-constraints.html#formulation-methods).
As of writing, this could be in the form of:

- [An LP (chord-line) formulation](https://linopy.readthedocs.io/en/latest/piecewise-linear-constraints.html#lp-chord-line-formulation)
- [An Incremental (Delta) formulation](https://linopy.readthedocs.io/en/latest/piecewise-linear-constraints.html#incremental-delta-formulation)
- [SOS2 (Convex combination)](https://linopy.readthedocs.io/en/latest/piecewise-linear-constraints.html#sos2-convex-combination)
- [Disjunctive (Disaggregated convex combi nation)](https://linopy.readthedocs.io/en/latest/piecewise-linear-constraints.html#disjunctive-disaggregated-convex-combination)

Each formulation will create a piecewise auxiliary decision variable of the form `<component>-<aux_variable>` in the optimisation problem (see the table [above](#defining-piecewise-data) for all possible `component` and `aux_variable` combinations).
For instance, a piecewise capital cost constraint applied to a generator will yield the variable `Generator-capital_cost_piecewise`, accessible at `n.model.variables["Generator-capital_cost_piecewise"]`.

Depending on the formulation used, [linopy][] will also create additional auxiliary variables and constraints.
For instance, the incremental (delta) formulation will lead to the auxiliary variables `<component>-<aux_variable>_delta` and `<component>-<aux_variable>_binary` and the constraints:

```py
<component>-<aux_variable>_delta_bound
<component>-<aux_variable>_fill_order
<component>-<aux_variable>_binary_order
<component>-<aux_variable>_link
```

These additional constraints and auxiliary variables will not find their way into the PyPSA network results or statistics.
You will, however, have access to the results `capital_cost_piecewise_opt` (static) and `marginal_cost_piecewise_opt` (dynamic) if applying piecewise costs.
Piecewise efficiencies will be applied directly to the appropriate dynamic result attribute (e.g. `p2` for a `Link` if the piecewise curve is applied to `efficiency2`).

You can override the default [linopy][] formulation by [configuring your own piecewise options](#configuring-piecewise-formulations).

### Configuring piecewise formulations

Our assumptions, and the `linopy` defaults, can be overridden when calling [n.optimize][pypsa.Network.optimize] by using the `piecewise_options` list argument.
Each dictionary in the list defines a formulation variation.
The options that you can override are defined in [pypsa.optimization.piecewise.PiecewiseOptions][].

!!! example
    By default, we do not pin piecewise costs to the curve, instead we allow costs to increase above those defined by the curve.
    With some piecewise curve shapes, this approach allows for a fully linear piecewise formulation.
    This should be fine in most cases; the optimisation is cost-minimising, so you will not actually get costs that deviate from the piecewise curve.

    If you want to pin piecewise costs to the curve, you need to use an equality sign.
    Assuming we are applying this to marginal costs of generators:

    ```python
    n.optimize(piecewise_options=[{"component": "Generator", "attribute": "marginal_cost", "sign": "=="}])
    ```

## Cumulative vs direct curves

Piecewise curves can be used in two different contexts in energy system modelling.
The choice is [configurable](#configuring-piecewise-formulations) but you should understand the difference before considering overriding defaults.

!!! info

    The default curve type can be overwritten using the boolean piecewise option `cumulative_attr` (True = cumulative, False = direct).
    See [pypsa.optimization.piecewise.PiecewiseOptions][] for more information.

### Cumulative curves

Here, the y-value at each point on the curve represents an _incremental_ contribution, so the total operating point $X$ is the integral of the curve up to $X$.
We assume that piecewise marginal operating and capital cost curves are cumulative.

!!! example

    For an increasing marginal cost curve, the increments are dispatched in cost order.
    A PyPSA `CCGT` generator might represent many individual assets, each with different marginal costs.
    At a 50% load rate, the cheapest half run at full output and the rest stay off, since that is the least-cost operating schedule.
    The value at $X$ is then the marginal cost of the generator at 50% load rate, the cost of the last increment.
    The total operating cost is the integral along the curve up to $X$.

### Direct curves

Here, the curve is read at the operating point and no integration takes place.
The value at the operating rate $X$ is the realised value directly, evaluated point by point rather than accumulated from the rates below it.

We assume that part-load efficiency curves are direct.
A cost curve can also be direct where its values are meant to be read at the operating point rather than accumulated.
In such a case, the total cost is the curve value at $X$ multiplied by $𝑋$.

!!! example

    Consider a component with a part-load efficiency curve:
    at a 50% load rate the asset operates at 50% of its nameplate capacity and the efficiency is the curve value at 50%, regardless of its efficiency at other load rates.
