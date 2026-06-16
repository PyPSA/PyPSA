<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Piecewise linearisation

If you have non-linear relationships between two decision variables, you may want to represent them as a [piecewise linear function](https://en.wikipedia.org/wiki/Piecewise_linear_function).
Common non-linear relationships found in energy systems include:

- **marginal cost curves**: Where the marginal cost of generating a unit of energy changes as the quantity of energy generated changes.
- **capital cost curves**: Where the capital cost of investing in the next unit of nominal capacity changes as the quantity nominal capacity changes.
- **part-load efficiency curves**: Where the efficiency of energy generation / conversion changes as the load rate ($\frac{\text{generation}{\text{nominal capacity}}$) changes.

## Defining piecewise data in PyPSA

A limited set of component attributes can be defined using piecewise curves:

{{ read_csv('../../../pypsa/components/piecewise.csv') }}

For each, they can be defined when using [pypsa.Network.add][] by either providing an appropriately formatted dictionary or [pandas.DataFrame][].
If a dictionary, keys should be x-values and values should be y-values at each of the _breakpoints_ of the curve being described.
If a DataFrame, the index should be breakpoint number, the columns should be the x- and y-axis names (e.g. `p_pu` and `marginal_cost`), and the values should be the x-/y-axis values at each breakpoint.

!!! question "What is a _breakpoint_?"

    A breakpoint is a point along the piecewise curve where two straight lines of different gradients meet.
    We define breakpoints in terms of integer values (breakpoint 0, 1, 2, etc.).

## Inspecting piecewise data in PyPSA

Piecewise data is stored in a component `piecewise` dictionary-like object, much like `dynamic` data.
For instance, to access marginal_cost piecewise data for generators, you can call `n.c.generators.piecewise.marginal_cost`.
The returned [pandas.DataFrame][] will be of the same form as when [defining an input attribute as a DataFrame](#defining-piecewise-data-in-pypsa), with an added column level with the names of the components for which you've defined piecewise data.

## Configuring piecewise formulations in PyPSA

When applying the piecewise constraints to the optimisation problem, we make some assumptions about how they should be formulated.
For the most part, we leave [linopy][] to decide on the best formulation and define the piecewise constraints in such a way that the simplest form of the constraint can feasibly be chosen.

Our assumptions, and the `linopy` defaults, can be overridden when calling [pypsa.Network.optimize][] by using the `piecewise_options` list argument.
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

## Merit order vs. spot curves

Piecewise curves can be used in two different contexts in energy system modelling.
The choice is [configurable](#configuring-piecewise-formulations-in-pypsa) but you should understand the difference before considering overriding defaults.

### Merit order curves

Here, one PyPSA component might represent a fleet of assets (e.g. a single `CCGT` generator represents 100 real assets in the given zone).
If the assets have different marginal costs, you would want to operate them from least to most expensive as you increase total generation.
In our example, a 50% load rate would mean the 50 least expensive assets are operating at 100% load since they incur the least cost in operation.

The piecewise curve in this instance is based on combining the increasing marginal costs of the fleet in increasing order.
We start by paying the cheaper costs and then build up to paying the most expensive cost.
The marginal operating cost at the optimal dispatch rate _is_ the marginal cost of the generator, but the total operating cost is the integral of operating costs along the piecewise curve up to that point.

We assume that piecewise marginal operating and capital cost curves are merit order curves.

### Non-merit order curves

Here, one PyPSA component might represent a single asset with a non-linear operating characteristic.
Unlike merit order curves, we do not care about the nature of the curve except at the point we are operating.
So, at a 50% load rate, this asset is operating at 50% of its nameplate capacity and it doesn't matter what the piecewise curve says about its characteristics at 10%, 25%, etc. load rate.

Therefore, the piecewise curve in this instance is _not_ integrated up to the operating point; instead, it is taken at face value at each operating point.

We assume that part-load efficiency curves are direct. A cost curve can also be direct where its values are meant to be read at the operating point rather than accumulated.
