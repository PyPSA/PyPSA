<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Load

The [`Load`][pypsa.components.Loads] components attach to a single bus and represent a demand for the
[`Bus`][pypsa.components.Buses] carrier they are connected to. With inverted sign, they can also be used
to model an exogenous supply. For "AC" buses, they act as a PQ load. If $p>0$
the load is consuming active power from the bus and if $q>0$ it is consuming
reactive power (i.e. behaving like an inductor).

## Passive and active loads

Loads are active components which mirror the [`Generator`][pypsa.components.Generators] with the opposite sign. Which of the two a load is follows from its `p_set`:

- **Passive load**: `p_set` is given for all optimized snapshots (a static `p_set` applies to all snapshots). The load enters the nodal balance as a constant, exactly like loads always did, and no optimization variable is created.
- **Active load**: `p_set` is NaN (the default) for some or all snapshots. The consumption becomes the decision variable `Load-p` within `[p_min_pu, p_max_pu] * p_nom`. Snapshots where `p_set` is given are fixed to it.

A load with a complete `p_set` still receives a (fixed) variable if it uses a feature that requires one: capacity expansion (`p_nom_extendable`), unit commitment (`committable`), modularity (`p_nom_mod`), ramp limits, volume limits (`e_sum_min`, `e_sum_max`) or a non-zero `marginal_cost`/`marginal_cost_quadratic`.

Active loads support the operational features of generators, including capacity expansion, unit commitment and ramp limits. A load with NaN `p_set` and no capacity (the default `p_nom=0`, not extendable) consumes nothing and also enters the model as a constant (zero where `p_set` is NaN).

## Marginal costs and sign

The linear `marginal_cost` of a load is applied to the power injected at the bus, `sign * p`. With the default `sign=-1`, a positive `marginal_cost` is the price the consumer is willing to pay for 1 MWh, which enters the objective as revenue:

$$\min \; \ldots - \text{marginal\_cost} \cdot p + \text{marginal\_cost\_quadratic} \cdot p^2$$

The quadratic term applies to the squared injection, so a positive `marginal_cost_quadratic` models a decreasing willingness to pay `marginal_cost - 2 * marginal_cost_quadratic * p` (elastic demand). Capital, start-up, shut-down and stand-by costs are costs in either direction and do not follow `sign`.

``` py
n.add("Load", "data-centre", bus="bus", p_nom=100, marginal_cost=80)  # pays up to 80 €/MWh
n.add("Load", "elastic", bus="bus", p_nom=500, marginal_cost=120, marginal_cost_quadratic=0.1)
```

!!! note "Relation to generators with negative sign"

    Price-responsive demand used to be modelled as a [`Generator`][pypsa.components.Generators] with `sign=-1` and a negative `marginal_cost`. `Load(marginal_cost=a, marginal_cost_quadratic=b)` is equivalent to `Generator(sign=-1, marginal_cost=-a, marginal_cost_quadratic=b)`. For generators, stores and storage units, `marginal_cost` is currently applied to the dispatch variable regardless of `sign`; this changes in PyPSA 2.0, when all components apply it to the power injected at the bus. A `FutureWarning` is raised for components with `sign < 0` and non-zero `marginal_cost`.

{{ read_csv('../../../pypsa/data/component_attrs/loads.csv', disable_numparse=True) }}
