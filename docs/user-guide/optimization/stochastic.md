<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Stochastic Optimization

## Overview

Stochastic optimization in PyPSA enables modeling and solving energy system planning problems under uncertainty. This capability addresses scenarios where parameters such as fuel prices, renewable energy availability, demand patterns, or technology costs are uncertain at the time investment decisions must be made.

PyPSA implements a **two-stage stochastic programming framework** with scenario trees, allowing users to optimize investment decisions (first-stage) that are feasible across multiple possible future realizations (scenarios) of uncertain parameters and minimize expected system costs.

## Risk-neutral Two-Stage Stochastic Programming

### Mathematical Formulation

!!! note "Background"

    The stochastic optimization problem in PyPSA follows the standard two-stage risk-neutral stochastic programming formulation:

    $$
    \begin{align}
    \min_{\mathbf{x}} \quad & \mathbf{c}^\top \mathbf{x} + \sum_{s \in \mathcal{S}} \pi_s Q(\mathbf{x}, \boldsymbol{\xi}_s) \\
    \text{s.t.} \quad & \mathbf{A} \mathbf{x} = \mathbf{b} \\
    & \mathbf{x} \geq 0
    \end{align}
    $$

    Where:

    - $\mathbf{x}$: First-stage (here-and-now) decision variables (e.g., investment decisions in component capacities)
    - $\boldsymbol{\xi}_s$: Random parameter realizations in scenario $s$
    - $\pi_s$: Probability of scenario $s$ occurring
    - $Q(\mathbf{x}, \boldsymbol{\xi}_s)$: Second-stage recourse function for scenario $s$
    - $\mathcal{S}$: Set of all scenarios

    The second-stage problem $Q(\mathbf{x}, \boldsymbol{\xi}_s)$ represents:

    $$
    \begin{align}
    Q(\mathbf{x}, \boldsymbol{\xi}_s) = \min_{\mathbf{y}_s} \quad & \mathbf{q}_s^\top \mathbf{y}_s \\
    \text{s.t.} \quad & \mathbf{W} \mathbf{y}_s = \mathbf{h}_s - \mathbf{T} \mathbf{x} \\
    & \mathbf{y}_s \geq 0
    \end{align}
    $$

    Where:

    - $\mathbf{y}_s$: Second-stage (wait-and-see) decision variables for scenario $s$ (e.g., energy dispatch decisions)
    - $\mathbf{q}_s$: Second-stage cost coefficients for scenario $s$
    - $\mathbf{W}$: Recourse matrix (coefficient matrix for second-stage variables $\mathbf{y}_s$)
    - $\mathbf{T}$: Technology matrix (coefficient matrix for first-stage variables $\mathbf{x}$ in second-stage constraints)
    - $\mathbf{h}_s$: Right-hand-side vector for scenario $s$

    The second-stage constraints $\mathbf{W} \mathbf{y}_s = \mathbf{h}_s - \mathbf{T} \mathbf{x}$ show how first-stage decisions $\mathbf{x}$ affect the feasible region for second-stage variables $\mathbf{y}_s$ in each scenario.

In PyPSA, the first-stage cost $\mathbf{c}^\top \mathbf{x}$ is the sum of the capital expenditures and the second-stage cost $Q$ is the operational cost of each scenario (see [objective](objective.md)):

<!-- md:spec total_cost -->

The scenario tree in PyPSA splits into **here-and-now** investment decisions (t=0) and **wait-and-see** dispatch decisions (t=1):

| Stage | Decision Type | Variables | Information | Examples |
|-------|---------------|-----------|-------------|-----------|
| **t=0** | **Here-and-now** | Investment decisions ($\mathbf{x}$) - Common across scenarios | Uncertain parameters unknown, probability distribution known | Generator capacity, transmission expansion, storage capacity |
| **t=1** | **Wait-and-see** | Dispatch decisions ($\mathbf{y}_s$) - Scenario-specific | Uncertain parameters revealed, perfect information for scenario $s$ | Dispatch, storage operation, load shedding |


!!! note "Limitation: Investment and dispatch stages are fixed"

    Currently, PyPSA only considers dispatch decisions as second-stage variables and investment decisions as first-stage variables. Support for second-stage investment decisions (e.g., scenario-dependent capacity additions as recourse measures) as well as first-stage dispatch decisions (e.g., for stochastic optimisation in purely operational settings with forecast uncertainty) is planned for future releases.

!!! note "Consideration of risk preferences and robust optimisation"

    The default stochastic programming implementation in PyPSA minimizes expected costs across scenarios weighted by their probabilities. This approach considers risk-neutral decision making.
    PyPSA also supports changing risk preference through Conditional Value at Risk (CVaR)-based risk-averse optimization, allowing users to account for extreme outcomes and tail risks in their optimization.

For a comprehensive treatment of two-stage stochastic programming theory and methods, see Birge and Louveaux (2011).[^1]

### Implementation

Let us consider a single-node capacity expansion model in the style of [model.energy](https://model.energy). This stylized model calculates the cost of meeting an hourly electricity demand time series from a combination of wind power, solar power, battery and hydrogen storage as well as load shedding. See this [example](../../examples/capacity-expansion-planning-single-node.ipynb). We give wind a small marginal cost so that the split between wind and solar curtailment is unique. We add a backup power plant as additional technology option and solve the model with given load and renewable availability profiles as a deterministic problem. Afterwards, we inspect the optimal objective value and expanded capacities.

``` py
>>> import pypsa
>>>
>>> n = pypsa.examples.model_energy()
>>> n.generators.loc["wind", "marginal_cost"] = 0.01
>>> n.add(
...     "Generator",
...     "backup_generator",
...     carrier="fossil backup",
...     bus="electricity",
...     p_nom_extendable=True,
...     efficiency=0.5,
...     marginal_cost=150,
...     capital_cost=46000,
... )
>>>
>>> n.optimize(log_to_console=False)
('ok', 'optimal')
>>>
>>> cap_deterministic = n.statistics.optimal_capacity().round(1)
>>> cap_deterministic
component    carrier
Generator    fossil backup       6271.2
             load shedding      10901.2
             solar              22071.0
             wind               21467.3
StorageUnit  battery storage     9305.5
dtype: float64
>>>
>>> obj_deterministic = n.objective / 1e9
>>> obj_deterministic  # doctest: +ELLIPSIS
5.845...
```

### Scenario Definition

Now suppose we want to consider the case that with a 20% probability, a volcano erupts which reduces the solar capacity factor time series uniformly by 90%. For such a scenario, we can use PyPSA's stochastic optimization functionality. First, we need to define the scenarios with probabilities that must sum to 1:

``` py
>>> n.set_scenarios({"volcano": 0.2, "no_volcano": 0.8})
```

We can check that the scenarios have been set correctly:

``` py
>>> n.has_scenarios
True
```

``` py
>>> n.scenarios
Index(['volcano', 'no_volcano'], dtype='str', name='scenario')
```

``` py
>>> n.scenario_weightings
            weight
scenario
volcano        0.2
no_volcano     0.8
```

The key change after calling `n.set_scenarios()` is that all component data is broadcasted across all scenarios. All component `pandas.DataFrame` objects gain a "scenario" dimension as outermost index level.

``` py
>>> n.generators[["bus", "marginal_cost", "efficiency"]]  # doctest: +ELLIPSIS
                                     bus  marginal_cost  efficiency
scenario   name
volcano    load shedding     electricity        2000.00         1.0
           wind              electricity           0.01         1.0
           solar             electricity           0.00         1.0
           backup_generator  electricity         150.00         0.5
no_volcano load shedding     electricity        2000.00         1.0
           wi...
```

``` py
>>> n.generators_t.p_max_pu.head(3)
scenario            volcano         no_volcano
name                  solar    wind      solar    wind
snapshot
2019-01-01 00:00:00     0.0  0.1846        0.0  0.1846
2019-01-01 03:00:00     0.0  0.3146        0.0  0.3146
2019-01-01 06:00:00     0.0  0.4957        0.0  0.4957
```

!!! warning

    - Scenarios must be set **after** adding all network components.
    - They are **immutable** once set.

### Parameter Updates

So far, all data is the same across scenarios. Now that we have a separate index level for
the scenarios, we can modify scenario-specific parameters. In our case, we want to reduce the solar capacity factor in the "volcano" scenario by 90%:

``` py
>>> # Apply volcano impact by reducing solar capacity factor by 90%
>>> n.generators_t.p_max_pu.loc[:, ("volcano", "solar")] *= 0.1
>>>
>>> # Check the impact on a summer day (July 21)
>>> n.generators_t.p_max_pu.loc["2019-07-21"]  # doctest: +ELLIPSIS
scenario            volcano         no_volcano
name                  solar    wind      solar    wind
snapshot
2019-07-21 00:00:00  0.0000  0.2201      0.000  0.2201
2019-07-21 03:00:00  0.0004  0.1682      0.004  0.1682
2019-07-21 06:00:00  0.0268  0.0715      0.268  0.0715
2019-07-21 09:00:00  0.0263  0.1164      0.263  0.1164
2019-07-21 12:00:00  0.0301  0.1646      0.301  0.1646
2019-07-21 15:00:00  0.0228  0.1181      0.228  0.1181
2019-07-21 18:00:00  0.0004  0.0702      0.004  0.0702
2019-07-21 21:00:00  0.0000  0.0896      0.000  0.0896
```

!!! note

    Attributes which define the topology or the structure of the optimisation problem must be identical across scenarios, e.g. `bus`, `bus0`, `bus1`, ..., `carrier`, `p_nom_extendable`, `p_nom_set`, `committable`, `build_year`, `lifetime`, `active` or the `delay` and `cyclic_delay` attributes of links and processes (including their per-port variants, e.g. `delay0`, `delay2`, `cyclic_delay2`). The same holds for the bus attribute `v_nom` and the effective impedances `x_pu_eff` and `r_pu_eff` of lines and transformers, since the Kirchhoff voltage law uses one set of impedances for all scenarios. The consistency check raises an error otherwise.

### Optimization

When we now call `n.optimize()`, the network is solved as a stochastic problem considering all defined scenarios and their respective probabilities.

``` py
>>> n.optimize()
('ok', 'optimal')
```

PyPSA creates stochastic optimization models by reformulating the two-stage problem as a large-scale deterministic equivalent with scenario-indexed variables and constraints:

``` py
>>> n.model
Linopy LP model
===============
<BLANKLINE>
Variables:
----------
 * Generator-p (name, scenario, snapshot)
 * Generator-p_nom (name)
 * Link-p (name, scenario, snapshot)
 * Link-p_nom (name)
 * StorageUnit-p_dispatch (name, scenario, snapshot)
 * StorageUnit-p_nom (name)
 * StorageUnit-p_store (name, scenario, snapshot)
 * StorageUnit-state_of_charge (name, scenario, snapshot)
 * Store-e (name, scenario, snapshot)
 * Store-e_nom (name)
 * Store-p (name, scenario, snapshot)
 * Store-p_store (name, scenario, snapshot)
<BLANKLINE>
Expressions:
------------
<empty>
<BLANKLINE>
Constraints:
------------
 * Bus-nodal_balance (name, scenario, snapshot)
 * Generator-ext-p-lower (name, scenario, snapshot)
 * Generator-ext-p_nom-lower (name)
 * Generator-ext-p-upper (name, scenario, snapshot)
 * Generator-fix-p-lower (name, scenario, snapshot)
 * Generator-fix-p-upper (name, scenario, snapshot)
 * Link-ext-p-lower (name, scenario, snapshot)
 * Link-ext-p_nom-lower (name)
 * Link-ext-p-upper (name, scenario, snapshot)
 * StorageUnit-energy_balance (name, scenario, snapshot)
 * StorageUnit-ext-p_dispatch-lower (name, scenario, snapshot)
 * StorageUnit-ext-p_dispatch-upper (name, scenario, snapshot)
 * StorageUnit-ext-p_nom-lower (name)
 * StorageUnit-ext-p_store-lower (name, scenario, snapshot)
 * StorageUnit-ext-p_store-upper (name, scenario, snapshot)
 * StorageUnit-ext-state_of_charge-lower (name, scenario, snapshot)
 * StorageUnit-ext-state_of_charge-upper (name, scenario, snapshot)
 * Store-energy_balance (name, scenario, snapshot)
 * Store-ext-e-lower (name, scenario, snapshot)
 * Store-ext-e_nom-lower (name)
 * Store-ext-e-upper (name, scenario, snapshot)
 * Store-ext-p_store-lower (name, scenario, snapshot)
 * Store-ext-p_store-upper (name, scenario, snapshot)
<BLANKLINE>
Status:
-------
ok
```

As investment variables (i.e. `p_nom`, `s_nom`, `e_nom`) are scenario-independent first-stage decisions, they do not have a scenario dimension (i.e. are common across all scenarios). This is also called non-anticipativity constraint. Operational variables, on the other hand, are fully scenario-indexed. All constraints are duplicated across scenarios with scenario-specific parameters and the objective function weights scenario costs by their probabilities.

The model of a network without scenarios has the same layout: its operational variables and constraints carry a `scenario` dimension with the one label `"base"`, e.g. `n.model["Generator-p"].sel(scenario="base")`. The results written to the network have no `scenario` level.

!!! note "Problem Scaling Properties"

    - **Investment variables** scale as O(components)
    - **Operational variables** scale as O(scenarios × components × snapshots)
    - **Constraint matrix size** grows almost linearly with the number of scenarios.

Optimization expressions from `n.optimize.expressions`, e.g. for custom constraints, carry a `scenario` dimension with scenario-specific coefficients, e.g. `n.optimize.expressions.capex()` uses the capital costs of each scenario.


### Evaluation

After solving the stochastic optimization problem, we can evaluate the results. The optimal capacities for each scenario can, among other ways, be accessed via `n.statistics.optimal_capacity()`:

``` py
>>> cap_stochastic = (
...     n.statistics.optimal_capacity().round(1).unstack(level="scenario")
... )
>>> cap_stochastic
scenario                      no_volcano  volcano
component   carrier
Generator   fossil backup          7159.2   7159.2
            load shedding         10901.2  10901.2
            solar                 16359.1  16359.1
            wind                  25326.7  25326.7
StorageUnit battery storage        5075.2   5075.2
```

Note that the optimal capacities are the same across scenarios for investment variables, reflecting the non-anticipativity constraint. However, compared to the previous deterministic model, the optimal capacities are different:

``` py
>>> cap_stochastic.iloc[:, 0].div(cap_deterministic).round(2)
component    carrier
Generator    fossil backup      1.14
             load shedding      1.00
             solar              0.74
             wind               1.18
StorageUnit  battery storage    0.55
dtype: float64
```

For example, we see less solar and battery storage, but more wind and fossil backup generation.
The system costs are also 7% larger because the model has to rely more on backup generation to remain feasible.

``` py
>>> n.objective / 1e9 / obj_deterministic  # doctest: +ELLIPSIS
1.067...
```

Finally, we can also see in the energy balance that more fossil backup generation is used in case the volcano erupts to compensate for the reduced solar generation:

``` py
>>> n.statistics.energy_balance().div(1e6).round(2).unstack(level="scenario").xs("electricity", level="bus_carrier")  # doctest: +ELLIPSIS
scenario                      no_volcano  volcano
component   carrier
Generator   fossil backup          12.15    22.91
            load shedding           0.01     0.04
            solar                  19.21     1.93
            wind                   35.16    41.48
Load        -                     -66.27   -66.27
StorageUnit battery storage        -0.26    -0.10
```

### Metrics

When working with stochastic optimization, it can be useful to quantify the value of accounting for uncertainty compared to deterministic approaches. There are several such value of information (VOI) metrics.

**Expected Value of Perfect Information (EVPI)**:

!!! note "Background"

    $$\text{EVPI} = \mathbb{E}[\text{WS}] - \text{SP}$$

The EVPI measures the maximum value of perfect information about uncertain parameters. It compares the expected cost of the wait-and-see (WS) solutions—where the decision maker knows which scenario will occur before making any decisions—against the stochastic programming (SP) solution. The EVPI represents an upper bound on what should be paid for improved forecasting or information gathering systems. By definition, EVPI is always non-negative, since perfect information cannot increase expected costs.

**Expected Cost of Ignoring Uncertainty (ECIU) / Value of Stochastic Solution (VSS)**:

!!! note "Background"

    $$\text{ECIU} = \text{EEV} - \text{SP}$$

The ECIU quantifies the cost of ignoring uncertainty by comparing the expected cost of the expected value (EEV) solution—where decisions are made using mean parameter values—against the stochastic programming solution. This metric demonstrates the value of using stochastic optimization instead of deterministic optimization with expected parameter values. A higher ECIU suggests that stochastic modeling provides substantial improvements over deterministic approaches. The ECIU is always non-negative.

The elements $\text{WS}$, $\text{SP}$ and $\text{EVPI}$ satisfy the following inequality chain:

!!! note "Background"

    $$\text{WS} \leq \text{SP} \leq \text{EEV}$$

For detailed treatment of these measures and their economic interpretation, see Birge and Louveaux (2011), Chapter 4.[^1]


## Risk Preferences with Conditional Value-at-Risk (CVaR)

The risk-neutral stochastic optimization introduced above minimizes expected systems costs and can leave the system exposed to rare but expensive outcomes.

PyPSA also supports risk-averse stochastic optimization using Conditional Value-at-Risk (CVaR). CVaR is a risk measure that captures the expected cost in the worst-case tail of the distribution. It adds a convex penalty term for expensive outcomes, thereby shifting investments towards solutions that hedge against extreme scenarios. Users control how much of the worst-case tail to consider and how strongly to weight it relative to the expected cost optimization.

### Mathematical Formulation

The risk-averse stochastic optimization extends the risk-neutral formulation by including CVaR in the objective, which weights the operational costs of the scenarios:

<!-- md:spec risk_weighted_opex -->

Where:

- $\alpha \in (0,1)$ is the confidence level that determines the probability threshold for the worst-case scenarios to be considered in the CVaR calculation. A higher $\alpha$ places more emphasis on extreme outcomes. CVaR averages costs over the worst $(1-\alpha)$ share of the distribution. At $\alpha = 1$, the denominator is zero (undefined). At $\alpha = 0$, the confidence level covers the entire distribution and the formulation degenerates (equivalent to risk-neutral expectation).

- $\omega \in [0,1]$ is the tail weighting. It balances expected cost and tail cost by controlling how strongly the identified worst-case scenarios are weighted in a convex combination. The closed interval includes meaningful endpoints: $\omega=0$ corresponds to risk-neutral optimization, while $\omega=1$ corresponds to pure CVaR minimization.

!!! note "Background"

    The CVaR at confidence level $\alpha$ is defined as the expected loss conditional on being in the worst $1-\alpha$ of outcomes:

    $$
    \mathit{CVaR}_\alpha(Q) = \mathbb{E}[\,Q(\mathbf{x},\boldsymbol{\xi}_s)\;|\;Q(\mathbf{x},\boldsymbol{\xi}_s)\geq \,\mathit{VaR}_\alpha(Q)].
    $$

    It can equivalently be written as an optimization problem, see Rockafellar & Uryasev (2002)[^2]:

    $$
    \mathit{CVaR}_\alpha(Q) = \min_{\mathit{VaR} \in \mathbb{R}} \Big[ \mathit{VaR} + \frac{1}{1-\alpha} \sum_{s \in \mathcal{S}} \pi_s \max\{Q(\mathbf{x},\boldsymbol{\xi}_s)-\mathit{VaR},0\} \Big].
    $$

To embed CVaR in a linear program, some auxiliary variables and constraints are introduced to enforce the definition:

Variables:

- $\mathit{VaR}$ (free): the Value-at-Risk cutoff at level $\alpha$.
- $\mathit{CVaR}$ (free): the CVaR at level $\alpha$.
- $\mathit{CVaR}^{\mathrm{a}}_s \geq 0$: excess loss of scenario $s$ above $\mathit{VaR}$.

Constraints:

<!-- md:spec CVaR_excess CVaR_def -->

At the optimal solution, the auxiliary variable $\mathit{VaR}$ takes the role of the $\alpha$-quantile of scenario costs, i.e. the Value-at-Risk at level $\alpha$. Scenarios with costs above $\mathit{VaR}$ have $\mathit{CVaR}^{\mathrm{a}}_s > 0$ and thus contribute to the tail. The CVaR variable then represents the expected cost conditional on being in this worst $1-\alpha$ fraction of scenarios. In this way, the optimization automatically identifies which scenarios belong to the tail and penalizes them in the objective.

!!! note "Robust-like optimization"
    A robust "optimize against the worst scenario" setting is approximated by choosing $\omega=1$ and setting $\alpha = 1 - \pi_{\text{worst scenario}}$, so that the tail includes only the worst case.

!!! note "Auxiliary variables"
    The auxiliary variables are not written back to component tables; they are internal to the optimization model.

??? note "Symbols"

    <!-- md:spec-legend total_cost risk_weighted_opex CVaR_excess CVaR_def -->

!!! warning "Limitations"
    CVaR is not available with quadratic marginal costs, since the resulting quadratic constraints are not supported by `linopy`.

### Implementation

Continuing the volcano example from above, risk preferences can be explored using the `n.set_risk_preference()` method. Let's compare different risk attitudes:

```py
# Risk-neutral baseline (expected cost)
>>> n.optimize()

# CVaR with moderate risk aversion
>>> n.set_risk_preference(alpha=0.8, omega=0.5)
>>> n.optimize()

# Edge case: CVaR with omega=0 equals risk-neutral
>>> n.set_risk_preference(alpha=0.8, omega=0.0)
>>> n.optimize()

# Edge case: CVaR capturing the results of the worst-case scenario
# (omega=1, alpha so only worst scenario is in tail)
>>> p_worst = float(n.scenario_weightings.loc["volcano", "weight"])  # 0.2
>>> n.set_risk_preference(alpha=1 - p_worst, omega=1.0)
>>> n.optimize()
```

If no risk preference is set, the model reverts to risk-neutral stochastic optimization.

#### Results and Interpretation

With moderate risk aversion (`alpha=0.8, omega=0.5`), the optimization shifts investments to hedge against expensive tail scenarios:

```py
>>> cap_cvar = (
...     n.statistics.optimal_capacity().round(1).unstack(level="scenario")
... )
>>> cap_diff = cap_cvar.iloc[:, 0] - cap_stochastic.iloc[:, 0]
>>> print("Capacity diff (omega=0.5 - neutral) [MW]:")
>>> print(cap_diff.round(2))
component    carrier
Generator    fossil backup       910.6
             load shedding         0.0
             solar             -9233.3
             wind               3534.0
StorageUnit  battery storage   -3884.4
```

Economic interpretation: CVaR penalizes costly tail operations (worst-scenario OPEX, such as load shedding or expensive peakers). The model invests more upfront to reduce exposure to these rare but severe outcomes. In this case, the economic hedge is to reduce solar (which is potentially affected by the volcano scenario) and instead increase wind capacity paired with more backup generation. Battery storage needs are also substantially reduced due to less solar and economics of battery storage cycle. Put short, *risk aversion hedges against the risky scenario by shifting costs from tail OPEX to upfront CAPEX.*

### Metrics

The insurance premium measures the increase in expected total system costs relative to the risk-neutral solution.

!!! note "Background"

    $$
    \mathrm{Premium}(\alpha, \omega) = \big( \mathbf{c}^\top \mathbf{x} + \sum_{s \in \mathcal{S}} \pi_s Q(\mathbf{x},\boldsymbol{\xi}_s) \big)_{(\alpha,\omega)} - \big( \mathbf{c}^\top \mathbf{x} + \sum_{s \in \mathcal{S}} \pi_s Q(\mathbf{x},\boldsymbol{\xi}_s) \big)_{(\omega=0)}
    $$

In other words, $\mathrm{Premium}(\alpha, \omega)$ is the additional expected cost of applying a hedging strategy (or of choosing a more robust portfolio/policy) compared to a baseline.

The tail risk reduction measures the decrease in CVaR at confidence level $\alpha$.

!!! note "Background"

    $$
    \Delta \,\mathit{CVaR}_{\alpha} = \mathit{CVaR}_{\alpha} \big(Q(\mathbf{x},\boldsymbol{\xi}_s)\big)_{(\omega=0)} - \mathit{CVaR}_{\alpha} \big(Q(\mathbf{x},\boldsymbol{\xi}_s)\big)_{(\alpha,\omega)}
    $$

In other words, $\Delta \,\mathit{CVaR}_{\alpha}$ is the amount of tail risk that is reduced thanks to the hedging strategy.

A useful combined indicator is the risk-hedging cost, which measures the additional expected cost per unit of tail risk avoided:

!!! note "Background"

    $$
    \mathrm{RHC}(\alpha, \omega) = \frac{\text{Premium}(\alpha, \omega)}{\Delta \,\mathit{CVaR}_\alpha}
    $$

For the volcano example with `alpha=0.8, omega=0.5`, we can calculate the insurance premium:

```py
>>> n.set_risk_preference(alpha=0.8, omega=0.5)
>>> n.optimize()
>>> obj_cvar = n.objective / 1e9
>>> n.set_risk_preference(alpha=0.8, omega=0.0)
>>> n.optimize()
>>> obj_risk_neutral = n.objective / 1e9
>>> premium = obj_cvar - obj_risk_neutral
>>> print(f"Insurance Premium: {premium:.4f} billion EUR ({premium / obj_risk_neutral * 100:.2f}%)")
Insurance Premium: 0.5062 billion EUR (8.11%)
```

Here the decision maker pays an 8% increase in expected system costs to achieve a more robust portfolio that hedges against low-probability, high-impact tail risks such as the volcano scenario with severe solar output reduction.


## Examples

<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Stochastic Optimization**

    Demonstrates investment planning under uncertainty with scenario-based two-stage stochastic optimization.

    [:octicons-arrow-right-24: Go to example](../../examples/stochastic-optimization.ipynb)

</div>

[^1]: Birge, J. R., & Louveaux, F. (2011). [Introduction to Stochastic Programming](https://link.springer.com/book/10.1007/978-1-4614-0237-4).

[^2]: Rockafellar, R. T., & Uryasev, S. (2002). [Conditional Value-at-Risk for General Loss Distributions](https://doi.org/10.1016/S0378-4266(02)00271-6). Journal of Banking & Finance, 26(7), 1443–1471.
