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

The stochastic optimization problem in PyPSA follows the standard two-stage risk-neutral stochastic programming formulation:

$$
\begin{align}
\min_{x} \quad & c^T x + \sum_{s \in S} p_s Q(x, \xi_s) \\
\text{s.t.} \quad & A x = b \\
& x \geq 0
\end{align}
$$

Where:

- $x$: First-stage (here-and-now) decision variables (e.g., investment decisions in component capacities)
- $\xi_s$: Random parameter realizations in scenario $s$
- $p_s$: Probability of scenario $s$ occurring
- $Q(x, \xi_s)$: Second-stage recourse function for scenario $s$
- $S$: Set of all scenarios

The second-stage problem $Q(x, \xi_s)$ represents:

$$
\begin{align}
Q(x, \xi_s) = \min_{y_s} \quad & q_s^T y_s \\
\text{s.t.} \quad & W y_s = h_s - T x \\
& y_s \geq 0
\end{align}
$$

Where:

- $y_s$: Second-stage (wait-and-see) decision variables for scenario $s$ (e.g., energy dispatch decisions)
- $q_s$: Second-stage cost coefficients for scenario $s$
- $W$: Recourse matrix (coefficient matrix for second-stage variables $y_s$)
- $T$: Technology matrix (coefficient matrix for first-stage variables $x$ in second-stage constraints)
- $h_s$: Right-hand-side vector for scenario $s$

The second-stage constraints $W y_s = h_s - T x$ show how first-stage decisions $x$ affect the feasible region for second-stage variables $y_s$ in each scenario.

The scenario tree in PyPSA splits into **here-and-now** investment decisions (t=0) and **wait-and-see** dispatch decisions (t=1):

| Stage | Decision Type | Variables | Information | Examples |
|-------|---------------|-----------|-------------|-----------|
| **t=0** | **Here-and-now** | Investment decisions ($x$) - Common across scenarios | Uncertain parameters unknown, probability distribution known | Generator capacity, transmission expansion, storage capacity |
| **t=1** | **Wait-and-see** | Dispatch decisions ($y_s$) - Scenario-specific | Uncertain parameters revealed, perfect information for scenario $s$ | Dispatch, storage operation, load shedding |


!!! note "Limitation: Investment and dispatch stages are fixed"

    Currently, PyPSA only considers dispatch decisions as second-stage variables and investment decisions as first-stage variables. Support for second-stage investment decisions (e.g., scenario-dependent capacity additions as recourse measures) as well as first-stage dispatch decisions (e.g., for stochastic optimisation in purely operational settings with forecast uncertainty) is planned for future releases.

!!! note "Consideration of risk preferences and robust optimisation"

    The default stochastic programming implementation in PyPSA minimizes expected costs across scenarios weighted by their probabilities. This approach considers risk-neutral decision making. 
    PyPSA also supports changing risk preference through Conditional Value at Risk (CVaR)-based risk-averse optimization, allowing users to account for extreme outcomes and tail risks in their optimization.

For a comprehensive treatment of two-stage stochastic programming theory and methods, see Birge and Louveaux (2011).[^1]

### Implementation

Let us consider a single-node capacity expansion model in the style of [model.energy](https://model.energy). This stylized model calculates the cost of meeting an hourly electricity demand time series from a combination of wind power, solar power, battery and hydrogen storage as well as load shedding. See this [example](../../examples/capacity-expansion-planning-single-node.ipynb). We add a gas power plant as additional technology option and solve the model with given load and renewable availability profiles as a deterministic problem. Afterwards, we inspect the optimal objective value and expanded capacities.

``` py
>>> import pypsa
>>>
>>> n = pypsa.examples.model_energy()
>>> n.add(
...     "Generator",
...     "gas",
...     carrier="gas",
...     bus="electricity",
...     p_nom=5000,
...     efficiency=0.5,
...     marginal_cost=100,
... )
>>>
>>> n.optimize(log_to_console=False)
('ok', 'optimal')
>>>
>>> cap_deterministic = n.statistics.optimal_capacity().div(1e3).round(1)
>>> cap_deterministic
component    carrier
Generator    gas                   5.0
             load shedding        10.9
             solar                22.8
             wind                 20.2
Link         electrolysis          0.1
             turbine               1.4
StorageUnit  battery storage      11.0
Store        hydrogen storage    245.4
dtype: float64
>>>
>>> obj_deterministic = n.objective / 1e9
>>> obj_deterministic  # doctest: +ELLIPSIS
5.3574849734203...
```

### Scenario Definition

Now suppose we want to consider the case that with a 10% probability, a volcano erupts which reduces the solar capacity factor time series uniformly by 70%. For such a scenario, we can use PyPSA's stochastic optimization functionality. First, we need to define the scenarios with probabilities that must sum to 1:

``` py
>>> n.set_scenarios({"volcano": 0.1, "no_volcano": 0.9})
```

We can check that the scenarios have been set correctly:

``` py
>>> n.has_scenarios
True
```

``` py
>>> n.scenarios
Index(['volcano', 'no_volcano'], dtype='object', name='scenario')
```

``` py
>>> n.scenario_weightings
            weight
scenario          
volcano        0.1
no_volcano     0.9
```

The key change after calling `n.set_scenarios()` is that all component data is broadcasted across all scenarios. All component `pandas.DataFrame` objects gain a "scenario" dimension as outermost index level.

``` py
>>> n.generators[["bus", "marginal_cost", "efficiency"]]  # doctest: +ELLIPSIS
                                  bus  marginal_cost  efficiency
scenario   name
volcano    load shedding  electricity         2000.0         1.0
           wind           electricity            0.0         1.0
           solar          electricity            0.0         1.0
           gas            electricity          100.0         0.5
no_volcano load shedding  electricity         2000.0         1.0
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
the scenarios, we can modify scenario-specific parameters. In our case, we want to reduce the solar capacity factor in the "volcano" scenario by 70%:

``` py
>>> n.generators_t.p_max_pu.loc[:, ("volcano", "solar")] *= 0.3
>>> n.generators_t.p_max_pu.loc["2019-06-21"]
scenario            volcano         no_volcano        
name                  solar    wind      solar    wind
snapshot                                              
2019-06-21 00:00:00  0.0000  0.1240      0.000  0.1240
2019-06-21 03:00:00  0.0051  0.0843      0.017  0.0843
2019-06-21 06:00:00  0.0864  0.0441      0.288  0.0441
2019-06-21 09:00:00  0.1710  0.0405      0.570  0.0405
2019-06-21 12:00:00  0.1482  0.0432      0.494  0.0432
2019-06-21 15:00:00  0.0720  0.0347      0.240  0.0347
2019-06-21 18:00:00  0.0030  0.0775      0.010  0.0775
2019-06-21 21:00:00  0.0000  0.1455      0.000  0.1455
```

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
 * Generator-p_nom (name)
 * Link-p_nom (name)
 * Store-e_nom (name)
 * StorageUnit-p_nom (name)
 * Generator-p (scenario, name, snapshot)
 * Link-p (scenario, name, snapshot)
 * Store-e (scenario, name, snapshot)
 * StorageUnit-p_dispatch (scenario, name, snapshot)
 * StorageUnit-p_store (scenario, name, snapshot)
 * StorageUnit-state_of_charge (scenario, name, snapshot)
 * Store-p (scenario, name, snapshot)
<BLANKLINE>
Constraints:
------------
 * Generator-ext-p_nom-lower (name, scenario)
 * Link-ext-p_nom-lower (name, scenario)
 * Store-ext-e_nom-lower (name, scenario)
 * StorageUnit-ext-p_nom-lower (name, scenario)
 * Generator-fix-p-lower (scenario, name, snapshot)
 * Generator-fix-p-upper (scenario, name, snapshot)
 * Generator-ext-p-lower (scenario, name, snapshot)
 * Generator-ext-p-upper (scenario, name, snapshot)
 * Link-ext-p-lower (scenario, name, snapshot)
 * Link-ext-p-upper (scenario, name, snapshot)
 * Store-ext-e-lower (scenario, name, snapshot)
 * Store-ext-e-upper (scenario, name, snapshot)
 * StorageUnit-ext-p_dispatch-lower (scenario, name, snapshot)
 * StorageUnit-ext-p_dispatch-upper (scenario, name, snapshot)
 * StorageUnit-ext-p_store-lower (scenario, name, snapshot)
 * StorageUnit-ext-p_store-upper (scenario, name, snapshot)
 * StorageUnit-ext-state_of_charge-lower (scenario, name, snapshot)
 * StorageUnit-ext-state_of_charge-upper (scenario, name, snapshot)
 * Bus-nodal_balance (name, scenario, snapshot)
 * StorageUnit-energy_balance (scenario, name, snapshot)
 * Store-energy_balance (scenario, name, snapshot)
<BLANKLINE>
Status:
-------
ok
```

As investment variables (i.e. `p_nom`, `s_nom`, `e_nom`) are scenario-independent first-stage decisions, they do not have a scenario dimension (i.e. are common across all scenarios). This is also called non-anticipativity constraint. Operational variables, on the other hand, are fully scenario-indexed. All constraints are duplicated across scenarios with scenario-specific parameters and the objective function weights scenario costs by their probabilities.

!!! note "Problem Scaling Properties"

    - **Investment variables** scale as O(components)
    - **Operational variables** scale as O(scenarios × components × snapshots)
    - **Constraint matrix size** grows almost linearly with the number of scenarios.


### Evaluation

After solving the stochastic optimization problem, we can evaluate the results. The optimal capacities for each scenario can, among other ways, be accessed via `n.statistics.optimal_capacity()`:

``` py
>>> cap_stochastic = (
...     n.statistics.optimal_capacity().div(1e3).round(1).unstack(level="scenario")
... )
>>> cap_stochastic
scenario                      no_volcano  volcano
component   carrier                              
Generator   gas                      5.0      5.0
            load shedding           10.9     10.9
            solar                   21.1     21.1
            wind                    22.0     22.0
Link        electrolysis             0.3      0.3
            turbine                  1.5      1.5
StorageUnit battery storage         10.2     10.2
Store       hydrogen storage       221.0    221.0
```

Note that the optimal capacities are the same across scenarios for investment variables, reflecting the non-anticipativity constraint. However, compared to the previous deterministic model, the optimal capacities are different:

``` py
>>> cap_stochastic.iloc[:, 0].div(cap_deterministic).round(2)
component    carrier         
Generator    gas                 1.00
             load shedding       1.00
             solar               0.93
             wind                1.09
Link         electrolysis        3.00
             turbine             1.07
StorageUnit  battery storage     0.93
Store        hydrogen storage    0.90
dtype: float64
```

For example, we see less solar and batteries and more wind and hydrogen storage.
The system costs are also 3% larger because the model has to hedge against uncertainty to remain feasible.

``` py
>>> n.objective / 1e9 / obj_deterministic  # doctest: +ELLIPSIS
1.03...
```

Finally, we can also see in the energy balance that more gas is used in case the volcano erupts to compensate for the reduced solar generation. Additionally, wind curtailment is reduced:

``` py
>>> n.statistics.energy_balance().div(1e6).round(2).unstack(level="scenario")  # doctest: +ELLIPSIS
scenario                                 no_volcano  volcano
component   carrier         bus_carrier
Generator   gas             electricity       10.14    21.20
            load shedding   electricity        0.06     0.28
...
```

### Metrics

When working with stochastic optimization, it can be useful to quantify the value of accounting for uncertainty compared to deterministic approaches. There are several such value of information (VOI) metrics.

**Expected Value of Perfect Information (EVPI)**:

$$\text{EVPI} = \mathbb{E}[\text{WS}] - \text{SP}$$

The EVPI measures the maximum value of perfect information about uncertain parameters. It compares the expected cost of the wait-and-see (WS) solutions—where the decision maker knows which scenario will occur before making any decisions—against the stochastic programming (SP) solution. The EVPI represents an upper bound on what should be paid for improved forecasting or information gathering systems. By definition, EVPI is always non-negative, since perfect information cannot increase expected costs. 

**Expected Cost of Ignoring Uncertainty (ECIU) / Value of Stochastic Solution (VSS)**:

$$\text{ECIU} = \text{EEV} - \text{SP}$$

The ECIU quantifies the cost of ignoring uncertainty by comparing the expected cost of the expected value (EEV) solution—where decisions are made using mean parameter values—against the stochastic programming solution. This metric demonstrates the value of using stochastic optimization instead of deterministic optimization with expected parameter values. A higher ECIU suggests that stochastic modeling provides substantial improvements over deterministic approaches. The ECIU is always non-negative.

The elements $\text{WS}$, $\text{SP}$ and $\text{EVPI}$ satisfy the following inequality chain:

$$
\text{WS} \leq \text{SP} \leq \text{EEV}
$$

For detailed treatment of these measures and their economic interpretation, see Birge and Louveaux (2011), Chapter 4.[^1]


## Risk Preferences with Conditional Value-at-Risk (CVaR)

The risk-neutral stochastic optimization introduced above minimizes expected systems costs and can leave the system exposed to rare but expensive outcomes.

PyPSA also supports risk-averse stochastic optimization using Conditional Value-at-Risk (CVaR). CVaR is a risk measure that captures the expected cost in the worst-case tail of the distribution. It adds a convex penalty term for expensive outcomes, thereby shifting investments towards solutions that hedge against extreme scenarios. Users control how much of the worst-case tail to consider and how strongly to weight it relative to the expected cost optimization.

### Mathematical Formulation

The risk-averse stochastic optimization extends the risk-neutral formulation by including CVaR in the objective:

$$
\min_x \quad c^T x \;+\; (1-\omega)\sum_{s \in S} p_s Q(x,\xi_s) \;+\; \omega \,\mathrm{CVaR}_\alpha.
$$

Where:

- $\alpha \in (0,1)$ is the confidence level that determines the probability threshold for the worst-case scenarios to be considered in the CVaR calculation. A higher $\alpha$ places more emphasis on extreme outcomes. CVaR averages costs over the worst $(1-\alpha)$ share of the distribution. At $\alpha = 1$, the denominator is zero (undefined). At $\alpha = 0$, the confidence level covers the entire distribution and the formulation degenerates (equivalent to risk-neutral expectation).

- $\omega \in [0,1]$ is the tail weighting. It balances expected cost and tail cost by controlling how strongly the identified worst-case scenarios are weighted in a convex combination. The closed interval includes meaningful endpoints: $\omega=0$ corresponds to risk-neutral optimization, while $\omega=1$ corresponds to pure CVaR minimization.

The CVaR at confidence level $\alpha$ is defined as the expected loss conditional on being in the worst $1-\alpha$ of outcomes:  

$$
\mathrm{CVaR}_\alpha(Q) = \mathbb{E}[\,Q(x,\xi_s)\;|\;Q(x,\xi_s)\geq \,\textrm{VaR}_\alpha(Q)].
$$

It can equivalently be written as an optimization problem, see Rockafellar & Uryasev (2002)[^2]:  

$$
\mathrm{CVaR}_\alpha(Q) = \min_{\theta \in \mathbb{R}} \Big[ \theta + \frac{1}{1-\alpha} \sum_{s \in S} p_s \max\{Q(x,\xi_s)-\theta,0\} \Big].
$$

To embed CVaR in a linear program, some auxiliary variables and constraints are introduced to enforce the definition:

Variables:

- $\theta$ (free): the Value-at-Risk cutoff at level $\alpha$.
- $\mathrm{CVaR}_\alpha$ (free): the CVaR at level $\alpha$.
- $a_s \geq 0$: excess loss of scenario $s$ above $\theta$.  

Constraints:

$$
a_s \ge Q(x,\xi_s) - \theta \quad \forall s \in S,
$$

$$
	\theta + \frac{1}{1-\alpha} \sum_{s \in S} p_s a_s \;\le\; \mathrm{CVaR}_\alpha.
$$

At the optimal solution, the auxiliary variable $\theta$ takes the role of the $\alpha$-quantile of scenario costs, i.e. the Value-at-Risk at level $\alpha$. Scenarios with costs above $\theta$ have $a_s > 0$ and thus contribute to the tail. The CVaR variable then represents the expected cost conditional on being in this worst $1-\alpha$ fraction of scenarios. In this way, the optimization automatically identifies which scenarios belong to the tail and penalizes them in the objective.

!!! note "Robust-like optimization"
    A robust "optimize against the worst scenario" setting is approximated by choosing $\omega=1$ and setting $\alpha = 1 - p(\text{worst scenario})$, so that the tail includes only the worst case.

!!! note "Auxiliary variables"
    The auxiliary variables are not written back to component tables; they are internal to the optimization model.

!!! warning "Limitations"
    CVaR is not available with quadratic marginal costs, since the resulting quadratic constraints are not supported by `linopy`.

### Implementation

Continuing the volcano example from above, risk preferences can be explored using the `n.set_risk_preference()` method. Let's compare different risk attitudes:

```py
# Risk-neutral baseline (expected cost)
>>> n.optimize()

# CVaR with moderate risk aversion
>>> n.set_risk_preference(alpha=0.9, omega=0.5)
>>> n.optimize()

# Edge case: CVaR with omega=0 equals risk-neutral
>>> n.set_risk_preference(alpha=0.9, omega=0.0)
>>> n.optimize()

# Edge case: CVaR capturing the results of the worst-case scenario 
# (omega=1, alpha so only worst scenario is in tail)
>>> p_worst = float(n.scenario_weightings.loc["volcano", "weight"])  # 0.1
>>> n.set_risk_preference(alpha=1 - p_worst, omega=1.0)
>>> n.optimize()
```

If no risk preference is set, the model reverts to risk-neutral stochastic optimization.

#### Results and Interpretation

With moderate risk aversion (`alpha=0.9, omega=0.5`), the optimization shifts investments to hedge against expensive tail scenarios:

```py
>>> cap_cvar = (
...     n.statistics.optimal_capacity().div(1e3).round(1).unstack(level="scenario")
... )
>>> cap_diff = cap_cvar.iloc[:, 0] - cap_stochastic.iloc[:, 0]
>>> print("Capacity diff (omega=0.5 - neutral) [GW]:")
>>> print(cap_diff.round(2))
component    carrier         
Generator    load shedding          0.00
             solar                 10.88
             wind                   4.64
Link         electrolysis           0.59
             turbine                0.36
StorageUnit  battery storage       -1.31
Store        hydrogen storage    1579.99
```

Economic interpretation: CVaR penalizes costly tail operations (worst-scenario OPEX, such as load shedding or expensive peakers). The model invests more upfront to reduce exposure to these rare but severe outcomes. In this case, the cheapest hedge is to overbuild solar and complement it with long-duration hydrogen storage, even though solar itself is affected in the "volcano" scenario. This combination minimizes worst-case costs.
In other words: *risk aversion hedges against the risky scenario by shifting costs from tail OPEX to upfront CAPEX.*

### Metrics

The insurance premium measures the increase in expected total system costs relative to the risk-neutral solution.

$$
\mathrm{Premium}(\alpha, \omega) = \big( c^T x + \sum_{s \in S} p_s Q(x,\xi_s) \big)_{(\alpha,\omega)} - \big( c^T x + \sum_{s \in S} p_s Q(x,\xi_s) \big)_{(\omega=0)}
$$

In other words, $\mathrm{Premium}(\alpha, \omega)$ is the additional expected cost of applying a hedging strategy (or of choosing a more robust portfolio/policy) compared to a baseline.

The tail risk reduction measures the decrease in CVaR at confidence level $\alpha$.

$$
\Delta \,\mathrm{CVaR}_{\alpha} = \mathrm{CVaR}_{\alpha} \big(Q(x,\xi_s)\big)_{(\omega=0)} - \mathrm{CVaR}_{\alpha} \big(Q(x,\xi_s)\big)_{(\alpha,\omega)}
$$

In other words, $\Delta \,\mathrm{CVaR}_{\alpha}$ the amount of tail risk that is reduced thanks to the hedging strategy.

A useful combined indicator is the risk-hedging cost, which measures the additional expected cost per unit of tail risk avoided:

$$
	\mathrm{RHC}(\alpha, \omega) = \frac{\text{Premium}(\alpha, \omega)}{\Delta \,\mathrm{CVaR}_\alpha}
$$




## Examples

<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Stochastic Optimization**

    Demonstrates investment planning under uncertainty with scenario-based two-stage stochastic optimization.

    [:octicons-arrow-right-24: Go to example](../../examples/stochastic-optimization.ipynb)

</div>

[^1]: Birge, J. R., & Louveaux, F. (2011). [Introduction to Stochastic Programming](https://link.springer.com/book/10.1007/978-1-4614-0237-4).

[^2]: Rockafellar, R. T., & Uryasev, S. (2002). [Conditional Value-at-Risk for General Loss Distributions](https://doi.org/10.1016/S0378-4266(02)00271-6). Journal of Banking & Finance, 26(7), 1443–1471.