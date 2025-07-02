# Stochastic Optimization in PyPSA

## Overview

Stochastic optimization in PyPSA enables modeling and solving power system planning problems under uncertainty. This capability addresses real-world scenarios where parameters such as fuel prices, renewable energy availability, demand patterns, or technology costs are uncertain at the time investment decisions must be made.

PyPSA implements a **two-stage stochastic programming framework** with scenario trees, allowing users to optimize investment decisions (first-stage) that are robust across multiple possible future realizations (scenarios) of uncertain parameters.

## Mathematical Formulation

### Two-Stage Stochastic Programming

The stochastic optimization problem in PyPSA follows the standard two-stage stochastic programming formulation:

$$
\begin{align}
\min_{x} \quad & c^T x + \sum_{s \in S} p_s Q(x, \xi_s) \\
\text{s.t.} \quad & A x = b \\
& x \geq 0
\end{align}
$$

Where:

- $x$: First-stage (here-and-now) decision variables (e.g., investment decisions in generation/transmission/storage capacities)
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

For a comprehensive treatment of two-stage stochastic programming theory and methods, see Birge and Louveaux (2011) [1].

### Scenario Tree Structure

PyPSA uses a **two-stage scenario tree** where:

1. **Root Node (t=0)**: **Here-and-now decisions** - Investment decisions (capacity planning) made under uncertainty
2. **Scenario Nodes (t=1)**: **Wait-and-see decisions** - Operational decisions (dispatch, storage) made after uncertainty realization*

*Currently, PyPSA implements operational decisions (dispatch, storage operation) as second-stage variables. Support for recourse investment decisions (e.g., scenario-dependent capacity additions) is planned for future releases.

**Scenario Tree Representation:**

| Stage | Decision Type | Variables | Information | Examples |
|-------|---------------|-----------|-------------|-----------|
| **t=0** | **Here-and-now** | Investment ($x$) - Common across scenarios | Uncertain parameters unknown, probability distribution known | Generator capacity, transmission lines, storage capacity |
| **t=1** | **Wait-and-see** | Operations ($y_s$) - Scenario-specific | Uncertain parameters revealed, perfect information for scenario $s$ | Dispatch, storage operation, load shedding |

**Information structure**: Decisions at t=0 use only probability distributions; decisions at t=1 use realized parameter values.

## Operations Research Foundation

### Value of Information Measures

PyPSA enables calculation of standard stochastic programming metrics for evaluating solution quality and the value of uncertainty modeling:

**Expected Value of Perfect Information (EVPI)**:

$$\text{EVPI} = \mathbb{E}[\text{WS}] - \text{SP}$$

The EVPI measures the maximum value of perfect information about uncertain parameters. It compares the expected cost of the wait-and-see (WS) solutions—where the decision maker knows which scenario will occur before making any decisions—against the stochastic programming (SP) solution. The EVPI represents an upper bound on what should be paid for improved forecasting or information gathering systems.

**Expected Cost of Ignoring Uncertainty (ECIU) / Value of Stochastic Solution (VSS)**:

$$\text{ECIU} = \text{EEV} - \text{SP}$$

The ECIU quantifies the cost of ignoring uncertainty by comparing the expected cost of the expected value (EEV) solution—where decisions are made using mean parameter values—against the stochastic programming solution. This metric demonstrates the value of using stochastic optimization instead of deterministic optimization with expected parameter values.

For detailed treatment of these measures and their economic interpretation, see Birge and Louveaux (2011) [1], Chapter 4.

### Robust Optimization Connection

Stochastic programming, as implemented in PyPSA, minimizes expected cost across scenarios weighted by their probabilities. This approach provides probabilistic guarantees about system performance and enables risk-neutral decision making. While PyPSA currently focuses on expectation-based optimization, the framework can be extended to incorporate risk measures such as Conditional Value at Risk (CVaR) or worst-case scenario considerations to achieve more risk-averse solutions.* 

*Support for risk measures and robust optimization variants is planned in the future release.

## PyPSA Implementation

### Stochastic Network Methods and Properties

PyPSA provides several methods and properties for working with stochastic networks:

**Core Methods:**

- `n.set_scenarios(scenarios)`: Define scenarios with probabilities (must sum to 1)
- `n.optimize()`: Solve stochastic optimization problem

**Network Properties:**

- `n.scenarios`: pandas Series containing scenario names and probabilities
- `n.has_scenarios`: Boolean indicating if network has scenarios defined

**Data Structure:**

- All component DataFrames gain scenario dimension as outermost index level
- Static data: `n.generators.loc[('scenario_name', 'generator_name')]`
- Time-series data: `n.generators_t.p.loc[:, ('scenario_name', 'generator_name')]`

**Important Notes:**
- Scenarios are immutable once set (changing scenarios not yet supported)
- Scenarios must be set **after** adding network components
- Probability values must sum to exactly 1.0

### Implementation Example

The following example demonstrates stochastic optimization for capacity planning under gas price uncertainty using PyPSA API.

```python
import pypsa
import pandas as pd

# Configuration
gas_price_scenarios = {"low": 40, "med": 70, "high": 100}  # EUR/MWh_th
scenario_probabilities = {"low": 0.4, "med": 0.3, "high": 0.3}
base_scenario = "low"  # Reference scenario for network setup

# Technology specifications
def annuity(life, rate):
    """Capital recovery factor for annualizing investment costs."""
    return rate / (1 - (1 + rate) ** -life) if rate else 1 / life

tech_data = {
    "solar": {"profile": "solar", "investment": 1e6, "marginal_cost": 0.01},
    "wind": {"profile": "onwind", "investment": 2e6, "marginal_cost": 0.02}, 
    "gas": {"investment": 7e5, "efficiency": 0.6},
    "lignite": {"investment": 1.3e6, "efficiency": 0.4, "marginal_cost": 130},
}

# Convert to annualized costs (3% discount rate, 25-year lifetime, 3% FOM)
fom_rate, discount_rate, lifetime = 0.03, 0.03, 25
for tech in tech_data.values():
    tech["capital_cost"] = (annuity(lifetime, discount_rate) + fom_rate) * tech["investment"]

# Load time series data
ts_url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
ts = pd.read_csv(ts_url, index_col=0, parse_dates=True).resample("3h").asfreq()

# Step 1: Build base network with reference scenario
n = pypsa.Network()
n.set_snapshots(ts.index)
n.snapshot_weightings = pd.Series(3, index=ts.index)  # 3-hour time steps

# Add bus and load
n.add("Bus", "DE")
n.add("Load", "DE_load", bus="DE", p_set=1)  # 1 MW constant load

# Add renewable generators (capacity investments)
for tech in ["solar", "wind"]:
    cfg = tech_data[tech]
    n.add("Generator", tech,
          bus="DE", p_nom_extendable=True,
          p_max_pu=ts[cfg["profile"]],
          capital_cost=cfg["capital_cost"],
          marginal_cost=cfg["marginal_cost"])

# Add conventional generators (capacity investments)
for tech in ["gas", "lignite"]:
    cfg = tech_data[tech]
    # Use base scenario gas price for initial setup
    marginal_cost = (gas_price_scenarios[base_scenario] / cfg["efficiency"] 
                    if tech == "gas" else cfg["marginal_cost"])
    
    n.add("Generator", tech,
          bus="DE", p_nom_extendable=True,
          efficiency=cfg.get("efficiency"),
          capital_cost=cfg["capital_cost"],
          marginal_cost=marginal_cost)

# Step 2: Add scenarios AFTER creating base network
n.set_scenarios(scenario_probabilities)

# Step 3: Set scenario-specific parameters (operational uncertainty)
for scenario in gas_price_scenarios:
    if scenario != base_scenario:
        gas_price = gas_price_scenarios[scenario]
        gas_marginal_cost = gas_price / tech_data["gas"]["efficiency"]
        n.generators.loc[(scenario, "gas"), "marginal_cost"] = gas_marginal_cost

# Step 4: Solve stochastic optimization
status, condition = n.optimize(solver_name="highs")
print(f"Optimization status: {status}")
print(f"Total expected cost: {n.objective/1e6:.1f} M€/year")

# Step 5: Analyze results
print("\nOptimal Capacity Mix (MW):")
# Investment decisions are scenario-independent
for tech in ["solar", "wind", "gas", "lignite"]:
    capacity = n.generators.p_nom_opt.loc[(base_scenario, tech)]
    print(f"  {tech.capitalize()}: {capacity:.1f} MW")
```

**Key Implementation Points:**

1. **Network Construction**: Build base network with reference scenario parameters
2. **Scenario Definition**: Use `set_scenarios()` after adding all components
3. **Parameter Updates**: Modify scenario-specific parameters (e.g., gas prices) 
4. **Optimization**: Single call to `optimize()` solves full stochastic problem

### Model Structure and Variables

PyPSA creates a stochastic optimization model by reformulating the two-stage problem as a large-scale deterministic equivalent with scenario-indexed variables and constraints. The model structure can be inspected using PyPSA's optimization module:

```python
# Create and inspect the optimization model
n.optimize.create_model()
print(n.model)
```


*Investment Variables* (scenario-independent):
- `Generator-p_nom`: `(component,)` - Generator capacity decisions
- `Line-s_nom`: `(component,)` - Transmission line capacity decisions
- `Store-e_nom`: `(component,)` - Storage energy capacity decisions

*Operational Variables* (scenario-specific):
- `Generator-p`: `(scenario, component, snapshot)` - Generator dispatch decisions
- `Line-s`: `(scenario, component, snapshot)` - Line flow decisions
- `StorageUnit-state_of_charge`: `(scenario, component, snapshot)` - Storage state decisions


**Key Model Properties:**
- Investment variables have **no scenario dimension** (non-anticipativity constraint)
- Operational variables are **fully scenario-indexed** 
- Constraints are duplicated across scenarios with scenario-specific parameters
- Objective function weights scenario costs by their probabilities

### EVPI and ECIU Calculation Examples

PyPSA enables calculation of value of information measures:

```python
def calculate_evpi(n_stochastic):
    """Calculate Expected Value of Perfect Information."""
    # Store stochastic solution
    sp_cost = n_stochastic.objective
    
    # Solve wait-and-see problems for each scenario
    ws_costs = {}
    for scenario, prob in n_stochastic.scenarios.items():
        # Create deterministic network for this scenario
        n_ws = pypsa.Network()
        # ... setup network with scenario-specific parameters
        # (copy components from n_stochastic but use single scenario)
        
        n_ws.optimize()
        ws_costs[scenario] = n_ws.objective
    
    # Calculate expected wait-and-see cost
    expected_ws = sum(prob * ws_costs[scenario] 
                     for scenario, prob in n_stochastic.scenarios.items())
    
    evpi = expected_ws - sp_cost
    return evpi, ws_costs

def calculate_eciu(n_stochastic):
    """Calculate Expected Cost of Ignoring Uncertainty."""
    # Create deterministic network with expected parameter values
    n_eev = pypsa.Network()
    # ... setup with mean gas price across scenarios
    
    # Solve Expected Value Problem
    n_eev.optimize() 
    eev_capacities = n_eev.generators.p_nom_opt
    
    # Test EEV solution against each scenario
    total_expected_cost = 0
    for scenario, prob in n_stochastic.scenarios.items():
        n_test = pypsa.Network()
        # ... setup with scenario parameters and EEV capacities (fixed)
        n_test.optimize()
        total_expected_cost += prob * n_test.objective
    
    eciu = total_expected_cost - n_stochastic.objective
    return eciu
```

## Validation and Consistency Checks

### Built-in Scenario Validation

The `set_scenarios()` method in `pypsa/network/index.py` performs several automatic validation checks:

```python
# Check if scenarios already exist (not supported to change)
if self.has_scenarios:
    raise NotImplementedError("Changing scenarios not yet supported")

# Ensure either scenarios or kwargs provided, but not both
if scenarios is None and not kwargs:
    raise ValueError("Must pass either scenarios or keyword arguments")
if kwargs and scenarios is not None:
    raise ValueError("Cannot pass both scenarios and keyword arguments")

# Probabilities must sum exactly to 1
if scenarios.sum() != 1:
    raise ValueError(f"Sum of weights must equal 1. Current sum: {scenarios.sum()}")
```

### Unit Tests for Stochastic Functionality

PyPSA's comprehensive test suite in `test/test_stochastic.py` validates the stochastic optimization functionality:

**1. Network Properties and Setup** (`test_network_properties`)
```python
# Scenario configuration validation
n.set_scenarios({"low": 0.33, "medium": 0.34, "high": 0.33})
assert len(n.scenarios) == 3
assert n.has_scenarios
assert abs(n.scenarios.sum() - 1.0) < 1e-10  # Probability normalization

# Data structure integrity
p_set = n.get_switchable_as_dense("Load", "p_set")
assert p_set.columns.names == ["scenario", "component"]
assert p_set.loc[:, scenario].shape[0] == len(n.snapshots)  # Correct time dimension

# Network representation includes scenario information
assert "Scenarios:" in repr(n)
```

**2. Component Function Compatibility** (`test_component_functions`)
```python
# Component functions work with stochastic networks
assert isinstance(n.branches(), pd.DataFrame)
assert isinstance(n.passive_branches(), pd.DataFrame)
assert isinstance(n.controllable_branches(), pd.DataFrame)
```

**3. Network Analysis Methods** (`test_calculate_dependent_values`, `test_cycles`)
```python
# Dependent value calculations (e.g., line reactances)
n.calculate_dependent_values()
assert n.lines.x_pu_eff.notnull().all()

# Cycle analysis for meshed networks
C = n.cycles()
assert isinstance(C, pd.DataFrame)
assert C.notnull().all().all()  # No NaN values in cycle matrix
```

**4. Optimization Model Structure** (`test_model_creation`)
```python
# Investment variables are scenario-independent (first-stage decisions)
assert n.model.variables["Generator-p_nom"].dims == ("component",)

# Operational variables include scenario dimension (second-stage decisions)
assert n.model.variables["Generator-p"].dims == ("scenario", "component", "snapshot")

# Constraint dimensional structure verification
investment_constraints = {"Generator-ext-p_nom-lower"}  # (component, scenario)
operational_constraints = {"Generator-ext-p-lower", "Generator-ext-p-upper"}  # (scenario, component, snapshot)
nodal_balance = {"Bus-nodal_balance"}  # (component, scenario, snapshot)

# Each constraint type has appropriate dimensional structure
for constraint_name in investment_constraints:
    dims = set(n.model.constraints[constraint_name].sizes.keys()) - {"_term"}
    assert dims == {"component", "scenario"}

for constraint_name in operational_constraints:
    dims = set(n.model.constraints[constraint_name].sizes.keys()) - {"_term"}
    assert dims == {"scenario", "component", "snapshot"}
```

**5. Optimization Solution Quality** (`test_optimization_simple`, `test_optimization_advanced`)
```python
# Basic stochastic optimization convergence
n.optimize.create_model()
status, _ = n.optimize(solver_name="highs")
assert status == "ok"

# Advanced component integration (storage, HVDC links)
n.set_scenarios({"low": 0.5, "high": 0.5})
status, _ = n.optimize(solver_name="highs")
assert status == "ok"  # Complex networks solve successfully
```

**6. Solution Accuracy Validation** (`test_solved_network_simple`)
```python
# Benchmark comparison for known test cases
# Gas price scenarios: {low: 40, med: 70, high: 100} €/MWh_th
n.generators.loc[("medium", "gas"), "marginal_cost"] = 70 / efficiency
n.generators.loc[("high", "gas"), "marginal_cost"] = 100 / efficiency

status, _ = n.optimize(solver_name="highs")
assert status == "ok"

# Solution accuracy against pre-computed benchmarks
equal(n.generators.p_nom_opt.loc["low", :], benchmark.generators.p_nom_opt, decimal=2)
equal(n.objective, benchmark.objective, decimal=2)
```

**7. Statistics and Post-Processing** (`test_statistics`, `test_statistics_plot`)
```python
# Stochastic network statistics calculation
ds = n.statistics.installed_capacity()
assert isinstance(ds.index, pd.MultiIndex)
assert "scenario" in ds.index.names
assert not ds.empty

# Statistics aggregation across scenarios
stats = n.statistics()
supply_df = n.statistics.supply(aggregate_time=False)
assert "scenario" in supply_df.index.names

# Visualization compatibility
n.statistics.installed_capacity.plot.bar()
```

**8. Multi-period Stochastic Integration** (`test_solved_network_multiperiod`)
```python
# Combined investment periods and scenarios
n.investment_periods = [2020, 2030]
n.set_scenarios({"high": 0.5, "low": 0.5})

# Scenario-specific time series for multi-period optimization
n.loads_t.p_set = scenario_indexed_dataframe  # (snapshots × scenarios × components)
status, condition = n.optimize(multi_investment_periods=True)
assert status == "ok"

# Energy balance verification per scenario and period
for scenario in ["high", "low"]:
    gen_output = n.generators_t.p.loc[:, (scenario, slice(None))].sum().sum()
    load_demand = n.loads_t.p_set.loc[:, (scenario, "load1")].sum()
    assert abs(gen_output - load_demand) < 1e-1

# Cross-scenario capacity comparison
gen_high = n.generators_t.p.loc[:, ("high", slice(None))].sum().sum()
gen_low = n.generators_t.p.loc[:, ("low", slice(None))].sum().sum()
assert gen_high > gen_low  # High demand scenario requires more generation
```

**9. Single-Scenario Equivalence** (`validate_single_scenario`)
```python
# Deterministic solution reference
status_det, _ = n.optimize()  # Solve without scenarios
obj_det = n.objective
capacity_det = n.generators.p_nom_opt.loc["gen1"]

# Convert to single-scenario stochastic formulation
n.set_scenarios(["scenario"])
n.loads_t.p_set = scenario_indexed_load_data

# Stochastic solution should be numerically identical
status_stoch, _ = n.optimize()
assert abs(n.objective - obj_det) < 1e-6
assert abs(n.generators.p_nom_opt.loc[("scenario", "gen1")] - capacity_det) < 1e-6

# Data structure consistency
assert len(n.scenarios) == 1
assert "scenario" in n.generators_t.p.columns.get_level_values("scenario")
```

**Test Coverage Summary:**
- **Network setup and data structures**: Scenario assignment, probability validation, data indexing
- **Component compatibility**: Standard PyPSA functions work with stochastic networks  
- **Model formulation**: Variable/constraint dimensions match two-stage formulation
- **Optimization robustness**: Convergence for simple and complex network topologies
- **Solution accuracy**: Numerical validation against known benchmarks
- **Feature integration**: Multi-period planning, statistics, visualization compatibility
- **Edge cases**: Single-scenario equivalence to deterministic optimization

## Computational Performance

### Problem Complexity Analysis

The computational complexity of stochastic optimization in PyPSA exhibits distinct scaling characteristics that require careful consideration for large-scale applications.

**Problem Scaling Properties**

- **Investment variables**: Scale as O(|components|)
- **Operational variables**: Scale as O(|scenarios| × |components| × |snapshots|)
- **Constraint matrix structure**: The two-stage stochastic program produces a block-angular matrix. Its size grows linearly with the number of scenarios.

**Memory consumption** increases with the number of scenarios due to:

- Linear growth in the number of variables and constraints
- Solver working memory requirements that may scale super-linearly, depending on the solution algorithm
- Internal solver data structures (e.g. factorization fill-in, basis management) that can exhibit non-linear scaling behavior


### Advanced Decomposition Methods

**PyPSA-SMS++ Integration:**
The [Resilient project](https://resilient-project.github.io/) is developing PyPSA-SMS++, a specialized API that will enable **advanced decomposition algorithms** integrated into PyPSA workflow. The goal is to support large-scale stochastic optimization problems directly within the PyPSA framework.

## Practical Example

For a comprehensive, hands-on demonstration of stochastic optimization in PyPSA, see the example notebook:

**[Stochastic Optimization Example](../examples/stochastic-optimization.ipynb)**

## References

[1] Birge, J. R., & Louveaux, F. (2011). *Introduction to Stochastic Programming* (2nd ed.). Springer Science & Business Media.
