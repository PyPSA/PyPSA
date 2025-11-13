<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Weightings

A practical guide to understanding and using weightings in PyPSA optimization.

## What Are Weightings?

Weightings are scaling factors that control how time is represented in your optimization model. They answer two key questions:

1. **How much does each snapshot contribute to total costs/emissions?**
2. **How are costs/emissions scaled across investment periods?**

## Quick Reference

| Weighting | What It Does | Typical Value |
|-----------|--------------|---------------|
| `snapshot_weightings.objective` | Scales operational costs in objective function | 1.0 for hourly data |
| `snapshot_weightings.stores` | Hours elapsed for storage energy balance | 1.0 for hourly snapshots |
| `snapshot_weightings.generators` | Scales emissions/energy for global constraints | 1.0 for hourly data |
| `investment_period_weightings.objective` | NPV discount factor for each period | Sum of discount factors |
| `investment_period_weightings.years` | Period duration for emission budgets | 10 for 2030-2039 |

## Snapshot Weightings

Accessed via `n.snapshot_weightings`, a DataFrame with three columns:

### 1. Objective (`snapshot_weightings.objective`)

**Used for:** Operational costs in the objective function (marginal costs, storage costs, stand-by costs)

**Formula:**
```
# Single investment period
Total OPEX = sum_over_snapshots(
    marginal_cost × dispatch × snapshot_weightings.objective[snapshot]
)

# Multi-investment periods
Total OPEX = sum_over_periods(
    sum_over_snapshots(
        marginal_cost × dispatch × snapshot_weightings.objective
    ) × investment_period_weightings.objective[period]
)
```

**When to scale:**
- Full year (8760 hours): Use `1.0` for each hourly snapshot
- Representative days: Scale so total weighted hours = 8760
- Multiple years per investment period: Scale to represent one year. Sum of weights across investment period = 8760
- Rule of thumb is sum of weights for a year (non-multi investment period model) or investment period should always = 8760

### 2. Stores (`snapshot_weightings.stores`)

**Used for:** Storage energy balance (hours elapsed between snapshots)

**Formula:**
```
energy[t] = efficiency^hours × energy[t-1] - hours × power[t]
where hours = snapshot_weightings.stores
```

**Critical rule:** This represents **physical time**, not economic weighting!

**When to scale:**
- Hourly snapshots: Always `1.0` (1 hour elapsed)
- 3-hourly snapshots: Use `3.0` (3 hours elapsed)
- Never scale for representative periods (still 1.0 for hourly data)

### 3. Generators (`snapshot_weightings.generators`)

**Used for:** Global constraints (emissions, transmission limits, primary energy)

**Formula:**
```
# Single investment period
Total emissions = sum_over_snapshots(
    dispatch × emission_rate × snapshot_weightings.generators[snapshot]
)

# Multi-investment periods
Total emissions = sum_over_periods(
    sum_over_snapshots(
        dispatch × emission_rate × snapshot_weightings.generators[snapshot]
    ) × investment_period_weightings.years[period]
)
```

**When to scale:**
- Full year (8760 hours): Use `1.0` for each hourly snapshot
- Representative days: Scale so total weighted hours = 8760
- Multiple years per investment period: Scale to represent one year. Sum of weights across investment period = 8760
- Rule of thumb is sum of weights for a year (non-multi investment period model) or investment period should always = 8760

## Investment Period Weightings

For multi-investment period optimization. Accessed via `n.investment_period_weightings`.

### 1. Objective (`investment_period_weightings.objective`)

**Used for:** Discounting costs (both CAPEX and OPEX) to Net Present Value

**How it's applied:**
```
# Investment costs (CAPEX) - snapshot weightings NOT used
Total CAPEX = sum_over_periods(
    capital_cost × capacity × investment_period_weightings.objective[period]
)

# Multi-investment periods
Total OPEX = sum_over_periods(
    sum_over_snapshots(
        marginal_cost × dispatch × snapshot_weightings.objective[snapshot]
    ) × investment_period_weightings.objective[period]
)
```

**How to calculate:**

**Note:** By suming the discount factor across the period years we are effectively weighting by period length.

```python
# Setup investment periods
investment_periods = [2030, 2040, 2050]
discount_rate = 0.05
period_duration = 10  # Each period represents 10 years
model_start_year = 2030  # Base year for discounting

# Calculate and assign NPV factors for each period
for period in investment_periods:
    # Calculate years from model start
    years_from_start = period - model_start_year  # 0 for 2030, 10 for 2040, 20 for 2050

    # Discount factors for each year in the period
    discounts = [1 / (1 + discount_rate)**t
                 for t in range(years_from_start, years_from_start + period_duration)]

    # Sum of discount factors = NPV weighting for this period
    npv_factor = sum(discounts)

    # Assign as the investment period objective weighting
    n.investment_period_weightings.loc[period, 'objective'] = npv_factor

# Results:
# Period 2030 (years 0-9):  objective = 8.11
# Period 2040 (years 10-19): objective = 4.98
# Period 2050 (years 20-29): objective = 3.06
```

### 2. Years (`investment_period_weightings.years`)

**Used for:** Scaling time-dependent global constraints (emission budgets)

**How it's applied:**
```
Total emissions = sum_over_periods(
    sum_over_snapshots(
        dispatch × emission_rate × snapshot_weightings.generators[snapshot]
    ) × investment_period_weightings.years[period]
)
```

**What it represents:** Duration of the investment period in years.

## Common Use Cases

### Case 1: Full Year Hourly Simulation (Single Period)

```python
# 8760 hourly snapshots
n.set_snapshots(pd.date_range("2030-01-01", freq="h", periods=8760))

# Default weightings are perfect
n.snapshot_weightings.objective     # All 1.0
n.snapshot_weightings.stores        # All 1.0
n.snapshot_weightings.generators    # All 1.0

# Verify: sum = 8760 hours ✓
```

### Case 2: Representative Days (Single Period)

```python
# 4 representative days (one per season)
n.set_snapshots(pd.date_range("2030-01-01", freq="h", periods=96))

# Each day represents 365/4 = 91.25 days
days_per_rep_day = 365 / 4

n.snapshot_weightings.objective = days_per_rep_day   # Scale costs
n.snapshot_weightings.stores = 1.0                    # Physical time!
n.snapshot_weightings.generators = days_per_rep_day  # Scale emissions

# Verify: sum = 96 × 91.25 = 8760 hours ✓
```

**Key:** Stores weighting stays 1.0 because each snapshot is still 1 hour apart physically.

### Case 3: Multi-Investment Period (Standard)

```python
# 3 periods, each with 1 year of representative data
periods = [2030, 2040, 2050]

# Create multi-index snapshots
for period in periods:
    year_snapshots = pd.date_range(f"{period}-01-01", freq="h", periods=8760)
    # ... add to MultiIndex

n.set_snapshots(snapshots)
n.investment_periods = periods

# Snapshot weightings: represent one year
n.snapshot_weightings.objective = 1.0
n.snapshot_weightings.stores = 1.0
n.snapshot_weightings.generators = 1.0

# Investment period weightings
discount_rate = 0.05
period_duration = 10  # Each period represents 10 years
model_start_year = 2030  # Base year for discounting

for period in periods:
    # Calculate years from model start
    years_from_start = period - model_start_year  # 0 for 2030, 10 for 2040, 20 for 2050

    # NPV factor: sum of discount factors
    discounts = [1/(1+discount_rate)**(years_from_start+t) for t in range(period_duration)]
    n.investment_period_weightings.loc[period, 'objective'] = sum(discounts)
    n.investment_period_weightings.loc[period, 'years'] = period_duration

# Period 2030: objective = 8.11, years = 10
# Period 2040: objective = 4.98, years = 10
# Period 2050: objective = 3.06, years = 10
```

### Case 4: Multi-Year Data Per Investment Period

**Goal:** Include 3 years of weather data per period to capture variability, while keeping only 2-3 investment decision points.

```python
# Setup: 2 periods, each with 3 years of data
periods = [2030, 2040]
years_per_period = 3
period_duration = 10  # Each period represents 10 actual years

# Create snapshots
for period in periods:
    for year_offset in [0, 1, 2]:  # 2030, 2031, 2032 for period 2030
        year = period + year_offset
        year_snapshots = pd.date_range(f"{year}-01-01", freq="h", periods=8760)
        # ... add to MultiIndex (period, timestamp)

# STEP 1: Scale snapshot weightings to ONE YEAR
annual_hours = 8760
snapshots_per_period = years_per_period * 8760
scaling_factor = annual_hours / snapshots_per_period

n.snapshot_weightings.objective = scaling_factor     # ~0.33
n.snapshot_weightings.stores = 1.0                   # Physical time!
n.snapshot_weightings.generators = scaling_factor    # ~0.33 (SAME!)

# Verify: 0.33 × 26,280 snapshots = 8,760 hours = 1 year ✓

# STEP 2: Set NPV factors (sum of discount factors)
discount_rate = 0.05
model_start_year = 2030  # Base year for discounting

for period in periods:
    # Calculate years from model start
    years_from_start = period - model_start_year  # 0 for 2030, 10 for 2040

    # NPV factor: sum of discount factors
    discounts = [1/(1+discount_rate)**(years_from_start+t) for t in range(period_duration)]
    n.investment_period_weightings.loc[period, 'objective'] = sum(discounts)

# STEP 3: Set period duration
n.investment_period_weightings.years = period_duration

# Result:
# - Costs: annual_cost × NPV_factor = properly discounted ✓
# - Emissions: annual_emissions × 10 = period_emissions ✓
