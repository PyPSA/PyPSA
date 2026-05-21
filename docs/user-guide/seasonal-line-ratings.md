<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Seasonal Line Ratings

Transmission lines have different thermal capacities in summer and winter, because conductor ampacity is cooling-limited. TenneT, RTE, National Grid, and CAISO all publish per-line summer / winter ratings derived from IEEE 738 / IEC 61597 conductor models.

The helper [`apply_seasonal_line_ratings`][pypsa.apply_seasonal_line_ratings] broadcasts a per-line `(summer, winter)` rating table onto `n.lines.s_nom` and `n.lines_t.s_max_pu` so the seasonal envelope is visible to the optimiser.

## When to use it

- A TSO publishes per-line summer / winter MVA ratings and you want them in the dispatch problem.
- You already model an N-1 margin via `n.lines.s_max_pu` and want to compose a seasonal factor on top of it.
- You want a small, dependency-free utility, not a JAO loader or DLR engine.

## Minimal example

```python
import pandas as pd
import pypsa

n = pypsa.Network()
n.set_snapshots(pd.date_range("2025-01-01", periods=8760, freq="h"))
n.add("Bus", ["a", "b"])
n.add("Line", "a-b", bus0="a", bus1="b", x=0.1, s_nom=1000)

ratings = pd.DataFrame(
    {"summer": [800], "winter": [1000]},
    index=["a-b"],
)
pypsa.apply_seasonal_line_ratings(n, ratings)

# s_nom is set to the winter envelope; summer hours get a 0.8 factor.
float(n.lines.at["a-b", "s_nom"])         # 1000.0
n.lines_t.s_max_pu["a-b"].iloc[3000]      # 0.8 (mid-summer hour)
```

`s_nom` is set to `max(summer, winter)`. The per-snapshot `s_max_pu` carries `min(summer, winter) / max(summer, winter)` on summer hours and `1.0` elsewhere.

## Composing with an N-1 margin

By default the helper multiplies the seasonal factor into any pre-existing `s_max_pu` so N-1 margins survive:

```python
n.lines.at["a-b", "s_max_pu"] = 0.9   # static N-1 margin
pypsa.apply_seasonal_line_ratings(n, ratings, compose=True)

# Summer hour:  0.9 * 0.8 = 0.72
# Winter hour:  0.9 * 1.0 = 0.9
```

Pre-existing dynamic series in `n.lines_t.s_max_pu` compose element-wise. Pass `compose=False` to overwrite instead.

## Southern hemisphere

The default `summer_months=(4, 5, 6, 7, 8, 9)` is northern-hemisphere. Override for SH networks:

```python
pypsa.apply_seasonal_line_ratings(
    n, ratings, summer_months=(10, 11, 12, 1, 2, 3),
)
```

## Convention: summer is the lower rating

The helper treats `min(summer, winter)` as the derating factor regardless of which column is smaller, so data that flips the convention still works. But the published-TSO convention is **summer = lower rating** (warmer ambient air, lower cooling, lower ampacity). Stick to it unless you have a specific reason.

## Out of scope

The helper is intentionally policy-light:

- It does not load JAO / TenneT / RTE feeds; the caller assembles `ratings`.
- It does not detect grid upgrades or replace ratings when a line is reinforced.
- It does not implement dynamic line rating (DLR) from weather time series.

These belong upstream of the helper. Pass in whatever rating table your data pipeline produces.

## Raises

- `TypeError` if `n.snapshots` is not a `DatetimeIndex`.
- `KeyError` if `ratings` references a line not in `n.lines.index`, or is missing the `summer` / `winter` columns.
- `ValueError` if any rating is non-positive or `summer_months` is empty.
