<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Seasonal Line Ratings

Transmission lines have different thermal capacities in summer and winter, because conductor ampacity is cooling-limited. Grid operators usually publish per-line summer / winter ratings, often derived from IEEE conductor models.

The [`Lines.apply_seasonal_rating`][pypsa.components._types.lines.Lines.apply_seasonal_rating] method broadcasts a per-line `(summer, winter)` scaling to `n.lines_t.s_max_pu` to properly reflect on the two seasonal operation modes.

As illustrated below, use the function in an already initialized network with date-time indexed snapshots.

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
n.c.lines.apply_seasonal_rating(ratings)

# s_nom is left unchanged; s_max_pu carries rating / s_nom per season.
float(n.lines.at["a-b", "s_nom"])         # 1000.0 (unchanged)
n.lines_t.s_max_pu["a-b"].iloc[3000]      # 0.8 (mid-summer hour: 800 / 1000)
```

Each snapshot's `s_max_pu` is `rating / s_nom` for the matching season: `800 / 1000 = 0.8` on summer hours and `1000 / 1000 = 1.0` elsewhere. A rating above `s_nom` yields `s_max_pu > 1`.

## Composing with an N-1 margin

By default the method multiplies the seasonal scaling into any pre-existing `s_max_pu` so N-1 margins survive:

```python
n.lines.at["a-b", "s_max_pu"] = 0.9   # static N-1 margin
n.c.lines.apply_seasonal_rating(ratings, compose=True)

# Summer hour:  0.9 * 0.8 = 0.72
# Winter hour:  0.9 * 1.0 = 0.9
```

Pre-existing dynamic series in `n.lines_t.s_max_pu` compose element-wise. Pass `compose=False` to overwrite instead.

## Southern hemisphere

The default `summer_months=(4, 5, 6, 7, 8, 9)` is northern-hemisphere. Override for SH networks:

```python
n.c.lines.apply_seasonal_rating(
    ratings, summer_months=(10, 11, 12, 1, 2, 3),
)
```

## Convention: summer is the lower rating

Each column is applied to its own season, so a data set that flips the convention still works. The published-TSO convention is **summer = lower rating** (warmer ambient air, less cooling, lower ampacity). Stick to it unless you have a specific reason.

## Raises

- `TypeError` if `n.snapshots` is not a `DatetimeIndex`.
- `KeyError` if `ratings` references a line not in `n.lines.index`, or is missing the `summer` / `winter` columns.
- `ValueError` if any rating is non-positive, `summer_months` is empty, or a rated line has a non-positive `s_nom`.
