# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Numerical scaling helper."""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Model, QuadraticExpression
from linopy.constraints import CSRConstraint

logger = logging.getLogger(__name__)

ColumnClass = Literal["energy", "cost", "none"]

# magnitude window every scaled quantity is pulled into, per category
WINDOW: dict[str, tuple[float, float]] = {
    "matrix": (1e-3, 1e6),
    "cost": (1e-2, 1e6),
    "bound": (1e-2, 1e6),
    "rhs": (1e-2, 1e6),
}
WEIGHT: dict[str, float] = {"matrix": 2.0, "cost": 1.0, "bound": 1.0, "rhs": 1.0}
VIOL_WEIGHT = 10.0
EPS_ONE = 1e-3
G_MAX = 40

_DIMENSIONLESS_SUFFIXES = (
    "-status",
    "-start_up",
    "-shut_down",
    "-maintenance",
    "-maintenance_start",
    "-maintenance_status",
    "-n_mod",
)
_COST_COLUMNS = {"CVaR-a", "CVaR-theta", "CVaR", "objective_constant"}
_COST_SUFFIXES = ("-marginal_cost_piecewise", "-capital_cost_piecewise")
_CLASSES: tuple[ColumnClass, ...] = ("energy", "cost", "none")


class ScalingFactors(NamedTuple):
    """Unit factors per column class and linopy row factors per constraint group."""

    energy: float = 1.0
    cost: float = 1.0
    constraint_factors: dict[str, float] = {}  # noqa: RUF012


def _positive(name: str, v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float, np.number)):
        msg = f"scaling factor {name!r} must be numeric, got {v!r}"
        raise TypeError(msg)
    if not np.isfinite(v) or v <= 0:
        msg = f"scaling factor {name!r} must be positive and finite, got {v!r}"
        raise ValueError(msg)
    return float(v)


def resolve_scaling(
    scaling: bool | dict | None,
) -> ScalingFactors | Literal[True] | None:
    """Turn the `scaling` argument into `None` (off), `True` (tune) or a spec."""
    if scaling is False or scaling is None:
        return None
    if scaling is True:
        return True
    if not isinstance(scaling, dict):
        msg = f"scaling must be a bool or dict, got {type(scaling).__name__}"
        raise TypeError(msg)
    spec = ScalingFactors(**scaling)  # TypeError on unknown keys
    if not isinstance(spec.constraint_factors, dict):
        msg = f"scaling key 'constraint_factors' must be a dict, got {spec.constraint_factors!r}"
        raise TypeError(msg)
    return ScalingFactors(
        _positive("energy", spec.energy),
        _positive("cost", spec.cost),
        {k: _positive(k, v) for k, v in spec.constraint_factors.items()},
    )


def classify_columns(m: Model) -> dict[str, ColumnClass]:
    """Classify every variable group as energy, cost or dimensionless."""
    classes: dict[str, ColumnClass] = {}
    for name, var in m.variables.items():
        attrs = var.attrs
        if (
            attrs.get("integer")
            or attrs.get("binary")
            or name.endswith(_DIMENSIONLESS_SUFFIXES)
            or name == "Transformer-phase_shift"
        ):
            classes[name] = "none"
        elif name in _COST_COLUMNS or name.endswith(_COST_SUFFIXES):
            classes[name] = "cost"
        else:
            classes[name] = "energy"
    return classes


def _label_classes(m: Model, classes: dict[str, ColumnClass]) -> np.ndarray:
    """Lookup table from variable label to its index in `_CLASSES`.

    linopy hands every variable group a contiguous label range.
    """
    ranges = {name: var.range for name, var in m.variables.items()}
    out = np.full(max((end for _, end in ranges.values()), default=0), 2, np.int8)
    for name, (start, end) in ranges.items():
        out[start:end] = _CLASSES.index(classes[name])
    return out


Ranges = dict[tuple[str, ColumnClass], tuple[float, float]]


def _group_ranges(m: Model, classes: dict[str, ColumnClass]) -> Ranges:
    """Nonzero |coeff| range `(min, max)` per `(constraint group, column class)`."""
    label_cls = _label_classes(m, classes)
    out: Ranges = {}
    for name, con in m.constraints.items():
        d = con.data
        live = ((d["labels"] != -1) & (d["vars"] != -1) & (d["coeffs"] != 0)).values
        absc = np.abs(d["coeffs"].values)
        cls_of = label_cls[d["vars"].values]
        for i, cls in enumerate(_CLASSES):
            vals = absc[live & (cls_of == i)]
            if vals.size:
                out[(name, cls)] = (float(vals.min()), float(vals.max()))
    return out


def _quantities(
    m: Model, classes: dict[str, ColumnClass], ranges: Ranges
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Collect the numbers the tuner has to keep in its window.

    Per constraint group the extreme coefficients and rhs, per variable group
    the extreme bounds, per column class the extreme objective coefficients.
    Returns their log2 values, a table saying by which ILP exponent (columns
    `energy`, `cost` and one per constraint group) each of them is shifted,
    and the category deciding its window.
    """
    names = list(m.constraints)
    if clash := {"energy", "cost"} & set(names):
        msg = f"constraint group names clash with unit names: {sorted(clash)}"
        raise ValueError(msg)
    rows: list[tuple[float, dict[str, float], str]] = []  # (value, shift, category)

    def add(vals: np.ndarray, shift: dict[str, float], cat: str) -> None:
        """Add the min and max of `vals`, shifted by the exponents in `shift`."""
        if vals.size:
            rows.extend([(vals.min(), shift, cat), (vals.max(), shift, cat)])

    def unit(cls: ColumnClass, sign: float = 1.0) -> dict[str, float]:
        return {} if cls == "none" else {cls: sign}

    for (name, cls), (lo, hi) in ranges.items():
        add(np.array([lo, hi]), {name: 1.0, **unit(cls)}, "matrix")
    for name, con in m.constraints.items():
        d = con.data
        rhs = _magnitudes(d["rhs"].values[d["labels"].values != -1])
        add(rhs, {name: 1.0}, "rhs")
    for name, var in m.variables.items():
        d = var.data
        live = d["labels"].values != -1
        lower, upper = d["lower"].values[live], d["upper"].values[live]
        if np.array_equal(lower, upper):
            continue  # fixed, presolved away, must not steer the ILP
        bounds = _magnitudes(np.concatenate([lower, upper]))
        add(bounds, unit(classes[name], -1.0), "bound")
    obj = m.objective.expression.data
    coeffs, labels = obj["coeffs"].values.ravel(), obj["vars"].values.ravel()
    cls_of = _label_classes(m, classes)[labels]
    for i, cls in enumerate(_CLASSES):
        shift = unit(cls)
        shift["cost"] = shift.get("cost", 0.0) - 1.0
        add(_magnitudes(coeffs[(labels != -1) & (cls_of == i)]), shift, "cost")

    logv = np.log2([v for v, _, _ in rows])
    A = pd.DataFrame([s for _, s, _ in rows], index=pd.RangeIndex(len(rows), name="q"))
    A = A.reindex(columns=pd.Index(["energy", "cost", *names], name="g")).fillna(0.0)
    return logv, A, [c for _, _, c in rows]


def _magnitudes(v: np.ndarray) -> np.ndarray:
    """Finite nonzero |v|."""
    v = np.abs(v)
    return v[np.isfinite(v) & (v != 0)]


def _row_units(ranges: Ranges, names: list[str]) -> list[ColumnClass]:
    """Classify each constraint group by unit.

    Cost wins over energy so that piecewise cost rows follow the objective.
    """
    return [
        next((c for c in ("cost", "energy") if (name, c) in ranges), "none")
        for name in names
    ]


def _unscalable(m: Model) -> str | None:
    """Reason the model cannot be scaled, or None."""
    if any(c.is_indicator for _, c in m.constraints.items()):
        return "model has indicator constraints"
    if any(isinstance(c, CSRConstraint) for _, c in m.constraints.items()):
        # frozen constraints have read-only scaling, the user froze them on purpose
        return "model has frozen constraints"
    if isinstance(m.objective.expression, QuadraticExpression):
        return "model has a quadratic objective"
    return None


def resolve_factors(
    m: Model, spec: ScalingFactors | Literal[True]
) -> ScalingFactors | None:
    """Factors to apply for a resolved `scaling` argument, None if unscalable.

    `True` tunes energy and cost units by the ILP with rows tied to their
    unit. Given factors are used verbatim, rows not listed in
    `constraint_factors` get the inverse of their unit factor.
    """
    names = list(m.constraints)
    if reason := _unscalable(m):
        logger.warning("scaling skipped: %s", reason)
        return None
    if spec is True:
        return choose_factors(m, constraint_factors=False)
    unknown = set(spec.constraint_factors) - set(names)
    if unknown:
        msg = f"unknown constraint group(s) in scaling: {sorted(unknown)}"
        raise ValueError(msg)
    classes = classify_columns(m)
    units = _row_units(_group_ranges(m, classes), names)
    unit_factor = {"energy": spec.energy, "cost": spec.cost, "none": 1.0}
    rows = {
        name: spec.constraint_factors.get(name, 1.0 / unit_factor[u])
        for name, u in zip(names, units, strict=True)
    }
    return ScalingFactors(spec.energy, spec.cost, rows)


def choose_factors(m: Model, constraint_factors: bool) -> ScalingFactors:
    """Pick pow2 factors by an ILP over the model's per-group ranges.

    Minimises the weighted log2 spread per category plus window violations,
    with a small pull of every exponent towards zero. Without
    `constraint_factors` every row exponent follows its columns (cost rows
    `-cost`, energy rows `-energy`, else 0), which is a plain change of
    units. With it each constraint group gets a free exponent on top.
    """
    names = list(m.constraints)
    classes = classify_columns(m)
    ranges = _group_ranges(m, classes)
    logv, A, cats = _quantities(m, classes, ranges)
    if not len(logv):
        return ScalingFactors(1.0, 1.0, dict.fromkeys(names, 1.0))

    q, unknowns = A.index, A.columns
    cat = pd.Index(sorted(set(cats)), name="cat")
    cat_q = xr.DataArray(cats, coords=[q])
    weight_q = xr.DataArray([VIOL_WEIGHT * WEIGHT[c] for c in cats], coords=[q])
    weight_cat = xr.DataArray([WEIGHT[c] for c in cat], coords=[cat])
    wlo, whi = (
        xr.DataArray(v, coords=[q]) for v in np.log2([WINDOW[c] for c in cats]).T
    )

    ilp = Model()
    g = ilp.add_variables(-G_MAX, G_MAX, coords=[unknowns], integer=True)
    t = ilp.add_variables(0, coords=[unknowns])  # |g|
    lo = ilp.add_variables(coords=[cat])
    hi = ilp.add_variables(coords=[cat])
    s_lo = ilp.add_variables(0, coords=[q])
    s_hi = ilp.add_variables(0, coords=[q])

    scaled = (xr.DataArray(A) * g).sum("g") + xr.DataArray(logv, coords=[q])
    ilp.add_constraints(scaled - lo.sel(cat=cat_q) >= 0)
    ilp.add_constraints(scaled - hi.sel(cat=cat_q) <= 0)
    ilp.add_constraints(scaled + s_lo >= wlo)
    ilp.add_constraints(scaled - s_hi <= whi)
    ilp.add_constraints(t - g >= 0)
    ilp.add_constraints(t + g >= 0)
    if not constraint_factors:
        # row exponent + unit exponent = 0 per group: coefficients keep their
        # value, rhs moves with the unit, exactly what rescaling inputs did
        T = pd.DataFrame(0.0, index=pd.Index(names, name="k"), columns=unknowns)
        for name, u in zip(names, _row_units(ranges, names), strict=True):
            T.loc[name, name] = 1.0
            if u != "none":
                T.loc[name, u] = 1.0
        ilp.add_constraints((xr.DataArray(T) * g).sum("g") == 0)
    ilp.add_objective(
        (weight_cat * (hi - lo)).sum()
        + (weight_q * (s_lo + s_hi)).sum()
        + EPS_ONE * t.sum()
    )
    status, condition = ilp.solve("highs", io_api="direct", log_to_console=False)
    if status != "ok":
        msg = f"scaling ILP failed: {condition}"
        raise RuntimeError(msg)
    sol = 2.0 ** np.rint(g.solution.to_pandas())
    return ScalingFactors(
        float(sol["energy"]), float(sol["cost"]), sol[names].astype(float).to_dict()
    )


def apply_factors(m: Model, factors: ScalingFactors) -> None:
    """Set linopy's `scaling` factors on the model.

    A unit factor divides its columns and the objective, row factors are
    applied as given. linopy scales at export and maps results back itself.
    """
    classes = classify_columns(m)
    col = {"energy": 1.0 / factors.energy, "cost": 1.0 / factors.cost, "none": 1.0}
    for name, var in m.variables.items():
        var.scaling = col[classes[name]]
    for name, con in m.constraints.items():
        con.scaling = factors.constraint_factors.get(name, 1.0)
    m.objective.scaling = 1.0 / factors.cost
    rows = list(factors.constraint_factors.values())
    logger.info(
        "scaling: energy %g, cost %g, row factors [%g, %g] over %d groups",
        factors.energy,
        factors.cost,
        min(rows, default=1.0),
        max(rows, default=1.0),
        len(rows),
    )
