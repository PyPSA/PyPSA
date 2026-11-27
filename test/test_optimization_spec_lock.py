# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Lock of the spec keys and of a digest of the model each case builds.

A refactor of the spec build must keep both. Record them again with
`python test/test_optimization_spec_lock.py` where a change is intended.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import scipy.sparse
from linopy import Model
from test_flow_based import triangle
from test_optimization_spec_equivalence import CASES as SOLVED
from test_optimization_spec_equivalence import (
    EXACT_MILP,
    _multi_period,
    _scigrid_de,
    _single_bus,
)

import pypsa
from pypsa.optimization.spec.build import fingerprint, spec, written

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr
    from linopy.constraints import Constraint
    from linopy.variables import Variable

DATA = Path(__file__).parent / "data" / "equivalence"
SPEC_KEYS = DATA / "spec_keys.json"
DIGESTS = DATA / "model_digests.json"
REGENERATE = "python test/test_optimization_spec_lock.py"
VARIANTS = {"default": False, "linearized_unit_commitment": True}


def risk_averse_growth_limit() -> None:
    n = _multi_period(storage=True, scenarios=True)
    n.c.carriers.static.at["gencarrier", "max_growth"] = 100
    n.set_risk_preference(alpha=0.9, omega=0.5)
    n.optimize(multi_investment_periods=True)


def flow_based() -> None:
    triangle(snapshots=2).optimize()


def maintenance_modules_ramps() -> None:
    n = _single_bus(6, [50, 80, 60, 40, 70, 90])
    n.add("Bus", "fuel")
    n.add(
        "Generator",
        "maintained",
        bus="bus",
        p_nom=100,
        marginal_cost=10,
        committable=True,
        maintainable=True,
        maintenance_duration=2,
    )
    n.add(
        "Generator",
        "modular",
        bus="bus",
        p_nom_extendable=True,
        p_nom_mod=20,
        p_nom_max=100,
        capital_cost=5,
        marginal_cost=30,
        e_sum_max=200,
        maintainable=True,
        maintenance_duration=1,
    )
    n.add("Generator", "fuel", bus="fuel", p_nom=200, p_set=40)
    n.add(
        "Link",
        "turbine",
        bus0="fuel",
        bus1="bus",
        p_nom=100,
        efficiency={0.0: 0.0, 0.5: 0.4, 1.0: 0.5},
        ramp_limit_up=0.3,
        ramp_limit_down=0.3,
    )
    n.add("Store", "tank", bus="fuel", e_nom=50, e_cyclic=True)
    n.optimize(solver_options=EXACT_MILP)


def committable_process() -> None:
    n = _single_bus(4, [50, 70, 0, 60])
    n.add("Bus", "h2")
    n.add("Generator", "h2", bus="h2", p_nom=200, marginal_cost=10)
    n.add(
        "Process",
        "fuel_cell",
        bus0="h2",
        bus1="bus",
        rate0=-1,
        rate1=0.6,
        p_nom=150,
        committable=True,
        p_min_pu=0.2,
        start_up_cost=10,
        min_up_time=2,
        ramp_limit_up=0.5,
    )
    n.optimize(solver_options=EXACT_MILP)


def scigrid_de_expansion() -> None:
    n = _scigrid_de()
    lines, transformers = n.c.lines.static, n.c.transformers.static
    lines.loc[lines.index[::2], "s_nom_extendable"] = True
    lines["v_ang_max"] = 30.0
    transformers.loc[transformers.index[::2], "s_nom_extendable"] = True
    transformers["v_ang_max"] = 30.0
    transformers.loc[transformers.index[:2], "phase_shift_min"] = -30.0
    transformers.loc[transformers.index[:2], "phase_shift_max"] = 30.0
    outages = lines.index[:2]
    n.optimize.optimize_security_constrained(n.snapshots[0], branch_outages=outages)


CASES: dict[str, Callable[[], Any]] = {
    **{
        name: SOLVED[name]
        for name in [
            "storage_hvdc_operational_limit",
            "scigrid_de_transmission_losses",
            "model_energy",
            "multi_period_expansion_limits",
            "multi_period_storage_primary_energy",
            "unit_commitment",
            "unit_commitment_link",
            "unit_commitment_extendable",
            "unit_commitment_linearized",
            "piecewise_costs",
            "multiport_process",
        ]
    },
    **{
        case.__name__: case
        for case in [
            risk_averse_growth_limit,
            flow_based,
            maintenance_modules_ramps,
            committable_process,
            scigrid_de_expansion,
        ]
    },
}


class _Built(Exception):  # noqa: N818
    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model


def _stop(model: Model, *args: Any, **kwargs: Any) -> None:
    raise _Built(model)


def build(case: str) -> Model:
    """Return the model a case builds, stopped before the solve."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Model, "solve", _stop)
        try:
            CASES[case]()
        except _Built as built:
            return built.model
    msg = f"The case {case!r} solved no model."
    raise AssertionError(msg)


def _hash(entry: Any) -> str:
    if isinstance(entry, dict):
        entry = {k: v for k, v in entry.items() if k != "description"}
    text = json.dumps(entry, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def spec_keys(linearized_unit_commitment: bool) -> dict[str, dict[str, str]]:
    """Return the fingerprint of each key of the spec, by kind."""
    keys: dict[str, dict[str, str]] = {}
    for key, entry in written(spec(linearized_unit_commitment).to_dict()).items():
        kind, _, name = key.partition(".")
        hashed = _hash(entry) if linearized_unit_commitment else fingerprint(key)
        keys.setdefault(kind, {})[name or kind] = hashed
    return {kind: dict(sorted(named.items())) for kind, named in sorted(keys.items())}


def _number(value: float) -> float:
    return float(f"{value:.10g}")


def _sum(values: np.ndarray) -> float:
    return _number(math.fsum(values[np.isfinite(values)].tolist()))


def _terms(
    rows: np.ndarray, columns: np.ndarray, coeffs: np.ndarray
) -> tuple[int, float]:
    held = columns != -1
    matrix = scipy.sparse.coo_array((coeffs[held], (rows[held], columns[held])))
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    return matrix.nnz, _sum(np.abs(matrix.data))


def _on(array: xr.DataArray, labels: xr.DataArray) -> np.ndarray:
    return array.broadcast_like(labels).transpose(*labels.dims).values


def _bounds(side: str, values: np.ndarray) -> dict[str, Any]:
    return {f"finite_{side}": int(np.isfinite(values).sum()), side: _sum(values)}


def _variable(var: Variable) -> dict[str, Any]:
    labels = var.labels
    built = labels.values != -1
    entries = int(built.sum())
    return {
        "dims": list(labels.dims),
        "entries": entries,
        "binary": entries if var.attrs["binary"] else 0,
        "integer": entries if var.attrs["integer"] else 0,
        **_bounds("lower", _on(var.lower, labels)[built]),
        **_bounds("upper", _on(var.upper, labels)[built]),
        "scaling": _sum(_on(var.data.scaling, labels)[built]),
    }


def _constraint(con: Constraint) -> dict[str, Any]:
    labels = con.labels
    built = (labels.values != -1).ravel()
    dims = (*labels.dims, "_term")
    columns = con.vars.transpose(*dims).values.reshape(built.size, -1)[built]
    coeffs = con.coeffs.transpose(*dims).values.reshape(built.size, -1)[built]
    rows = np.repeat(np.arange(len(columns)), columns.shape[1])
    nonzeros, magnitude = _terms(rows, columns.ravel(), coeffs.ravel())
    signs = Counter(_on(con.sign, labels).ravel()[built].tolist())
    return {
        "dims": list(labels.dims),
        "rows": int(built.sum()),
        "nonzeros": nonzeros,
        "abs_coeffs": magnitude,
        "rhs": _sum(_on(con.rhs, labels).ravel()[built]),
        "signs": dict(sorted(signs.items())),
        "scaling": _sum(_on(con.data.scaling, labels).ravel()[built]),
    }


def _objective(m: Model) -> dict[str, Any]:
    expression = m.objective.expression
    assert not m.objective.is_quadratic, "The digest reads linear objectives only."
    columns = expression.vars.values.ravel()
    coeffs = expression.coeffs.values.ravel()
    terms, magnitude = _terms(np.zeros_like(columns), columns, coeffs)
    return {"terms": terms, "abs_coeffs": magnitude, "sense": m.objective.sense}


def digest(m: Model) -> dict[str, Any]:
    """Return counts and sums of the model that hold whatever order it is built in."""
    return {
        "variables": {k: _variable(m.variables[k]) for k in sorted(m.variables)},
        "constraints": {
            k: _constraint(m.constraints[k]) for k in sorted(m.constraints)
        },
        "objective": _objective(m),
    }


def changes(expected: Any, found: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Return a line for each leaf that is added, removed or changed from *expected* to *found*."""
    if isinstance(expected, dict) and isinstance(found, dict):
        keys = sorted(expected.keys() | found.keys())
        return [
            c for k in keys for c in changes(expected.get(k), found.get(k), (*path, k))
        ]
    where = " > ".join(path)
    if expected == found:
        return []
    if expected is None:
        return [f"added {where}"]
    if found is None:
        return [f"removed {where}"]
    return [f"changed {where}: {expected} (lock) -> {found} (now)"]


def _assert_unchanged(expected: Any, found: Any, root: str) -> None:
    if lines := changes(expected, found, (root,)):
        hint = f"Record the files again with `{REGENERATE}` if the change is intended."
        pytest.fail("\n".join([*lines, hint]), pytrace=False)


@pytest.mark.parametrize("variant", VARIANTS)
def test_spec_keys_match_lock(variant: str) -> None:
    expected = json.loads(SPEC_KEYS.read_text())[variant]
    _assert_unchanged(expected, spec_keys(VARIANTS[variant]), variant)


@pytest.mark.parametrize("case", CASES)
def test_model_digest_matches_lock(case: str) -> None:
    expected = json.loads(DIGESTS.read_text()).get(case)
    _assert_unchanged(expected, digest(build(case)), case)


def _dumps(data: Any, depth: int, indent: str = "") -> str:
    if depth == 0 or not isinstance(data, dict):
        return json.dumps(data)
    inner = indent + "  "
    items = [
        f"{inner}{json.dumps(k)}: {_dumps(v, depth - 1, inner)}"
        for k, v in data.items()
    ]
    return "{\n" + ",\n".join(items) + f"\n{indent}}}"


if __name__ == "__main__":
    pypsa.options.debug.runtime_verification = True
    for name in pypsa.options.future._children:
        if pypsa.options.get_option(f"future.{name}") is None:
            pypsa.options.set_option(f"future.{name}", False)
    keys = {variant: spec_keys(v) for variant, v in VARIANTS.items()}
    digests = {case: digest(build(case)) for case in CASES}
    SPEC_KEYS.write_text(_dumps(keys, 3) + "\n")
    DIGESTS.write_text(_dumps(digests, 3) + "\n")
