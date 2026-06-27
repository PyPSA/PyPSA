# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Numerical scaling helper."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pypsa import Network

# - `energy` (MW / MWh), default 1e3
# - `cost` (€), default 1e3
# - `emissions` (tCO2), default 1e6
#
# (energy, cost, emissions)
_NOM = re.compile(r"(?!v_nom)(\w+_nom(_min|_max|_set|_mod)?|nom_(min|max)_\w+)")
_COLUMN_EXPONENTS = {
    # power flows (MW), energy stocks (MWh), capacities (*_nom, MW/MWh)
    "p_set": (-1, 0, 0),
    "q_set": (-1, 0, 0),
    "inflow": (-1, 0, 0),
    "p_dispatch_set": (-1, 0, 0),
    "p_store_set": (-1, 0, 0),
    "e_initial": (-1, 0, 0),
    "e_set": (-1, 0, 0),
    "state_of_charge_initial": (-1, 0, 0),
    "state_of_charge_set": (-1, 0, 0),
    "max_growth": (-1, 0, 0),
    "e_sum_min": (-1, 0, 0),
    "e_sum_max": (-1, 0, 0),
    _NOM: (-1, 0, 0),
    # cost per quantity (€/MW, €/MWh)
    "marginal_cost": (1, -1, 0),
    "spill_cost": (1, -1, 0),
    "marginal_cost_storage": (1, -1, 0),
    "capital_cost": (1, -1, 0),
    # cost on a dimensionless binaries
    "start_up_cost": (0, -1, 0),
    "shut_down_cost": (0, -1, 0),
    "stand_by_cost": (0, -1, 0),
    # quadratic cost
    "marginal_cost_quadratic": (2, -1, 0),
    # carrier emissions (tCO2/MWh)
    "co2_emissions": (1, 0, -1),
}


class Scaler(NamedTuple):
    """Energy, cost and emissions base-unit factors plus unscaling rules."""

    energy: float
    cost: float
    emissions: float

    @classmethod
    def resolve(cls, scaling: bool | dict | None) -> Scaler | None:
        """Resolve the `scaling` argument into a `Scaler` or `None`."""
        if scaling is False or scaling is None:
            return None
        defaults = {
            "energy": 1e3,
            "cost": 1e3,
            "emissions": 1e6,
        }
        if scaling is True:
            return cls(**defaults)
        if not isinstance(scaling, dict):
            msg = f"scaling must be a bool or dict, got {type(scaling).__name__}"
            raise TypeError(msg)
        unknown = set(scaling) - set(defaults)
        if unknown:
            msg = (
                f"unknown scaling factor(s) {sorted(unknown)}; "
                f"valid keys are {sorted(defaults)}"
            )
            raise ValueError(msg)
        for k, v in scaling.items():
            if not isinstance(v, (int, float)):
                msg = f"scaling factor {k!r} must be numeric, got {v!r}"
                raise TypeError(msg)
        return cls(**{k: float(scaling.get(k, d)) for k, d in defaults.items()})

    def _factor(self, exponents: tuple[int, int, int]) -> float:
        """Product of the base units raised to the given exponents."""
        e, c, m = exponents
        return self.energy**e * self.cost**c * self.emissions**m

    def _column_factor(self, col: str) -> float | None:
        """Multiplicative factor for an input column, or None to leave unchanged."""
        exponents = _COLUMN_EXPONENTS.get(col) or (
            _COLUMN_EXPONENTS[_NOM] if _NOM.fullmatch(col) else None
        )
        return self._factor(exponents) if exponents else None

    def variable_factor(self, attr: str) -> float:
        """Factor that converts a scaled solution variable back to original units."""
        # Dimensionless variables (commitment status, module counts) stay unscaled.
        dimensionless = {"status", "start_up", "shut_down", "n_mod"}
        return 1.0 if attr in dimensionless else self.energy

    def dual_factor(self, prefix: str, suffix: str, n: Network) -> float:
        """Output factor for a constraint dual."""
        if prefix == "GlobalConstraint":
            gc = n.global_constraints
            if suffix in gc.index:
                gc_type = gc.at[suffix, "type"]
                if gc_type == "transmission_expansion_cost_limit":
                    return 1.0
                if gc_type == "primary_energy":
                    return self.cost / self.emissions
        return self.cost / self.energy

    @contextmanager
    def applied(self, n: Network) -> Iterator[None]:
        """Scale `n`'s inputs in place for the block, restoring exact originals on exit."""
        static, dynamic, constant = {}, {}, None
        for c in n.components:
            for col in c.static.columns:
                f = self._column_factor(col)
                if f is not None:
                    static[c.name, col] = c.static[col].copy()
                    c.static[col] = c.static[col] * f
            for col, df in c.dynamic.items():
                f = self._column_factor(col)
                if f is not None and df.shape[1]:
                    dynamic[c.name, col] = df.copy()
                    c.dynamic[col] = df * f

        # Scale each GlobalConstraint RHS budget by its unit (as in dual_factor)
        gc = n.global_constraints
        if len(gc):
            constant = gc["constant"].copy()
            is_cost = gc["type"] == "transmission_expansion_cost_limit"
            is_em = gc["type"] == "primary_energy"
            gc.loc[is_cost, "constant"] /= self.cost
            gc.loc[is_em, "constant"] /= self.emissions
            gc.loc[~(is_cost | is_em), "constant"] /= self.energy

        # Expose the active factors so raw constants can be scaled (e.g. p_init)
        n._scaling = self._asdict()
        try:
            yield
        finally:
            n._scaling = {"energy": 1.0, "cost": 1.0, "emissions": 1.0}
            for (cname, col), original in static.items():
                n.components[cname].static[col] = original
            for (cname, col), original in dynamic.items():
                n.components[cname].dynamic[col] = original
            if constant is not None:
                n.global_constraints["constant"] = constant
