# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Links components module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components
from pypsa.constants import RE_PORTS_GE_2

if TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr

logger = logging.getLogger(__name__)


@patch_add_docstring
class Links(Components):
    """Links components class.

    This class is used for link components. All functionality specific to
    links is implemented here. Functionality for all components is implemented in
    the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.links
    'Link' Components
    -----------------
    Attached to PyPSA Network 'AC-DC-Meshed'
    Components: 4

    """

    _operational_variables = ["p"]

    def get_bounds_pu(
        self,
        attr: str = "p",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for links.

        <!-- md:badge-version v1.0.0 -->

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "p"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = f"Bounds can only be retrieved for operational attributes. For links those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        return self.da.p_min_pu, self.da.p_max_pu

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str = "",
        overwrite: bool = False,
        return_names: bool | None = None,
        **kwargs: Any,
    ) -> pd.Index | None:
        """Wrap Components.add() and docstring is patched via decorator."""
        return super().add(
            name=name,
            suffix=suffix,
            overwrite=overwrite,
            return_names=return_names,
            **kwargs,
        )

    @property
    def additional_ports(self) -> list[str]:
        """Identify additional link ports (bus connections) beyond predefined ones.

        Returns
        -------
        list of strings
            List of additional link ports. E.g. ["2", "3"] for bus2, bus3.

        Also see
        ---------
        pypsa.Components.ports

        Examples
        --------
        >>> n = pypsa.Network() # doctest: +SKIP
        >>> n.add("Link", "link1", bus0="bus1", bus1="bus2", bus2="bus3") # doctest: +SKIP
        Index(['link1'], dtype='object')
        >>> n.components.links.additional_ports # doctest: +SKIP
        ['2']

        """
        return [
            match.group(1)
            for col in self.static.columns
            if (match := RE_PORTS_GE_2.search(col))
        ]

    @staticmethod
    def _delay_positions(
        weights: np.ndarray, delay: float, is_cyclic: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute source positions from a numpy weight array."""
        n = len(weights)

        #  cumulative start time of snapshot i
        tau = np.concatenate(([0.0], np.cumsum(weights[:-1])))
        src_time = tau - delay

        if is_cyclic:
            total = float(weights.sum())
            if total <= 0:
                msg = (
                    "Cyclic weighted link delay requires a positive total "
                    "snapshot weighting over the optimized snapshots."
                )
                raise ValueError(msg)
            src_time = np.mod(src_time, total)
            valid = np.ones(n, dtype=bool)
        else:
            valid = src_time >= 0

        src = np.searchsorted(tau, src_time, side="right") - 1
        src = np.clip(src, 0, n - 1).astype(int)

        # Warn if delay doesn't align with snapshot boundaries (sub/super-snapshot)
        if valid.any() and np.any(tau[src[valid]] != src_time[valid]):
            logger.warning(
                "Link delay %g does not align exactly with snapshot weighting "
                "boundaries and will be rounded to the nearest snapshot "
                "boundary.",
                delay,
            )

        return src, valid

    @staticmethod
    def get_delay_source_indexer(
        snapshots: pd.Index,
        weightings: pd.Series,
        delay: int,
        is_cyclic: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get per-snapshot source positions for link delays.

        Delay is interpreted in elapsed time units. For each target snapshot
        `t`, the source is the latest snapshot `s` such that
        `tau[s] <= tau[t] - delay`, where `tau` are cumulative weighting
        starts. In cyclic mode, source times wrap around the horizon. In
        non-cyclic mode, targets without a valid source are marked invalid.

        For multi-investment period snapshots (MultiIndex), delays are applied
        independently per period since investment periods are not temporally
        adjacent.

        Parameters
        ----------
        snapshots : pd.Index
            Snapshot index of the optimization horizon.
        weightings : pd.Series
            Generator snapshot weightings (`n.snapshot_weightings.generators`),
            defining the duration of each snapshot in time units.
        delay : int
            Delivery delay in the same time units as `weightings`.
        is_cyclic : bool
            If True, delayed times wrap around the horizon. If False,
            snapshots without a valid source are marked invalid.

        Returns
        -------
        src_positions : np.ndarray of int
            Index position of the source snapshot for each target snapshot.
        valid : np.ndarray of bool
            False for target snapshots that have no valid source
            (non-cyclic mode only).

        Examples
        --------
        >>> snapshots = pd.RangeIndex(6)
        >>> weightings = pd.Series(1.0, index=snapshots)
        >>> src, valid = Links.get_delay_source_indexer(snapshots, weightings, delay=2, is_cyclic=True)
        >>> src
        array([4, 5, 0, 1, 2, 3])
        >>> valid
        array([ True,  True,  True,  True,  True,  True])
        >>> src, valid = Links.get_delay_source_indexer(snapshots, weightings, delay=2, is_cyclic=False)
        >>> src
        array([0, 0, 0, 1, 2, 3])
        >>> valid
        array([False, False,  True,  True,  True,  True])

        """
        n_snapshots = len(snapshots)
        if n_snapshots == 0:
            return np.array([], dtype=int), np.array([], dtype=bool)
        if delay <= 0:
            return np.arange(n_snapshots, dtype=int), np.ones(n_snapshots, dtype=bool)

        # For multi-investment periods, apply delay per period independently
        if isinstance(snapshots, pd.MultiIndex):
            src = np.empty(n_snapshots, dtype=int)
            valid = np.empty(n_snapshots, dtype=bool)
            offset = 0
            for period in snapshots.unique("period"):
                w = weightings[snapshots.get_loc(period)].to_numpy().astype(float)
                p_src, p_valid = Links._delay_positions(w, delay, is_cyclic)
                src[offset : offset + len(w)] = p_src + offset
                valid[offset : offset + len(w)] = p_valid
                offset += len(w)
            return src, valid

        w = weightings.reindex(snapshots).astype(float).to_numpy()
        return Links._delay_positions(w, delay, is_cyclic)
