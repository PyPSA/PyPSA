# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""MultiPort components module."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pypsa.components.components import Components
from pypsa.constants import RE_PORTS_GE_2

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


logger = logging.getLogger(__name__)


class Multiport(Components):
    """Multiport components class.

    This class is used for shared functionality for components with multiple ports.

    See Also
    --------
    [pypsa.Components][]

    """

    @property
    def additional_ports(self) -> list[str]:
        """Identify additional component ports (bus connections) beyond predefined ones.

        Returns
        -------
        list of strings
            List of additional ports. E.g. ["2", "3"] for bus2, bus3.

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

    @property
    def _output_ports(self) -> list[str]:
        return self.ports

    def _port_suffix(self, port: str) -> str:
        return port

    @property
    @abstractmethod
    def _coefficient_attr(self) -> str: ...

    @property
    def _delay_config(self) -> dict[str, tuple[pd.Series | int, pd.Series | bool]]:
        """Get delay and cyclic_delay configuration for each output port.

        Returns
        -------
        dict
            Mapping of port_suffix to (delays, cyclics) pair. If no delay columns
            exist, returns scalar values (0, True).

        """
        config = {}
        for port in self._output_ports:
            suffix = self._port_suffix(port)
            delay_col = f"delay{suffix}"
            cyclic_col = f"cyclic_delay{suffix}"

            if delay_col in self.static.columns:
                config[suffix] = (self.static[delay_col], self.static[cyclic_col])
            else:
                config[suffix] = (0, True)
        return config

    def _iter_balance_args(
        self, sns: Sequence
    ) -> Iterator[tuple[str, Any, pd.Index, int, bool]]:
        """Iterate over all balance arguments, separating immediate and delayed.

        Yields
        ------
        bus_col : str
            Bus column name (e.g., "bus0", "bus1")
        coeff : xr.DataArray
            Coefficient values for the component
        names : pd.Index
            Component names with this delay configuration
        delay : int
            Delay in time units (0 for immediate, >0 for delayed)
        is_cyclic : bool
            Whether delay wraps around the horizon

        """
        if self.empty:
            return

        active = self.active_assets

        for port in self._output_ports:
            suffix = self._port_suffix(port)
            coeff = self.da[f"{self._coefficient_attr}{suffix}"].sel(snapshot=sns)
            delays, cyclics = self._delay_config[suffix]

            for (d, cyc), group in self.static.assign(
                _delay=delays, _cyclic=cyclics
            ).groupby(["_delay", "_cyclic"]):
                delay_int = int(d)

                names = group.index
                if isinstance(names, pd.MultiIndex):
                    names = names.get_level_values("name").unique()
                names = names.intersection(active)

                if not names.empty:
                    yield (
                        f"bus{port}",
                        coeff.sel(name=names),
                        names,
                        delay_int,
                        bool(cyc),
                    )

    @staticmethod
    def _delay_positions(
        weights: np.ndarray, delay: int, is_cyclic: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute source positions from a numpy weight array."""
        n = len(weights)

        tau = np.concatenate(([0.0], np.cumsum(weights[:-1])))
        src_time = tau - delay

        if is_cyclic:
            total = float(weights.sum())
            if total <= 0:
                msg = (
                    "Cyclic weighted delay requires a positive total "
                    "snapshot weighting over the optimized snapshots."
                )
                raise ValueError(msg)
            src_time = np.mod(src_time, total)
            valid = np.ones(n, dtype=bool)
        else:
            valid = src_time >= 0

        src = np.searchsorted(tau, src_time, side="right") - 1
        src = np.clip(src, 0, n - 1).astype(int)

        if valid.any() and np.any(tau[src[valid]] != src_time[valid]):
            logger.warning(
                "Delay %g does not align exactly with snapshot weighting "
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
        """Get per-snapshot source positions for component delays.

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
        >>> src, valid = Multiport.get_delay_source_indexer(snapshots, weightings, delay=2, is_cyclic=True)
        >>> src
        array([4, 5, 0, 1, 2, 3])
        >>> valid
        array([ True,  True,  True,  True,  True,  True])
        >>> src, valid = Multiport.get_delay_source_indexer(snapshots, weightings, delay=2, is_cyclic=False)
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

        if isinstance(snapshots, pd.MultiIndex):
            src = np.empty(n_snapshots, dtype=int)
            valid = np.empty(n_snapshots, dtype=bool)
            offset = 0
            for period in snapshots.unique("period"):
                w = weightings[snapshots.get_loc(period)].to_numpy().astype(float)
                p_src, p_valid = Multiport._delay_positions(w, delay, is_cyclic)
                src[offset : offset + len(w)] = p_src + offset
                valid[offset : offset + len(w)] = p_valid
                offset += len(w)
            return src, valid

        w = weightings.reindex(snapshots).astype(float).to_numpy()
        return Multiport._delay_positions(w, delay, is_cyclic)
