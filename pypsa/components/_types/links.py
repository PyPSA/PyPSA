# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Links components module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components
from pypsa.constants import RE_PORTS_GE_2

if TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr


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

    class DelayGroup(NamedTuple):
        """Grouped delay configuration for a link port."""

        delay: int
        is_cyclic: bool
        names: pd.Index

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

    def split_by_port_delay(
        self, port: str, *, subset: pd.Index | None = None
    ) -> tuple[pd.Index, list[DelayGroup]]:
        """Split links of a port into non-delayed assets and delayed groups.

        Parameters
        ----------
        port : str
            Port id like ``"1"``, ``"2"``, ...
        subset : pd.Index, optional
            Optional subset of link names to process.

        Returns
        -------
        tuple[pd.Index, list[DelayGroup]]
            First element contains non-delayed link names. Second element contains
            delay-groups, where each group has common ``delay`` and
            ``is_cyclic`` values.

        """
        i_suffix = "" if port == "1" else port
        delay_attr = f"delay{i_suffix}"
        cyclic_attr = f"cyclic_delay{i_suffix}"

        names = (
            self.static.index
            if subset is None
            else subset.intersection(self.static.index)
        )
        if delay_attr not in self.static.columns:
            return names, []

        delays = self.static.loc[names, delay_attr]
        has_delay = delays > 0
        non_delayed = delays[~has_delay].index
        if not has_delay.any():
            return non_delayed, []

        if cyclic_attr in self.static.columns:
            cyclic = self.static.loc[names, cyclic_attr]
        else:
            cyclic = pd.Series(True, index=names)

        grouped: list[Links.DelayGroup] = []
        for d in delays[has_delay].unique():
            delay_names = delays[has_delay & (delays == d)].index
            is_cyclic = cyclic.loc[delay_names]
            for cyc_val in is_cyclic.unique():
                cyc_names = is_cyclic[is_cyclic == cyc_val].index
                grouped.append(Links.DelayGroup(int(d), bool(cyc_val), cyc_names))

        return non_delayed, grouped

    @staticmethod
    def get_delay_source_indexer(
        snapshots: pd.Index,
        weightings: pd.Series,
        delay: int,
        is_cyclic: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get per-snapshot source positions for link delays.

        Delay is interpreted in elapsed time units. For each target snapshot
        ``t``, the source is the latest snapshot ``s`` such that
        ``tau[s] <= tau[t] - delay``, where ``tau`` are cumulative weighting starts.
        """
        n_snapshots = len(snapshots)
        if n_snapshots == 0:
            return np.array([], dtype=int), np.array([], dtype=bool)
        if delay <= 0:
            src = np.arange(n_snapshots, dtype=int)
            return src, np.ones(n_snapshots, dtype=bool)

        weights = weightings.reindex(snapshots).astype(float).to_numpy()
        if (weights < 0).any():
            msg = "Negative snapshot weightings are not supported for link delays."
            raise ValueError(msg)

        tau = np.concatenate(([0.0], np.cumsum(weights[:-1])))
        src_time = tau - float(delay)

        if is_cyclic:
            total = float(weights.sum())
            if total <= 0:
                msg = (
                    "Cyclic weighted link delay requires a positive total "
                    "snapshot weighting over the optimized snapshots."
                )
                raise ValueError(msg)
            src_time = np.mod(src_time, total)
            valid = np.ones(n_snapshots, dtype=bool)
        else:
            valid = src_time >= 0

        src = np.searchsorted(tau, src_time, side="right") - 1
        src = np.clip(src, 0, n_snapshots - 1).astype(int)
        return src, valid
