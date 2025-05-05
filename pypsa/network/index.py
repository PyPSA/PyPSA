"""
Array module of PyPSA components.

Contains logic to combine static and dynamic pandas DataFrames to single xarray
DataArray for each variable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from pypsa.network.abstract import _NetworkABC

logger = logging.getLogger(__name__)


class _NetworkIndex(_NetworkABC):
    """
    Helper class for components array methods.

    Class only inherits to Components and should not be used directly.
    """

    # ----------------
    # Snapshots
    # ----------------

    def set_snapshots(
        self,
        snapshots: Sequence,
        default_snapshot_weightings: float = 1.0,
        weightings_from_timedelta: bool = False,
    ) -> None:
        """
        Set the snapshots/time steps and reindex all time-dependent data.

        Snapshot weightings, typically representing the hourly length of each snapshot,
        is filled with the `default_snapshot_weighintgs` value, or uses the timedelta
        of the snapshots if `weightings_from_timedelta` flag is True, and snapshots are
        of type `pd.DatetimeIndex`.

        This will reindex all components time-dependent DataFrames
        (:py:meth:`pypsa.Network.dynamic`). NaNs are filled with the default value for
        that quantity.

        Parameters
        ----------
        snapshots : list, pandas.Index or pd.MultiIndex
            All time steps.
        default_snapshot_weightings: float
            The default weight for each snapshot. Defaults to 1.0.
        weightings_from_timedelta: bool
            Wheter to use the timedelta of `snapshots` as `snapshot_weightings` if
            `snapshots` is of type `pd.DatetimeIndex`.  Defaults to False.

        Returns
        -------
        None

        """
        # Check if snapshots contain timezones
        if isinstance(snapshots, pd.DatetimeIndex) and snapshots.tz is not None:
            msg = (
                "Numpy datetime64[ns] objects with timezones are not supported and are "
                "thus not allowed in snapshots. Please pass timezone-naive timestamps "
                "(e.g. via ds.values)."
            )
            raise ValueError(msg)

        if isinstance(snapshots, pd.MultiIndex):
            if snapshots.nlevels != 2:
                msg = "Maximally two levels of MultiIndex supported"
                raise ValueError(msg)
            sns = snapshots.rename(["period", "timestep"])
            sns.name = "snapshot"
            self._snapshots = sns
        else:
            self._snapshots = pd.Index(snapshots, name="snapshot")

        if len(self._snapshots) == 0:
            raise ValueError("Snapshots must not be empty.")

        self.snapshot_weightings = self.snapshot_weightings.reindex(
            self._snapshots, fill_value=default_snapshot_weightings
        )

        if isinstance(snapshots, pd.DatetimeIndex) and weightings_from_timedelta:
            hours_per_step = (
                snapshots.to_series()
                .diff(periods=1)
                .shift(-1)
                .ffill()  # fill last value by assuming same as the one before
                .apply(lambda x: x.total_seconds() / 3600)
            )
            self._snapshot_weightings = pd.DataFrame(
                {c: hours_per_step for c in self._snapshot_weightings.columns}
            )
        elif not isinstance(snapshots, pd.DatetimeIndex) and weightings_from_timedelta:
            logger.info(
                "Skipping `weightings_from_timedelta` as `snapshots`is not of type `pd.DatetimeIndex`."
            )

        for component in self.all_components:
            dynamic = self.dynamic(component)
            attrs = self.components[component]["attrs"]

            for k in dynamic.keys():
                if dynamic[k].empty:  # avoid expensive reindex operation
                    dynamic[k].index = self._snapshots
                elif k in attrs.default[attrs.varying]:
                    if isinstance(dynamic[k].index, pd.MultiIndex):
                        dynamic[k] = dynamic[k].reindex(
                            self._snapshots, fill_value=attrs.default[attrs.varying][k]
                        )
                    else:
                        # Make sure to keep timestep level in case of MultiIndex
                        dynamic[k] = dynamic[k].reindex(
                            self._snapshots,
                            fill_value=attrs.default[attrs.varying][k],
                            level="timestep",
                        )
                else:
                    dynamic[k] = dynamic[k].reindex(self._snapshots)

        # NB: No need to rebind dynamic to self, since haven't changed it

    @property
    def snapshots(self) -> pd.Index | pd.MultiIndex:
        """
        Snapshots dimension of the network.

        If snapshots are a pandas.MultiIndex, the first level are investment periods
        and the second level are timesteps. If snapshots are single indexed, the only
        level is timesteps.

        Returns
        -------
        pd.Index or pd.MultiIndex
            Snapshots of the network, either as a single index or a multi-index.

        See Also
        --------
        pypsa.networks.Network.timesteps : Get the timestep level only.
        pypsa.networks.Network.periods : Get the period level only.

        Notes
        -----
        Note that Snapshots are a dimension, while timesteps and and periods are
        only levels of the snapshots dimension, similar to coords in xarray.
        This is because timesteps and periods are not necessarily unique or complete
        across snapshots.
        """
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots: Sequence) -> None:
        """
        Setter for snapshots dimension.

        Parameters
        ----------
        snapshots : Sequence


        See Also
        --------
        pypsa.networks.Network.snapshots : Getter method
        pypsa.networks.Network.set_snapshots : Setter method
        """
        self.set_snapshots(snapshots)

    # Timesteps
    # ---------
    @property
    def timesteps(self) -> pd.Index:
        """
        Timestep level of snapshots dimension.

        If snapshots is single indexed, timesteps and snapshots yield the same result.
        Otherwise only the timestep level will be returned.

        Returns
        -------
        pd.Index
            Timesteps of the network.

        See Also
        --------
        pypsa.networks.Network.snapshots : Get the snapshots dimension.
        pypsa.networks.Network.periods : Get the period level only.

        """
        if "timestep" in self.snapshots.names:
            return self.snapshots.get_level_values("timestep").unique()
        else:
            return self.snapshots.rename("timestep")

    @timesteps.setter
    def timesteps(self, timesteps: Sequence) -> None:
        """
        Setter for timesteps level of snapshots dimension.

        .. warning::
            Setting `timesteps` is not supported. Please set `snapshots` instead.

        Parameters
        ----------
        timesteps : Sequence

        Also see
        --------
        pypsa.networks.Network.timesteps : Getter method
        """
        msg = "Setting `timesteps` is not supported. Please set `snapshots` instead."
        raise NotImplementedError(msg)

    # Periods
    # ---------

    def set_investment_periods(self, periods: Sequence) -> None:
        """
        Set the investment periods of the network.

        If the network snapshots are a pandas.MultiIndex, the investment periods
        have to be a subset of the first level. If snapshots are a single index,
        they and all time-series are repeated for all periods. This changes
        the network snapshots to be a MultiIndex (inplace operation) with the first
        level being the investment periods and the second level the snapshots.

        Parameters
        ----------
        n : pypsa.Network
        periods : list
            List of periods to be selected/initialized.

        Returns
        -------
        None.

        """
        periods_ = pd.Index(periods, name="period")
        if periods_.empty:
            return
        if not (
            pd.api.types.is_integer_dtype(periods_)
            and periods_.is_unique
            and periods_.is_monotonic_increasing
        ):
            raise ValueError(
                "Investment periods are not strictly increasing integers, "
                "which is required for multi-period investment optimisation."
            )
        if isinstance(self.snapshots, pd.MultiIndex):
            if not periods_.isin(self.snapshots.unique("period")).all():
                raise ValueError(
                    "Not all investment periods are in level `period` of snapshots."
                )
            if len(periods_) < len(self.snapshots.unique(level="period")):
                raise NotImplementedError(
                    "Investment periods do not equal first level values of snapshots."
                )
        else:
            # Convenience case:
            logger.info(
                "Repeating time-series for each investment period and "
                "converting snapshots to a pandas.MultiIndex."
            )
            names = ["period", "timestep"]
            for c in self.components.values():
                for k in c.dynamic.keys():
                    c.dynamic[k] = pd.concat(
                        {p: c.dynamic[k] for p in periods_}, names=names
                    )
                    c.dynamic[k].index.name = "snapshot"

            self._snapshots = pd.MultiIndex.from_product(
                [periods_, self.snapshots], names=names
            )
            self._snapshots.name = "snapshot"
            self._snapshot_weightings = pd.concat(
                {p: self.snapshot_weightings for p in periods_}, names=names
            )
            self._snapshot_weightings.index.name = "snapshot"

        self.investment_period_weightings = self.investment_period_weightings.reindex(
            self.periods, fill_value=1.0
        ).astype(float)

    @property
    def periods(self) -> pd.Index:
        """
        Periods level of snapshots dimension.

        If snapshots is single indexed, periods will always be empty, since there no
        investment periods without timesteps are defined. Otherwise only the period
        level will be returned.

        Returns
        -------
        pd.Index
            Periods of the network.

        See Also
        --------
        pypsa.networks.Network.snapshots : Get the snapshots dimension.
        pypsa.networks.Network.timesteps : Get the timestep level only.

        """
        if "period" in self.snapshots.names:
            return self.snapshots.get_level_values("period").unique()
        else:
            return pd.Index([], name="period")

    @periods.setter
    def periods(self, periods: Sequence) -> None:
        """
        Setter for periods level of snapshots dimension.

        Parameters
        ----------
        periods : Sequence

        Also see
        --------
        pypsa.networks.Network.periods : Getter method
        pypsa.networks.Network.set_investment_periods : Setter method
        """
        self.set_investment_periods(periods)

    @property
    def has_periods(self) -> bool:
        """
        Check if network has investment periods assigned to snapshots dimension.

        Returns
        -------
        bool
            True if network has investment periods, otherwise False.

        See Also
        --------
        pypsa.networks.Network.snapshots : Snapshots dimension of the network.
        pypsa.networks.Network.periods : Periods level of snapshots dimension.
        """
        return not self.periods.empty

    @property
    def investment_periods(self) -> pd.Index:
        """
        Periods level of snapshots dimension.

        If snapshots is single indexed, periods will always be empty, since there no
        investment periods without timesteps are defined. Otherwise only the period
        level will be returned.

        .. Note :: Alias for :py:meth:`pypsa.Network.periods`.

        Returns
        -------
        pd.Index
            Investment periods of the network.

        See Also
        --------
        pypsa.networks.Network.snapshots : Get the snapshots dimension.
        pypsa.networks.Network.periods : Get the snapshots dimension.
        pypsa.networks.Network.timesteps : Get the timestep level only.

        """
        return self.periods

    @investment_periods.setter
    def investment_periods(self, periods: Sequence) -> None:
        """
        Setter for periods level of snapshots dimension.

        .. Note :: Alias for :py:meth:`pypsa.Network.periods`.

        Parameters
        ----------
        periods : Sequence

        Also see
        --------
        pypsa.networks.Network.periods : Getter method
        pypsa.networks.Network.set_investment_periods : Setter method
        """
        self.periods = periods

    @property
    def has_investment_periods(self) -> bool:
        """
        Check if network has investment periods assigned to snapshots dimension.

        .. Note :: Alias for :py:meth:`pypsa.Network.has_periods`.

        Returns
        -------
        bool
            True if network has investment periods, otherwise False.

        See Also
        --------
        pypsa.networks.Network.snapshots : Snapshots dimension of the network.
        pypsa.networks.Network.periods : Periods level of snapshots dimension.
        """
        return self.has_periods

    # Snapshot weightings
    # -------------------

    @property
    def snapshot_weightings(self) -> pd.DataFrame:
        """
        Weightings applied to each snapshots during the optimization (LOPF).

        * Objective weightings multiply the operational cost in the
          objective function.

        * Generator weightings multiply the impact of all generators
          in global constraints, e.g. multiplier of GHG emmissions.

        * Store weightings define the elapsed hours for the charge, discharge
          standing loss and spillage of storage units and stores in order to
          determine the state of charge.
        """
        return self._snapshot_weightings

    @snapshot_weightings.setter
    def snapshot_weightings(self, df: pd.DataFrame) -> None:
        assert df.index.equals(self.snapshots), (
            "Weightings not defined for all snapshots."
        )
        if isinstance(df, pd.Series):
            logger.info("Applying weightings to all columns of `snapshot_weightings`")
            df = pd.DataFrame({c: df for c in self._snapshot_weightings.columns})
        self._snapshot_weightings = df

    @property
    def investment_period_weightings(self) -> pd.DataFrame:
        """
        Weightings applied to each investment period during the optimization
        (LOPF).

        * Objective weightings are multiplied with all cost coefficients in the
          objective function of the respective investment period
          (e.g. to include a social discount rate).

        * Years weightings denote the elapsed time until the subsequent investment period
          (e.g. used for global constraints CO2 emissions).
        """
        return self._investment_period_weightings

    @investment_period_weightings.setter
    def investment_period_weightings(self, df: pd.DataFrame) -> None:
        assert df.index.equals(self.investment_periods), (
            "Weightings not defined for all investment periods."
        )
        if isinstance(df, pd.Series):
            logger.info(
                "Applying weightings to all columns of `investment_period_weightings`"
            )
            df = pd.DataFrame(
                {c: df for c in self._investment_period_weightings.columns}
            )
        self._investment_period_weightings = df

    # -----------
    # Scenarios
    # -----------

    def set_scenarios(
        self,
        scenarios: dict | pd.Series | Sequence | None = None,
        weights: float | pd.Series | None = None,
        **kwargs: Any,
    ) -> None:
        # Validate input
        if scenarios is None and weights is None and not kwargs:
            msg = (
                "You must pass either `scenarios` (with weights) or keyword arguments "
                "to set_scenarios."
            )
            raise ValueError(msg)
        if kwargs and (scenarios is not None or weights is not None):
            msg = (
                "You can pass scenarios either via `scenarios`/`weights` or via "
                "keyword arguments, but not both."
            )
            raise ValueError(msg)
        if isinstance(scenarios, dict | pd.Series) and weights is not None:
            msg = (
                "When passing a dict or pandas.Series to `scenarios`, their values "
                "are used as weights. Therefore `weights` must be None."
            )
            raise ValueError(msg)
        if weights is not None and len(weights) != len(scenarios):
            msg = "Length of `weights` must be equal to the length of `scenarios`."
            raise ValueError(msg)

        if isinstance(scenarios, dict):
            scenarios = pd.Series(scenarios)
        elif isinstance(scenarios, pd.Series):
            pass
        elif isinstance(scenarios, Sequence) and weights is not None:
            scenarios = pd.Series(weights, index=scenarios)
        elif isinstance(scenarios, Sequence) and weights is None:
            scenarios = pd.Series(
                [1 / len(scenarios)] * len(scenarios), index=scenarios
            )
        elif kwargs:
            scenarios = pd.Series(kwargs)

        if scenarios.sum() != 1:
            msg = (
                "The sum of the weights in `scenarios` must be equal to 1. "
                f"Current sum: {scenarios.sum()}"
            )
            raise ValueError(msg)

        scenarios = scenarios.rename("scenario")
        scenarios.index = scenarios.index.astype(str)

        for c in self.components.values():
            c.static = pd.concat(
                {scen: c.static for scen in scenarios.index}, names=["scenario"]
            )
            for k, v in c.dynamic.items():
                c.dynamic[k] = pd.concat(
                    {scen: v for scen in scenarios.index}, names=["scenario"], axis=1
                )

        self._scenarios = scenarios

    @property
    def scenarios(self) -> pd.Series:
        return self._scenarios

    @scenarios.setter
    def scenarios(self, scenarios: dict | pd.Series | Sequence) -> None:
        self.set_scenarios(scenarios)

    @property
    def has_scenarios(self) -> bool:
        """
        Boolean indicating if the network has scenarios defined.
        """
        return len(self.scenarios) > 0
