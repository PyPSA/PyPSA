"""Network index module.

Contains single mixin class which is used to inherit to [pypsa.Networks] class.
Should not be used directly.

Index methods and properties are used to access the different index levels, set them
and convert the Network accordingly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from pypsa.network.abstract import _NetworkABC

logger = logging.getLogger(__name__)


class NetworkIndexMixin(_NetworkABC):
    """Mixin class for network index methods.

    Class only inherits to [pypsa.Network][] and should not be used directly.
    All attributes and methods can be used within any Network instance.
    """

    _risk_preference: dict[str, float] | None

    # ----------------
    # Snapshots
    # ----------------

    def set_snapshots(
        self,
        snapshots: Sequence,
        default_snapshot_weightings: float = 1.0,
        weightings_from_timedelta: bool = False,
    ) -> None:
        """Set the snapshots/time steps and reindex all time-dependent data.

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

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.set_snapshots(pd.date_range("2015-01-01", freq="h", periods=3))
        >>> n.snapshots
        DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
                       '2015-01-01 02:00:00'],
                      dtype='datetime64[ns]', name='snapshot', freq='h')

        """
        # Check if snapshots contain timezones
        if isinstance(snapshots, pd.DatetimeIndex) and snapshots.tz is not None:
            msg = (
                "Numpy datetime64[ns] objects with timezones are not supported and are "
                "thus not allowed in snapshots. Please pass timezone-naive timestamps "
                "(e.g. via ds.values)."
            )
            raise ValueError(msg)

        # Always create normal pd.Index, never pd.RangeIndex
        if isinstance(snapshots, range):
            snapshots = list(snapshots)

        if isinstance(snapshots, pd.MultiIndex):
            if snapshots.nlevels != 2:
                msg = "Maximally two levels of MultiIndex supported"
                raise ValueError(msg)
            sns = snapshots.rename(["period", "timestep"])
            sns.name = "snapshot"
        else:
            sns = pd.Index(snapshots, name="snapshot")

        if len(sns) == 0:
            msg = "Snapshots must not be empty."
            raise ValueError(msg)

        self._snapshots_data = self._snapshots_data.reindex(
            sns, fill_value=default_snapshot_weightings
        )

        if isinstance(snapshots, pd.DatetimeIndex) and weightings_from_timedelta:
            hours_per_step = (
                snapshots.to_series()
                .diff(periods=1)
                .shift(-1)
                .ffill()  # fill last value by assuming same as the one before
                .apply(lambda x: x.total_seconds() / 3600)
            )
            self._snapshots_data = pd.DataFrame(
                dict.fromkeys(self._snapshots_data.columns, hours_per_step)
            )
        elif not isinstance(snapshots, pd.DatetimeIndex) and weightings_from_timedelta:
            logger.info(
                "Skipping `weightings_from_timedelta` as `snapshots`is not of type `pd.DatetimeIndex`."
            )

        for component in self.all_components:
            dynamic = self.dynamic(component)
            attrs = self.components[component]["attrs"]

            for k in dynamic:
                if dynamic[k].empty:  # avoid expensive reindex operation
                    dynamic[k].index = self.snapshots
                elif k in attrs.default[attrs.varying]:
                    if isinstance(dynamic[k].index, pd.MultiIndex):
                        dynamic[k] = dynamic[k].reindex(
                            self.snapshots, fill_value=attrs.default[attrs.varying][k]
                        )
                    else:
                        # Make sure to keep timestep level in case of MultiIndex
                        dynamic[k] = dynamic[k].reindex(
                            self.snapshots,
                            fill_value=attrs.default[attrs.varying][k],
                            level="timestep",
                        )
                else:
                    dynamic[k] = dynamic[k].reindex(self.snapshots)

        # NB: No need to rebind dynamic to self, since haven't changed it

    @property
    def snapshots(self) -> pd.Index | pd.MultiIndex:
        """Snapshots dimension of the network.

        If snapshots are a pandas.MultiIndex, the first level are investment periods
        and the second level are timesteps. If snapshots are single indexed, the only
        level is timesteps.

        Returns
        -------
        pd.Index or pd.MultiIndex
            Snapshots of the network, either as a single index or a multi-index.

        See Also
        --------
        [pypsa.Network.timesteps][] : Get the timestep level only.
        [pypsa.Network[] : Get the period level only.

        Notes
        -----
        Note that Snapshots are a dimension, while timesteps and and periods are
        only levels of the snapshots dimension, similar to coords in xarray.
        This is because timesteps and periods are not necessarily unique or complete
        across snapshots.

        Examples
        --------
        >>> n.snapshots # doctest: +ELLIPSIS
        DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
                       '2015-01-01 02:00:00', '2015-01-01 03:00:00',
                      ...
                      dtype='datetime64[ns]', name='snapshot', freq=None)

        """
        return self._snapshots_data.index

    @snapshots.setter
    def snapshots(self, snapshots: Sequence) -> None:
        """Setter for snapshots dimension.

        Parameters
        ----------
        snapshots : Sequence
            Snapshots to be set.

        See Also
        --------
        [pypsa.Network.snapshots][] : Getter method
        [pypsa.Network.set_snapshots][] : Setter method

        """
        self.set_snapshots(snapshots)

    # Timesteps (Coordinate of Snapshots)
    # ---------
    @property
    def timesteps(self) -> pd.Index:
        """Timestep level of snapshots dimension.

        If snapshots is single indexed, timesteps and snapshots yield the same result.
        Otherwise only the timestep level will be returned.

        Returns
        -------
        pd.Index
            Timesteps of the network.

        See Also
        --------
        [pypsa.Network.snapshots][] : Get the snapshots dimension.
        [pypsa.Network.periods][] : Get the period level only.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.set_snapshots(pd.date_range("2015-01-01", freq="h", periods=3))

        For a Network without investment periods, the timesteps are identical to the
        snapshots:
        >>> n.timesteps
        DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
                       '2015-01-01 02:00:00'],
                      dtype='datetime64[ns]', name='snapshot', freq='h')
        >>> n.snapshots
        DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
                       '2015-01-01 02:00:00'],
                      dtype='datetime64[ns]', name='snapshot', freq='h')

        For a Network with investment periods, the timesteps are are the unqiue set
        of timesteps in across all investment periods:
        >>> n.investment_periods = [1, 2]
        >>> n.timesteps
        DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
                       '2015-01-01 02:00:00'],
                      dtype='datetime64[ns]', name='timestep', freq=None)
        >>> n.snapshots
        MultiIndex([(1, '2015-01-01 00:00:00'),
                (1, '2015-01-01 01:00:00'),
                (1, '2015-01-01 02:00:00'),
                (2, '2015-01-01 00:00:00'),
                (2, '2015-01-01 01:00:00'),
                (2, '2015-01-01 02:00:00')],
               name='snapshot')

        """
        if "timestep" in self.snapshots.names:
            return self.snapshots.get_level_values("timestep").drop_duplicates()
        return self.snapshots

    @timesteps.setter
    def timesteps(self, timesteps: Sequence) -> None:
        """Setter for timesteps level of snapshots dimension.

        .. warning::
            Setting `timesteps` is not supported. Please set `snapshots` instead.

        Parameters
        ----------
        timesteps : Sequence
            Timesteps to be set.

        Also see
        --------
        pypsa.Network.timesteps : Getter method

        """
        msg = "Setting `timesteps` is not supported. Please set `snapshots` instead."
        raise NotImplementedError(msg)

    # Investment Periods (Coordinate of Snapshots)
    # ---------

    def set_investment_periods(self, periods: Sequence) -> None:
        """Set the investment periods of the network.

        If the network snapshots are a pandas.MultiIndex, the investment periods
        have to be a subset of the first level. If snapshots are a single index,
        they and all time-series are repeated for all periods. This changes
        the network snapshots to be a MultiIndex (inplace operation) with the first
        level being the investment periods and the second level the snapshots.

        Parameters
        ----------
        periods : list
            List of periods to be selected/initialized.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.set_snapshots(pd.date_range("2015-01-01", freq="h", periods=3))
        >>> n.snapshots
        DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
                       '2015-01-01 02:00:00'],
                      dtype='datetime64[ns]', name='snapshot', freq='h')
        >>> n.investment_periods = [1, 2]
        >>> n.snapshots
        MultiIndex([(1, '2015-01-01 00:00:00'),
                (1, '2015-01-01 01:00:00'),
                (1, '2015-01-01 02:00:00'),
                (2, '2015-01-01 00:00:00'),
                (2, '2015-01-01 01:00:00'),
                (2, '2015-01-01 02:00:00')],
               name='snapshot')

        """
        periods_ = pd.Index(periods, name="period")
        if periods_.empty:
            return
        if not (
            pd.api.types.is_integer_dtype(periods_)
            and periods_.is_unique
            and periods_.is_monotonic_increasing
        ):
            msg = (
                "Investment periods are not strictly increasing integers, "
                "which is required for multi-period investment optimisation."
            )
            raise ValueError(msg)
        if isinstance(self.snapshots, pd.MultiIndex):
            if not periods_.isin(self.snapshots.unique("period")).all():
                msg = "Not all investment periods are in level `period` of snapshots."
                raise ValueError(msg)
            if len(periods_) < len(self.snapshots.unique(level="period")):
                msg = "Investment periods do not equal first level values of snapshots."
                raise NotImplementedError(msg)
        else:
            # Convenience case:
            logger.info(
                "Repeating time-series for each investment period and "
                "converting snapshots to a pandas.MultiIndex."
            )
            names = ["period", "timestep"]
            for component in self.all_components:
                dynamic = self.dynamic(component)

                for k in dynamic:
                    dynamic[k] = pd.concat(
                        dict.fromkeys(periods_, dynamic[k]), names=names
                    )
                    dynamic[k].index.name = "snapshot"

            sns = pd.MultiIndex.from_product([periods_, self.snapshots], names=names)
            sns.name = "snapshot"
            self._snapshots_data = pd.concat(
                dict.fromkeys(periods_, self.snapshot_weightings), names=names
            )
            self._snapshots_data.index.name = "snapshot"

        self.investment_period_weightings = self.investment_period_weightings.reindex(
            self.periods, fill_value=1.0
        ).astype(float)

    @property
    def periods(self) -> pd.Index:
        """Periods level of snapshots dimension.

        If snapshots is single indexed, periods will always be empty, since there no
        investment periods without timesteps are defined. Otherwise only the period
        level will be returned.

        Returns
        -------
        pd.Index
            Periods of the network.

        See Also
        --------
        [pypsa.Network.snapshots][] : Get the snapshots dimension.
        [pypsa.Network.timesteps][] : Get the timestep level only.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", "bus") # doctest: +SKIP
        >>> n.snapshots = pd.date_range("2015-01-01", freq="h", periods=3)
        >>> n.snapshots
        DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
                       '2015-01-01 02:00:00'],
                      dtype='datetime64[ns]', name='snapshot', freq='h')

        Add investment periods:
        >>> n.periods = [1, 2]
        >>> n.periods
        Index([1, 2], dtype='int64', name='period')

        Which will also cast snapshots to a MultiIndex:
        >>> n.snapshots
        MultiIndex([(1, '2015-01-01 00:00:00'),
                (1, '2015-01-01 01:00:00'),
                (1, '2015-01-01 02:00:00'),
                (2, '2015-01-01 00:00:00'),
                (2, '2015-01-01 01:00:00'),
                (2, '2015-01-01 02:00:00')],
               name='snapshot')

        """
        if "period" in self.snapshots.names:
            return self.snapshots.get_level_values("period").drop_duplicates()
        return pd.Index([], name="period")

    @periods.setter
    def periods(self, periods: Sequence) -> None:
        """Setter for periods level of snapshots dimension.

        Parameters
        ----------
        periods : Sequence
            Investment periods to be set.
        Also see
        --------
        pypsa.Network.periods : Getter method
        pypsa.Network.set_investment_periods : Setter method

        """
        self.set_investment_periods(periods)

    @property
    def has_periods(self) -> bool:
        """Check if network has investment periods assigned to snapshots dimension.

        Returns
        -------
        bool
            True if network has investment periods, otherwise False.

        See Also
        --------
        [pypsa.Network.snapshots][] : Snapshots dimension of the network.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", "bus") # doctest: +SKIP
        >>> n.snapshots = pd.date_range("2015-01-01", freq="h", periods=3)
        >>> n.has_periods
        False

        Add investment periods:
        >>> n.periods = [1, 2]
        >>> n.has_periods
        True

        """
        return not self.periods.empty

    @property
    def investment_periods(self) -> pd.Index:
        """Periods level of snapshots dimension.

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
        [pypsa.Network.snapshots][] : Get the snapshots dimension.
        [pypsa.Network.periods][] : Get the snapshots dimension.
        [pypsa.Network.timesteps][] : Get the timestep level only.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", "bus") # doctest: +SKIP
        >>> n.snapshots = pd.date_range("2015-01-01", freq="h", periods=3)
        >>> n.snapshots
        DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
                       '2015-01-01 02:00:00'],
                      dtype='datetime64[ns]', name='snapshot', freq='h')

        Add investment periods:
        >>> n.investment_periods = [1, 2]
        >>> n.investment_periods
        Index([1, 2], dtype='int64', name='period')

        Which will also cast snapshots to a MultiIndex:
        >>> n.snapshots
        MultiIndex([(1, '2015-01-01 00:00:00'),
                (1, '2015-01-01 01:00:00'),
                (1, '2015-01-01 02:00:00'),
                (2, '2015-01-01 00:00:00'),
                (2, '2015-01-01 01:00:00'),
                (2, '2015-01-01 02:00:00')],
               name='snapshot')

        """
        return self.periods

    @investment_periods.setter
    def investment_periods(self, periods: Sequence) -> None:
        """Setter for periods level of snapshots dimension.

        .. Note :: Alias for :py:meth:`pypsa.Network.periods`.

        Parameters
        ----------
        periods : Sequence
            Investment periods to be set.

        Also see
        --------
        pypsa.Network.periods : Getter method
        pypsa.Network.set_investment_periods : Setter method

        """
        self.periods = periods

    @property
    def has_investment_periods(self) -> bool:
        """Check if network has investment periods assigned to snapshots dimension.

        .. Note :: Alias for :py:meth:`pypsa.Network.has_periods`.

        Returns
        -------
        bool
            True if network has investment periods, otherwise False.

        See Also
        --------
        [pypsa.Network.snapshots][] : Snapshots dimension of the network.
        [pypsa.Network.periods][] : Periods level of snapshots dimension.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", "bus") # doctest: +SKIP
        >>> n.snapshots = pd.date_range("2015-01-01", freq="h", periods=3)
        >>> n.has_investment_periods
        False

        Add investment periods:
        >>> n.periods = [1, 2]
        >>> n.has_investment_periods
        True

        """
        return self.has_periods

    # Snapshot weightings
    # -------------------

    @property
    def snapshot_weightings(self) -> pd.DataFrame:
        """Weightings applied to each snapshots during the optimization.

        * Objective weightings are factors on the operational cost in the
          objective function.

        * Store weightings define the elapsed hours for the charge, discharge
          standing loss and spillage of storage units and stores in order to
          determine the state of charge.

        * Generator weightings are factors for the contribution of generators
          to global constraints, e.g. emission limits, and energy balances.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", "bus") # doctest: +SKIP
        >>> n.snapshots = pd.date_range("2015-01-01", freq="h", periods=3)
        >>> n.snapshot_weightings
                         objective  stores  generators
        snapshot
        2015-01-01 00:00:00        1.0     1.0         1.0
        2015-01-01 01:00:00        1.0     1.0         1.0
        2015-01-01 02:00:00        1.0     1.0         1.0

        Change the snapshot weightings:
        >>> n.snapshot_weightings.objective = [5, 7, 9]
        >>> n.snapshot_weightings
                         objective  stores  generators
        snapshot
        2015-01-01 00:00:00          5     1.0         1.0
        2015-01-01 01:00:00          7     1.0         1.0
        2015-01-01 02:00:00          9     1.0         1.0

        """
        return self._snapshots_data

    @snapshot_weightings.setter
    def snapshot_weightings(self, df: pd.DataFrame) -> None:
        if not df.index.equals(self.snapshots):
            msg = "Weightings not defined for all snapshots."
            raise ValueError(msg)

        if isinstance(df, pd.Series):
            logger.info("Applying weightings to all columns of `snapshot_weightings`")
            df = pd.DataFrame(dict.fromkeys(self._snapshots_data.columns, df))
        df.index.names = self.snapshots.names
        self._snapshots_data = df

    @property
    def investment_period_weightings(self) -> pd.DataFrame:
        """Weightings applied to each investment period during the optimization (LOPF).

        Objective weightings are multiplied with all cost coefficients in the
        objective function of the respective investment period (e.g. to include a
        social discount rate).
        Years weightings denote the elapsed time until the subsequent investment period
        (e.g. used for global constraints CO2 emissions).

        Examples
        --------
        Create a network with investment periods:
        >>> n = pypsa.Network()
        >>> n.add("Bus", "bus") # doctest: +SKIP
        >>> n.snapshots = pd.date_range("2015-01-01", freq="h", periods=2)
        >>> n.investment_periods = [1, 2]

        >>> n.investment_period_weightings
                objective  years
        period
        1             1.0    1.0
        2             1.0    1.0

        Change the investment period weightings:
        >>> n.investment_period_weightings.objective = [5, 7]
        >>> n.investment_period_weightings.years = [1, 2]
        >>> n.investment_period_weightings
                objective  years
        period
        1             5    1
        2             7    2

        """
        return self._investment_periods_data

    @investment_period_weightings.setter
    def investment_period_weightings(self, df: pd.DataFrame) -> None:
        if not df.index.equals(self.investment_periods):
            msg = "Weightings not defined for all investment periods."
            raise ValueError(msg)
        if isinstance(df, pd.Series):
            logger.info(
                "Applying weightings to all columns of `investment_period_weightings`"
            )
            df = pd.DataFrame(dict.fromkeys(self._investment_periods_data.columns, df))
        self._investment_periods_data = df

    # -----------
    # Scenarios
    # -----------

    def set_scenarios(
        self,
        scenarios: dict | Sequence | pd.Series | pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """Set scenarios for the network to create a stochastic network.

        Parameters
        ----------
        scenarios : dict, Sequence, pd.Series, optional
            Scenarios to set for the network.
        **kwargs : Any
            Alternative way to set scenarios via keyword arguments.
            E.g. `n.set_scenarios(low=0.5, high=0.5)`.

        """
        # Validate input
        if self.has_scenarios:
            msg = (
                "Changing scenarios on a network that already has scenarios defined is not "
                "yet supported."
            )
            raise NotImplementedError(msg)
        if scenarios is None and not kwargs:
            msg = (
                "You must pass either `scenarios` or keyword arguments "
                "to set_scenarios."
            )
            raise ValueError(msg)
        if kwargs and scenarios is not None:
            msg = (
                "You can pass scenarios either via `scenarios` or via "
                "keyword arguments, but not both."
            )
            raise ValueError(msg)

        if isinstance(scenarios, dict):
            scenarios_ = pd.Series(scenarios)
        elif isinstance(scenarios, pd.Series):
            scenarios_ = scenarios
        elif isinstance(scenarios, pd.DataFrame):
            if scenarios.shape[1] != 1:
                msg = "`scenarios` DataFrame must have exactly one column."
                raise ValueError(msg)
            scenarios_ = scenarios.iloc[:, 0]
        elif isinstance(scenarios, Sequence):
            scenarios_ = pd.Series(
                [1 / len(scenarios)] * len(scenarios), index=scenarios
            )
        elif kwargs:
            scenarios_ = pd.Series(kwargs)
        else:
            msg = "Invalid type for `scenarios`. Must be dict, pd.DataFrame, pd.Series, or Sequence. "
            raise TypeError(msg)

        if abs(scenarios_.sum() - 1) > 1e-5:
            msg = (
                "The sum of the weights in `scenarios` must be equal to 1. "
                f"Current sum: {scenarios_.sum()}"
            )
            raise ValueError(msg)

        scenarios_ = scenarios_.rename("weight")
        scenarios_.index = scenarios_.index.astype(str)
        scenarios_.index.name = "scenario"

        for c in self.components.values():
            c.static = pd.concat(
                dict.fromkeys(scenarios_.index, c.static), names=["scenario"]
            )
            for k, v in c.dynamic.items():
                c.dynamic[k] = pd.concat(
                    dict.fromkeys(scenarios_.index, v), names=["scenario"], axis=1
                )

        self._scenarios_data = scenarios_.to_frame()

    @property
    def scenarios(self) -> pd.Index:
        """Get the scenarios index for the network.

        Returns
        -------
        pd.Index
            The scenarios index for the network.

        """
        return self._scenarios_data.index

    @scenarios.setter
    def scenarios(self, scenarios: dict | pd.Series | Sequence) -> None:
        self.set_scenarios(scenarios)

    @property
    def scenario_weightings(self) -> pd.DataFrame:
        """Get the scenario weightings for the network.

        Returns
        -------
        pd.DataFrame
            The scenario weightings as a DataFrame with 'weight' column.

        """
        return self._scenarios_data

    @property
    def has_scenarios(self) -> bool:
        """Boolean indicating if the network has scenarios defined."""
        return len(self._scenarios_data) > 0

    # -----------
    # Risk Preferences (CVaR)
    # -----------

    def set_risk_preference(self, alpha: float, omega: float) -> None:
        """Set risk aversion preferences for stochastic optimization using CVaR formulation.

        Parameters
        ----------
        alpha : float
            Risk tail parameter (confidence level). Must be between 0 and 1.
            Common values are 0.05 (5% worst outcomes) or 0.1 (10% worst outcomes).
            Lower values focus on more extreme (worse) outcomes.
        omega : float
            Risk preference parameter (risk aversion weight). Must be between 0 and 1.
            - omega = 0: Risk-neutral optimization (traditional expected value)
            - omega > 0: Risk-averse optimization (penalizes variance/tail risk)
            - omega = 1: Maximum risk aversion (pure CVaR optimization)
            Higher values indicate more risk aversion.

        Raises
        ------
        ValueError
            If alpha is not between 0 and 1, or if omega is not between 0 and 1.
        RuntimeError
            If the network does not have scenarios defined.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.set_scenarios({"low": 0.3, "medium": 0.4, "high": 0.3})
        >>> n.set_risk_preference(alpha=0.05, omega=0.1)  # 5% CVaR with moderate risk aversion

        Notes
        -----
        This method must be called after `set_scenarios()` as CVaR formulation
        requires stochastic scenarios to be defined. The CVaR formulation will
        add auxiliary variables and constraints to the optimization model during
        the model building phase.

        References
        ----------
        Uryasev, S. (2000). Conditional value-at-risk: optimization algorithms
        and applications. Financial Engineering News, 14(2), 1-5.

        """
        # Validate that scenarios are defined
        if not self.has_scenarios:
            msg = (
                "Risk preferences can only be set for stochastic networks. "
                "Please call set_scenarios() first to define scenarios."
            )
            raise RuntimeError(msg)

        # Validate parameters
        if not (0 < alpha < 1):
            msg = f"Alpha must be between 0 and 1, got {alpha}"
            raise ValueError(msg)

        if not (0 <= omega <= 1):
            msg = f"Omega must be between 0 and 1, got {omega}"
            raise ValueError(msg)

        # Store risk preferences
        self._risk_preference = {"alpha": alpha, "omega": omega}

    @property
    def risk_preference(self) -> dict[str, float] | None:
        """Get the risk preference parameters for the network.

        Returns
        -------
        dict[str, float] | None
            Dictionary containing 'alpha' and 'omega' parameters if risk preferences
            are set, None otherwise.

        """
        return self._risk_preference

    @property
    def has_risk_preference(self) -> bool:
        """Boolean indicating if the network has risk preferences defined."""
        return self._risk_preference is not None
