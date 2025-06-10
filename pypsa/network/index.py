"""Network index module.

Contains single mixin class which is used to inherit to [pypsa.Networks] class.
Should not be used directly.

Index methods and properties are used to access the different index levels, set them
and convert the Network accordingly.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from pypsa.network.abstract import _NetworkABC

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class NetworkIndexMixin(_NetworkABC):
    """Mixin class for network index methods.

    Class only inherits to [pypsa.Network][] and should not be used directly.
    All attributes and methods can be used within any Network instance.
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
            msg = "Snapshots must not be empty."
            raise ValueError(msg)

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
                dict.fromkeys(self._snapshot_weightings.columns, hours_per_step)
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
        return self._snapshots

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
            return self.snapshots.get_level_values("timestep").unique()
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

            self._snapshots = pd.MultiIndex.from_product(
                [periods_, self.snapshots], names=names
            )
            self._snapshots.name = "snapshot"
            self._snapshot_weightings = pd.concat(
                dict.fromkeys(periods_, self.snapshot_weightings), names=names
            )
            self._snapshot_weightings.index.name = "snapshot"

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
            return self.snapshots.get_level_values("period").unique()
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
        """Weightings applied to each snapshots during the optimization (LOPF).

        * Objective weightings multiply the operational cost in the
          objective function.

        * Generator weightings multiply the impact of all generators
          in global constraints, e.g. multiplier of GHG emmissions.

        * Store weightings define the elapsed hours for the charge, discharge
          standing loss and spillage of storage units and stores in order to
          determine the state of charge.

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
        return self._snapshot_weightings

    @snapshot_weightings.setter
    def snapshot_weightings(self, df: pd.DataFrame) -> None:
        if not df.index.equals(self.snapshots):
            msg = "Weightings not defined for all snapshots."
            raise ValueError(msg)

        if isinstance(df, pd.Series):
            logger.info("Applying weightings to all columns of `snapshot_weightings`")
            df = pd.DataFrame(dict.fromkeys(self._snapshot_weightings.columns, df))
        self._snapshot_weightings = df

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
        return self._investment_period_weightings

    @investment_period_weightings.setter
    def investment_period_weightings(self, df: pd.DataFrame) -> None:
        if not df.index.equals(self.investment_periods):
            msg = "Weightings not defined for all investment periods."
            raise ValueError(msg)
        if isinstance(df, pd.Series):
            logger.info(
                "Applying weightings to all columns of `investment_period_weightings`"
            )
            df = pd.DataFrame(
                dict.fromkeys(self._investment_period_weightings.columns, df)
            )
        self._investment_period_weightings = df
