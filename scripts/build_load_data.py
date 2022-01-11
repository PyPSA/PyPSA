# SPDX-FileCopyrightText: : 2020 @JanFrederickUnnewehr, The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""

This rule downloads the load data from `Open Power System Data Time series <https://data.open-power-system-data.org/time_series/>`_. For all countries in the network, the per country load timeseries with suffix ``_load_actual_entsoe_transparency`` are extracted from the dataset. After filling small gaps linearly and large gaps by copying time-slice of a given period, the load data is exported to a ``.csv`` file.

Relevant Settings
-----------------

.. code:: yaml

    snapshots:

    load:
        interpolate_limit:
        time_shift_for_large_gaps:
        manual_adjustments:


.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`load_cf`

Inputs
------


Outputs
-------

- ``resource/time_series_60min_singleindex_filtered.csv``:


"""

import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging

import pandas as pd
import numpy as np
import dateutil
from pandas import Timedelta as Delta


def load_timeseries(fn, years, countries, powerstatistics=True):
    """
    Read load data from OPSD time-series package version 2020-10-06.

    Parameters
    ----------
    years : None or slice()
        Years for which to read load data (defaults to
        slice("2018","2019"))
    fn : str
        File name or url location (file format .csv)
    countries : listlike
        Countries for which to read load data.
    powerstatistics: bool
        Whether the electricity consumption data of the ENTSOE power
        statistics (if true) or of the ENTSOE transparency map (if false)
        should be parsed.

    Returns
    -------
    load : pd.DataFrame
        Load time-series with UTC timestamps x ISO-2 countries
    """
    logger.info(f"Retrieving load data from '{fn}'.")

    pattern = 'power_statistics' if powerstatistics else '_transparency'
    pattern = f'_load_actual_entsoe_{pattern}'
    rename = lambda s: s[:-len(pattern)]
    date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)
    return (pd.read_csv(fn, index_col=0, parse_dates=[0], date_parser=date_parser)
            .filter(like=pattern)
            .rename(columns=rename)
            .dropna(how="all", axis=0)
            .rename(columns={'GB_UKM' : 'GB'})
            .filter(items=countries)
            .loc[years])


def consecutive_nans(ds):
    return (ds.isnull().astype(int)
            .groupby(ds.notnull().astype(int).cumsum()[ds.isnull()])
            .transform('sum').fillna(0))


def fill_large_gaps(ds, shift):
    """
    Fill up large gaps with load data from the previous week.

    This function fills gaps ragning from 3 to 168 hours (one week).
    """
    shift = Delta(shift)
    nhours = shift / np.timedelta64(1, 'h')
    if (consecutive_nans(ds) > nhours).any():
        logger.warning('There exist gaps larger then the time shift used for '
                       'copying time slices.')
    time_shift = pd.Series(ds.values, ds.index + shift)
    return ds.where(ds.notnull(), time_shift.reindex_like(ds))


def nan_statistics(df):
    def max_consecutive_nans(ds):
        return (ds.isnull().astype(int)
                  .groupby(ds.notnull().astype(int).cumsum())
                  .sum().max())
    consecutive = df.apply(max_consecutive_nans)
    total = df.isnull().sum()
    max_total_per_month = df.isnull().resample('m').sum().max()
    return pd.concat([total, consecutive, max_total_per_month],
                 keys=['total', 'consecutive', 'max_total_per_month'], axis=1)


def copy_timeslice(load, cntry, start, stop, delta):
    start = pd.Timestamp(start)
    stop = pd.Timestamp(stop)
    if start-delta in load.index and stop in load.index and cntry in load:
        load.loc[start:stop, cntry] = load.loc[start-delta:stop-delta, cntry].values


def manual_adjustment(load, powerstatistics):
    """
    Adjust gaps manual for load data from OPSD time-series package.

    1. For the ENTSOE power statistics load data (if powerstatistics is True)

    Kosovo (KV) and Albania (AL) do not exist in the data set. Kosovo gets the
    same load curve as Serbia and Albania the same as Macdedonia, both scaled
    by the corresponding ratio of total energy consumptions reported by
    IEA Data browser [0] for the year 2013.

    2. For the ENTSOE transparency load data (if powerstatistics is False)

    Albania (AL) and Macedonia (MK) do not exist in the data set. Both get the
    same load curve as Montenegro,  scaled by the corresponding ratio of total energy
    consumptions reported by  IEA Data browser [0] for the year 2016.

    [0] https://www.iea.org/data-and-statistics?country=WORLD&fuel=Electricity%20and%20heat&indicator=TotElecCons


    Parameters
    ----------
    load : pd.DataFrame
        Load time-series with UTC timestamps x ISO-2 countries
    powerstatistics: bool
        Whether argument load comprises the electricity consumption data of
        the ENTSOE power statistics or of the ENTSOE transparency map

    Returns
    -------
    load : pd.DataFrame
        Manual adjusted and interpolated load time-series with UTC
        timestamps x ISO-2 countries
    """

    if powerstatistics:
        if 'MK' in load.columns:
            if 'AL' not in load.columns or load.AL.isnull().values.all():
                load['AL'] = load['MK'] * (4.1 / 7.4)
        if 'RS' in load.columns:
            if 'KV' not in load.columns or load.KV.isnull().values.all():
                load['KV'] = load['RS'] * (4.8 / 27.)

        copy_timeslice(load, 'GR', '2015-08-11 21:00', '2015-08-15 20:00', Delta(weeks=1))
        copy_timeslice(load, 'AT', '2018-12-31 22:00', '2019-01-01 22:00', Delta(days=2))
        copy_timeslice(load, 'CH', '2010-01-19 07:00', '2010-01-19 22:00', Delta(days=1))
        copy_timeslice(load, 'CH', '2010-03-28 00:00', '2010-03-28 21:00', Delta(days=1))
        # is a WE, so take WE before
        copy_timeslice(load, 'CH', '2010-10-08 13:00', '2010-10-10 21:00', Delta(weeks=1))
        copy_timeslice(load, 'CH', '2010-11-04 04:00', '2010-11-04 22:00', Delta(days=1))
        copy_timeslice(load, 'NO', '2010-12-09 11:00', '2010-12-09 18:00', Delta(days=1))
        # whole january missing
        copy_timeslice(load, 'GB', '2009-12-31 23:00', '2010-01-31 23:00', Delta(days=-364))

    else:
        if 'ME' in load:
            if 'AL' not in load and 'AL' in countries:
                load['AL'] = load.ME * (5.7/2.9)
            if 'MK' not in load and 'MK' in countries:
                load['MK'] = load.ME * (6.7/2.9)
        copy_timeslice(load, 'BG', '2018-10-27 21:00', '2018-10-28 22:00', Delta(weeks=1))

    return load


if __name__ == "__main__":

    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_load_data')

    configure_logging(snakemake)

    config = snakemake.config
    powerstatistics = config['load']['power_statistics']
    interpolate_limit = config['load']['interpolate_limit']
    countries = config['countries']
    snapshots = pd.date_range(freq='h', **config['snapshots'])
    years = slice(snapshots[0], snapshots[-1])
    time_shift = config['load']['time_shift_for_large_gaps']

    load = load_timeseries(snakemake.input[0], years, countries, powerstatistics)

    if config['load']['manual_adjustments']:
        load = manual_adjustment(load, powerstatistics)

    logger.info(f"Linearly interpolate gaps of size {interpolate_limit} and less.")
    load = load.interpolate(method='linear', limit=interpolate_limit)

    logger.info("Filling larger gaps by copying time-slices of period "
                f"'{time_shift}'.")
    load = load.apply(fill_large_gaps, shift=time_shift)

    assert not load.isna().any().any(), (
        'Load data contains nans. Adjust the parameters '
        '`time_shift_for_large_gaps` or modify the `manual_adjustment` function '
        'for implementing the needed load data modifications.')

    load.to_csv(snakemake.output[0])

