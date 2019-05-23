#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

main script for the wind_power_icing project.
This script
* reads in the pre-prepared anonymized forecast data and station data,
* trains random forecast regressors on it.
* evalutes the forecasts and makes plots

uses: production curves computed with compute_production_curves.py

Note: this script is designed to work on Unix-like systems.
it uses os.system commands for creating output directories that very probably dont work on windows


@author: sebastian
"""
import pickle
import os

import pandas as pd
import seaborn as sns
from pylab import plt
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

stations = ['A', 'B', 'C', 'D', 'E', 'F']

os.system('mkdir -p plots')

name='standard'

# load the parametes for the production curves
curve_params = pickle.load(open('optimal_sigmoid_params.pkl','rb'))

def production_curve(wspd, a, b, y_scale):
    return y_scale *1.0 / (1.0 + np.exp(-a*(wspd-b)))

def read_station_data():
    """ read anonymized station data """

    res = []
    for season in ['201312-201402', '201409-201501']:

        ## read production data. each station has its own file
        for station in stations:
            ifile = 'data/anonymized/' + station + '_' + season + '_power_obs.txt'

            _df = pd.read_csv(ifile, sep=';', skipinitialspace=True)

            # the time column is called "# ymdh", which is unhandy. rename it to "date"
            _df = _df.rename(columns={'# ymdh': 'datetime'})

            # add a column with station name
            _df['station'] = station

            res.append(_df)

    station_df = pd.concat(res)

    # compute potential production
    station_df['potpower'] = np.array([production_curve(ws, **curve_params[station]) for ws, station in
              zip(station_df['ff'], station_df['station'])])


    station_df['ploss_abs'] = station_df['potpower'] - station_df['power']
    station_df['ploss_abs'] = station_df['ploss_abs'].clip(lower=0)
    station_df['ploss'] = (station_df['ploss_abs'] / station_df['potpower'] * 100).copy()
    station_df['ploss'] = station_df['ploss'].clip(lower=0,upper=100)


    # reformat the dates, and set them as index
    station_df['datetime'] = pd.to_datetime(station_df['datetime'], format='%Y%m%d%H')
    station_df.index = pd.DatetimeIndex(station_df['datetime'])

    return station_df


def read_fc_data():
    """read forecast data from .h5 file"""

    fc_data_path='data/anonymized'

    fc_file1 = fc_data_path+'/anonymized_forecast_data_winter2013_2014_from_dat.h5'
    fc_df1 = pd.read_hdf(fc_file1)
    fc_file2 = fc_data_path+'/anonymized_forecast_data_winter2014_2015_from_dat.h5'
    fc_df2 = pd.read_hdf(fc_file2)

    fc_df = pd.concat([fc_df1, fc_df2])
    # convert K to degree C
    fc_df['tk'] = fc_df['tk'] - 273.15

    # restrict to forecasts initialized at 06:00, because these are the forecasts that are long
    hour = 6
    print('only retaining forecast initialized at hour ', hour)
    fc_df = fc_df[pd.DatetimeIndex(fc_df.date_init).hour == hour]

    return fc_df


def combine_fc_and_station_data(lead_hour):
    '''
    combine forecast and station data
    this is different for each lead_hour, because the station data needs to be shifted
    accordingly (we need to have the station data that is valid at the initialization time of the forecast)

    :param lead_hour:
    :return: combinded dataframe
    '''

    fc = fc_df[fc_df['lead_hour'] == lead_hour]

    # set date_valid as index
    fc = fc.set_index('date_valid')

    # combine the obs and forecast dataframes
    # at the moment, the foreccast data and the target (ploss) have the right time.
    # however ,also the observations at the stations are at the same date as the forecast_valid at the moment
    # In an operational setting, we would not have these measuremente. However, we would have the measurements
    # up to date_init. To reflect this, we remove the raw measurements, and replace them by ones that are shifted
    # in time.

    res = []
    for station in stations:
        _fc = fc[fc['station'] == station]
        _obs = station_df[station_df['station'] == station]
        _target = _obs[['ploss', 'ploss_abs', 'power','potpower']]
        # rename the target to not confuse it with the measurements
        # at initialization time. the target vars are these that contain the measurements at valid time,
        # and we will use them for evalution of the forecasts
        _target = _target.rename(columns={'ploss': 'ploss_target', 'ploss_abs': 'ploss_abs_target',
                                          'power':'power_target','potpower':'potpower_target'})
        # in order to avoid a second station column, reomve
        # the station column from _obs
        _obs = _obs.drop(columns='station')

        # _obs has as index datetime, but also as variable column. this gets confusing, therefore delete the
        # variable column
        _obs = _obs.drop(columns='datetime')

        # there is some missing data in the obs, this is inconvenient
        # therefore, fill it up with nans
        # first create a daterange with hourly steps, this are the "expected" dates
        full_dates = pd.DatetimeIndex(pd.date_range(_obs.index[0], _obs.index[-1], freq='h'))
        print('filling up ', len(full_dates) - len(_obs), ' values in obs')
        # now reindex _obs , filling up all missing dates with NAN
        _obs = _obs.reindex(full_dates, fill_value=np.NaN)

        # the observations should now have hourly timestep
        # verify this
        if not _obs.index.to_series().diff()[1:].unique() / np.timedelta64(1, 'h') == 1:
            raise ValueError('observations should have hourly timesteps!!')

        # now shift the observations to correspond to the initial time
        # now shift backward (minus lead_hour steps)
        _obs = _obs.shift(-lead_hour)
        merged = pd.concat([_obs, _fc, _target], axis=1, join='inner')

        res.append(merged)

    merged = pd.concat(res)

    # this is now our datset ready for training!
    # for convenience, lets call it df, and drop all NaN
    df = merged.dropna()


    return df


def train_test_split_chunks(df, chunk_length=10):
    """
    splitting into train and test set. the train set will also contain the dev set.
    :param df:
    :param chunk_length: lenght ot 1 chunk of samples
    :return: train_df, test_df
    """

    test_fraction = 0.2

    # compute number of chunks in the whole dataset
    n_samples = len(df)
    n_chunks = n_samples//chunk_length  #// makes a floor division


    n_chunks_test = int(n_chunks * test_fraction)
    n_chunks_train = n_chunks - n_chunks_test
    chunk_idcs_all = np.arange(n_chunks)
    # to get the same for all lead_times, we use a seed for the random generator
    np.random.seed(10)
    # now get a random selection of chunk indices for the train set
    chunk_idcs_train = np.random.choice(chunk_idcs_all, size=n_chunks_train, replace=False)
    # now get the complement, these are then the indices for the test set
    chunk_idcs_test = chunk_idcs_all[~np.isin(chunk_idcs_all, chunk_idcs_train)]

    # do some sanity checks
    assert(np.all(np.isin(chunk_idcs_all, np.concatenate([chunk_idcs_train, chunk_idcs_test]))))
    assert(~np.all(np.isin(chunk_idcs_test,chunk_idcs_train)))

    # now convert the chunk idcs to dataset indices.
    # for this first creates a nested list containing all indices of the dataset, arranged
    # in the appropriate way
    idcs_chunks = np.array([np.arange(i*chunk_length,(i+1)*chunk_length) for i in range(n_chunks)])
    # this returns a 2d-array of indices (n_chunks_train x chunk_length, therefore we flatten
    # it immediately after selecting
    idcs_train = idcs_chunks[chunk_idcs_train].flatten()
    idcs_test = idcs_chunks[chunk_idcs_test].flatten()

    # sanity checks
    assert(np.all(~np.isin(idcs_train,idcs_test)))
    assert(np.all(~np.isin(idcs_test,idcs_train)))

    df_train = df.iloc[idcs_train]
    df_test = df.iloc[idcs_test]

    print('train set has ',len(df_train), 'samples')
    print('test set has ',len(df_test), 'samples')

    # now df_train and df_test are internally copys of df. This sometimes causes problems when changing
    # the data later (pandas warning "A value is trying to be set on a copy of a slice from a DataFrame.
    # Try using .loc[row_indexer,col_indexer] = value").
    # to avoid this, we just explicitely copy the dataframes. since we dont have memory constraints we can just
    # do this
    df_train = df_train.copy()
    df_test = df_test.copy()

    return df_train, df_test


fc_df = read_fc_data()
station_df = read_station_data()

# get a list with all forecast variables. these are the keys of the forecast
# df, but without the "meta"-info date_valid, date_init, ead_hour and station
fc_vars = fc_df.drop(columns=['date_valid', 'date_init', 'lead_hour', 'station']).keys()
obs_vars = station_df.drop(columns=['datetime', 'station']).keys()

input_vars = list(fc_vars) + list(obs_vars)
print('using the following input vats', input_vars)
target_var = 'ploss_target'
print('using ', target_var,' as target var')

lead_hour=42

# arrange the data for this lead time
df = combine_fc_and_station_data(lead_hour)

df_train, df_test = train_test_split_chunks(df)

# bias correction of windspeed forecast.
# 'ws' is the windspeed forecast, and 'ff' is the measured windspeed at the stations
# we do the bias correction for each station, and obviosuly compute the bias only
# over the train set, and then apply it to the test set
wspd_obs_mean_per_station = df_train.groupby('station')['ff'].mean()
wspd_fc_mean_per_station = df_train.groupby('station')['ws'].mean()
bias_per_station = wspd_fc_mean_per_station - wspd_obs_mean_per_station
# now do the correction.]
# for this we need to expand the bias, e.g. make a list that has the correct bias (=correct station) for each sample
expanded_bias_train = [bias_per_station[station] for station in df_train['station']]
expanded_bias_test = [bias_per_station[station] for station in df_test['station']]
print('applying bias correction wo windspeed')
df_train['ws'] = df_train['ws'] - expanded_bias_train
df_test['ws'] = df_test['ws'] - expanded_bias_test

X_train = df_train[input_vars]
y_train = df_train[target_var]
X_test = df_test[input_vars]
y_test = df_test[target_var]


pipe = Pipeline(memory=None,
                steps=[('normalization', StandardScaler(copy=True, with_mean=True, with_std=True)),
                       ('rf', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=250,
                                                    max_features='auto', max_leaf_nodes=None,
                                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                                    min_samples_leaf=1, min_samples_split=2,
                                                    min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
                                                    oob_score=False, random_state=None, verbose=0,
                                                    warm_start=False))])


pipe.fit(X_train, y_train)

# now use this estimator on the test data
y_test_predicted = pipe.predict(X_test)

# put the predictions, the true values and some more information in a dataframe
# specifically, we want to have all the target_vars because we need them for evaluation
# additionally, we need the windspeed forecast in order to compute the forecast potential production
df= pd.DataFrame({'truth':y_test, 'prediction':y_test_predicted,
                   'station':df_test['station'], 'power_target':df_test['power_target'],
                   'potpower_target':df_test['potpower_target'], 'ws':df_test['ws'],
                   'ploss_abs_target':df_test['ploss_abs_target']})



rf = pipe.named_steps['rf']

var_names = X_train.keys()
var_importances = rf.feature_importances_
assert(len(var_names) == len(var_importances))


#%%------------------- now make plots
sns.set_context('notebook', font_scale=1.0, rc={"lines.linewidth": 3.})
sns.set_palette('colorblind')
plt.rcParams['savefig.bbox'] = 'tight'
# make plots lead_hour vs X



plt.figure(figsize=(7,4))
sns.barplot(var_names, var_importances * 100, color=sns.color_palette()[0])
plt.ylabel('importance [%]')
sns.despine()
plt.xticks(rotation=90)
plt.savefig('plots/feature_importance.pdf')

