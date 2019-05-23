#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

additional script for the wind_power_icing project, training individual regressors for each station
This script
* reads in the pre-prepared anonymized forecast data and station data,
* trains random forecast regressors on it.
* evalutes the forecasts and makes plots

Deviations from main script: for each station, an individual regressor is trained

uses: production curves computed with compute_production_curves.py

Note: this script is designed to work on Unix-like systems.
it uses os.system commands that very probably dont work on windows


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

name = 'trained_per_station' # for plot names

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

res_all_lead_times = []
lead_hours = np.arange(0,42+1,1)
for lead_hour in lead_hours:

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

    # now train for each station individually
    # for this, first rename df_test and df_train
    df_test_allstations = df_test
    df_train_allstations = df_train
    res = []
    for station in stations:

        # select the right station, using the new pands query syntax
        df_train = df_train_allstations.query('station == @station')
        df_test = df_test_allstations.query('station == @station')
        X_train = df_train[input_vars]
        y_train = df_train[target_var]
        X_test = df_test[input_vars]
        y_test = df_test[target_var]
        # now train a couple of estimators  with the configuration that came out of the tuning
        N_trains = 10

        # here we compute the esemble mean prediction immediately, otherwise it is a bit tricky with the split up
        # stations
        res_ensemble = []
        for i in range(N_trains):

            # we have to define the pipeline in the loop, then it is initialized in every iteration (which we want,
            # otherwise we train the already trained model)
            # here I copied the output from the tuning script, therefore all params are explicitely set, even though
            # most of them use the default values

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
            res_ensemble.append(y_test_predicted)

        ensmean = np.mean(res_ensemble, axis=0)

        # put the predictions, the true values and some more information in a dataframe
        # specifically, we want to have all the target_vars because we need them for evaluation
        # additionally, we need the windspeed forecast in order to compute the forecast potential production
        _df= pd.DataFrame({'truth':y_test, 'prediction':ensmean,
                           'station':df_test['station'], 'power_target':df_test['power_target'],
                           'potpower_target':df_test['potpower_target'], 'ws':df_test['ws'],
                           'ploss_abs_target':df_test['ploss_abs_target']})

        # persistence forecast
        _df['persistence'] = X_test['ploss']  ## note: this does not work when target is ploss_abs!!

        res.append(_df)

    # combine all members
    res = pd.concat(res)


    res['month'] = res.index.month

    res['lead_hour'] = lead_hour

    res_all_lead_times.append(res)



final_df = pd.concat(res_all_lead_times)



final_df['mse'] = (final_df['truth'] - final_df['prediction']) ** 2
final_df['abse'] = (final_df['truth'] - final_df['prediction']).abs()
final_df['persistence_mse'] = (final_df['persistence'] - final_df['truth'])**2
final_df['persistence_abse'] = (final_df['persistence'] - final_df['truth']).abs()

final_df['abse_diff'] = final_df['abse'] - final_df['persistence_abse']
final_df['mse_diff'] = final_df['mse'] - final_df['persistence_mse']

#%%------------------- now make plots
sns.set_context('notebook', font_scale=1.0, rc={"lines.linewidth": 3.})
sns.set_palette('colorblind')
plt.rcParams['savefig.bbox'] = 'tight'
# make plots lead_hour vs X

# if wen wanto to use the built-in bootstrapping methods form seaborn for computing the uncertainty
# in RMSE, we canot use the standard method, because one cannot average RMSE.
# however, we can do it vie the MSE: we put the MSE in the plitting fucntion, and then we dont use the standrad
# estimator, but a costum one, that computes the squareroot of the mean. we can pass it in as a lambda function
ci=90

plt.figure(figsize=(7,4))
sns.lineplot('lead_hour', 'abse', data=final_df, label='random forest', ci=ci)
sns.lineplot('lead_hour', 'persistence_abse', data=final_df, label='persistence', ci=ci)
sns.lineplot('lead_hour', 'abse_diff', data=final_df, label='diff', ci=ci)
plt.axhline(0)
sns.despine()
plt.legend()
plt.ylabel('MAE relative power loss [%]')
plt.savefig('plots/lead_hour_vs_abserror_'+name+'.svg')

plt.figure(figsize=(7,4))
sns.lineplot('lead_hour', 'mse', data=final_df,estimator=lambda x:np.sqrt(np.mean(x)), label='random forest', ci=ci)
sns.lineplot('lead_hour', 'persistence_mse', data=final_df,estimator=lambda x:np.sqrt(np.mean(x)), label='persistence', ci=ci)
sns.despine()
plt.ylabel('RMSE relative power loss [%]')
plt.legend()
plt.savefig('plots/lead_hour_vs_rmse_'+name+'.svg')

plt.figure(figsize=(7,4))
sns.lineplot('lead_hour', 'mse', data=final_df,estimator=lambda x:np.sqrt(np.mean(x)), ci=ci,
             hue='station')
sns.despine()
plt.ylabel('RMSE relative power loss [%]')
plt.legend()
plt.savefig('plots/lead_hour_vs_rmse_station_'+name+'.svg')


plt.figure(figsize=(7,4))
sns.lineplot('lead_hour', 'abse', data=final_df, hue='station', ci=ci)
sns.despine()
plt.savefig('plots/lead_hour_vs_abserror_stations_'+name+'.svg')



#%%-------------------
# scatter and barplots for individual lead_hours
for lead_hour in lead_hours:

    df_sub = final_df[final_df['lead_hour']==lead_hour]
    pred = df_sub['prediction']
    truth = df_sub['truth']
    mse_test = metrics.mean_squared_error(truth, pred)
    rmse_test = np.sqrt(mse_test)
    print('mse on test data:', mse_test)
    print('rmse on test data', rmse_test)

    r2 = np.corrcoef(truth, pred)[0, 1] ** 2

    plt.figure()
    sns.regplot(pred, truth, ci=ci)
    sns.despine()
    plt.ylabel('ploss test')
    plt.xlabel('ploss test predicted')
    plt.xlim((-5,100))
    plt.ylim((-5, 100))
    plt.text(0.1, 0.9, 'R^2={:.2f}'.format(r2), transform=plt.gca().transAxes)
    plt.text(0.1, 0.85, 'RMSE={:.2f}'.format(rmse_test), transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, 'N={:.2f}'.format(len(y_test)), transform=plt.gca().transAxes)
    plt.savefig('plots/'+target_var + '_vs_' + target_var + '_predicted_on_test_set_lead_hour' + \
                str(lead_hour) + '_ensmean_'+name+'.svg')



    plt.figure()
    # two plit both mse and psersistenc_mse in one barplot, we need to rearrange
    # and melt the dataframe
    subsub_df = df_sub[['station','mse','persistence_mse']].rename(columns={'mse':'rf_mse'})
    subsub_df = subsub_df.melt(id_vars='station',var_name='fc_type', value_name='mse')
    sns.barplot('station','mse',data=subsub_df,hue='fc_type', estimator=lambda x:np.sqrt(np.mean(x)), ci=ci)
    sns.despine()
    plt.ylabel('RMSE relative production loss [%]')
    plt.savefig('plots/barplot_station_vs_rmse'+str(lead_hour) + '_'+name+'.svg')

    plt.figure()
    # two plit both mse and psersistenc_mse in one barplot, we need to rearrange
    # and melt the dataframe
    subsub_df = df_sub[['station', 'abse', 'persistence_abse', 'abse_diff']].rename(
        columns={'abse': 'random forest',   'persistence_abse':'persistence', 'abse_diff':'diff'})
    subsub_df = subsub_df.melt(id_vars='station', var_name='fc_type', value_name='abse')
    sns.barplot('station', 'abse', data=subsub_df, hue='fc_type',ci=ci)
    sns.despine()
    plt.ylabel('MAE relative production loss [%]')
    plt.savefig('plots/barplot_station_vs_abse' + str(lead_hour) + '_'+name+'.svg')


    plt.figure()
    sns.barplot('month','mse',data=df_sub, estimator=lambda x:np.sqrt(np.mean(x)), ci=ci)
    sns.despine()
    plt.ylabel('RMSE forecast production loss [%]')
    plt.savefig('plots/barplot_month_vs_abse'+str(lead_hour) + '_'+name+'.svg')

    plt.close('all')


#%%-------------------
# plot error vs real power.

lead_hour = 42
plt.figure()
sns.regplot('power_target','abse', data = final_df[final_df['lead_hour']==lead_hour], ci=ci)
sns.despine()
plt.savefig('plots/power_vs_error_leadhour'+str(lead_hour)+'_'+name+'.svg')

#%%-------------------



# compute the potential prodcution based on the windspeed forecast
# since the params very by station, we loop over all samples and compute it individually
pot_power_pred = np.array([production_curve(ws,**curve_params[station]) for ws,station in
                           zip(final_df['ws'], final_df['station'])])

# add to dataframe
final_df['pot_power_prediction'] = pot_power_pred
final_df['power_prediction'] = (1-final_df['prediction']/100) * final_df['pot_power_prediction']
final_df['mse_power'] = (final_df['power_target'] - final_df['power_prediction'])**2
final_df['abse_power'] = (final_df['power_target'] - final_df['power_prediction']).abs()
final_df['power_prediction_persistence'] = (1-final_df['persistence']/100) * final_df['pot_power_prediction']
final_df['mse_power_persistence'] = (final_df['power_prediction_persistence'] - final_df['power_target'])**2
final_df['abse_power_persistence'] = (final_df['power_prediction_persistence'] - final_df['power_target']).abs()
final_df['mse_power_only_nwp'] = (final_df['power_target'] - final_df['pot_power_prediction'])**2
final_df['abse_power_only_nwp'] = (final_df['power_target'] - final_df['pot_power_prediction']).abs()

final_df['abse_power_diff'] = final_df['abse_power'] - final_df['abse_power_persistence']

plt.figure(figsize=(7,4))
sns.lineplot('lead_hour', 'mse_power', data=final_df,estimator=lambda x:np.sqrt(np.mean(x)), label='random forest',
             ci=ci)
sns.lineplot('lead_hour', 'mse_power_persistence', data=final_df,estimator=lambda x:np.sqrt(np.mean(x)),
             label='persistence', ci=ci)
sns.lineplot('lead_hour', 'mse_power_only_nwp', data=final_df,estimator=lambda x:np.sqrt(np.mean(x)), label='only NWP',
             ci=ci)
sns.despine()
plt.ylabel('RMSE power prediction [kW]')
plt.legend()
plt.savefig('plots/lead_hour_vs_rmse_power_'+name+'.svg')

plt.figure(figsize=(7,4))
sns.lineplot('lead_hour', 'abse_power', data=final_df, label='random forest',
             ci=ci)
sns.lineplot('lead_hour', 'abse_power_persistence', data=final_df , label='persistence', ci=ci)
sns.lineplot('lead_hour', 'abse_power_diff', data=final_df, label='diff', ci=ci)
sns.lineplot('lead_hour', 'abse_power_only_nwp', data=final_df, label='only NWP', ci=ci)
sns.despine()
plt.ylabel('MAE power prediction [kW]')
plt.legend()
plt.axhline(0)
plt.savefig('plots/lead_hour_vs_abse_power_'+name+'.svg')


# now compute error of potential production loss forecast
final_df['abs_ploss_pred'] = final_df['pot_power_prediction'] * final_df['prediction']/100
final_df['mse_abs_power_loss'] = (final_df['ploss_abs_target'] - final_df['abs_ploss_pred'])**2
final_df['abse_abs_power_loss'] = (final_df['ploss_abs_target'] - final_df['abs_ploss_pred']).abs()
final_df['abs_ploss_persistence'] = final_df['pot_power_prediction'] * final_df['persistence']/100
final_df['mse_abs_power_loss_persistence'] = (final_df['ploss_abs_target'] - final_df['abs_ploss_persistence'])**2
final_df['abse_abs_power_loss_persistence'] = (final_df['ploss_abs_target'] - final_df['abs_ploss_persistence']).abs()
final_df['abse_abs_power_diff'] = final_df['abse_abs_power_loss'] - final_df['abse_abs_power_loss_persistence']


plt.figure(figsize=(7,4))
sns.lineplot('lead_hour', 'mse_abs_power_loss', data=final_df,estimator=lambda x:np.sqrt(np.mean(x)),
             label='random forest', ci=ci)
sns.lineplot('lead_hour', 'mse_abs_power_loss_persistence', data=final_df,estimator=lambda x:np.sqrt(np.mean(x)),
             label='persistence', ci=ci)
sns.despine()
plt.ylabel('RMSE power loss prediction [kW]')
plt.legend()
plt.savefig('plots/lead_hour_vs_rmse_power_loss_'+name+'.svg')

plt.figure(figsize=(7,4))
sns.lineplot('lead_hour', 'abse_abs_power_loss', data=final_df, label='random forest', ci=ci)
sns.lineplot('lead_hour', 'abse_abs_power_loss_persistence', data=final_df, label='persistence', ci=ci)
sns.lineplot('lead_hour', 'abse_abs_power_diff', data=final_df, label='diff', ci=ci)
sns.despine()
plt.ylabel('MAE power loss prediction [kW]')
plt.axhline(0)
plt.legend()
plt.savefig('plots/lead_hour_vs_abse_power_loss_'+name+'.svg')
