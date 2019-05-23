#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:49:37 2019

tuning script. Tries out different methods and saves the best one.

currently configured to fun 32 parallel jobs, reduce n_jobs if you want to run it on a laptop.

@author: sebastian, March 2019
"""

import pickle

import pandas as pd
import seaborn as sns
from pylab import plt
import numpy as np
from sklearn import svm, linear_model, ensemble, preprocessing, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import neural_network



n_jobs = 32  # maximum number of parallesl jobs in the gridsearch
stations = ['A', 'B', 'C', 'D', 'E', 'F']
lead_hour=42

def read_station_data():
    """ read station data """

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

    # compute absolute production loss
    station_df['ploss_abs'] = station_df['ploss'] * station_df['potpower']/100.

    # reformat the dates, and set them as index
    station_df['datetime'] = pd.to_datetime(station_df['datetime'], format='%Y%m%d%H')
    station_df.index = pd.DatetimeIndex(station_df['datetime'])

    return station_df


def read_fc_data():
    """read forecast data from .h5 file"""

    #fc_data_path='/climstorage/sebastian/wind_power_icing/data/'
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
        # and we will use them for evalution of the ofrecasts
        _target = _target.rename(columns={'ploss': 'ploss_target', 'ploss_abs': 'ploss_abs_target',
                                          'power':'power_target','potpower':'potpower_target'})
        # in order to avoid a second station column, reomve
        # the station column from _obs
        _obs = _obs.drop(columns='station')

        # _obs has as index datetime, but also as variable column. this gets confusing, therefore delete the
        # variable column
        _obs = _obs.drop(columns='datetime')

        # there is some missing data in the obs, this is inconveneint
        # therefore, fill it up with nans
        # first crreate a daterange with hourly steps, this are the "expected" dates
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
    # to get the same test train split in the tuning and in the main training afterwards,
    # we use a fixed seed
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
    # it immediately after seledting
    idcs_train = idcs_chunks[chunk_idcs_train].flatten()
    idcs_test = idcs_chunks[chunk_idcs_test].flatten()

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

# plot the autocorrelation of the target variable. This will help us
# in deciding how to split up the data into test and training data

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

df = combine_fc_and_station_data(lead_hour)

df_train, df_test = train_test_split_chunks(df)





# plot autocorrelation
y = df[target_var]
plt.figure()
plt.acorr(y, maxlags=59)
plt.xlabel('lag [h]')
plt.ylabel('autocorrelation hourly '+target_var)
plt.savefig('acorr_hourly_'+target_var+'.svg')


y_daily = y.resample('1d').mean().dropna()
plt.figure()
plt.acorr(y_daily, maxlags=30)
plt.xlabel('lag [days]')
plt.ylabel('autocorrelation daily mean '+target_var)
plt.savefig('acorr_dailymean_'+target_var+'.svg')

y_daily = y.iloc[::24]
plt.figure()
plt.acorr(y_daily, maxlags=30)
plt.xlabel('lag [days]')
plt.ylabel('autocorrelation daily snapshot '+target_var)
plt.savefig('acorr_dailysnapshot_'+target_var+'.svg')


input_vars = list(fc_vars) + list(obs_vars)

target_var = 'ploss_target'



X_train = df_train[input_vars]
y_train = df_train[target_var]

X_test = df_test[input_vars]
y_test = df_test[target_var]
# now we test different methods. for reach method, we create a pipeline consisting
# of a StandardScaler that scales each input variable to unit mean and variance, and a classifier
# for each classifier, we define a paramtergrid. Then we do a full gridsearchover this parametergrid,
# using 3-KFold cross validation

configs = [
    {'name':'linear',
     'pipe':Pipeline(steps=[('normalization',preprocessing.StandardScaler()),
                            ('lm',linear_model.LinearRegression())]),
     'param_grid': {}},

    {'name':'svr',
     'pipe':Pipeline(steps=[('normalization',preprocessing.StandardScaler()),
                            ('svr',svm.SVR())]),
     'param_grid': {'svr__kernel':['rbf','linear'],
                    'svr__C':[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10],
                    }
     },

    {'name':'rf',
     'pipe':Pipeline(steps=[('normalization',preprocessing.StandardScaler()),
                            ('rf', ensemble.RandomForestRegressor())]),
     'param_grid': {'rf__n_estimators':[10,50,100,500,1000],
                    'rf__max_depth':[None]+list(np.arange(1,101)*10), # not sure what to set here best
                    }
     },
    {'name': 'nn',
     'pipe': Pipeline(steps=[('normalization', preprocessing.StandardScaler()),
                             ('nn', neural_network.MLPRegressor(solver='adam'))]),
     'param_grid': {'nn__activation': ['relu', 'logistic'],
                    'nn__learning_rate_init': [0.001,0.003,0.01,0.03],
                    'nn__hidden_layer_sizes': [ 2,4,8,10,20,50,100 ]  # we only use 1-layer networks
                    }
     },

]

#%%
regressors = []
for config in configs:
    print('fitting ', config['name'])

    # do a gridsearch with the standard crossvalidation strategy, which is a 3 Kfold split.
    reg = GridSearchCV(config['pipe'], param_grid=config['param_grid'],
                           n_jobs=n_jobs, scoring='neg_mean_squared_error', verbose=2)
    reg.fit(X_train, y_train)

    preds = reg.predict(X_train)

    regressors.append({'name': config['name'], 'cv': reg, 'predictions': preds})

best_score_per_regression_type = []
for reg in regressors:
    # get the score of the best estimator
    # the best_score attribute contains the score of the best estimator, averaged
    # over the 3 folds
    print (reg['cv'].best_score_)
    # conver tot RMSE
    mse =  -reg['cv'].best_score_
    rmse = np.sqrt(mse)
    print(rmse)
    best_score_per_regression_type.append(mse)

# now select the best overall method
idx_best_regression_type = np.argmin(best_score_per_regression_type)

best_cv = regressors[idx_best_regression_type]['cv']
print('best found estimator:')
print(best_cv.best_estimator_)
print('mse in training cross-validation:',-best_cv.best_score_)
print('rmse in training cross-validation:',np.sqrt(-best_cv.best_score_))


#%%
final_estimator = best_cv.best_estimator_

# save the final estimator
estimator_file = 'wind_power_icing_final_estimator_from_gridsearch.pkl'
pickle.dump(final_estimator, open(estimator_file,'wb'))
# save the configuration as text file
print(final_estimator, file=open('wind_power_icing_final_estimator_config.txt','w'))

# now use this estimator on the test data
y_test_predicted = final_estimator.predict(X_test)

mse_test = metrics.mean_squared_error(y_test, y_test_predicted)
rmse_test = np.sqrt(mse_test)
print('mse on test data:', mse_test)
print('rmse on test data', rmse_test)

r2 = np.corrcoef(y_test_predicted, y_test)[0,1]**2

plt.figure()
sns.regplot(y_test_predicted, y_test)
plt.ylabel('ploss test')
plt.xlabel('ploss test predicted')
plt.text(0.1,0.9,'R^2={:.2f}'.format(r2), transform = plt.gca().transAxes)
plt.text(0.1,0.85,'RMSE={:.2f}'.format(rmse_test), transform = plt.gca().transAxes)
plt.text(0.1,0.8,'N={:.2f}'.format(len(y_test)), transform = plt.gca().transAxes)
plt.savefig(target_var+'_vs_'+target_var+'_predicted_on_test_set_lead_hour'+str(lead_hour)+'.svg')
