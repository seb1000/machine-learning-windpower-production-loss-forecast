#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts uses the station data to compute production curves (production as a function of windspeed)

@Author: Sebastian Scher, March 2019
"""

import pickle
import os

import pandas as pd
import seaborn as sns
from pylab import plt
import numpy as np

from scipy.optimize import curve_fit

os.system('mkdir -p plots')
stations = ['A', 'B', 'C', 'D', 'E', 'F']
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


station_df = read_station_data()




# select only warm days

data = station_df.query('temp>5')

plt.figure()
sns.scatterplot('ff','power', data = data, hue='station')


# now fit a sigmoid function to the data for each staton


def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))



optimal_simgmoid_params = {}

for station in station_df['station'].unique():
    sub = data[data['station'] == station]
    X, y = sub['ff'], sub['power']

    y_scale = y.max()

    y_scaled = y / y_scale

    # using the curve fit function from scipy did not really work (dont know why, probably because the
    # 2 params for the simgmoid function have very different sensitivity)
    # we replace it by a brute-force full grid search over a an b
    # popt, pcov = curve_fit(fsigmoid, X, y_scale, method='trf', ftol=1e-12, p0=[0.8,7],
    #                        xtol=1e-12, gtol=1e-30,max_nfev=10000*len(y),
    #                        bounds=([0.7, 6.],[0.9, 9.]), verbose=2, loss='cauchy',x_scale=[10,1]
    #                        )
    a_vals = np.linspace(0.7,1,100)
    b_vals = np.linspace(6, 10, 100)
    res = []
    # try out all combinations af a and b
    for a in a_vals:
        print(a)
        for b in b_vals:
            pred = fsigmoid(X, a, b)
            # compute mse
            res.append(np.mean((pred - y_scaled)**2))

    res = np.array(res)
    # get the combination with lowest mse
    best_idx = np.argmin(res)
    a_b_comb = [(a,b) for a in a_vals for b in b_vals]
    a,b = a_b_comb[best_idx]

    optimal_simgmoid_params[station] = {'a':a,'b':b,'y_scale':y_scale}



sns.set_context('notebook', font_scale=1.0, rc={"lines.linewidth": 1.5})
sns.set_palette('colorblind')
plt.rcParams['savefig.bbox'] = 'tight'
plt.figure()
for i,station in enumerate(data['station'].unique()):

    p = optimal_simgmoid_params[station]
    xvals = np.linspace(0,20,1000)
    ypred = fsigmoid(xvals, p['a'],p['b']) * p['y_scale']

    sub = data[data['station'] == station]
    X, y = sub['ff'], sub['power']

    color = sns.color_palette()[i]
    plt.plot(xvals,ypred, color=color, alpha=0.8)
    plt.scatter(X,y, color=color,label=station, s=4,alpha=0.6, edgecolors='face')

plt.ylabel('production [kW]')
plt.xlabel('windspeed [m/s]')
sns.despine()
plt.legend()
plt.savefig('plots/potential_production_fit.pdf')


pickle.dump(optimal_simgmoid_params, open('optimal_sigmoid_params.pkl','wb'))
