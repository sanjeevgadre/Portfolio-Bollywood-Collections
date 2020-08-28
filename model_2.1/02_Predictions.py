#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:04:09 2020

@author: sanjeev
"""
#%% Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle

#%% Get Data and Model Params
movie_master = pd.read_pickle('../data/movie_master_en.pkl')
cpi_master = pd.read_csv('../data/CPI.csv')
weekly_master = pd.read_hdf('../data/weekly_master.h5')


# Adjusting the first week revenue to account for entertainment and service tax
fwr = movie_master['india-first-week'] * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

# Getting the run length of films
df = weekly_master.groupby('movie_id').sum().iloc[:, 3:]
runlen = df.apply(lambda x: np.count_nonzero(x), axis = 1)
runlen.reset_index(drop = True, inplace = True)

# Getting exhibitor share
exsh = movie_master['india-nett-gross'] - movie_master['india-distributor-share']
exsh[1094] = 1             # hack

# Params for best fits
with open('../model_1/runlen_best_param.pkl', 'r+b') as handle:
    runlen_best_param = pickle.loads(handle.read())

with open('./totrev_best_param.pkl', 'r+b') as handle:
    totrev_best_param = pickle.loads(handle.read())

# with open('./exsh_best_param.pkl', 'r+b') as handle:
#     exsh_best_param = pickle.loads(handle.read())

# Set up Train and Test indices
train_idx, test_idx = train_test_split(movie_master.index, random_state = 1970)


#%% Predicting the Run Length
print('RUN LENGTH PREDICTION MODEL PERFORMANCE')

X_train = movie_master.loc[train_idx, ['release_year', 'screens']]
X_train = pd.concat([X_train, fwr[train_idx]], axis = 1)
X_train.columns = ['release_year', 'screens', 'fwr']
Y_train = runlen[train_idx]

gb_est_runlen = GradientBoostingRegressor(random_state = 1970).set_params(**runlen_best_param)
gb_mod_runlen = gb_est_runlen.fit(X_train, Y_train)

X_test = movie_master.loc[test_idx, ['release_year', 'screens']]
X_test = pd.concat([X_test, fwr[test_idx]], axis = 1)
X_test.columns = ['release_year', 'screens', 'fwr']
Y_test = runlen[test_idx]

Y_test_hat = gb_mod_runlen.predict(X_test)
#Y_test_hat = np.floor(Y_test_hat)

for i in range(len(Y_test_hat)):
    if Y_test_hat[i] < 1: Y_test_hat[i] = 1
    if Y_test_hat[i] > 11: Y_test_hat[i] = 11

err_mae = np.abs(Y_test - Y_test_hat)/Y_test

intervals = np.arange(0.25, 0.56, 0.1)
for interval in intervals:
    cnt = len([x for x in err_mae if x < interval])
    cnt = 100*cnt/len(err_mae)
    print('Percentage of estimates for test set that are off by less than %.0f%% from true value: %.2f' % (100*interval, cnt))
    
runlen_test_hat = pd.Series(Y_test_hat, name = 'run_len', index = test_idx)
    
#%% Predicting Total Revenue
print('TOTAL NETT GROSS PREDICTION MODEL PERFORMANCE')

X_train = movie_master.loc[train_idx, ['runtime', 'screens']]
X_train = pd.concat([X_train, fwr[train_idx], runlen[train_idx]], axis = 1)
X_train.columns = ['runtime', 'screens', 'fwr', 'runlen']
Y_train = movie_master.loc[train_idx, 'india-nett-gross']

gb_est_totrev = GradientBoostingRegressor(random_state = 1970).set_params(**totrev_best_param)
gb_mod_totrev = gb_est_totrev.fit(X_train, Y_train)

X_test = movie_master.loc[test_idx, ['runtime', 'screens']]
X_test = pd.concat([X_test, fwr[test_idx], runlen_test_hat], axis = 1)
X_test.columns = ['runtime', 'screens', 'fwr', 'runlen']
Y_test = movie_master.loc[test_idx, 'india-nett-gross']

Y_test_hat = gb_mod_totrev.predict(X_test)

err_mae = np.abs(Y_test - Y_test_hat)/Y_test

intervals = np.arange(0.25, 0.56, 0.1)
for interval in intervals:
    cnt = len([x for x in err_mae if x < interval])
    cnt = 100*cnt/len(err_mae)
    print('Percentage of estimates for test set that are off by less than %.0f%% from true value: %.2f' % (100*interval, cnt))
    
totrev_test_hat = pd.Series(Y_test_hat, name = 'tot_rev', index = test_idx)

#%% Predicting Exhibitor Revenue
print('EXHIBITOR NETT GROSS PREDICTION MODEL PERFORMANCE')

X_train = movie_master.loc[train_idx, ['budget', 'screens', 'india-footfalls', 'india-nett-gross']]
X_train = pd.concat([X_train, fwr[train_idx], runlen[train_idx]], axis = 1)
X_train.columns = ['budget', 'screens', 'india-footfalls', 'india-nett-gross', 'fwr', 'runlen']
Y_train = exsh[train_idx]

gb_est_exsh = GradientBoostingRegressor(random_state = 1970).set_params(**exsh_best_param)
gb_mod_exsh = gb_est_exsh.fit(X_train, Y_train)

X_test = movie_master.loc[test_idx, ['budget', 'screens']]
X_test = pd.concat([X_test, ff_test_hat, totrev_test_hat, fwr[test_idx], runlen_test_hat], axis = 1)
X_test.columns = ['budget', 'screens', 'india-footfalls', 'india-nett-gross', 'fwr', 'runlen']
Y_test = exsh[test_idx]

Y_test_hat = gb_mod_exsh.predict(X_test)

err_mae = np.abs(Y_test - Y_test_hat)/Y_test

intervals = np.arange(0.25, 0.56, 0.1)
for interval in intervals:
    cnt = len([x for x in err_mae if x < interval])
    cnt = 100*cnt/len(err_mae)
    print('Percentage of estimates for test set that are off by less than %.0f%% from true value: %.2f' % (100*interval, cnt))
    
exsh_test_hat = pd.Series(Y_test_hat, name = 'ex_share', index = test_idx)
