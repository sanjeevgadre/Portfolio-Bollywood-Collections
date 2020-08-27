#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:22:06 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pickle

#%% Loading Data
movie_master = pd.read_pickle('../data/movie_master_en.pkl')
cpi_master = pd.read_csv('../data/CPI.csv')
weekly_master = pd.read_hdf('../data/weekly_master.h5')

# Adjusting the first week revenue to account for entertainment and service tax
fwr = movie_master['india-first-week'] * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

# Getting the run length of films
df = weekly_master.groupby('movie_id').sum().iloc[:, 3:]
run_len = df.apply(lambda x: np.count_nonzero(x), axis = 1)
run_len.reset_index(drop = True, inplace = True)

#%% Run Length v/s Release Week conditioned on Budget
X = movie_master.loc[:, ['release_week', 'budget']]
X['release_week'] = X['release_week'].astype('float')
Y = run_len

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Run Length v/s Genre conditioned on Year
X = movie_master.loc[:, ['release_year', 'genre']]
X['release_year'] = X['release_year'].astype('float')
X = pd.get_dummies(X)
Y = run_len

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Run Length v/s First Week Revenue conditioned on Budget, Screens and Year
X = movie_master.loc[:, ['budget', 'screens', 'release_year']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['budget', 'release_year', 'screens', 'fwr']
X = X.astype('float')
Y = run_len
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Run Length v/s Screens conditioned on Budget, Year and First Week Revenue
X = movie_master.loc[:, ['budget', 'screens', 'release_year']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['budget', 'release_year', 'screens', 'fwr']
X = X.astype('float')
Y = run_len
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Run Length v/s Runtime conditioned on Budget and Year
X = movie_master.loc[:, ['release_year', 'budget', 'runtime']]
X['release_year'] = X['release_year'].astype('float')
Y = run_len
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Run Length v/s Budget, conditioned on Year, Screens, Runtime and First Week Revenue
X = movie_master.loc[:, ['release_year', 'budget', 'screens', 'runtime']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'budget', 'screens', 'runtime', 'fwr']
X = X.astype('float')
Y = run_len
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Run Length v/s Year, conditioned on Screens, Runtime, Budget and First Week Revenue
X = movie_master.loc[:, ['release_year', 'budget', 'screens', 'runtime']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'budget', 'screens', 'runtime', 'fwr']
X = X.astype('float')
Y = run_len
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Run Length Models - Investigating Likely Response Functin

X = movie_master.loc[:, ['release_year', 'screens', 'runtime']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'screens', 'runtime', 'fwr']
X = X.astype('float')
Y = run_len

# Should not standardize when doing heteroscedacidy analysis as log transformations can lead to NANs

## Investigating heteroscedacity - studentized residuals v/s predicted values
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
Y_hat = model.predict(X)
studentized_residuals = model.get_influence().resid_studentized_internal

plt.figure()
Y_hat = Y_hat
plt.scatter(Y_hat, studentized_residuals)
plt.grid(True, which = 'major', axis = 'y')
plt.ylabel('Studentized Residuals')
plt.xlabel('Predicted Run Length in Weeks')
plt.show()

#--> Clearly there is no heteroscedacity. 

## Investigating likely non-linear response function - residuals v/s predicted values

residuals = Y_hat - Y
plt.figure(figsize = (15, 5))
sns.set(style="whitegrid")
axs = sns.residplot(Y_hat, residuals, lowess = True)
axs.set_xlabel('Predicted Run Length in Weeks')
axs.set_ylabel('Residuals')
plt.show()

#--> Perhaps a linear response function

#%% Run Length Prediction Models - Tree Based
X = movie_master.loc[:, ['release_year', 'budget', 'screens', 'runtime']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'budget', 'screens', 'runtime', 'fwr']
X = X.astype('float')
Y = run_len

rf_est = RandomForestRegressor(random_state = 1970)
gb_est = GradientBoostingRegressor(random_state = 1970)

## Using MAE as evaluation metric
rf_param_grid = {'n_estimators' : [100, 500, 2500], 'criterion' : ['mse', 'mae'],
                 'max_depth' : [1, 2, 4], 'max_features' : [0.33, 0.66, 1]}
rf_mod = GridSearchCV(rf_est, param_grid = rf_param_grid, 
                      scoring = 'neg_mean_absolute_error', n_jobs = -1,
                      error_score = 'raise').fit(X, Y)

print('The best fit Random Forest Regressor reports a cross-validated MAE of %.4f' % -rf_mod.best_score_)
print('The parameters for the best fit model are %s' % rf_mod.best_params_)

gb_param_grid = {'n_estimators' : [100, 500, 2500], 'learning_rate' : [0.01, 0.001], 
                 'criterion' : ['mse', 'mae'], 'max_depth' : [1, 2, 4], 
                 'max_features' : [0.33, 0.66, 1]}
gb_mod = GridSearchCV(gb_est, param_grid = gb_param_grid, 
                      scoring = 'neg_mean_absolute_error', n_jobs = -1, 
                      error_score = 'raise').fit(X, Y)

print('The best fit Gradient Boosted Tree Regressor reports a cross-validated MAE of %.4f' % -gb_mod.best_score_)
print('The parameters for the best fit model are %s' % gb_mod.best_params_)

# Run Length Prediction Model - Linear

lr_est = LinearRegression()
lr_mod = cross_val_score(lr_est, X, Y, scoring = 'neg_mean_absolute_error')

print('The Linear Regression Model reports a cross validated MAE of %.4f' % -lr_mod.mean())

## Gradient Boosted Ensemble delivers best results (lower MAE)

#%% Prediction Model Performance

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1970)

gb_est_best = GradientBoostingRegressor(random_state = 1970).set_params(**gb_mod.best_params_)

gb_mod_best = gb_est_best.fit(X_train, Y_train)
print('The R^2 for the model using test data: %.4f' % gb_mod_best.score(X_test, Y_test))

Y_test_hat = gb_mod_best.predict(X_test)

for i in range(len(Y_test_hat)):
    if Y_test_hat[i] < 1: Y_test_hat[i] = 1
    if Y_test_hat[i] > 11: Y_test_hat[i] = 11

err_mae = np.abs(Y_test - Y_test_hat)/Y_test

intervals = np.arange(0.25, 0.56, 0.1)
for interval in intervals:
    cnt = len([x for x in err_mae if x < interval])
    cnt = 100*cnt/len(err_mae)
    print('Percentage of estimates for test set that are off by less than %.0f%% from true value: %.2f' % (100*interval, cnt))

#%% Saving the best fit ensemble parameters
with open('./runlen_best_param.pkl', 'w+b') as handle:
    pickle.dump(gb_mod.best_params_, handle)
