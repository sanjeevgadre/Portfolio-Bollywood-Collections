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


#%% Loading Data
movie_master = pd.read_pickle('./data/movie_master_en.pkl')
cpi_master = pd.read_csv('./data/CPI.csv')

# Adjusting the first week revenue to account for entertainment and service tax
fwr = movie_master['india-first-week'] * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

#%% First Week Revenue v/s Release Week conditioned on Budget
X = movie_master.loc[:, ['release_week', 'budget']]
X['release_week'] = X['release_week'].astype('float')
Y = fwr

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% First Week Revenue v/s Genre, conditioned on Year
X = movie_master.loc[:, ['release_year', 'genre']]
X['release_year'] = X['release_year'].astype('float')
X = pd.get_dummies(X)
Y = fwr

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% First Week Revenue v/s Screens conditioned on Budget and Year
X = movie_master.loc[:, ['release_year', 'budget', 'screens']]
X['release_year'] = X['release_year'].astype('float')
Y = fwr
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% First Week Revenue v/s Runtime conditioned on Budget and Year
X = movie_master.loc[:, ['release_year', 'budget', 'runtime']]
X['release_year'] = X['release_year'].astype('float')
Y = fwr
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% First Week Revenue v/s Budget, conditioned on Inflation and Screens
X = movie_master.loc[:, ['inf_adj_fct', 'budget', 'screens']]
Y = fwr
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% First Week Revenue v/s Inflation conditioned on Year and Budget
X = movie_master.loc[:, ['release_year', 'inf_adj_fct', 'budget']]
X = X.astype('float')
Y = fwr
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% First Week Revenue v/s Year, conditioned on Inflation, Screens and Budget
X = movie_master.loc[:, ['release_year', 'inf_adj_fct', 'budget', 'screens']]
X = X.astype('float')
Y = fwr
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% First Week Models - Investigating Likely Response Function

X = movie_master.loc[:, ['release_year', 'inf_adj_fct''budget', 'screens']]
Y = fwr

# Should not standardize when doing heteroscedacidy analysis as log transformations can lead to NANs

## Investigating heteroscedacity - studentized residuals v/s predicted values
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
Y_hat = model.predict(X)
studentized_residuals = model.get_influence().resid_studentized_internal

plt.figure()
Y_hat = Y_hat/10000000
plt.scatter(Y_hat, studentized_residuals)
plt.grid(True, which = 'major', axis = 'y')
plt.ylabel('Studentized Residuals')
plt.xlabel('Predicted First Week Revenue in Rs. crores')
plt.show()

#--> Clearly there is heteroscedacity. To eliminate heteroscedacity, we try fitting a model using the log values of Y
Y = np.log(Y)
model = sm.OLS(Y, X).fit()
Y_hat = model.predict(X)
studentized_residuals = model.get_influence().resid_studentized_internal

plt.figure()
plt.scatter(Y_hat, studentized_residuals)
plt.grid(True, which = 'major', axis = 'y')
plt.ylabel('Studentized Residuals')
plt.xlabel('Predicted Log Value of First Week Revenue')
plt.show()

#--> Does eliminate heteroscedacity. Should use log values of Y for further investigation and modelling

## Investigating likely non-linear response function - residuals v/s predicted values

residuals = Y_hat - Y
plt.figure(figsize = (15, 5))
sns.set(style="whitegrid")
axs = sns.residplot(Y_hat, residuals, lowess = True)
axs.set_xlabel('Predicted Log Value of First Week Revenue')
axs.set_ylabel('Residuals')
plt.show()

#--> Clearly a non-linear response function

#%% First Week Revenue Prediction Models - Tree Based

X = movie_master.loc[:, ['release_year', 'inf_adj_fct', 'budget', 'screens']]
Y = fwr

rf_est = RandomForestRegressor(random_state = 1970)
gb_est = GradientBoostingRegressor(random_state = 1970)

rf_param_grid = {'n_estimators' : [100, 500, 2500], 'criterion' : ['mse', 'mae'],
                 'max_depth' : [1, 2, 4], 'max_features' : [0.33, 0.66, 1]}
rf_mod = GridSearchCV(rf_est, param_grid = rf_param_grid, 
                      scoring = 'neg_mean_absolute_error', n_jobs = -1,
                      error_score = 'raise').fit(X, Y)

print('The best fit Random Forest Regressor reports a cross-validated MAE of %.0f' % -rf_mod.best_score_)
print('The parameters for the best fit model are %s' % rf_mod.best_params_)

gb_param_grid = {'n_estimators' : [100, 500, 2500], 'learning_rate' : [0.01, 0.001], 
                 'criterion' : ['mse', 'mae'], 'max_depth' : [1, 2, 4], 
                 'max_features' : [0.33, 0.66, 1]}
gb_mod = GridSearchCV(gb_est, param_grid = gb_param_grid, 
                      scoring = 'neg_mean_absolute_error', n_jobs = -1, 
                      error_score = 'raise').fit(X, Y)

print('The best fit Gradient Boosted Tree Regressor reports a cross-validated MAE of %.0f' % -gb_mod.best_score_)
print('The parameters for the best fit model are %s' % gb_mod.best_params_)

## The Random Forest Ensemble provides a better model (lower MAE)

#%% First Week Revenue Prediction Model Performance

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1970)

rf_est_best = RandomForestRegressor(random_state = 1970).set_params(**rf_mod.best_params_)
rf_mod_best = rf_est_best.fit(X_train, Y_train)
print('The R^2 for the model using test data: %.4f' % rf_mod_best.score(X_test, Y_test))

Y_test_hat = rf_mod_best.predict(X_test)
err_mae = np.abs(Y_test - Y_test_hat)/Y_test

intervals = np.arange(0.25, 0.56, 0.1)
for interval in intervals:
    cnt = len([x for x in err_mae if x < interval])
    cnt = 100*cnt/len(err_mae)
    print('Percentage of estimates for test set that are off by less than %.0f%% from true value: %.2f' % (100*interval, cnt))

#%% First Week Revenue Prediction Models - Tree Based

X = movie_master.loc[:, ['screens']]
Y = fwr

rf_est = RandomForestRegressor(random_state = 1970)
gb_est = GradientBoostingRegressor(random_state = 1970)

rf_param_grid = {'n_estimators' : [100, 500, 2500], 'criterion' : ['mse', 'mae'],
                 'max_depth' : [1, 2, 4], 'max_features' : [0.33, 0.66, 1]}
rf_mod = GridSearchCV(rf_est, param_grid = rf_param_grid, 
                      scoring = 'neg_mean_absolute_error', n_jobs = -1,
                      error_score = 'raise').fit(X, Y)

print('The best fit Random Forest Regressor reports a cross-validated MAE of %.0f' % -rf_mod.best_score_)
print('The parameters for the best fit model are %s' % rf_mod.best_params_)

gb_param_grid = {'n_estimators' : [100, 500, 2500], 'learning_rate' : [0.01, 0.001], 
                 'criterion' : ['mse', 'mae'], 'max_depth' : [1, 2, 4], 
                 'max_features' : [0.33, 0.66, 1]}
gb_mod = GridSearchCV(gb_est, param_grid = gb_param_grid, 
                      scoring = 'neg_mean_absolute_error', n_jobs = -1, 
                      error_score = 'raise').fit(X, Y)

print('The best fit Gradient Boosted Tree Regressor reports a cross-validated MAE of %.0f' % -gb_mod.best_score_)
print('The parameters for the best fit model are %s' % gb_mod.best_params_)

## The Gradient Boosted Ensemble provides a better model (lower MAE)

#%% First Week Revenue Prediction Model Performance

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1970)

gb_est_best = GradientBoostingRegressor(random_state = 1970).set_params(**gb_mod.best_params_)
gb_mod_best = gb_est_best.fit(X_train, Y_train)
print('The R^2 for the model using test data: %.4f' % gb_mod_best.score(X_test, Y_test))

Y_test_hat = gb_mod_best.predict(X_test)
err_mae = np.abs(Y_test - Y_test_hat)/Y_test

intervals = np.arange(0.25, 0.56, 0.1)
for interval in intervals:
    cnt = len([x for x in err_mae if x < interval])
    cnt = 100*cnt/len(err_mae)
    print('Percentage of estimates for test set that are off by less than %.0f%% from true value: %.2f' % (100*interval, cnt))
