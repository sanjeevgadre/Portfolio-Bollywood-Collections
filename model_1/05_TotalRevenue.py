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
import pickle

#%% Loading Data
movie_master = pd.read_pickle('./data/movie_master_en.pkl')
cpi_master = pd.read_csv('./data/CPI.csv')
weekly_master = pd.read_hdf('./data/weekly_master.h5')

# Adjusting the first week revenue to account for entertainment and service tax
fwr = movie_master['india-first-week'] * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

# Getting the run length of films
df = weekly_master.groupby('movie_id').sum().iloc[:, 3:]
run_length = df.apply(lambda x: np.count_nonzero(x), axis = 1)
run_length.reset_index(drop = True, inplace = True)

#%% Total Nett Revenue v/s Release Week conditioned on Budget
X = movie_master.loc[:, ['release_week', 'budget']]
X['release_week'] = X['release_week'].astype('float')
Y = movie_master['india-nett-gross']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s Genre conditioned on Year and Footfalls
X = movie_master.loc[:, ['release_year', 'india-footfalls', 'genre']]
X['release_year'] = X['release_year'].astype('float')
X = pd.get_dummies(X)
Y = movie_master['india-nett-gross']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s Footfalls, conditioned on Year, Genre, Screens, Runtime, First Week Revenue and Run Length
X = movie_master.loc[:, ['release_year', 'genre', 'screens', 'runtime', 'india-footfalls']]
X = pd.concat([X, fwr, run_length], axis = 1)
X.columns = ['release_year', 'genre', 'screens', 'runtime', 
             'india-footfalls', 'fwr', 'runlen']
X['release_year'] = X['release_year'].astype('float')
X = pd.get_dummies(X)
Y = movie_master['india-nett-gross']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s Run Length conditioned on Year, Screens and First Week Revenue
X = movie_master.loc[:, ['release_year', 'budget', 'screens', 'runtime', 'india-footfalls']]
X = pd.concat([X, fwr, run_length], axis = 1)
X.columns = ['release_year', 'budget', 'screens', 'runtime', 
             'india-footfalls', 'fwr', 'runlen']
X['release_year'] = X['release_year'].astype('float')
X = pd.get_dummies(X)
Y = movie_master['india-nett-gross']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s First Week Revenue conditioned on Year, Inflation, Budget, Screens and Footfalls
X = movie_master.loc[:, ['release_year', 'inf_adj_fct', 'budget', 'screens', 
                         'india-footfalls']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'inflation', 'budget', 'screens', 
                         'india-footfalls', 'fwr']
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['india-nett-gross']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s Screens conditioned on Year, Budget, Screens, Footfalls and First Week Revenue
X = movie_master.loc[:, ['release_year', 'budget', 'screens', 'india-footfalls']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'budget', 'screens', 'india-footfalls', 'fwr']
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['india-nett-gross']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s Runtime conditioned on Year, Budget and Footfalls
X = movie_master.loc[:, ['release_year', 'budget', 'india-footfalls', 'runtime']]
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['india-nett-gross']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s Budget, conditioned on Year, Inflation, Screens and First Week Revenue
X = movie_master.loc[:, ['release_year', 'inf_adj_fct', 'budget', 'screens']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'inflation', 'budget', 'screens', 'fwr']
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['india-nett-gross']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s Inflation conditioned on Year and First Week Revenue
X = movie_master.loc[:, ['release_year', 'inf_adj_fct']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'inflation', 'fwr']
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['india-nett-gross']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s Year, conditioned on Screens, First Week Revenue and Footfalls
X = movie_master.loc[:, ['release_year', 'screens', 'india-footfalls']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'screens', 'india-footfalls', 'fwr']
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['india-nett-gross']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue - Features of Likely Predictive Model
X = movie_master.loc[:, ['release_year', 'screens', 'india-footfalls']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'screens', 'india-footfalls', 'fwr']
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['india-nett-gross']

# Should not standardize when doing heteroscedacidy analysis as log transformations can lead to NANs

## Investigating heteroscedacity - studentized residuals v/s predicted values
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
Y_hat = model.predict(X)
studentized_residuals = model.get_influence().resid_studentized_internal

plt.figure()
Y_hat = Y_hat/1000000
plt.scatter(Y_hat, studentized_residuals)
plt.grid(True, which = 'major', axis = 'y')
plt.ylabel('Studentized Residuals')
plt.xlabel('Predicted Total Nett Revenue')
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
plt.xlabel('Predicted Log Value of Total Nett Revenue')
plt.show()

#--> Does eliminate heteroscedacity. Should use log values of Y for further investigation and modelling


## Investigating likely non-linear response function - residuals v/s predicted values

residuals = Y_hat - Y
plt.figure(figsize = (15, 5))
sns.set(style="whitegrid")
axs = sns.residplot(Y_hat, residuals, lowess = True)
axs.set_xlabel('Predicted Log Values of Total Nett Revenue')
axs.set_ylabel('Residuals')
plt.show()

#--> Likely non-linear response function

#%% Total Nett Revenue Prediction Models - Tree Based
X = movie_master.loc[:, ['release_year', 'screens', 'india-footfalls']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'screens', 'india-footfalls', 'fwr']
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['india-nett-gross']

rf_est = RandomForestRegressor(random_state = 1970)
gb_est = GradientBoostingRegressor(random_state = 1970)

## Using MAE as evaluation metric
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

## Boosted Tree Ensemble delivers best results (lower MAE)

#%% Total Nett Revenue Prediction Model Performance

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
    
#%% Saving the best fit ensemble parameters
with open('./totrev_best_param1.pkl', 'w+b') as handle:
    pickle.dump(gb_mod.best_params_, handle)