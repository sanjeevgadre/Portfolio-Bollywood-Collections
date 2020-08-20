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
movie_master = pd.read_pickle('./data/movie_master_en.pkl')
cpi_master = pd.read_csv('./data/CPI.csv')
weekly_master = pd.read_hdf('./data/weekly_master.h5')

# Adjusting the first week revenue to account for entertainment and service tax
fwr = movie_master['india-first-week'] * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

# Getting the run length of films
df = weekly_master.groupby('movie_id').sum().iloc[:, 3:]
run_len = df.apply(lambda x: np.count_nonzero(x), axis = 1)
run_len.reset_index(drop = True, inplace = True)

#%% Run Length v/s Release Week
corr = movie_master['release_week'].corr(run_len, method = 'spearman')
print('%.4f' % corr)

#%% Run Length v/s Genre conditioned on Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = run_len[indx]
    corr = Y.corr(X, method = 'spearman')
    corr_lst.append(corr)

print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(-1, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.axhline(y = -0.3, color = 'b', linestyle='--')
plt.title('Spearman Correlation: Run Length v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/runlen/02_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Run Length v/s First Week Revenue conditioned on Budget and Screens
X = movie_master.loc[:, ['budget', 'screens']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['budget', 'screens', 'fwr']
Y = run_len
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Run Length v/s Screens conditioned on Budget
X = movie_master.loc[:, ['budget', 'screens']]
Y = run_len
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Run Length v/s Runtime conditioned on Budget and Year
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'runtime']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = run_len[indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['runtime'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: Run Length v/s Runtime')
plt.grid(axis = 'y')
plt.savefig('./figs/runlen/05_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Run Length v/s Budget, conditioned on Year
years = movie_master['release_year'].unique()
corr_lst = []
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, 'budget']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = run_len[indx]
    corr = X.corr(Y, method = 'spearman')
    corr_lst.append(corr)
    
print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.title('Spearman Correlation: Run Length v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/runlen/06_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Run Length v/s Year
X = movie_master['release_year']
Y = run_len
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on F : %.4f' % corr)

#%% Run Length Models - Investigating Likely Response Functin

X = movie_master.loc[:, ['release_year', 'screens']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'screens', 'fwr']
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
X = movie_master.loc[:, ['release_year', 'screens']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['release_year', 'screens', 'fwr']
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
Y_test_hat = np.floor(Y_test_hat)
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
    