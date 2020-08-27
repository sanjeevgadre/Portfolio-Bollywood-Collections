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
movie_master = pd.read_pickle('../data/movie_master_en.pkl')
cpi_master = pd.read_csv('../data/CPI.csv')
weekly_master = pd.read_hdf('../data/weekly_master.h5')

# Adjusting the first week revenue to account for entertainment and service tax
fwr = movie_master['india-first-week'] * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

# Getting the run length of films
df = weekly_master.groupby('movie_id').sum().iloc[:, 3:]
run_length = df.apply(lambda x: np.count_nonzero(x), axis = 1)
run_length.reset_index(drop = True, inplace = True)

# Calculating Exhibitor Share (Total Nett Gross - Distributor Share)
ex_share = movie_master['india-nett-gross'] - movie_master['india-distributor-share']

#%% Exhibitor Share v/s Release Week conditioned on Budget
X = movie_master.loc[:, ['release_week', 'budget']]
X['release_week'] = X['release_week'].astype('float')
Y = ex_share

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Exhibitor Share v/s Genre conditioned on Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = ex_share[indx]
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
plt.title('Spearman Correlation: Exhibitor Share v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/es/02_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Exhibitor Share v/s Footfalls, conditioned on Year, Budget, First Week Revenue and Run Length
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'india-footfalls']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    X = pd.concat([X, fwr[indx], run_length[indx], ], axis = 1)
    X.columns = ['budget', 'footfalls', 'fwr', 'run_len']
    Y = ex_share[indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['footfalls'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: Exhibitor Share v/s Footfalls')
plt.grid(axis = 'y')
plt.savefig('./figs/es/03_COR.jpg', dpi = 'figure')
plt.show()
plt.close()


#%% Exhibitor Share v/s Run Length conditioned on Year, Screens and First Week Revenue
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['screens']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    X = pd.concat([X, fwr[indx], run_length[indx]], axis = 1)
    X.columns = ['screens', 'fwr', 'run_len']
    Y = ex_share[indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['run_len'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: Exhibitor Share v/s Run Length')
plt.grid(axis = 'y')
plt.savefig('./figs/es/04_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Exhibitor Share v/s First Week Revenue conditioned on Budget and Screens
X = movie_master.loc[:, ['budget', 'screens']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['budget', 'screens', 'fwr']
Y = ex_share
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Exhibitor Share v/s Screens conditioned on Budget
X = movie_master.loc[:, ['budget', 'screens']]
Y = ex_share
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Exhibitor Share v/s Runtime conditioned on Budget and Year
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'runtime']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = ex_share[indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['runtime'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: Exhibitor Share v/s Runtime')
plt.grid(axis = 'y')
plt.savefig('./figs/es/07_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Exhibitor Share v/s Budget, conditioned on Year
years = movie_master['release_year'].unique()
corr_lst = []
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, 'budget']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = ex_share[indx]
    corr = X.corr(Y, method = 'spearman')
    corr_lst.append(corr)
    
print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.title('Spearman Correlation: Exhibitor Share v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/es/08_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Exhibitor Share v/s Year, conditioned on Inflation
X = movie_master['release_year']
Y = ex_share * movie_master['inf_adj_fct']
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on E, adjusted for inflation, : %.4f' % corr)

X = movie_master['budget']
Y = ex_share
corr = X.corr(Y, method = 'spearman')
print('Total Effect of B on E : %.4f' % corr)

## First Week Revenue v/s Year
X = movie_master['release_year']
Y = ex_share
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on F : %.4f' % corr)

#%% Exhibitor Share v/s Total Nett Revenue, conditioned on Footfalls, Run Length, First Week Revenue, Release Screens and Budget
X = movie_master.loc[:, ['budget', 'screens', 'india-nett-gross', 'india-footfalls']]
X = pd.concat([X, fwr, run_length], axis = 1)
X.columns = ['budget', 'screens', 'india-nett-gross', 'india-footfalls', 'fwr', 'run_len']
Y = ex_share

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Exhibitor Share - Features of Likely Predictive Model
X = movie_master.loc[:, ['budget', 'india-footfalls', 'screens', 'india-nett-gross']]
X = pd.concat([X, fwr, run_length], axis = 1)
X.columns = ['budget', 'india-footfalls', 'screens', 'india-nett-gross', 'fwr', 'runlen']
Y = ex_share
Y[1094] = 1             # hack

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
plt.xlabel('Predicted Exhibitor Share')
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
plt.xlabel('Predicted Log Value of Exhibitor Share')
plt.show()

#--> Does eliminate heteroscedacity. Should use log values of Y for further investigation and modelling


## Investigating likely non-linear response function - residuals v/s predicted values

residuals = Y_hat - Y
plt.figure(figsize = (15, 5))
sns.set(style="whitegrid")
axs = sns.residplot(Y_hat, residuals, lowess = True)
axs.set_xlabel('Predicted Log Values of Exhibitor Share')
axs.set_ylabel('Residuals')
plt.show()

#--> Likely non-linear response function


#%% Exhibitor Share Prediction Models - Tree Based
X = movie_master.loc[:, ['budget', 'india-footfalls', 'screens', 'india-nett-gross']]
X = pd.concat([X, fwr, run_length], axis = 1)
X.columns = ['budget', 'india-footfalls', 'screens', 'india-nett-gross', 'fwr', 'runlen']
Y = ex_share

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

## Boosted Tree Ensemble delivers beter results (lower MAE)

#%% Exhibitor Share Prediction Model Performance

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1970)

gb_est_best = GradientBoostingRegressor(random_state = 1970).set_params(**gb_mod.best_params_)
gb_mod_best = gb_est.fit(X_train, Y_train)
print('The R^2 for the model using test data: %.4f' % gb_mod_best.score(X_test, Y_test))

Y_test_hat = gb_mod_best.predict(X_test)

err_mae = np.abs(Y_test - Y_test_hat)/Y_test

intervals = np.arange(0.25, 0.56, 0.1)
for interval in intervals:
    cnt = len([x for x in err_mae if x < interval])
    cnt = 100*cnt/len(err_mae)
    print('Percentage of estimates for test set that are off by less than %.0f%% from true value: %.2f' % (100*interval, cnt))

#%% Saving the best fit ensemble parameters
with open('./exsh_best_param.pkl', 'w+b') as handle:
    pickle.dump(gb_mod.best_params_, handle)