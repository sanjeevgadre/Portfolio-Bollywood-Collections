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

#%% Footfalls v/s Release Week
corr = movie_master['release_week'].corr(movie_master['india-footfalls'], method = 'spearman')
print('%.4f' % corr)

#%% Footfalls v/s Genre conditioned on Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = movie_master['india-footfalls'][indx]
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
plt.title('Spearman Correlation: Footfalls v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/ff/02_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Footfalls v/s Run Length conditioned on Year, Screens and First Week Revenue
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['screens']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    X = pd.concat([X, fwr[indx], run_length[indx]], axis = 1)
    X.columns = ['screens', 'fwr', 'run_len']
    Y = movie_master['india-footfalls'][indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['run_len'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: Footfalls v/s Run Length')
plt.grid(axis = 'y')
plt.savefig('./figs/ff/03_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Footfalls v/s First Week Revenue conditioned on Budget and Screens
X = movie_master.loc[:, ['budget', 'screens']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['budget', 'screens', 'fwr']
Y = movie_master['india-footfalls']
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Footfalls v/s Screens conditioned on Budget
X = movie_master.loc[:, ['budget', 'screens']]
Y = movie_master['india-footfalls']
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Footfalls v/s Runtime conditioned on Budget and Year
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'runtime']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = movie_master['india-footfalls'][indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['runtime'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: Footfalls v/s Runtime')
plt.grid(axis = 'y')
plt.savefig('./figs/ff/06_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Footfalls v/s Budget, conditioned on Year
years = movie_master['release_year'].unique()
corr_lst = []
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, 'budget']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = movie_master['india-footfalls'][indx]
    corr = X.corr(Y, method = 'spearman')
    corr_lst.append(corr)
    
print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.title('Spearman Correlation: Footfalls v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/ff/07_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Footfalls v/s Year
X = movie_master['release_year']
Y = movie_master['india-footfalls']
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on F : %.4f' % corr)

#%% Footfalls - Features of Likely Predictive Model

X = movie_master.loc[:, ['release_year', 'budget']]
X = pd.concat([X, fwr, run_length], axis = 1)
X.columns = ['release_year', 'budget', 'fwr', 'run_len']
Y = movie_master['india-footfalls']

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
plt.xlabel('Predicted Footfalls')
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
plt.xlabel('Predicted Log Value of Footfalls')
plt.show()

#--> Does eliminate heteroscedacity. Should use log values of Y for further investigation and modelling


## Investigating likely non-linear response function - residuals v/s predicted values

residuals = Y_hat - Y
plt.figure(figsize = (15, 5))
sns.set(style="whitegrid")
axs = sns.residplot(Y_hat, residuals, lowess = True)
axs.set_xlabel('Predicted Log Values of Footfalls')
axs.set_ylabel('Residuals')
plt.show()

#--> Perhaps a non-linear response function


#%% Footfalls Prediction Models - Tree Based
X = movie_master.loc[:, ['release_year', 'budget']]
X = pd.concat([X, fwr, run_length], axis = 1)
X.columns = ['release_year', 'budget', 'fwr', 'run_len']
Y = movie_master['india-footfalls']

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

## Boosted Tree Ensemble delivers best results (lower MAE)

#%% Footfalls Prediction Model Performance

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
with open('./ff_best_param.pkl', 'w+b') as handle:
    pickle.dump(gb_mod.best_params_, handle)
    
