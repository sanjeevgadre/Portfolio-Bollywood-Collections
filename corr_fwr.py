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
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


#%% Loading Data
movie_master = pd.read_pickle('./data/movie_master_en.pkl')
cpi_master = pd.read_csv('./data/CPI.csv')

# Adjusting the first week revenue to account for entertainment and service tax
fwr = movie_master['india-first-week'] * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

#%% FIRST WEEK REVENUE
## First Week Revenue v/s Release Week
corr = movie_master['release_week'].corr(fwr, method = 'spearman')
print('%.4f' % corr)

## First Week Revenue v/s Genre
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = fwr[indx]
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
plt.title('Spearman Correlation: First Week Revenue v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/fwr/f_g_cond_y.jpg', dpi = 'figure')
plt.show()
plt.close()

## First Week Revenue v/s Screens conditioned on Budget
X = movie_master.loc[:, ['budget', 'screens']]
Y = fwr
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## First Week Revenue v/s Runtime conditioned on Budget and Year
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'runtime']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = fwr[indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['runtime'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: First Week Revenue v/s Runtime')
plt.grid(axis = 'y')
plt.savefig('./figs/fwr/f_r_cond_y_b.jpg', dpi = 'figure')
plt.show()
plt.close()

## First Week Revenue v/s Budget, conditioned on Year and Inflation
years = movie_master['release_year'].unique()
corr_lst = []
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, 'budget']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = fwr[indx]
    corr = X.corr(Y, method = 'spearman')
    corr_lst.append(corr)
    
print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.title('Spearman Correlation: First Week Revenue v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/fwr/f_b_cond_y_i.jpg', dpi = 'figure')
plt.show()
plt.close()

## First Week Revenue v/s Year, conditioned on Inflation
X = movie_master['release_year']
Y = fwr * movie_master['inf_adj_fct']
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on F, adjusted for inflation, : %.4f' % corr)

X = movie_master['budget']
Y = fwr
corr = X.corr(Y, method = 'spearman')
print('Total Effect of B on F : %.4f' % corr)

## First Week Revenue v/s Year
X = movie_master['release_year']
Y = fwr
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on F : %.4f' % corr)

#%% First Week Models - statsmodels
X = movie_master.loc[:, ['budget', 'screens']]
X = sm.add_constant(X)
Y = fwr
model = sm.OLS(Y, X).fit()
print(model.summary())


X = movie_master.loc[:, ['budget', 'screens']]
X = (X - X.mean())/X.std()
Y = fwr
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% First Week Models - sklearn

X = movie_master.loc[:, ['budget', 'screens']]
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

## Investigating Effect of Unit change in Causes on Response function
X = movie_master.loc[:, ['budget', 'screens']]
Y = fwr                # Do we need to eliminate heteroscedacity

# Standardizing Variables
X_std = (X - X.mean())/X.std()
Y_std = (Y - Y.mean())/Y.std()

model = sm.OLS(Y_std, X_std).fit()
print(model.summary())

imp_unt_chg = model.params['budget']*Y.std()/X['budget'].std()
# imp_unt_chg = np.exp(imp_unt_chg)
print('Impact of unit increase in budget on first week revenue %.2f' % imp_unt_chg)

imp_unt_chg = model.params['screens']*Y.std()/X['screens'].std()
# imp_unt_chg = np.exp(imp_unt_chg)
print('Impact of unit increase in screens on first week revenue %.2f' % imp_unt_chg)

#%% First Week Revenue Prediction Models - Tree Based

X = movie_master.loc[:, ['budget', 'screens']]
Y = fwr

rf_est = RandomForestRegressor(random_state = 1970)
gb_est = GradientBoostingRegressor(random_state = 1970)

## Using RMSE as evaluation metric
rf_param_grid = {'n_estimators' : [100, 500, 2500], 
                 'max_features' : [0.33, 0.66, 1]}
rf_mod = GridSearchCV(rf_est, param_grid = rf_param_grid, scoring = 'neg_root_mean_squared_error', n_jobs = -1,
                      error_score = 'raise').fit(X, Y)

print('The best fit Random Forest Regressor reports a cross-validated RMSE of %.0f' % -rf_mod.best_score_)
print('The parameters for the best fit model are %s' % rf_mod.best_params_)

gb_param_grid = {'n_estimators' : [100, 500, 2500], 'learning_rate' : [0.01, 0.001], 
                 'max_depth' : [1, 2, 4], 'max_features' : [0.33, 0.66, 1]}
gb_mod = GridSearchCV(gb_est, param_grid = gb_param_grid, scoring = 'neg_root_mean_squared_error', n_jobs = -1, 
                      error_score = 'raise').fit(X, Y)

print('The best fit Gradient Boosted Tree Regressor reports a cross-validated RMSE of %.0f' % -gb_mod.best_score_)
print('The parameters for the best fit model are %s' % gb_mod.best_params_)

## The Boosted Tree Ensemble provides a better model (lower RMSE)

# Prediction Model Performance

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1970)

gb_est = GradientBoostingRegressor(random_state = 1970).set_params(**gb_mod.best_params_)
gb_mod = gb_est.fit(X_train, Y_train)
print('The R^2 for the model using test data: %.4f' % gb_mod.score(X_test, Y_test))

Y_test_hat = gb_mod.predict(X_test)
err_rmse = (Y_test_hat - Y_test)/Y_test

# plt.figure(figsize = (13,7))
# plt.hist(err_rmse, bins = 500, histtype = 'step', cumulative = True, density = True)
# plt.xticks(np.arange(0, err_rmse.max() + 0.3, .25), rotation = 45, fontsize = 10)
# plt.yticks(np.arange(0, 1.1, .1), rotation = 45, fontsize = 10)
# plt.grid(b= True, axis = 'both', which = 'both')
# plt.show()

#%%

## Using MAE as evaluation metric
rf_param_grid = {'n_estimators' : [100, 500, 2500], 
                 'max_features' : [0.33, 0.66, 1]}
rf_mod = GridSearchCV(rf_est, param_grid = rf_param_grid, scoring = 'neg_mean_absolute_error', n_jobs = -1,
                      error_score = 'raise').fit(X, Y)

print('The best fit Random Forest Regressor reports a cross-validated MAE of %.0f' % -rf_mod.best_score_)
print('The parameters for the best fit model are %s' % rf_mod.best_params_)

gb_param_grid = {'n_estimators' : [100, 500, 2500], 'learning_rate' : [0.01, 0.001], 
                 'max_depth' : [1, 2, 4], 'max_features' : [0.33, 0.66, 1]}
gb_mod = GridSearchCV(gb_est, param_grid = gb_param_grid, scoring = 'neg_mean_absolute_error', n_jobs = -1, 
                      error_score = 'raise').fit(X, Y)

print('The best fit Gradient Boosted Tree Regressor reports a cross-validated MAE of %.0f' % -gb_mod.best_score_)
print('The parameters for the best fit model are %s' % gb_mod.best_params_)

## The Random Forest Ensemble provides a better model (lower MAE)


# Prediction Model Performance
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1970)

rf_est = RandomForestRegressor(random_state = 1970).set_params(**rf_mod.best_params_)
rf_mod = rf_est.fit(X_train, Y_train)
print('The R^2 for the model using test data: %.4f' % rf_mod.score(X_test, Y_test))

Y_test_hat = rf_mod.predict(X_test)
err_mae = np.abs(Y_test_hat - Y_test)/Y_test

## The MAE model's predictions are less off the mark than RMSE model projections