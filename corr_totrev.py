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
weekly_master = pd.read_hdf('./data/weekly_master.h5')

# Adjusting the first week revenue to account for entertainment and service tax
fwr = movie_master['india-first-week'] * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

# Getting the run length of films
df = weekly_master.groupby('movie_id').sum().iloc[:, 3:]
run_length = df.apply(lambda x: np.count_nonzero(x), axis = 1)
run_length.reset_index(drop = True, inplace = True)

#%% Total Nett Revenue v/s Release Week
corr = movie_master['release_week'].corr(movie_master['india-nett-gross'], method = 'spearman')
print('%.4f' % corr)

#%% Total Nett Revenue v/s Genre conditioned on Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = movie_master['india-nett-gross'][indx]
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
plt.title('Spearman Correlation: Total Nett Revenue v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/tr/02_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Total Nett Revenue v/s Footfalls, conditioned on Year, Budget, First Week Revenue and Run Length
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'india-footfalls']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    X = pd.concat([X, fwr[indx], run_length[indx], ], axis = 1)
    X.columns = ['budget', 'footfalls', 'fwr', 'run_len']
    Y = movie_master['india-nett-gross'][indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['footfalls'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: Total Nett Revenue v/s Footfalls')
plt.grid(axis = 'y')
plt.savefig('./figs/tr/03_COR.jpg', dpi = 'figure')
plt.show()
plt.close()


#%% Total Nett Revenue v/s Run Length conditioned on Year, Screens and First Week Revenue
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['screens']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    X = pd.concat([X, fwr[indx], run_length[indx]], axis = 1)
    X.columns = ['screens', 'fwr', 'run_len']
    Y = movie_master['india-nett-gross'][indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['run_len'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: Total Nett Revenue v/s Run Length')
plt.grid(axis = 'y')
plt.savefig('./figs/tr/04_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Total Nett Revenue v/s First Week Revenue conditioned on Budget and Screens
X = movie_master.loc[:, ['budget', 'screens']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['budget', 'screens', 'fwr']
Y = movie_master['india-nett-gross']
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s Screens conditioned on Budget
X = movie_master.loc[:, ['budget', 'screens']]
Y = movie_master['india-nett-gross']
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Total Nett Revenue v/s Runtime conditioned on Budget and Year
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'runtime']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = movie_master['india-nett-gross'][indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['runtime'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: Total Nett Revenue v/s Runtime')
plt.grid(axis = 'y')
plt.savefig('./figs/tr/07_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Total Nett Revenue v/s Budget, conditioned on Year
years = movie_master['release_year'].unique()
corr_lst = []
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, 'budget']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = movie_master['india-nett-gross'][indx]
    corr = X.corr(Y, method = 'spearman')
    corr_lst.append(corr)
    
print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.title('Spearman Correlation: Total Nett Revenue v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/tr/08_COR.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Total Nett Revenue v/s Year, conditioned on Inflation
X = movie_master['release_year']
Y = movie_master['india-nett-gross'] * movie_master['inf_adj_fct']
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on T, adjusted for inflation, : %.4f' % corr)

X = movie_master['budget']
Y = movie_master['india-nett-gross']
corr = X.corr(Y, method = 'spearman')
print('Total Effect of B on E : %.4f' % corr)

## Total Nett Revenue v/s Year
X = movie_master['release_year']
Y = movie_master['india-nett-gross']
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on T : %.4f' % corr)

#%% Preliminary Stats Models
# X = movie_master.loc[:, ['release_year', 'budget', 'screens']]
# X = pd.concat([X, fwr], axis = 1)
# X.columns = ['release_year', 'budget', 'screens', 'fwr']
# X = sm.add_constant(X)
# Y = movie_master['india-nett-gross']
# model = sm.OLS(Y, X).fit()
# print(model.summary())


# X = movie_master.loc[:, ['release_year', 'budget', 'screens']]
# X = pd.concat([X, fwr], axis = 1)
# X.columns = ['release_year', 'budget', 'screens', 'fwr']
# X.iloc[:, 1:3] = (X.iloc[:, 1:3] - X.iloc[:, 1:3].mean())/X.iloc[:, 1:3].std()
# Y = movie_master['india-nett-gross']
# Y = (Y - Y.mean())/Y.std()

# model = sm.OLS(Y, X).fit()
# print(model.summary())


#%% Total Nett Revenue - Features of Likely Predictive Model
X = movie_master.loc[:, ['budget', 'india-footfalls', 'screens']]
X = pd.concat([X, fwr, run_length], axis = 1)
X.columns = ['budget', 'india-footfalls', 'screens', 'fwr', 'runlen']
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
X = movie_master.loc[:, ['budget', 'india-footfalls', 'screens']]
X = pd.concat([X, fwr, run_length], axis = 1)
X.columns = ['budget', 'india-footfalls', 'screens', 'fwr', 'runlen']
Y = movie_master['india-nett-gross']

rf_est = RandomForestRegressor(random_state = 1970)
gb_est = GradientBoostingRegressor(random_state = 1970)

## Using MAE as evaluation metric
rf_param_grid = {'n_estimators' : [100, 500, 2500], 
                 'max_features' : [0.33, 0.66, 1]}
rf_mod = GridSearchCV(rf_est, param_grid = rf_param_grid, 
                      scoring = 'neg_mean_absolute_error', n_jobs = -1,
                      error_score = 'raise').fit(X, Y)

print('The best fit Random Forest Regressor reports a cross-validated MAE of %.0f' % -rf_mod.best_score_)
print('The parameters for the best fit model are %s' % rf_mod.best_params_)

gb_param_grid = {'n_estimators' : [100, 500, 2500], 'learning_rate' : [0.01, 0.001], 
                 'max_depth' : [1, 2, 4], 'max_features' : [0.33, 0.66, 1]}
gb_mod = GridSearchCV(gb_est, param_grid = gb_param_grid, 
                      scoring = 'neg_mean_absolute_error', n_jobs = -1, 
                      error_score = 'raise').fit(X, Y)

print('The best fit Gradient Boosted Tree Regressor reports a cross-validated MAE of %.0f' % -gb_mod.best_score_)
print('The parameters for the best fit model are %s' % gb_mod.best_params_)

## Boosted Tree Ensemble delivers beter results (lower MAE)

#%% Total Nett Revenue Prediction Model Performance

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1970)

gb_est = GradientBoostingRegressor(random_state = 1970).set_params(**gb_mod.best_params_)
gb_mod = gb_est.fit(X_train, Y_train)
print('The R^2 for the model using test data: %.4f' % gb_mod.score(X_test, Y_test))

Y_test_hat = gb_mod.predict(X_test)

err_mae = np.abs(Y_test - Y_test_hat)/Y_test

intervals = np.arange(0.25, 0.56, 0.1)
for interval in intervals:
    cnt = len([x for x in err_mae if x < interval])
    cnt = 100*cnt/len(err_mae)
    print('Percentage of estimates for test set that are off by less than %.0f%% from true value: %.2f' % (100*interval, cnt))
