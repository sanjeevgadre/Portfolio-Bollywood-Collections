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
from scipy import stats
import statsmodels.api as sm

#%% Loading Data
movie_master = pd.read_pickle('./data/movie_master_en.pkl')
cpi_master = pd.read_csv('./data/CPI.csv')
# weekly_master = pd.read_hdf('./data/weekly_master.h5')

# Adjusting the first week revenue to account for entertainment and service tax
fwr = movie_master['india-first-week'] * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

# Calculating the distributors' share
disti_share = (movie_master['india-distributor-share']/movie_master['india-nett-gross'])*100

# # Getting the Total revenue and reindexing the resultant series
# swr = weekly_master.groupby('movie_id').sum()['week_2']
# swr.reset_index(drop = True, inplace = True)

# # # Adjusting the second week revenue to account for entertainment and service tax
# swr = swr * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

#%% TOTAL REVENUE
## Total Revenue v/s Release Week
corr = movie_master['release_week'].corr(movie_master['india-nett-gross'], method = 'spearman')
print('%.4f' % corr)

## Total Revenue v/s Genre conditioned on Year
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
plt.title('Spearman Correlation: Total Revenue v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/tr/tr_g_cond_y.jpg', dpi = 'figure')
plt.show()
plt.close()

## Total Revenue v/s First Week Revenue conditioned on Budget, Screens and Distributor Share

X = movie_master.loc[:, ['budget', 'screens']]
X = pd.concat([X, disti_share, fwr], axis = 1)
X.columns = ['budget', 'screens', 'disti_share', 'fwr']
Y = movie_master['india-nett-gross']
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Total Revenue v/s Distributor Share conditioned on Budget
X = pd.concat([movie_master['budget'], disti_share], axis = 1)
X.columns = ['budget', 'disti_share']
Y = movie_master['india-nett-gross']
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Total Revenue v/s Screens conditioned on Budget
X = movie_master.loc[:, ['budget', 'screens']]
Y = movie_master['india-nett-gross']
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Total Revenue v/s Runtime conditioned on Budget and Year
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
plt.title('p-values for Regression Coefficient: Total Revenue v/s Runtime')
plt.grid(axis = 'y')
plt.savefig('./figs/corr//tr/tr_r_cond_y_b.jpg', dpi = 'figure')
plt.show()
plt.close()

## Total Revenue v/s Budget, conditioned on Year and Inflation
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
plt.title('Spearman Correlation: Total Revenue v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/tr/tr_b_cond_y_i.jpg', dpi = 'figure')
plt.show()
plt.close()

## Total Revenue v/s Year, conditioned on Inflation
X = movie_master['release_year']
Y = movie_master['india-nett-gross'] * movie_master['inf_adj_fct']
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on TR, adjusted for inflation, : %.4f' % corr)

## Total Revenue v/s Budget
X = movie_master['budget']
Y = movie_master['india-nett-gross']
corr = X.corr(Y, method = 'spearman')
print('Total Effect of B on F : %.4f' % corr)

## Total Revenue v/s Year
X = movie_master['release_year']
Y = movie_master['india-nett-gross']
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on F : %.4f' % corr)

#%% Total Revenue Models
import numpy as np
import seaborn as sns

X = movie_master.loc[:, ['budget', 'screens']]
X = pd.concat([X, fwr, disti_share], axis = 1)
X.columns = ['budget', 'screens','fwr', 'disti_share']
Y = movie_master['india-nett-gross']

# Should not standardize when doing heteroscedacidy analysis as log transformations can lead to NANs
# X = (X - X.mean())/X.std()
# Y = (Y - Y.mean())/Y.std()

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
plt.xlabel('Predicted Total Revenue in Rs. crores')
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
plt.xlabel('Predicted Log Value of Total Revenue')
plt.show()

#--> Does eliminate heteroscedacity. Should use log values of Y for further investigation and modelling

## Investigating likely non-linear response function - residuals v/s predicted values

residuals = Y_hat - Y
plt.figure(figsize = (15, 5))
sns.set(style="whitegrid")
axs = sns.residplot(Y_hat, residuals, lowess = True)
axs.set_xlabel('Predicted Log Value of Total Revenue')
axs.set_ylabel('Residuals')
plt.show()

#--> Clearly a non-linear response function

## Investigating Effect of Unit change in Causes on Response function
X = movie_master.loc[:, ['budget', 'screens']]
X = pd.concat([X, fwr, disti_share], axis = 1)
X.columns = ['budget', 'screens','fwr', 'disti_share']
Y = movie_master['india-nett-gross']

# Standardizing Variables
X_std = (X - X.mean())/X.std()
Y_std = (Y - Y.mean())/Y.std()

model = sm.OLS(Y_std, X_std).fit()
print(model.summary())

imp_unt_chg = model.params['budget']*Y.std()/X['budget'].std()
print('Impact of unit increase in budget on total revenue %.2f' % imp_unt_chg)

imp_unt_chg = model.params['screens']*Y.std()/X['screens'].std()
print('Impact of unit increase in screens on total revenue %.2f' % imp_unt_chg)

imp_unt_chg = model.params['fwr']*Y.std()/X['fwr'].std()
print('Impact of unit increase in first week revenue on total revenue %.2f' % imp_unt_chg)

imp_unt_chg = model.params['disti_share']*Y.std()/X['disti_share'].std()
print('Impact of unit increase in distributor share on total revenue %.2f' % imp_unt_chg)



