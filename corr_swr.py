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
weekly_master = pd.read_hdf('./data/weekly_master.h5')

# Adjusting the first week revenue to account for entertainment and service tax
fwr = movie_master['india-first-week'] * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

# Getting the second week revenue and reindexing the resultant series
swr = weekly_master.groupby('movie_id').sum()['week_2']
swr.reset_index(drop = True, inplace = True)

# # Adjusting the second week revenue to account for entertainment and service tax
swr = swr * (movie_master['india-nett-gross']/movie_master['india-total-gross'])

#%% SECOND WEEK REVENUE
## Second Week Revenue v/s Release Week
corr = movie_master['release_week'].corr(swr, method = 'spearman')
print('%.4f' % corr)

## Second Week Revenue v/s Genre conditioned on Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = swr[indx]
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
plt.title('Spearman Correlation: Second Week Revenue v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/swr_g_cond_y.jpg', dpi = 'figure')
plt.show()
plt.close()

## Second Week Revenue v/s First Week Revenue conditioned on Budget, Screens and Distributor Share
disti_share = movie_master['india-distributor-share']/movie_master['india-total-gross']
X = movie_master.loc[:, ['budget', 'screens']]
X = pd.concat([X, disti_share, fwr], axis = 1)
X.columns = ['budget', 'screens', 'disti_share', 'fwr']
Y = swr
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Second Week Revenue v/s Distributor Share conditioned on Budget
X = pd.concat([movie_master['budget'], disti_share], axis = 1)
X.columns = ['budget', 'disti_share']
Y = swr
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Second Week Revenue v/s Screens conditioned on Budget
X = movie_master.loc[:, ['budget', 'screens']]
Y = swr
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Second Week Revenue v/s Runtime conditioned on Budget and Year
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'runtime']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = swr[indx]
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    model = sm.OLS(Y, X).fit()
    pvalue_lst.append(model.pvalues['runtime'])

plt.figure()
plt.scatter(years, pvalue_lst)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color='r', linestyle='--')
plt.title('p-values for Regression Coefficient: Second Week Revenue v/s Runtime')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/swr_r_cond_y_b.jpg', dpi = 'figure')
plt.show()
plt.close()

## Second Week Revenue v/s Budget, conditioned on Year and Inflation
years = movie_master['release_year'].unique()
corr_lst = []
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, 'budget']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = swr[indx]
    corr = X.corr(Y, method = 'spearman')
    corr_lst.append(corr)
    
print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.title('Spearman Correlation: Second Week Revenue v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/swr_b_cond_y_i.jpg', dpi = 'figure')
plt.show()
plt.close()

## Second Week Revenue v/s Year, conditioned on Inflation
X = movie_master['release_year']
Y = swr * movie_master['inf_adj_fct']
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on Swr, adjusted for inflation, : %.4f' % corr)

X = movie_master['budget']
Y = swr
corr = X.corr(Y, method = 'spearman')
print('Total Effect of B on F : %.4f' % corr)

## Second Week Revenue v/s Year
X = movie_master['release_year']
Y = swr
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on F : %.4f' % corr)


