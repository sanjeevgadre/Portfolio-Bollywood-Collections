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
plt.savefig('./figs/corr/f_g_cond_y.jpg', dpi = 'figure')
plt.show()
plt.close()

## First Week Revenue v/s Distributor Share conditioned on Budget
disti_share = movie_master['india-distributor-share']/movie_master['india-total-gross']
X = pd.concat([movie_master['budget'], disti_share], axis = 1)
X.columns = ['budget', 'disti_share']
Y = fwr
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

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
plt.savefig('./figs/corr/f_r_cond_y_b.jpg', dpi = 'figure')
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
plt.savefig('./figs/corr/f_b_cond_y_i.jpg', dpi = 'figure')
plt.show()
plt.close()

## First Week Revenue v/s Year, conditioned on Inflation
X = movie_master['release_year']
Y = fwr * movie_master['inf_adj_fct']
corr = X.corr(Y, method = 'spearman')
print('%.4f' % corr)

# First Week Revenue v/s Year
X = movie_master['release_year']
Y = movie_master['india-first-week']
corr = X.corr(Y, method = 'spearman')
print('%.4f' % corr)