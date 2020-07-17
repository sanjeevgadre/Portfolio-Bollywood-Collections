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


#%% FIRST WEEK REVENUE
## First Week Revenue v/s Release Week
corr = movie_master['release_week'].corr(movie_master['india-first-week'], method = 'spearman')
print('%.4f' % corr)

## First Week Revenue v/s Genre
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    Y = movie_master.loc[movie_master['release_year'] == year]['india-first-week']
    corr = Y.corr(X, method = 'spearman')
    corr_lst.append(corr)

print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.plot(years, corr_lst)
plt.ylim(-1, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.axhline(y = -0.3, color = 'b', linestyle='--')
plt.title('Spearman Correlation: First Week Revenue v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/fwr_g_X_y.jpg', dpi = 'figure')
plt.show()
plt.close()

## First Week Revenue v/s Distributor Share conditioned on Budget
disti_share = movie_master['india-distributor-share']/movie_master['india-total-gross']
X = pd.concat([movie_master['budget'], disti_share], axis = 1)
X.columns = ['budget', 'disti_share']
Y = movie_master['india-first-week']
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## First Week Revenue v/s Screens conditioned on Budget
X = movie_master.loc[:, ['budget', 'screens']]
Y = movie_master['india-first-week']
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## First Week Revenue v/s Runtime conditioned on Budget
X = movie_master.loc[:, ['budget', 'runtime']]
Y = movie_master['india-first-week']
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## First Week Revenue v/s Budget, conditioned on Year and Inflation
years = movie_master['release_year'].unique()
corr_lst = []
for year in years:
    inf_adj_fct = movie_master.loc[movie_master['release_year'] == year, 'inf_adj_fct'].iloc[0]
    X = movie_master.loc[movie_master['release_year'] == year, 'budget'] * inf_adj_fct
    Y = movie_master.loc[movie_master['release_year'] == year, 'india-first-week'] * inf_adj_fct
    corr = X.corr(Y, method = 'spearman')
    corr_lst.append(corr)
    
print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.plot(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.title('Spearman Correlation: First Week Revenue v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/fwr_g_X_y_i.jpg', dpi = 'figure')
plt.show()
plt.close()

## First Week Revenue v/s Year, conditioned on Inflation
X = movie_master['release_year']
Y = movie_master['india-first-week'] * movie_master['inf_adj_fct']
corr = X.corr(Y, method = 'spearman')
print('%.4f' % corr)

# First Week Revenue v/s Yeear
X = movie_master['release_year']
Y = movie_master['india-first-week']
corr = X.corr(Y, method = 'spearman')
print('%.4f' % corr)