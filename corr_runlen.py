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

# Getting the run length of films
df = weekly_master.groupby('movie_id').sum().iloc[:, 3:]
run_length = df.apply(lambda x: np.count_nonzero(x), axis = 1)
run_length.reset_index(drop = True, inplace = True)
#%% RUN LENGTH
## Run Length v/s Release Week
corr = movie_master['release_week'].corr(run_length, method = 'spearman')
print('%.4f' % corr)

## Run Length v/s Genre conditioned on Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = run_length[indx]
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

## Run Length v/s First Week Revenue conditioned on Budget and Screens
X = movie_master.loc[:, ['budget', 'screens']]
X = pd.concat([X, fwr], axis = 1)
X.columns = ['budget', 'screens', 'fwr']
Y = run_length
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Run Length v/s Screens conditioned on Budget
X = movie_master.loc[:, ['budget', 'screens']]
Y = run_length
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Run Length v/s Runtime conditioned on Budget and Year
pvalue_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'runtime']]
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = run_length[indx]
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

## Run Length v/s Budget, conditioned on Year
years = movie_master['release_year'].unique()
corr_lst = []
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, 'budget']
    indx = movie_master.loc[movie_master['release_year'] == year].index
    Y = run_length[indx]
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

## Run Length v/s Year
X = movie_master['release_year']
Y = run_length
corr = X.corr(Y, method = 'spearman')
print('Total Effect of Y on F : %.4f' % corr)

#%%


