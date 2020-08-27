#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:17:44 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
from scipy import stats
import statsmodels.api as sm


#%% Loading Data
movie_master = pd.read_pickle('../data/movie_master_en.pkl')
cpi_master = pd.read_csv('../data/CPI.csv')

#%% INFLATION
# Inflation v/s Release Year
corr = cpi_master['Year'].corr(cpi_master['CPI'], method = 'spearman')
print('%.4f' % corr)

#%% GENRE
## Genre v/s Release Year
crosstab = pd.crosstab(movie_master['release_year'], movie_master['genre'])
chi2, p_val, _, _ = stats.chi2_contingency(crosstab)
print('Chi-square statistic = %.2f' % chi2)
print('The p-value of the test = %.6f' % p_val)


#%% BUDGET
## Budget v/s Genre - conditioned on Release Year
X = movie_master.loc[:, ['release_year', 'genre']]
X['release_year'] = X['release_year'].astype('float')
X = pd.get_dummies(X)
Y = movie_master['budget']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Budget v/s Release Year, conditioned on Inflation
X = movie_master.loc[:, ['release_year', 'inf_adj_fct']]
X = X.astype('float')
Y = movie_master['budget']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% RUNTIME
## Runtime v/s Genre, conditioned for Year
X = movie_master.loc[:, ['release_year', 'genre']]
X['release_year'] = X['release_year'].astype('float')
X = pd.get_dummies(X)
Y = movie_master['runtime']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Runtime v/s Budget, conditioned for Year
X = movie_master.loc[:, ['release_year', 'budget']]
X = X.astype('float')
Y = movie_master['runtime']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% RELEASE WEEK
## Release Week v/s Genre, conditioned on Year
X = movie_master.loc[:, ['release_year', 'genre']]
X['release_year'] = X['release_year'].astype('float')
X = pd.get_dummies(X)
Y = movie_master['release_week']
Y = Y.astype('float')

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())


## Release Week v/s Runtime conditioned on Budget and Year
X = movie_master.loc[:, ['release_year', 'budget', 'runtime']]
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['release_week']
Y = Y.astype('float')

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Release Week v/s Budget 
    ## conditioned on Year
X = movie_master.loc[:, ['release_year', 'budget']]
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['release_week']
Y = Y.astype('float')

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% SCREENS
## Screens v/s Release Week, conditioned on Budget
X = movie_master.loc[:, ['release_week', 'budget']]
X['release_week'] = X['release_week'].astype('float')
Y = movie_master['screens']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Screens v/s Genre, conditioned on Year
X = movie_master.loc[:, ['release_year', 'genre']]
X['release_year'] = X['release_year'].astype('float')
X = pd.get_dummies(X)
Y = movie_master['screens']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Screens v/s Runtime conditioned on Year and Budget
X = movie_master.loc[:, ['release_year', 'budget', 'runtime']]
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['screens']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

## Screens v/s Budget conditioned on Year
X = movie_master.loc[:, ['release_year', 'budget']]
X['release_year'] = X['release_year'].astype('float')
Y = movie_master['screens']

X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Release Week v/s Release Year
corr = movie_master['release_year'].corr(movie_master['release_week'], method = 'spearman')
print('%.4f' % corr)