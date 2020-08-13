#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:17:44 2020

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
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['budget']
    Y = movie_master.loc[movie_master['release_year'] == year]['genre']
    corr = Y.corr(X, method = 'spearman')
    corr_lst.append(corr)

print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(-1, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.axhline(y = -0.3, color='b', linestyle='--')
plt.title('Spearman Rank Correlation: Budget v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/feats//corr/b_g_cond_y.jpg', dpi = 'figure')
plt.show()

# Budget v/s Release Year
## B v/s Y
corr = movie_master['release_year'].corr(movie_master['budget'], method = 'spearman')
print('%.4f' % corr)

## B v/s Y, conditioned on I
corr = movie_master['release_year'].corr(movie_master['budget_adj'], method = 'spearman')
print('%.4f' % corr)


#%% RUNTIME
## Runtime v/s Genre
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['runtime']
    Y = movie_master.loc[movie_master['release_year'] == year]['genre']
    corr = Y.corr(X, method = 'spearman')
    corr_lst.append(corr)

print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(-1, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.axhline(y = -0.3, color='b', linestyle='--')
plt.title('Spearman Rank Correlation: Runtime v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/feats/r_g_cond_y.jpg', dpi = 'figure')
plt.show()

## Runtime v/s Budget
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['budget']
    Y = movie_master.loc[movie_master['release_year'] == year]['runtime']
    corr = Y.corr(X, method = 'spearman')
    corr_lst.append(corr)

print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.title('Spearman Rank Correlation: Runtime v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/feats/r_b_cond_y.jpg', dpi = 'figure')
plt.show()
plt.close()

## Runtime v/s Release Year
corr = movie_master['release_year'].corr(movie_master['runtime'], method = 'spearman')
print('%.4f' % corr)

plt.figure()
plt.plot(movie_master['release_year'].unique(), movie_master.groupby('release_year')['runtime'].median())
plt.title('Median Runtime v/s Release Year')
plt.ylabel('Minutes')
plt.xlabel('Year of Release')
plt.savefig('./figs/corr/feats/r_y.jpg', dpi = 'figure')
plt.show()

#%% RELEASE WEEK
## Release Week v/s Genre
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    Y = movie_master.loc[movie_master['release_year'] == year]['release_week']
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
plt.title('Spearman Correlation: Release Week v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/feats/w_g_cond_y.jpg', dpi = 'figure')
plt.show()
plt.close()

## Release Week v/s Runtime conditioned on Budget and Year
X = movie_master.loc[:, ['runtime', 'budget']]
X = (X - X.mean())/X.std()
X = pd.concat([X, movie_master['release_year']], axis = 1)
Y = movie_master['release_week']

model = sm.MNLogit(Y.astype('float'), X.astype('float')).fit()
print('The Pseudo R Square for the fitted model: %.4f' % model.prsquared)

s = model.pvalues.loc['runtime',:]

plt.figure()
plt.scatter(s.index, s)
plt.axhline(y = 0.05, color = 'r', linestyle = '--')
plt.ylabel('p-values')
plt.xlabel('Release Week')
plt.xticks(np.arange(1, 54, 3))
plt.title('p-value for Coefficeint of Runtime v/s Release Week')
plt.savefig('./figs/feats/w_r_cond_b_y.jpg')
plt.show()
plt.close()

## Release Week v/s Budget 
    ## conditioned on Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['budget']
    Y = movie_master.loc[movie_master['release_year'] == year]['release_week']
    corr = Y.corr(X, method = 'spearman')
    corr_lst.append(corr)

print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

## Release Week v/s Budget
    ## I as Instrumental Variable

corr_ab = movie_master['inf_adj_fct'].corr(movie_master['release_week'], method = 'spearman')
corr_a = .2922
corr_b = corr_ab/corr_a

print('The Spearman Correlation Coeff: %.4f' % corr_b)

## Release Week v/s Release Year
corr = movie_master['release_year'].corr(movie_master['release_week'], method = 'spearman')
print('%.4f' % corr)

#%% SCREENS
## Screens v/s Release Week
corr = movie_master['release_week'].corr(movie_master['runtime'], method = 'spearman')
print('%.4f' % corr)

## Screens v/s Genre
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    Y = movie_master.loc[movie_master['release_year'] == year]['screens']
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
plt.title('Spearman Correlation: Screens v/s Genre')
plt.grid(axis = 'y')
plt.savefig('./figs/feats/s_g_cond_y.jpg', dpi = 'figure')
plt.show()
plt.close()

## Screens v/s Runtime conditioned on Year and Budget
years = movie_master['release_year'].unique()
p_val_lst = []
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'runtime']]
    Y = movie_master.loc[movie_master['release_year'] == year]['screens']
    X = (X - X.mean())/X.std()
    Y = (Y - Y.mean())/Y.std()
    
    model = sm.OLS(Y, X).fit()
    
    p_val = model.pvalues['runtime']
    p_val_lst.append(p_val)
    
plt.figure()
plt.scatter(years, p_val_lst)
plt.ylim(-0.1, 1)
plt.xlabel('Year of Release')
plt.ylabel('p values')
plt.axhline(y = 0.05, color = 'r', linestyle='--')
plt.title('p-values for Regression Coefficient: Screens v/s Runtime')
plt.grid(axis = 'y')
plt.savefig('./figs/feats/s_r_cond_b_y.jpg', dpi = 'figure')
plt.show()
plt.close()

## Screens v/s Budget
     ## conditioned on Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['budget']
    Y = movie_master.loc[movie_master['release_year'] == year]['screens']
    corr = Y.corr(X, method = 'spearman')
    corr_lst.append(corr)

print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.scatter(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.axhline(y = -0.3, color = 'b', linestyle='--')
plt.title('Spearman Correlation: Screens v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/feats/s_b_cond_y.jpg', dpi = 'figure')
plt.show()
plt.close()

## Screens v/s Release Year
corr = movie_master['release_year'].corr(movie_master['screens'], method = 'spearman')
print('%.4f' % corr)


# #%% EXHIBITOR SHARE
# exhibitor_share = 1- movie_master['india-distributor-share']/movie_master['india-nett-gross']

# ## Exhibitor Share v/s Release Week
# corr = movie_master['release_week'].corr(exhibitor_share, method = 'spearman')
# print('%.4f' % corr)

# ## Exhibitor Share v/s Genre conditioned on Year
# corr_lst = []
# years = movie_master['release_year'].unique()
# for year in years:
#     X = movie_master.loc[movie_master['release_year'] == year]['genre']
#     indx = movie_master.loc[movie_master['release_year'] == year].index
#     Y = exhibitor_share[indx]
#     corr = Y.corr(X, method = 'spearman')
#     corr_lst.append(corr)

# print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

# plt.figure()
# plt.scatter(years, corr_lst)
# plt.ylim(-1, 1)
# plt.xlabel('Year of Release')
# plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
# plt.axhline(y = 0.3, color = 'b', linestyle='--')
# plt.axhline(y = -0.3, color = 'b', linestyle='--')
# plt.title('Spearman Correlation: Exhibitor Share v/s Genre')
# plt.grid(axis = 'y')
# plt.savefig('./figs/corr/feats/e_g_cond_y.jpg', dpi = 'figure')
# plt.show()
# plt.close()

# ## Exhibitor Share v/s Screens conditioned on Budget
# X = movie_master.loc[:, ['budget', 'screens']]
# Y = exhibitor_share
# X = (X - X.mean())/X.std()
# Y = (Y - Y.mean())/Y.std()

# model = sm.OLS(Y, X).fit()
# print(model.summary())

# ## Exhibitor Share v/s Runtime conditioned on Year and Budget
# years = movie_master['release_year'].unique()
# p_val_lst = []
# for year in years:
#     X = movie_master.loc[movie_master['release_year'] == year, ['budget', 'runtime']]
#     indx = movie_master.loc[movie_master['release_year'] == year].index
#     Y = exhibitor_share[indx]
#     X = (X - X.mean())/X.std()
#     Y = (Y - Y.mean())/Y.std()
    
#     model = sm.OLS(Y, X).fit()
    
#     p_val = model.pvalues['runtime']
#     p_val_lst.append(p_val)
    
# plt.figure()
# plt.scatter(years, p_val_lst)
# plt.ylim(-0.1, 1)
# plt.xlabel('Year of Release')
# plt.ylabel('p values')
# plt.axhline(y = 0.05, color = 'r', linestyle='--')
# plt.title('p-values for Regression Coefficient: Exhibitor Share v/s Runtime')
# plt.grid(axis = 'y')
# plt.savefig('./figs/corr/feats/e_r_cond_y_b.jpg', dpi = 'figure')
# plt.show()
# plt.close()

# ## Exhibitor Share v/s Budget
#      ## conditioned on Year
# corr_lst = []
# years = movie_master['release_year'].unique()
# for year in years:
#     X = movie_master.loc[movie_master['release_year'] == year]['budget']
#     indx = movie_master.loc[movie_master['release_year'] == year].index
#     Y = exhibitor_share[indx]
#     corr = Y.corr(X, method = 'spearman')
#     corr_lst.append(corr)

# print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

# plt.figure()
# plt.scatter(years, corr_lst)
# plt.ylim(-1, 0)
# plt.xlabel('Year of Release')
# plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
# plt.axhline(y = 0.3, color = 'b', linestyle='--')
# plt.axhline(y = -0.3, color = 'b', linestyle='--')
# plt.title('Spearman Correlation: Exhibitor Share v/s Budget')
# plt.grid(axis = 'y')
# plt.savefig('./figs/corr/feats/e_b_cond_y.jpg', dpi = 'figure')
# plt.show()
# plt.close()

# ## Exhibitor Share v/s Release Year
# corr = movie_master['release_year'].corr(exhibitor_share, method = 'spearman')
# print('%.4f' % corr)

