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

#%% Release Year v/s Genre
crosstab = pd.crosstab(movie_master['release_year'], movie_master['genre'])
chi2, p_value, _, _ = stats.chi2_contingency(crosstab)
print('Chi-square statistic = %.2f' % chi2)
print('The p-value of the test = %.6f' % p_value)

#%% Genre v/s Budget - conditioned on Release Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['budget']
    Y = movie_master.loc[movie_master['release_year'] == year]['genre']
    corr = Y.corr(X, method = 'spearman')
    corr_lst.append(corr)

print('Average Spearman Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.plot(years, corr_lst)
plt.ylim(-1, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.axhline(y = -0.3, color='b', linestyle='--')
plt.title('Spearman Rank Correlation: Genre v/s Budget')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/genre_budget_stratified.jpg', dpi = 'figure')
plt.show()

#%% Release Year v/s Budget - conditioned on Inflation
corr = movie_master['release_year'].corr(movie_master['budget_adj'], method = 'spearman')
print('%.4f' % corr)

plt.figure()
plt.plot(movie_master['release_year'].unique(), movie_master.groupby('release_year')['budget_adj'].median()/1000000)
plt.title('Median Budget by Release Year')
plt.ylabel('Rupees in millions')
plt.xlabel('Year of Release')
plt.savefig('./figs/corr/ry_budget.jpg', dpi = 'figure')
plt.show()

#%% Release Year v/s Runtime
corr = movie_master['release_year'].corr(movie_master['runtime'], method = 'spearman')
print('%.4f' % corr)

plt.figure()
plt.plot(movie_master['release_year'].unique(), movie_master.groupby('release_year')['runtime'].median())
plt.title('Release Year v/s Median Runtime')
plt.ylabel('Minutes')
plt.xlabel('Year of Release')
plt.savefig('./figs/corr/ry_runtime.jpg', dpi = 'figure')
plt.show()

#%% Genre v/s Runtime - conditioned on Release Year
chi2_lst = []
pv_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    Y = movie_master.loc[movie_master['release_year'] == year]['runtime']
    crosstab = pd.crosstab(X, Y)
    chi2, p_value, _, _ = stats.chi2_contingency(crosstab)
    chi2_lst.append(chi2)
    pv_lst.append(p_value)

print('Average Chi-square statistic: %.4f' % np.mean(chi2_lst))

plt.figure()
plt.plot(years, pv_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color = 'r', linestyle='-')
plt.title('p-values for Chi-square test statistic: Genre v/s Runtime')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/genre_runtime_stratified.jpg', dpi = 'figure')
plt.show()

#%% Budget v/s Runtime - conditioned on Release Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['budget']
    Y = movie_master.loc[movie_master['release_year'] == year]['runtime']
    corr = Y.corr(X, method = 'pearson')
    corr_lst.append(corr)

print('Average Pearson Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.plot(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.title('Pearson Correlation Coefficient: Budget v/s Runtime')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/budget_runtime_stratified.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Release Year v/s Release Week
crosstab = pd.crosstab(movie_master['release_year'], movie_master['release_week'])
chi2, p_value, _, _ = stats.chi2_contingency(crosstab)
print('Chi-square statistic = %.2f' % chi2)
print('The p-value of the test = %.6f' % p_value)

#%% Runtime v/s Release Week
corr = movie_master['runtime'].corr(movie_master['release_week'], method = 'spearman')
print('%.4f' % corr)

#%% Genre v/s Release Week
crosstab = pd.crosstab(movie_master['genre'], movie_master['release_week'])
chi2, p_value, _, _ = stats.chi2_contingency(crosstab)
print('Chi-square statistic = %.2f' % chi2)
print('The p-value of the test = %.6f' % p_value)

#%% Budget v/s Release Week
corr = movie_master['budget'].corr(movie_master['release_week'], method = 'spearman')
print('%.4f' % corr)

#%% Budget v/s Release Week using Release Year as an Instrumental variable
# Regress Budget on Year
Y = movie_master['budget']
X = movie_master['release_year']

# We normalize X and Y as they are on different scales and then add a constant term for intercept. 
Y = (Y - Y.mean())/Y.std()
X = (X - X.mean())/X.std()
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
print(model.summary())

# Regress Release Week on Release Year
Y = movie_master['release_month'].astype('int')
Y = (Y - Y.mean())/Y.std()

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% Release Year v/s Screens
corr = movie_master['release_year'].corr(movie_master['screens'], method = 'spearman')
print('%.4f' % corr)

plt.figure()
plt.plot(movie_master['release_year'].unique(), movie_master.groupby('release_year')['screens'].median())
plt.title('Release Year v/s Median Screens')
plt.ylabel('Number of Screens')
plt.xlabel('Year of Release')
plt.savefig('./figs/corr/ry_screens.jpg', dpi = 'figure')
plt.show()

#%% Release Week v/s Screens
corr = movie_master['release_week'].astype('int').corr(movie_master['runtime'], method = 'spearman')
print('%.4f' % corr)

plt.figure()
plt.plot(np.sort(movie_master['release_week'].unique()), 
         movie_master.groupby('release_week')['runtime'].median())
plt.title('Release Week v/s Median Screens')
plt.ylabel('Number of Screens')
plt.xlabel('Week of Release')
plt.xticks(np.arange(1, 53, 5))
plt.savefig('./figs/corr/rw_screens.jpg', dpi = 'figure')
plt.show()

#%% Genre v/s Screens - conditioned on Release Year
chi2_lst = []
pv_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['genre']
    Y = movie_master.loc[movie_master['release_year'] == year]['screens']
    crosstab = pd.crosstab(X, Y)
    chi2, p_value, _, _ = stats.chi2_contingency(crosstab)
    chi2_lst.append(chi2)
    pv_lst.append(p_value)

print('Average Chi-square statistic: %.4f' % np.mean(chi2_lst))

plt.figure()
plt.plot(years, pv_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = 0.05, color = 'r', linestyle='-')
plt.title('p-values for Chi-square test statistic: Genre v/s Screens')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/genre_screen_stratified.jpg', dpi = 'figure')
plt.show()

#%% Budget v/s Screens - conditioned on Release Year
corr_lst = []
years = movie_master['release_year'].unique()
for year in years:
    X = movie_master.loc[movie_master['release_year'] == year]['budget']
    Y = movie_master.loc[movie_master['release_year'] == year]['screens']
    corr = Y.corr(X, method = 'pearson')
    corr_lst.append(corr)

print('Average Pearson Correlation Coeff: %.4f' % np.mean(corr_lst))

plt.figure()
plt.plot(years, corr_lst)
plt.ylim(0, 1)
plt.xlabel('Year of Release')
plt.axhline(y = np.mean(corr_lst), color='r', linestyle='-')
plt.axhline(y = 0.3, color = 'b', linestyle='--')
plt.title('Pearson Correlation Coefficient: Budget v/s Screen')
plt.grid(axis = 'y')
plt.savefig('./figs/corr/budget_screen_stratified.jpg', dpi = 'figure')
plt.show()
plt.close()

#%% Runtime v/s Screens - conditioned on Budget
Y = movie_master['screens']
X = movie_master.loc[:, ['budget', 'runtime']]

# We normalize X and Y as they are on different scales and then add a constant term for intercept. 
Y = (Y - Y.mean())/Y.std()
X = (X - X.mean())/X.std()
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
print(model.summary())

#%% DISTRIBUTOR SHARE
disti_share = movie_master['india-distributor-share']/movie_master['india-total-gross']

# Release Year v/s Distributor Share
corr = movie_master['release_year'].corr(disti_share, method = 'spearman')
print('%.4f' % corr)

plt.figure()
plt.plot(movie_master['release_year'].unique(), movie_master.groupby('release_year')['india_first_week_adj'].median()/1000000)
plt.title('Median First Week Revenue by Release Year (conditioned on Inflation)')
plt.ylabel('Rupees in millions')
plt.xlabel('Year of Release')
plt.savefig('./figs/corr/ry-fwr.jpg', dpi = 'figure')
plt.show()

# Release Week v/s Distributor Share
corr = movie_master['release_week'].astype('int').corr(disti_share, method = 'spearman')
print('%.4f' % corr)

# Genre v/s Distributor Share


#%% Release Year v/s First Week Revenue
corr = movie_master['release_year'].corr(movie_master['india_first_week_adj'], method = 'spearman')
print('%.4f' % corr)

plt.figure()
plt.plot(movie_master['release_year'].unique(), movie_master.groupby('release_year')['india_first_week_adj'].median()/1000000)
plt.title('Median First Week Revenue by Release Year (conditioned on Inflation)')
plt.ylabel('Rupees in millions')
plt.xlabel('Year of Release')
plt.savefig('./figs/corr/ry-fwr.jpg', dpi = 'figure')
plt.show()



#%% Genre v/s FWR - conditioned on Release Year
Y = movie_master.loc[movie_master['release_year'] == 2000]['india_first_week_adj']

X = pd.get_dummies(movie_master.loc[movie_master['release_year'] == 2000]['genre'])
X = sm.add_constant(X)
model = sm.OLS(Y/100000,X)
results = model.fit()
results.params
results.tvalues
#%% Year v/s Budget

movie_master['release_year'].corr(movie_master['budget_adj'], method = 'spearman')

plt.figure()
plt.plot(movie_master['release_year'].unique(), movie_master.groupby('release_year')['budget_adj'].median())
plt.show()

