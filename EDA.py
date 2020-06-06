#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:15:06 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Exploratory Data Analysis
movie_master = pd.read_pickle('./data/movie_master_en.pkl')

#%% Revenues

# Total revenues for the top 50 films of the year adjusted for inflation
s = movie_master.groupby(['release_year'])['india-adjusted-nett-gross'].sum()

plt.figure(figsize = (13, 5))
plt.plot(s.index, s/1000000000)
plt.title('Total revenues for the top 50 films of the year adjusted for inflation', fontweight = 'bold')
plt.ylabel('Rupees in billions')
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.grid(True)
plt.show()

# Share of total revenue of the top 50 films of the year by quartiles
years = movie_master['release_year'].unique()
col_names = ['Quartile-' + str(i) for i in np.arange(1, 5, 1)]
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year')['india-nett-gross']
    s.reset_index(drop = True, inplace = True)
    s = (s.cumsum()/s.sum()) * 100
    df.loc[df.index == year, col_names] = [s[12], s[24] - s[12], s[36] - s[24], s[49] - s[36]]

plt.figure(figsize = (13, 5))
for name in col_names:
    plt.plot(df.index, df[name], label = name)
plt.legend()
plt.title('Share of total revenue of the top 50 films of the year by quartiles', fontweight = 'bold')
plt.ylabel('Share in %')
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.grid(True)
plt.show()

#%% Footfalls

# Total footfalls for the top 50 films of the year
s = movie_master.groupby(['release_year'])['india-footfalls'].sum()

plt.figure(figsize = (13, 5))
plt.plot(s.index, s/1000000000)
plt.title('Total footfalls for the top 50 films of the year', fontweight = 'bold')
plt.ylabel('Footfalls in billions')
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.grid(True)
plt.show()

# Share of total footfalls of the top 50 films of the year by revenue quartiles
years = movie_master['release_year'].unique()
col_names = ['Quartile-' + str(i) for i in np.arange(1, 5, 1)]
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year')['india-footfalls']
    s.reset_index(drop = True, inplace = True)
    s = (s.cumsum()/s.sum()) * 100
    df.loc[df.index == year, col_names] = [s[12], s[24] - s[12], s[36] - s[24], s[49] - s[36]]

plt.figure(figsize = (13, 5))
for name in col_names:
    plt.plot(df.index, df[name], label = name)
plt.legend()
plt.title('Share of total footfalls of the top 50 films of the year by revenue quartiles', fontweight = 'bold')
plt.ylabel('Share in %')
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.grid(True)
plt.show()

#%% Budget

# Mean budget for top 50 films of the year by revenue quartiles
years = movie_master['release_year'].unique()
col_names = ['Quartile-' + str(i) for i in np.arange(1, 5, 1)]
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year')['budget']
    s.reset_index(drop = True, inplace = True)
    df.loc[df.index == year, col_names] = [s[:13].mean(), s[13:25].mean(), s[25:37].mean(), s[37:].mean()]

plt.figure(figsize = (13, 5))
for name in col_names:
    plt.plot(df.index, df[name]/1000000000, label = name)
plt.legend()
plt.title('Mean budget for top 50 films of the year by revenue quartiles', fontweight = 'bold')
plt.ylabel('Rupees in billions')
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.grid(True)
plt.show()

# Mean return-on-budget for top 50 films of the year by revenue quartiles
years = movie_master['release_year'].unique()
col_names = ['Quartile-' + str(i) for i in np.arange(1, 5, 1)]
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year')['ret_on_budget']
    s.reset_index(drop = True, inplace = True)
    df.loc[df.index == year, col_names] = [s[:13].mean(), s[13:25].mean(), s[25:37].mean(), s[37:].mean()]

plt.figure(figsize = (13, 5))
for name in col_names:
    plt.plot(df.index, df[name], label = name)
plt.legend()
plt.title('Mean return-on-budget for top 50 films of the year by revenue quartiles', fontweight = 'bold')
plt.ylabel('times of budget')
plt.ylim(0, 3.5)
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.grid(True)
plt.show()

# Maximum and minimum return-on-budget for top 10 films of the year
years = movie_master['release_year'].unique()
col_names = ['Quantile-1-Max', 'Quantile-1-Min']
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year')['ret_on_budget']
    s.reset_index(drop = True, inplace = True)
    df.loc[df.index == year, col_names] = [s[:10].max(), s[:10].min()]

plt.figure(figsize = (13, 5))
for name in col_names:
    plt.plot(df.index, df[name], label = name)
plt.legend()
plt.title('Maximum and minimum return-on-budget for top 10 films of the year', fontweight = 'bold')
plt.ylabel('times of budget')
plt.ylim(0, 5)
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.grid(True)
plt.show()

#%% Screens

# Median number of release screens for top 50 films of the year
s = movie_master.groupby(['release_year'])['screens'].median()

plt.figure(figsize = (13, 5))
plt.plot(s.index, s)
plt.title('Median number of release screens for top 50 films of the year', fontweight = 'bold')
plt.ylabel('# Screens')
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.grid(True)
plt.show()

# Median number of release screns for top 50 films of the year by revenue quartiles
years = movie_master['release_year'].unique()
col_names = ['Quartile-' + str(i) for i in np.arange(1, 5, 1)]
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year')['screens']
    s.reset_index(drop = True, inplace = True)
    df.loc[df.index == year, col_names] = [s[:13].median(), s[13:25].median(), s[25:37].median(), s[37:].median()]

plt.figure(figsize = (13, 5))
for name in col_names:
    plt.plot(df.index, df[name], label = name)
plt.legend()
plt.title('Median number of release screns for top 50 films of the year by revenue quartiles', fontweight = 'bold')
plt.ylabel('times of budget')
#plt.ylim(0, 3.5)
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.grid(True)
plt.show()

# Distribution of movie genres


#%% Scratch
s = movie_master.groupby('release_interval')['genre'].value_counts(sort = False, normalize = True)
s.loc[s.index.get_level_values('release_interval') == 1]
