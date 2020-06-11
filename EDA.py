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

#Factor to adjust historical rupee values to current values
#inf_fct_curr = movie_master['cpi'].max()

#%% Revenues

fig, ax = plt.subplots(nrows = 3, sharex = 'col', figsize = (10, 12))

s = movie_master.groupby('release_year')['india-total-gross'].sum()
ax[0].plot(s.index, s/1000000000)

years = movie_master['release_year'].unique()
col_names = ['Median: Top 25', 'Median: Next 25',
             'Share: Top 25', 'Share: Next 25']
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year')['india-total-gross']
    s.reset_index(drop = True, inplace = True)
    df.loc[df.index == year, col_names] = [s[:25].median(), s[25:].median(), 
                                           s[:25].sum()/s.sum(), s[25:].sum()/s.sum()]
for name in col_names[:2]:
    ax[1].plot(df.index, df[name]/1000000000, label = name)
        
for name in col_names[2:]:
    ax[2].plot(df.index, df[name]*100, label = name)

for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title('Total India revenues for the top 50 films of the year', fontweight = 'bold')
ax[1].set_title('Median India revenues for the top 50 films of the year', fontweight = 'bold')
ax[2].set_title('Share of total India revenues for the top 50 films of the year', fontweight = 'bold')

ax[0].set_ylabel('Rupees in billions', fontsize = 14)
ax[1].set_ylabel('Rupees in billions', fontsize = 14)
ax[2].set_ylabel('Share in %', fontsize = 14)

ax[2].set_xlabel('Year of Release', fontsize = 14)
ax[2].set_xticks(np.arange(1994, 2020, 1))
ax[2].set_xticklabels(np.arange(1994, 2020, 1), rotation = 45, fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/revenue.jpg', dpi = 'figure')

plt.show()
plt.close()

### Total revenues have expanded from ~Rs.4.9B in 1994 to ~Rs.48.8B in 2019, a CAGR of 9.24%

### Median revenues for
    ### the top 25 films expanded from ~Rs.98M in 1994 to ~Rs.1636M in 2019, a CAGR of 11.43%
    ### the next 25 films expanded from ~Rs.51M in 1994 to ~Rs.234M in 2019, a CAGR of 6.03%

### Top 25 films account for ~76% of 1994 revenues versus ~87% of 2019 revenues

#%% Footfalls

fig, ax = plt.subplots(nrows = 3, sharex = 'col', figsize = (10, 12))

s = movie_master.groupby('release_year')['india-footfalls'].sum()
ax[0].plot(s.index, s/1000000)

years = movie_master['release_year'].unique()
col_names = ['Median: Top 25', 'Median: Next 25',
             'Share: Top 25', 'Share: Next 25']
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year')['india-footfalls']
    s.reset_index(drop = True, inplace = True)
    df.loc[df.index == year, col_names] = [s[:25].median(), s[25:].median(), 
                                           s[:25].sum()/s.sum(), s[25:].sum()/s.sum()]
for name in col_names[:2]:
    ax[1].plot(df.index, df[name]/1000000, label = name)
        
for name in col_names[2:]:
    ax[2].plot(df.index, df[name]*100, label = name)

for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title('Total India footfalls for the top 50 films of the year', fontweight = 'bold')
ax[1].set_title('Median India footfalls for the top 50 films of the year', fontweight = 'bold')
ax[2].set_title('Share of India footfalls for the top 50 films of the year', fontweight = 'bold')

ax[0].set_ylabel('Footfalls in millions', fontsize = 14)
ax[1].set_ylabel('Footfalls in millions', fontsize = 14)
ax[2].set_ylabel('Share in %', fontsize = 14)

ax[2].set_xlabel('Year of Release', fontsize = 14)
ax[2].set_xticks(np.arange(1994, 2020, 1))
ax[2].set_xticklabels(np.arange(1994, 2020, 1), rotation = 45, fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/footfalls.jpg', dpi = 'figure')

plt.show()
plt.close()

### Total footfalls have contracted from ~431M to 307M, a CAGR of -1.3% and an overall contraction of ~26% from 1994. In 2004, at its lowest point footfalls were down to ~183M.

### Median footfalls for
    ### the top 25 films expanded from 9M in 1994 to 10M in 2019
    ### the next 25 films expanded from ~Rs.4.5M in 1994 to ~Rs.1.3M in 2019

### Top 25 films account for ~73% of 1994 footfalls versus ~87% of 2019 footfalls.

#%% Budgets

fig, ax = plt.subplots(nrows = 3, sharex = 'col', figsize = (10, 12))

s = movie_master.groupby('release_year')['budget'].sum()
ax[0].plot(s.index, s/1000000000)

years = movie_master['release_year'].unique()
col_names = ['Median: Top 25', 'Median: Next 25',
             'Share: Top 25', 'Share: Next 25']
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year')['budget']
    s.reset_index(drop = True, inplace = True)
    df.loc[df.index == year, col_names] = [s[:25].median(), s[25:].median(), 
                                           s[:25].sum()/s.sum(), s[25:].sum()/s.sum()]
for name in col_names[:2]:
    ax[1].plot(df.index, df[name]/1000000000, label = name)
        
for name in col_names[2:]:
    ax[2].plot(df.index, df[name]*100, label = name)

for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title('Total budget for the top 50 films of the year', fontweight = 'bold')
ax[1].set_title('Median budget for the top 50 films of the year', fontweight = 'bold')
ax[2].set_title('Share of total budget for the top 50 films of the year', fontweight = 'bold')

ax[0].set_ylabel('Rupees in billions', fontsize = 14)
ax[1].set_ylabel('Rupees in billions', fontsize = 14)
ax[2].set_ylabel('Share in %', fontsize = 14)

ax[2].set_xlabel('Year of Release', fontsize = 14)
ax[2].set_xticks(np.arange(1994, 2020, 1))
ax[2].set_xticklabels(np.arange(1994, 2020, 1), rotation = 45, fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/budget.jpg', dpi = 'figure')

plt.show()
plt.close()

### Total budget have expanded from ~Rs.4.9B in 1994 to ~Rs.48.8B in 2019, a CAGR of 9.24%

### Median budget for
    ### the top 25 films expanded from ~Rs.28M in 1994 to ~Rs.840M in 2019, a CAGR of 14%
    ### the next 25 films expanded from ~Rs.19M in 1994 to ~Rs.310M in 2019, a CAGR of 11%

### Top 25 films account for ~76% of 1994 budget versus ~87% of 2019 budget

#%% Screens

fig, ax = plt.subplots(nrows = 2, sharex = 'col', figsize = (10, 8))

intervals = movie_master['release_interval'].unique().astype('int')
col_names = ['Median: All', 'Median: Top Half', 'Median: Bottom Half']
df = pd.DataFrame(data = None, index = intervals, columns = col_names)
for interval in intervals:
    s = movie_master.query('release_interval == @interval').loc[:, ['india-total-gross', 'screens']]
    s.sort_values(by = ['india-total-gross'], inplace = True, ascending = False, 
                  ignore_index = True)
    s = s['screens']
    mid = int(len(s)/2)
    df.loc[df.index == interval, col_names] = [s.median(), s[:mid].median(), s[mid:].median()]

ax[0].plot(df.index, df['Median: All'])

for name in col_names[-2:]:
    ax[1].plot(df.index, df[name], label = name)
        
for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title('Median no. of release screens for top films in 3 year time intervals', fontweight = 'bold')
ax[1].set_title('Median no. of release screens for top films in 3 year time intervals - Stratified by Total Gross Revenue', fontweight = 'bold')

ax[0].set_ylabel('Number of screens', fontsize = 14)
ax[1].set_ylabel('Number of screens', fontsize = 14)

ax[1].set_xlabel('Year of Release', fontsize = 14)
ax[1].set_xticks(np.arange(1, 10, 1))
ax[1].set_xticklabels(['1994-1996', '1997-1999', '2000-2002', '2003-2005', '2006-2008', 
                       '2009-2011', '2012-2014', '2015-17', '2018-2019'], rotation = 45, 
                      fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/screens.jpg', dpi = 'figure')

plt.show()
plt.close()

### The median number of release screens have expanded from ~158 in 1994-197 to 1925 in 2018-19, a CAGR of 11% 

### Median number of release screens for films with total gross revenue
    ### above the median, expanded from 190 in 1994 to 2975 in 2019
    ### below the median, expanded from 130 in 1994 to 1425 in 2019
    
### In 1994-97, films grossing above the median had opened at ~50% more number of screens than films grossing below the median. In 2018-19 that number is ~109%.

#%% Runtime

fig, ax = plt.subplots(nrows = 2, sharex = 'col', figsize = (10, 8))

intervals = movie_master['release_interval'].unique().astype('int')
col_names = ['Median: All', 'Median: Top Half', 'Median: Bottom Half']
df = pd.DataFrame(data = None, index = intervals, columns = col_names)
for interval in intervals:
    s = movie_master.query('release_interval == @interval').loc[:, ['india-total-gross', 'runtime']]
    s.sort_values(by = ['india-total-gross'], inplace = True, ascending = False, 
                  ignore_index = True)
    s = s['runtime']
    mid = int(len(s)/2)
    df.loc[df.index == interval, col_names] = [s.median(), s[:mid].median(), s[mid:].median()]

ax[0].plot(df.index, df['Median: All'])

for name in col_names[-2:]:
    ax[1].plot(df.index, df[name], label = name)
        
for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title('Median runtime for top films in 3 year time intervals', fontweight = 'bold')
ax[1].set_title('Median runtime for top films in 3 year time intervals - Stratified by Total Gross Revenue', fontweight = 'bold')

ax[0].set_ylabel('Runtime in minutes', fontsize = 14)
ax[1].set_ylabel('Runtime in minutes', fontsize = 14)

ax[1].set_xlabel('Year of Release', fontsize = 14)
ax[1].set_xticks(np.arange(1, 10, 1))
ax[1].set_xticklabels(['1994-1996', '1997-1999', '2000-2002', '2003-2005', '2006-2008', 
                       '2009-2011', '2012-2014', '2015-17', '2018-2019'], rotation = 45, 
                      fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/runtime.jpg', dpi = 'figure')

plt.show()
plt.close()

### The median runtime has contracted from ~156 mins. in 1994-97 to 139 mins. in 2018-19, a drop of 17 minutes or ~10% 

### Median number of release screens for films with total gross revenue
    ### above the median, contracted from 160 mins. in 1994 to 141 mins in 2019, a drop of 19 mins. or ~12%
    ### below the median, contracted from 152 mins. in 1994 to 139 mins in 2019, a drop of 13 mins. or ~9%
    
### Over the 26 year period, films have shortened in runtime by about 10%. This shortening is minimally different for more-successful and less-successful films.

#%% Distributor's Share

fig, ax = plt.subplots(nrows = 2, sharex = 'col', figsize = (10, 8))

intervals = movie_master['release_interval'].unique().astype('int')
col_names = ['Median: All', 'Median: Top Half', 'Median: Bottom Half']
df = pd.DataFrame(data = None, index = intervals, columns = col_names)
for interval in intervals:
    s = movie_master.query('release_interval == @interval').loc[:, ['india-total-gross', 'india-distributor-share']]
    s.sort_values(by = ['india-total-gross'], inplace = True, ascending = False, 
                  ignore_index = True)
    s = s['india-distributor-share']/s['india-total-gross']
    mid = int(len(s)/2)
    df.loc[df.index == interval, col_names] = [s.median(), s[:mid].median(), s[mid:].median()]

ax[0].plot(df.index, df['Median: All']*100)

for name in col_names[-2:]:
    ax[1].plot(df.index, df[name]*100, label = name)
        
for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title("Distributors' share of total gross for top films in 3 year time intervals", fontweight = 'bold')
ax[1].set_title("Distributors' share of total gross for top films in 3 year time intervals - Stratified by Total Gross Revenue", fontweight = 'bold')

ax[0].set_ylabel('Share in Percentage', fontsize = 14)
ax[1].set_ylabel('Share in Percentage', fontsize = 14)

ax[1].set_xlabel('Year of Release', fontsize = 14)
ax[1].set_xticks(np.arange(1, 10, 1))
ax[1].set_xticklabels(['1994-1996', '1997-1999', '2000-2002', '2003-2005', '2006-2008', 
                       '2009-2011', '2012-2014', '2015-17', '2018-2019'], rotation = 45, 
                      fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/disti_share.jpg', dpi = 'figure')

plt.show()
plt.close()

### The median distributors' share has expanded from ~29% in 1994-97 to 38% in 2018-19, an increase of 9 percentage points 

### Median distributors' share for films with total gross revenue
    ### above the median, expanded from ~30% in 1994 to ~39% in 2019, an increase of 9 percentage points
    ### below the median, expanded from ~25% in 1994 to ~37% in 2019, an increase of 12 percentage points
    
### Over the 26 year period, distributors' share of total gross has significantly increased. Since around 2009, distributors are commanding the same share of a film's total gross irrespecive of how successful the film is. This isn't true of the period from 1994 to 2008, when the more successful films were parting with a much larger share of their total gross than the less successful films. One could draw possibly two conclusions:
    ### In recent times, a film has very little leverage in determining the distributors' share of the total gross
    ### In recent times, a distributors' share likely has no impact on a film's fortunes. That is starkly different from what is observerd in the earlier times.
    
#%% First Week Revenue

fig, ax = plt.subplots(nrows = 2, sharex = 'col', figsize = (10, 8))

intervals = movie_master['release_interval'].unique().astype('int')
col_names = ['Median: All', 'Median: Top Half', 'Median: Bottom Half']
df = pd.DataFrame(data = None, index = intervals, columns = col_names)
for interval in intervals:
    s = movie_master.query('release_interval == @interval').loc[:, ['india-total-gross', 'india-first-week']]
    s.sort_values(by = ['india-total-gross'], inplace = True, ascending = False, 
                  ignore_index = True)
    s = s['india-first-week']/s['india-total-gross']
    mid = int(len(s)/2)
    df.loc[df.index == interval, col_names] = [s.median(), s[:mid].median(), s[mid:].median()]

ax[0].plot(df.index, df['Median: All']*100)

for name in col_names[-2:]:
    ax[1].plot(df.index, df[name]*100, label = name)
        
for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title("First week gross as % of total gross for top films in 3 year time intervals", fontweight = 'bold')
ax[1].set_title("First week gross as % of total gross for top films in 3 year time intervals - Stratified by Total Gross Revenue", fontweight = 'bold')

ax[0].set_ylabel('Share in Percentage', fontsize = 14)
ax[1].set_ylabel('Share in Percentage', fontsize = 14)

ax[1].set_xlabel('Year of Release', fontsize = 14)
ax[1].set_xticks(np.arange(1, 10, 1))
ax[1].set_xticklabels(['1994-1996', '1997-1999', '2000-2002', '2003-2005', '2006-2008', 
                       '2009-2011', '2012-2014', '2015-17', '2018-2019'], rotation = 45, 
                      fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/first_week.jpg', dpi = 'figure')

plt.show()
plt.close()

### The first week's share has expanded from ~27% in 1994-97 to ~59% in 2018-19, an increase of 32 percentage points 

### Median first week's share for films with total gross revenue
    ### above the median, expanded from ~22% in 1994 to ~52% in 2019, an increase of 30 percentage points
    ### below the median, expanded from ~30% in 1994 to ~70% in 2019, an increase of 40 percentage points
    
### Over the 26 year period, a film's stay at the theaters has shortened, given how much of the total gross is earned in the first week. This is even more evident for the less successful films. If a film is deemed to be unsuccessful, it doesn't get to stay in the theatre. While this has been always true, it is now significantly more acute.

#%% Commentry

# The film industry is under pressure, growing revenues relatively slowly and lost over a quarter of its footfalls over the last 26 years. Add to it the fact that "winners" are cornering an increasing share of the revenue. This makes it critical to analyse what factors drive/predict a winner.

# We investigate the correlation of india total gross revenue to
    # budget, 
    # India first day collection
    # India first weekend collection
    # India first week collection
    # distributor's share of gross revenue, 
    # number of screens the film opens to, 
    # zeitgeist (genre of film, aggregated over release interval), 
    # release month, 
    # running time

#%% Correlations

['movie_id', 'title', 'release_date', 'runtime', 'genre', 'screens',
       'india-footfalls', 'budget', 'india-nett-gross',
       'india-adjusted-nett-gross', 'india-first-day', 'india-first-weekend',
       'india-first-week', 'india-total-gross', 'india-distributor-share',
       'worldwide-total-gross', 'release_year', 'release_week',
       'release_month', 'release_interval', 'cpi']


movie_master.loc[:, ['india-total-gross', 'genre']].corr() #.iloc[0, 1].round(2)
pd.crosstab(movie_master['genre'], movie_master['release_interval'], normalize = 1)

