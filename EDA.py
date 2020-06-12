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
#weekly_master = pd.read_hdf('./data/weekly_master.h5', key = 'df')

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

#%% Genre

df = pd.crosstab(movie_master['release_interval'], movie_master['genre'], normalize = 'index').round(4)
df = df*100

ax1 = df.plot(kind = 'bar', stacked = True, figsize = (10, 4), colormap = 'tab20_r', legend = False)
ax1.set_xlabel('Year of Release', fontsize = 14)
ax1.set_xticks(np.arange(0, 9, 1))
ax1.set_xticklabels(['1994-1996', '1997-1999', '2000-2002', '2003-2005', '2006-2008', 
                       '2009-2011', '2012-2014', '2015-17', '2018-2019'], rotation = 45, 
                      fontsize = 8)
ax1.set_ylabel('Share in Percentage', fontsize = 14)
ax1.set_title('Share of movie genres in top films in 3 year time intervals', fontweight = 'bold')
ax1.legend(ncol = 1, loc = (1.01, 0))
plt.savefig('./figs/genre_all.jpg', dpi = 'figure')


intervals = movie_master['release_interval'].unique()
mm_top = pd.DataFrame(data = None, columns = ['release_interval', 'genre'])
mm_bot = mm_top
for interval in intervals:
    mm_int = movie_master.query('release_interval == @interval').loc[:, ['release_interval', 'genre']]
    mid = int(len(mm_int)/2)
    mm_top = mm_top.append(mm_int.iloc[:mid, :], ignore_index = True)
    mm_bot = mm_bot.append(mm_int.iloc[mid:, :], ignore_index = True)
    
df = pd.crosstab(mm_top['release_interval'], mm_top['genre'], normalize = 'index').round(4)
df = df*100

ax2 = df.plot(kind = 'bar', stacked = True, figsize = (10, 4), colormap = 'tab20_r', legend = False)
ax2.legend(ncol = 1, loc = (1.01, 0))
ax2.set_xlabel('Year of Release', fontsize = 14)
ax2.set_xticks(np.arange(0, 9, 1))
ax2.set_xticklabels(['1994-1996', '1997-1999', '2000-2002', '2003-2005', '2006-2008', 
                       '2009-2011', '2012-2014', '2015-17', '2018-2019'], rotation = 45, 
                      fontsize = 8)
ax2.set_ylabel('Share in Percentage', fontsize = 14)
ax2.set_title('Share of movie genres in top films in 3 year time intervals - top hhalf', fontweight = 'bold')
ax2.legend(ncol = 1, loc = (1.01, 0))
plt.savefig('./figs/genre_top.jpg', dpi = 'figure')

df = pd.crosstab(mm_bot['release_interval'], mm_bot['genre'], normalize = 'index').round(4)
df = df*100

ax3 = df.plot(kind = 'bar', stacked = True, figsize = (10, 4), colormap = 'tab20_r', legend = False)
ax3.legend(ncol = 1, loc = (1.01, 0))
ax3.set_xlabel('Year of Release', fontsize = 14)
ax3.set_xticks(np.arange(0, 9, 1))
ax3.set_xticklabels(['1994-1996', '1997-1999', '2000-2002', '2003-2005', '2006-2008', 
                       '2009-2011', '2012-2014', '2015-17', '2018-2019'], rotation = 45, 
                      fontsize = 8)
ax3.set_ylabel('Share in Percentage', fontsize = 14)
ax3.set_title('Share of movie genres in top films in 3 year time intervals - bottom half ', fontweight = 'bold')
ax3.legend(ncol = 1, loc = (1.01, 0))
plt.savefig('./figs/genre_bot.jpg', dpi = 'figure')


### Different genres are popular at different time intervals. However, there is no discernable difference between genres across strata


#%% Release Month

df_all = pd.crosstab(movie_master['release_interval'], movie_master['release_month'], normalize = 'index').round(4)
df_all = df_all*100

ax1 = df_all.plot(kind = 'bar', stacked = True, figsize = (10, 4), colormap = 'tab20_r', legend = False)
ax1.set_xlabel('Year of Release', fontsize = 14)
ax1.set_xticks(np.arange(0, 9, 1))
ax1.set_xticklabels(['1994-1996', '1997-1999', '2000-2002', '2003-2005', '2006-2008', 
                       '2009-2011', '2012-2014', '2015-17', '2018-2019'], rotation = 45, 
                      fontsize = 8)
ax1.set_ylabel('Share in Percentage', fontsize = 14)
ax1.set_title('Share of movie genres in top films in 3 year time intervals', fontweight = 'bold')
ax1.legend(ncol = 1, loc = (1.01, 0))
plt.savefig('./figs/release_month_all.jpg', dpi = 'figure')


intervals = movie_master['release_interval'].unique()
mm_top = pd.DataFrame(data = None, columns = ['release_interval', 'release_month'])
mm_bot = mm_top
for interval in intervals:
    mm_int = movie_master.query('release_interval == @interval').loc[:, ['release_interval', 'release_month']]
    mid = int(len(mm_int)/2)
    mm_top = mm_top.append(mm_int.iloc[:mid, :], ignore_index = True)
    mm_bot = mm_bot.append(mm_int.iloc[mid:, :], ignore_index = True)
    
df_top = pd.crosstab(mm_top['release_interval'], mm_top['release_month'], normalize = 'index').round(4)
df_top = df_top*100

ax2 = df_top.plot(kind = 'bar', stacked = True, figsize = (10, 4), colormap = 'tab20_r', legend = False)
ax2.legend(ncol = 1, loc = (1.01, 0))
ax2.set_xlabel('Year of Release', fontsize = 14)
ax2.set_xticks(np.arange(0, 9, 1))
ax2.set_xticklabels(['1994-1996', '1997-1999', '2000-2002', '2003-2005', '2006-2008', 
                       '2009-2011', '2012-2014', '2015-17', '2018-2019'], rotation = 45, 
                      fontsize = 8)
ax2.set_ylabel('Share in Percentage', fontsize = 14)
ax2.set_title('Share of movie genres in top films in 3 year time intervals - top hhalf', fontweight = 'bold')
ax2.legend(ncol = 1, loc = (1.01, 0))
plt.savefig('./figs/release_month_top.jpg', dpi = 'figure')

df_bot = pd.crosstab(mm_bot['release_interval'], mm_bot['release_month'], normalize = 'index').round(4)
df_bot = df_bot*100

ax3 = df_bot.plot(kind = 'bar', stacked = True, figsize = (10, 4), colormap = 'tab20_r', legend = False)
ax3.legend(ncol = 1, loc = (1.01, 0))
ax3.set_xlabel('Year of Release', fontsize = 14)
ax3.set_xticks(np.arange(0, 9, 1))
ax3.set_xticklabels(['1994-1996', '1997-1999', '2000-2002', '2003-2005', '2006-2008', 
                       '2009-2011', '2012-2014', '2015-17', '2018-2019'], rotation = 45, 
                      fontsize = 8)
ax3.set_ylabel('Share in Percentage', fontsize = 14)
ax3.set_title('Share of movie genres in top films in 3 year time intervals - bottom half ', fontweight = 'bold')
ax3.legend(ncol = 1, loc = (1.01, 0))
plt.savefig('./figs/release_month_bot.jpg', dpi = 'figure')


### For the more successful films, the preferred months of release are Aug, Sep and Oct, the  not-preferred months are Feb and July and the rest are neutral
### For the less successful films, the preferred months of release are Sep and Nov, the  not-preferred months are Feb and Dec and the rest are neutral


#%% Correlations

# tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r

['movie_id', 'title', 'release_date', 'runtime', 'genre', 'screens',
       'india-footfalls', 'budget', 'india-nett-gross',
       'india-adjusted-nett-gross', 'india-first-day', 'india-first-weekend',
       'india-first-week', 'india-total-gross', 'india-distributor-share',
       'worldwide-total-gross', 'release_year', 'release_week',
       'release_month', 'release_interval', 'cpi']


movie_master.loc[:, ['india-total-gross', 'genre']].corr() #.iloc[0, 1].round(2)
foo = pd.crosstab(movie_master['release_month'], movie_master['release_interval'], normalize = 'columns').round(4)

#%% Scratch
df = movie_master.query('release_interval == 9').loc[:, ['movie_id', 'india-total-gross']]
df.sort_values(by = 'india-total-gross', inplace = True, ignore_index = True, 
                ascending = False)

col_names = ['month_1', 'month_2']
df_w = pd.DataFrame(data = None, columns = col_names)

for id_ in df['movie_id']:
    s = weekly_master.query('movie_id == @id_').iloc[:, 4:].sum()
    s = s.cumsum()/s.sum()
    s = {'month_1' : s[4]*100, 'month_2' : (s[8] - s[4])*100}
    df_w = df_w.append(s, ignore_index = True)
