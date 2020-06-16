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
weekly_master = pd.read_hdf('./data/weekly_master.h5', key = 'df')
region_master = pd.read_hdf('./data/region_master.h5', key = 'df')

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
ax[2].set_xticks(years)
ax[2].set_xticklabels(years, rotation = 45, fontsize = 8)

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
ax[2].set_xticks(years)
ax[2].set_xticklabels(years, rotation = 45, fontsize = 8)

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

years = movie_master['release_year'].unique()
col_names = ['Median: All', 'Median: Top Half', 'Median: Bottom Half']
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year').loc[:, ['india-total-gross', 'screens']]
    s.reset_index(drop = True, inplace = True)
    s = s['screens']
    mid = int(len(s)/2)
    df.loc[df.index == year, col_names] = [s.median(), s[:mid].median(), s[mid:].median()]

ax[0].plot(df.index, df['Median: All'])

for name in col_names[-2:]:
    ax[1].plot(df.index, df[name], label = name)
        
for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title('Median no. of release screens for top films of the year', fontweight = 'bold')
ax[1].set_title('Median no. of release screens for top films of the year - Stratified by Total Gross Revenue', fontweight = 'bold')

ax[0].set_ylabel('Number of screens', fontsize = 14)
ax[1].set_ylabel('Number of screens', fontsize = 14)

ax[1].set_xlabel('Year of Release', fontsize = 14)
ax[1].set_xticks(years)
ax[1].set_xticklabels(years, rotation = 45, fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/screens.jpg', dpi = 'figure')

plt.show()
plt.close()

### The median number of release screens have expanded from 140 in 1994 to 2325 in 2018-19, a CAGR of ~11.41% 

### Median number of release screens for films with total gross revenue
    ### above the median, expanded from 170 in 1994 to 3100 in 2019
    ### below the median, expanded from 135 in 1994 to 1500 in 2019
    
### In 1994, films grossing above the median had opened at ~26% more number of screens than films grossing below the median. In 2018-19 that number is ~107%.

#%% Runtime

fig, ax = plt.subplots(nrows = 2, sharex = 'col', figsize = (10, 8))

years = movie_master['release_year'].unique()
col_names = ['Median: All', 'Median: Top Half', 'Median: Bottom Half']
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year').loc[:, ['india-total-gross', 'runtime']]
    s.reset_index(drop = True, inplace = True)
    s = s['runtime']
    mid = int(len(s)/2)
    df.loc[df.index == year, col_names] = [s.median(), s[:mid].median(), s[mid:].median()]

ax[0].plot(df.index, df['Median: All'])

for name in col_names[-2:]:
    ax[1].plot(df.index, df[name], label = name)
        
for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title('Median runtime for top films of the year', fontweight = 'bold')
ax[1].set_title('Median runtime for top films of the year - Stratified by revenue', fontweight = 'bold')

ax[0].set_ylabel('Runtime in minutes', fontsize = 14)
ax[1].set_ylabel('Runtime in minutes', fontsize = 14)

ax[1].set_xlabel('Year of Release', fontsize = 14)
ax[1].set_xticks(years)
ax[1].set_xticklabels(years, rotation = 45, fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/runtime.jpg', dpi = 'figure')

plt.show()
plt.close()

### The median runtime has contracted from 156 mins. in 1994 to 140 mins. in 2019, a drop of 16 minutes or ~10% 

### Median number of release screens for films with total gross revenue
    ### above the median, contracted from 161 mins. in 1994 to 146 mins in 2019, a drop of 15 mins. or ~9%
    ### below the median, contracted from 152 mins. in 1994 to 134 mins in 2019, a drop of 18 mins. or ~12%
    
### Over the 26 year period, films have shortened in runtime by about 10%. This shortening is minimally different for more-successful and less-successful films.

#%% Distributors' share

fig, ax = plt.subplots(nrows = 2, sharex = 'col', figsize = (10, 8))

years = movie_master['release_year'].unique()
col_names = ['Median: All', 'Median: Top Half', 'Median: Bottom Half']
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year').loc[:, ['india-total-gross', 'india-distributor-share']]
    s.reset_index(drop = True, inplace = True)
    s = s['india-distributor-share']/s['india-total-gross']
    mid = int(len(s)/2)
    df.loc[df.index == year, col_names] = [s.median(), s[:mid].median(), s[mid:].median()]

ax[0].plot(df.index, df['Median: All']*100)

for name in col_names[-2:]:
    ax[1].plot(df.index, df[name]*100, label = name)
        
for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title("Distributors' share of total gross for top films of the year", fontweight = 'bold')
ax[1].set_title("Distributors' share of total gross for top films of the year - Stratified by Total Gross Revenue", fontweight = 'bold')

ax[0].set_ylabel('Share in Percentage', fontsize = 14)
ax[1].set_ylabel('Share in Percentage', fontsize = 14)

ax[1].set_xlabel('Year of Release', fontsize = 14)
ax[1].set_xticks(years)
ax[1].set_xticklabels(years, rotation = 45, fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/disti_share.jpg', dpi = 'figure')

plt.show()
plt.close()

### The median distributors' share has expanded from ~25% in 1994 to ~39% in 2019, an increase of 14 percentage points 

### Median distributors' share for films with total gross revenue
    ### above the median, expanded from ~29% in 1994 to ~40% in 2019, an increase of 11 percentage points
    ### below the median, expanded from ~23% in 1994 to ~39% in 2019, an increase of 16 percentage points
    
### Over the 26 year period, distributors' share of total gross has significantly increased. Since around 2010, distributors are commanding the same share of a film's total gross irrespecive of how successful the film is. This isn't true of the period from 1994 to 2009, when the more successful films were parting with a much larger share of their total gross than the less successful films. One could draw possibly two conclusions:
    ### In recent times, a film has very little leverage in determining the distributors' share of the total gross.
    ### In recent times, a distributors' share likely has no impact on a film's fortunes. That is different from likely happened in the earlier times.

#%% First Week Revenue

fig, ax = plt.subplots(nrows = 2, sharex = 'col', figsize = (10, 8))

years = movie_master['release_year'].unique()
col_names = ['Median: All', 'Median: Top Half', 'Median: Bottom Half']
df = pd.DataFrame(data = None, index = years, columns = col_names)
for year in years:
    s = movie_master.query('release_year == @year').loc[:, ['india-total-gross', 'india-first-week']]
    s.reset_index(inplace = True, drop = True)
    s = s['india-first-week']/s['india-total-gross']
    mid = int(len(s)/2)
    df.loc[df.index == year, col_names] = [s.median(), s[:mid].median(), s[mid:].median()]

ax[0].plot(df.index, df['Median: All']*100)

for name in col_names[-2:]:
    ax[1].plot(df.index, df[name]*100, label = name)
        
for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

ax[0].set_title("First week gross as % of total gross for top films of the year", fontweight = 'bold')
ax[1].set_title("First week gross as % of total gross for top films of the year - Stratified by Total Gross Revenue", fontweight = 'bold')

ax[0].set_ylabel('Share in Percentage', fontsize = 14)
ax[1].set_ylabel('Share in Percentage', fontsize = 14)

ax[1].set_xlabel('Year of Release', fontsize = 14)
ax[1].set_xticks(years)
ax[1].set_xticklabels(years, rotation = 45, fontsize = 8)

# Save before you Show. Show "creates" a new figure.
plt.savefig('./figs/first_week.jpg', dpi = 'figure')

plt.show()
plt.close()

### The first week's share has expanded from ~23% in 1994 to ~62% in 2019, an increase of 39 percentage points 

### Median first week's share for films with total gross revenue
    ### above the median, expanded from ~21% in 1994 to ~53% in 2019, an increase of 32 percentage points
    ### below the median, expanded from ~29% in 1994 to ~71% in 2019, an increase of 42 percentage points
    
### Over the 26 year period, a film's stay at the theaters has shortened, given how much of the total gross is earned in the first week. This is even more evident for the less successful films. If a film is deemed to be unsuccessful, it doesn't get to stay in the theatre. While this has been always true, it is now significantly more acute.

#%% Release Month

df_all = pd.crosstab(movie_master['release_year'], movie_master['release_month'], normalize = 'index').round(4)
df_all = df_all*100

years = movie_master['release_year'].unique()
mm_top = pd.DataFrame(data = None, columns = ['release_year', 'release_month'])
mm_bot = mm_top
for year in years:
    mm_year = movie_master.query('release_year == @year').loc[:, ['release_year', 'release_month']]
    mm_top = mm_top.append(mm_year.iloc[:25, :], ignore_index = True)
    mm_bot = mm_bot.append(mm_year.iloc[25:, :], ignore_index = True)
    
df_top = pd.crosstab(mm_top['release_year'], mm_top['release_month'], normalize = 'index').round(4)
df_top = df_top*100

df_bot = pd.crosstab(mm_bot['release_year'], mm_bot['release_month'], normalize = 'index').round(4)
df_bot = df_bot*100

fig, axs = plt.subplots(nrows = 3, sharex = 'col', figsize = (10, 12))

axs[0].plot(df_all.columns, df_all.mean())
axs[0].set_title('Share of release month for top films during 1994-2019', fontweight = 'bold')
axs[0].set_ylabel('Share in Percentage', fontsize = 14)
axs[0].grid(True)
axs[0].axhline(y = 100/12, color = 'r', linestyle = '--')

axs[1].plot(df_top.columns, df_top.mean())
axs[1].set_title('Share of release month for top films during 1994-2019 - Top Half', fontweight = 'bold')
axs[1].set_ylabel('Share in Percentage', fontsize = 14)
axs[1].grid(True)
axs[1].axhline(y = 100/12, color = 'r', linestyle = '--')

axs[2].plot(df_bot.columns, df_bot.mean())
axs[2].set_title('Share of release month for top films during 1994-2019 - Bottom Half', fontweight = 'bold')
axs[2].set_ylabel('Share in Percentage', fontsize = 14)
axs[2].grid(True)
axs[2].axhline(y = 100/12, color = 'r', linestyle = '--')
axs[2].set_xticks(np.arange(1, 13, 1))
axs[2].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation = 45, fontsize = 8)
axs[2].set_xlabel('Month of Release', fontsize = 14)

plt.savefig('./figs/release_month.jpg', dpi = 'figure')
plt.show()
plt.close()

### For the more successful films, the preferred months of release are Jun, Jul, Aug and Dec the  not-preferred months are Jan, Feb and Mar and the rest are neutral
### For the less successful films, the preferred months of release are Mar, Aug, Sep and Oct, the  not-preferred months are Feb, Jun and Dec and the rest are neutral

#%% Region

years = movie_master['release_year'].unique()
reg_count = np.shape(region_master)[0]
df_top = pd.DataFrame(data = None, index = years, columns = region_master['region'])
df_bot = pd.DataFrame(data = None, index = years, columns = region_master['region'])

for year in years:
    i = 0                   # index to capture the data for top 25 movies of the year
    s_top = [0]*reg_count
    s_bot = [0]*reg_count
    movie_ids = movie_master.query('release_year == @year')['movie_id']
    for movie_id in movie_ids:
        movie_reg_net_gross = weekly_master.query('movie_id == @movie_id')['net-gross'].tolist()
        if i < 25:          # adding the net-gross for the top 25 movies of a year
            s_top = [sum(x) for x in zip(s_top, movie_reg_net_gross)]
        else:
            s_bot = [sum(x) for x in zip(s_bot, movie_reg_net_gross)]
        i += 1

    s_top = [x/sum(s_top) for x in s_top]
    s_bot = [x/sum(s_bot) for x in s_bot]

    df_top.loc[df_top.index == year, :] = s_top
    df_bot.loc[df_bot.index == year, :] = s_bot

ncols = 3
nrows = np.ceil(region_master.shape[0]/ncols).astype('int')
idx = 0

fig, axs = plt.subplots(ncols = ncols, nrows = nrows, sharex = 'col', sharey = 'row', figsize = (15, 10))

for row in range(nrows):
    for col in range(ncols):
        if idx == len(df_top.columns):
            exit
        else:
            axs[row, col].plot(df_top.index, df_top.iloc[:, idx]*100, label = 'Top Half')
            axs[row, col].plot(df_bot.index, df_bot.iloc[:, idx]*100, label = 'Bottom Half')
            axs[row, col].set_title("Share of %s in net gross revenue" % df_top.columns[idx])
            axs[row, col].grid(True)
            axs[row, col].legend()
            idx += 1

fig.text(0.5, 0.001, 'Year of Release', ha = 'center', fontsize = 14)
fig.text(0.001, 0.5, 'Share in Percentage', va = 'center', fontsize = 14, rotation = 'vertical')
plt.tight_layout()

plt.savefig('./figs/region_share.jpg', dpi = 'figure')
plt.show()
plt.close()

### Share of revenue across regions have broadly stayed constant over the years.

### There are minimal differences between the top and bottom grossing movies when it comes to regional shares in revenue.

#%% Genres

years = movie_master['release_year'].unique()
genres = movie_master['genre'].unique().tolist()
genres.sort()
df_top = pd.DataFrame(data = None, index = years, columns = genres)
df_bot = pd.DataFrame(data = None, index = years, columns = genres)

for year in years:
    s = movie_master.query('release_year == @year')['genre']
    s.reset_index(drop = True, inplace = True)
    mid = int(len(s)/2)
    df_top.loc[df_top.index == year, :] = s[:mid].value_counts(sort = False, normalize = True).tolist()
    df_bot.loc[df_bot.index == year, :] = s[mid:].value_counts(sort = False, normalize = True).tolist()
    

ncols = 3
nrows = np.ceil(len(genres)/ncols).astype('int')
idx = 0

fig, axs = plt.subplots(ncols = ncols, nrows = nrows, sharex = 'col', sharey = 'row', figsize = (15, 10))

for row in range(nrows):
    for col in range(ncols):
        if idx == len(df_top.columns):
            exit
        else:
            axs[row, col].plot(df_top.index, df_top.iloc[:, idx]*100, label = 'Top Half')
            axs[row, col].plot(df_bot.index, df_bot.iloc[:, idx]*100, label = 'Bottom Half')
            axs[row, col].set_title("Share of %s movies released in a year" % df_top.columns[idx])
            axs[row, col].grid(True)
            axs[row, col].legend()
            idx += 1

fig.text(0.5, 0.001, 'Year of Release', ha = 'center', fontsize = 14)
fig.text(0.001, 0.5, 'Share in Percentage', va = 'center', fontsize = 14, rotation = 'vertical')
plt.tight_layout()

plt.savefig('./figs/genre_share.jpg', dpi = 'figure')
plt.show()
plt.close()

### There are noticable differences in the genres that are "preferred" in a year.
    ### Preference has shifted from Action movies to Drama movies
    ### Comedy movies maintain a constant share
    ### There was a growing preference to Love Story movies up until 2005 but that preference
    ### waning since then.
    ### Thriller movies are seeing a preference in recent years

### There are minimal differences between the top and bottom grossing movies when it comes to choice of genres.



