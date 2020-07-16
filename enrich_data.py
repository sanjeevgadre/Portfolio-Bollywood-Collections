#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:10:55 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
from datetime import datetime

#%% Enriching the movie_master
movie_master = pd.read_hdf('./data/movie_master.h5', key = 'df')
cpi_master = pd.read_csv('./data/CPI.csv')

# Re-index movie_master to get unique index values
movie_master.reset_index(drop = True, inplace = True)

# Extract release_year from release_date
movie_master['release_year'] = movie_master['release_date'].apply(lambda x: datetime.strptime(x, '%d %b %Y').year)
#movie_master['release_year'] = movie_master['release_year'].astype('category')

# Extract release_week from release_date and convert release_week to categorical data
movie_master['release_week'] = movie_master['release_date'].apply(lambda x: datetime.strptime(x, '%d %b %Y').strftime('%V'))
movie_master['release_week'] = movie_master['release_week'].astype('int')
#movie_master['release_week'] = movie_master['release_week'].astype('category')

# Extract release_month from release_date and convert release_month to categorical data
movie_master['release_month'] = movie_master['release_date'].apply(lambda x: datetime.strptime(x, '%d %b %Y').month)
movie_master['release_month'] = movie_master['release_month'].astype('int')
#movie_master['release_month'] = movie_master['release_month'].astype('category')

# Convert genre to categorical data
movie_master['genre'] = movie_master['genre'].astype('category')

'''# We divide the period 1994 to 2019 into intervals of 3 years (the final interval will have only 2 years). We assign a movie to an appropriate year_interval based on its release_year. Finally we convert the release_interval data type to int
base_year = 1994
incr = 3
intrvl = 1
for i in range(len(movie_master)):
    if movie_master.loc[i, 'release_year'] - base_year < incr:
        movie_master.loc[i, 'release_interval'] = intrvl
    else:
        intrvl = intrvl + 1
        base_year = base_year + incr
        movie_master.loc[i, 'release_interval'] = intrvl
movie_master['release_interval'] = movie_master['release_interval'].astype('int')'''
               
'''
During data download it was noticed that some movies were missing values for 'india-footfalls' and value 0 was imputed. We now impute a more appropriate value as follows:
    Calculate the mean ratio for 'india-footfalls'/'india-net-gross' for all movies in the same year as the movies with missing 'india-fooftalls' value.
    Use this ratio to impute a more appropriate value for movies with 'missing india-footfalls' '''

# Identify years with movies reportng 0 footfalls
zero_years = movie_master.query('`india-footfalls` == 0').loc[:, 'release_year'].drop_duplicates().tolist()

# Imputing the missing footfalls values
for year in zero_years:
    df = movie_master.query('release_year == @year and `india-footfalls` != 0').loc[:, ['india-nett-gross', 'india-footfalls']]
    df['ff_to_gross'] = df.apply(lambda x: x['india-footfalls']/x['india-nett-gross'], axis = 1)
    mean_ff_to_gross = df['ff_to_gross'].mean()
    df = movie_master.query('release_year == @year and `india-footfalls` == 0')
    for idx in df.index:
        movie_master.loc[idx, 'india-footfalls'] = movie_master.loc[idx, 'india-nett-gross'] * mean_ff_to_gross
        
''' 
We adjust for inflation four columns of movie_master: budget, india-total-gross, india-distributor-share and india-first-week. We capture the adjusted values in three new columns: india_total_gross_adj, india_distributor_share_adj and india_first_week_adj.

To adjust for inflation we use the CPI values from cpi_master
'''

cols = ['budget', 'india-total-gross', 'india-distributor-share', 'india-first-week']
new_cols = ['budget_adj', 'india_total_gross_adj', 'india_distributor_share_adj', 'india_first_week_adj']
      
# Get the CPI adjustment factor (inflation index) for each film
_max = cpi_master['CPI'].max()
movie_master['inf_adj_fct'] = [_max/cpi_master.query('Year == @x')['CPI'].item() for x in            movie_master['release_year']]

for c in range(len(cols)):
    movie_master[new_cols[c]] = movie_master.apply(lambda x: x[cols[c]]*x['inf_adj_fct'], axis = 1)
        
# Save the enriched movie_master file
movie_master.to_pickle('./data/movie_master_en.pkl')
