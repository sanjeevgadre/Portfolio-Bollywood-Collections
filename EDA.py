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

# Total revenues for the top 50 films of the year adjusted for inflation
df = movie_master.groupby(['release_year'])['india-adjusted-nett-gross'].sum()

plt.figure(figsize = (13, 5))
plt.plot(df.index, df/1000000000)
plt.title('# Total revenues for the top 50 films of the year adjusted for inflation', fontweight = 'bold')
plt.ylabel('Rupees in billions')
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.grid(True)
plt.show()

# Median return on budget for the top 50 films of the year
df = movie_master.groupby(['release_year'])['ret_on_budget'].agg([np.mean, np.median])

plt.figure(figsize = (13, 5))
plt.plot(df.index, (df['mean'] - 1) * 100, label = 'Mean Returns')
plt.plot(df.index, (df['median'] - 1) * 100, label = 'Median Returns')
plt.title('Mean and Median Returns on budget for the top 50 films of the year', fontweight = 'bold')
plt.ylabel('Returns in %')
plt.ylim(-70, 125)
plt.xlabel('Year of Release')
plt.xticks(np.arange(1994, 2020, 1), rotation = 45)
plt.legend()
plt.grid(True)
plt.show()

#%% Scratch
df = movie_master.groupby(['release_year'])['india-nett-gross', 'budget'].sum()
df['return_on_budget'] = df.apply(lambda x: x['india-nett-gross']/x['budget'], axis = 1)
