#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 09:01:57 2020

@author: sanjeev
"""

# %% Libraries
import numpy as np
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup

# %% Get Data
'''home = 'https://www.boxofficeindia.com/'
years = np.arange(2000, 2020, 1)
adj_nett_gross_url = home + 'india-adjusted-nett-gross.php'

years = [2000]      # To test subsequent code. To be deleted later

for y in years:
    params = {'year' : y}
    try:
        year_html = requests.get(adj_nett_gross_url, params = params)
        soup = BeautifulSoup(year_html.content, 'html.parser')
        movies = soup.find_all('a', class_ = 'anchormob')   # Finds all 'a' tags with class = 'anchormob'
        for movie in movies:                                # For each movie in that year
            movie_href = movie.get('href')
            movie_url = home + movie_href
            movie_html = requests.get(movie_url)
            soup = BeautifulSoup(movie_html.content, 'html.parser')
            
            title = soup.find('a', href = movie_href).text
            release_date = soup.find('span', class_ = 'redtext').text
            
            
    except ConnectionError:
        continue'''
        
# %% Step 1: Find the top movies for a given year
home = 'https://www.boxofficeindia.com/'
nett_gross_url = home + 'india-total-nett-gross.php'

cols = ['movie_id', 'title','release_date', 'runtime', 'genre', 'screens', 'india_footfalls', 
        'budget', 'india-nett-gross', 'india-adjusted-nett-gross', 'india-first-day', 
        'india-first-weekend', 'india-first-week', 'india-total-gross', 'india-distributor-share', 
        'worldwide-total-gross']
movie_df = pd.DataFrame(data = None, columns = cols)

cols = ['movie_id', 'mumbai_net_gross', 'mumbai_dist_net_gross', 'delhi_up_net_gross', 
        'delhi_up_dist_net_gross', 'east_punjab_net_gross', 'east_punjab_dist_net_gross', 
        'rajasthan_net_gross', 'rajasthan_dist_net_gross', 'cp_berar_net_gross',
       'cp_berar_dist_net_gross', 'ci_net_gross', 'ci_dist_net_gross',
       'nizam_net_gross', 'nizam_dist_net_gross', 'mysore_net_gross',
       'mysore_dist_net_gross', 'tn_kerla_net_gross', 'tn_kerla_dist_net_gross', 
       'bihar_net_gross', 'bihar_dist_net_gross', 'west_bengal_net_gross', 
       'west_bengal_dist_net_gross', 'assam_net_gross', 'assam_dist_net_gross', 
       'orrisa_net_gross', 'orrisa_dist_net_gross']
regn_cons_df = pd.DataFrame(data = None, columns = cols)

regn_lst = ['mumbai', 'delhi_up', 'east_punjab', 'rajasthan', 'cp_berar', 'ci', 
            'nizam', 'mysore', 'tn_kerla', 'bihar', 'west_bengal', 'assam', 'orrisa']
year = 2000                                                         # For now doing it for only one year

param = {'year' : year}
try:
    year_html = requests.get(nett_gross_url, params = param)
    year_soup = BeautifulSoup(year_html.content, 'html.parser')
    movies = year_soup.find_all('a', class_= 'anchormob')          # Finds the top 50 movies for the year
except ConnectionError:
    exit 

#%% Step 2: For a movie from the list formed in Step 1, scrape all the data (except weekly regional numbers). 

def tointeger(field):
    '''
    Change type from bs4.navigable.string to integers. Additionally strip the "," and "|" in the string.

    Parameters
    ----------
    field : bs4.navigable.string
        Numerical data scrapped from the webpage as a string.

    Returns
    -------
    field : int
        Appropriately formatted integer.

    '''
    field = str(field)
    field = field.replace(',', '')
    field = field.replace('|', '')
    field = int(field)
    return field

def val_in_soup(soup, string):
    '''
    Finds the value associated with the string (field_name) from the soup. 
    This is not a generic function and there are some added constraints on the tags that must be present in the search criterion.

    Parameters
    ----------
    soup   : BeautifulSoup
        The soup from which value is sought.
    string : str
        A string (field_name) for which value is sought.

    Returns
    -------
    field : TYPE
        Appropriately formatted integer.

    '''
    
    try:
        field = soup.find('a', href = string).find_next('img', src = 'images/rupeesim-black.png').next
        field = tointeger(field)
    except AttributeError:
        field = None
    return field


movie_id = 0
movie_lst = []
regn_cons_lst = []

for movie in movies[:1]:                                           # For now doing it for only one movie    
    movie_href = movie.get('href')
    movie_url = home + movie_href
    movie_html = requests.get(movie_url)
    movie_soup = BeautifulSoup(movie_html.content, 'html.parser')
    
    movie_id += 1
    
    dict_ = {'movie_id' : movie_id}
    
    try:
        title = movie_soup.find('a', href = re.compile('movieid')).text
        dict_['title'] = str(title)
    except AttributeError:
        dict_['title'] = None
    
    try:
        release_date = movie_soup.find('span', class_ = 'redtext').text
        dict_['release_date'] = str(release_date)
    except AttributeError:
        dict_['release_date'] = None
    
    try:
        runtime = movie_soup.find('a', href = re.compile('running-time.php')).next.next.next
        runtime = str(runtime)
        runtime = runtime.replace('min', '')
        dict_['runtime'] = int(runtime)
    except AttributeError:
        dict_['runtime'] = int(runtime)
    
    try:
        genre = movie_soup.find('a', href = re.compile('genre.php')).text
        dict_['genre'] = str(genre)
    except AttributeError:
        dict_['genre'] = None
    
    try:
        screens = movie_soup.find('a', href = re.compile('screens.php')).find_next('td', class_ = 'td_cst_wd').text
        dict_['screens'] = int(str(screens))
    except AttributeError:
        dict_['screens'] = None
    
    try:
        india_footfalls = movie_soup.find('a', href = re.compile('india-footfalls.php?')).find_next('td').find_next('td').text
        dict_['india_footfalls'] = tointeger(india_footfalls)
    except AttributeError:
        dict_['india_footfalls'] = None
        
    india_nett_gross = val_in_soup(movie_soup, 'net_box_office.php')
    dict_['india-nett-gross'] = tointeger(india_nett_gross)
    
    india_adj_nett_gross = val_in_soup(movie_soup, 'india-adjusted-nett-gross.php?fm=1')
    dict_['india-adjusted-nett-gross'] = tointeger(india_adj_nett_gross)
     
    fields = ['budget.php', 'india-first-day.php', 'india-first-weekend.php', 'india-first-week.php', 
              'india-total-gross.php', 'india-distributor-share.php', 'worldwide-total-gross.php']
    for field in fields:
        field_val = val_in_soup(movie_soup, field)
        dict_[field[:-4]] = field_val
    
    movie_lst.append(dict_)
       
    # Territory Consolidated Revenue Data
    dict_ = {'movie_id' : movie_id}
    for regn in regn_lst:
        phrase = 'net_box_office.php?cityName=' + regn
        field = regn + '_net_gross'
        dict_[field] = val_in_soup(movie_soup, phrase)
        
        phrase = 'india-distributor-share.php?cityName=' + regn
        field = regn + '_dist_net_gross'
        dict_[field] = val_in_soup(movie_soup, phrase)
    
    regn_cons_lst.append(dict_)
        
#%% Step 2a: Writing the scraped data to a file
movie_df = movie_df.append(movie_lst)
regn_cons_df = regn_cons_df.append(regn_cons_lst)



#%% Step 3: For a movie from the list formed in Step 1, scrape all the weekly regional data.

weeklies = movie_soup.find_all('a', href = re.compile('weekly-movies'))
movie_weekly_lst = np.zeros((len(regn_lst), len(weeklies)), dtype = int) # Array to hold the regionwise-weekly nett-gross

w = 0
for weekly in weeklies:                     # for now for one week
    weekly_url = home + weekly.get('href')
    weekly_html = requests.get(weekly_url)
    weekly_soup = BeautifulSoup(weekly_html.content, 'html.parser')
    r = 0
    for regn in regn_lst:
        phrase = 'week_report_india.php?type_key=' + weekly_url[70:] + '&cityName=' + regn
        movie_weekly_lst[r, w] = val_in_soup(weekly_soup, phrase) # for the week, for the region
        r +=1
    w +=1
    

#%% Scratch
print(movie_weekly_lst)
