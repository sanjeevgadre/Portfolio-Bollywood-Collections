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

# %% Helper Functions
def tointeger(field):
    '''
    Change type from bs4.navigable.string to integers. Additionally strips the "," and "|" in the string.

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
    field : int
        Appropriately formatted integer.

    '''
    
    try:
        field = soup.find('a', href = string).find_next('img', src = 'images/rupeesim-black.png').next
        field = tointeger(field)
    except AttributeError:
        field = None
    return field

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

regn_lst = ['mumbai', 'delhi_up', 'east_punjab', 'rajasthan', 'cp_berar', 'ci', 
            'nizam', 'mysore', 'tn_kerla', 'bihar', 'west_bengal', 'assam', 'orrisa']
dict_ = {'region' : regn_lst, 'region-id' : range(1, len(regn_lst)+1)}
regn_master = pd.DataFrame(data = dict_)

cols = ['movie-id', 'title','release-date', 'runtime', 'genre', 'screens', 'india-footfalls', 
        'budget', 'india-nett-gross', 'india-adjusted-nett-gross', 'india-first-day', 
        'india-first-weekend', 'india-first-week', 'india-total-gross', 'india-distributor-share', 
        'worldwide-total-gross']
movie_master = pd.DataFrame(data = None, columns = cols)

cols = ['movie-id', 'region-id', 'net-gross', 'dist-net-gross', 'week_1', 'week_2', 'week_3', 
        'week_4', 'week_5', 'week_6', 'week_7', 'week_8', 'week_9', 'week_ten', 'week_x11']
weekly_master = pd.DataFrame(data = None, columns = cols)


#%% Step 2: For a movie from the list formed in Step 1, populate movie_master. 

years = [2000]
movie_id = 0

for year in years:
    print('Now scrapping data for Year-%i' % year)
    
    param = {'year' : year}
    try:
        year_html = requests.get(nett_gross_url, params = param)
        year_soup = BeautifulSoup(year_html.content, 'html.parser')
        movies = year_soup.find_all('a', class_= 'anchormob')          # Finds the top 50 movies for the year
    except ConnectionError:
        exit 
        
    movie_lst = []
    weekly_lst = []
   
    for movie in movies[-1:]:                                               
        movie_href = movie.get('href')
        movie_url = home + movie_href
        movie_html = requests.get(movie_url)
        movie_soup = BeautifulSoup(movie_html.content, 'html.parser')
        
        movie_id += 1
        
        dict_ = {'movie-id' : movie_id}
        
        try:
            title = movie_soup.find('a', href = re.compile('movieid')).text
            dict_['title'] = str(title)
        except AttributeError:
            dict_['title'] = None
        
        try:
            release_date = movie_soup.find('span', class_ = 'redtext').text
            dict_['release-date'] = str(release_date)
        except AttributeError:
            dict_['release-date'] = None
        
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
            dict_['india-footfalls'] = tointeger(india_footfalls)
        except AttributeError:
            dict_['india-footfalls'] = None
            
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
        
        # Getting Weekly Data
        weeklies = movie_soup.find_all('a', href = re.compile('weekly-movies'))
        weekly_soup_lst = []
        
        for weekly in weeklies:
            weekly_url = home + weekly.get('href')
            weekly_html = requests.get(weekly_url)
            weekly_soup = BeautifulSoup(weekly_html.content, 'html.parser')
            weekly_soup_lst.append(weekly_soup)
                
        # Territory Consolidated Revenue Data
        
        for regn in regn_lst:
            dict_= {'movie-id' : movie_id}
            dict_['region-id'] = int(regn_master.loc[regn_master['region'] == regn, 'region-id'])
            
            phrase = 'net_box_office.php?cityName=' + regn
            dict_['net-gross'] = val_in_soup(movie_soup, phrase)
            
            phrase = 'india-distributor-share.php?cityName=' + regn
            dict_['dist-net-gross'] = val_in_soup(movie_soup, phrase)
            
            i = 4
            for w_soup in weekly_soup_lst:
                phrase = 'week_report_india.php?type_key=' + weekly_master.columns[i] + '&cityName=' + regn
                dict_[weekly_master.columns[i]] = val_in_soup(w_soup, phrase)
                i +=1
                      
            weekly_lst.append(dict_)
    

#%% Step 3: Writing the scraped data to a file
movie_master = movie_master.append(movie_lst, ignore_index = True)
weekly_master = weekly_master.append(weekly_lst, ignore_index = True)
    

