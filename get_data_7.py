#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 09:01:57 2020

@author: sanjeev
"""

# %% Libraries
import pandas as pd
import requests
import re
import tables
from bs4 import BeautifulSoup
from datetime import datetime

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

def val_in_soup(soup, string, tag = 'img'):
    '''
    Finds the value associated with the string from the soup. 
    This is not a generic function and there are some added constraints on the tags that must be present in the search criterion.

    Parameters
    ----------
    soup   : BeautifulSoup
        The soup from which value is sought.
    string : str
        A string for which value is sought.
    tag : str
        The tag that acts as an anchor when finding the required value. Can be either 'img' or 'td'.

    Returns
    -------
    field : int
        Appropriately formatted integer.

    '''
    
    if tag == 'img':
        try:
            field = soup.find('a', href = string).find_next('img', src = 'images/rupeesim-black.png').next
            field = tointeger(field)
        except AttributeError:
            field = 0
    else: 
        field = soup.find('a', href = string).find_next('td').text
        if field[:1] == '-':
            field = 0
        else:
            field = tointeger(field)
            
    return field

# %% Step 1: Setting up Parameters
home = 'https://www.boxofficeindia.com/'
nett_gross_url = home + 'india-total-nett-gross.php'

regn_lst = ['mumbai', 'delhi_up', 'east_punjab', 'rajasthan', 'cp_berar', 'ci', 
            'nizam', 'mysore', 'tn_kerla', 'bihar', 'west_bengal', 'assam', 'orrisa']
dict_ = {'region' : regn_lst, 'region-id' : range(1, len(regn_lst)+1)}
regn_master = pd.DataFrame(data = dict_)

years = [2010, 2011]
movie_id = int(datetime.now().strftime('%y%m%d%H%M'))   # Setting up a unique movie_id seed for each run


#%% Step 2: Scrapping Data for Each Year. 
for year in years:
    cols = ['movie-id', 'title','release-date', 'runtime', 'genre', 'screens', 'india-footfalls', 
        'budget', 'india-nett-gross', 'india-adjusted-nett-gross', 'india-first-day', 
        'india-first-weekend', 'india-first-week', 'india-total-gross', 'india-distributor-share', 
        'worldwide-total-gross']
    movie_master = pd.DataFrame(data = None, columns = cols, dtype = 'int')

    cols = ['movie-id', 'region-id', 'net-gross', 'dist-net-gross', 'week_1', 'week_2', 'week_3', 
        'week_4', 'week_5', 'week_6', 'week_7', 'week_8', 'week_9', 'week_ten', 'week_x11']
    weekly_master = pd.DataFrame(data = None, columns = cols, dtype = 'int')
    
    print('Now scraping data for Year-%i....' % year)
    
    param = {'year' : year}
    try:
        year_html = requests.get(nett_gross_url, params = param)
        year_soup = BeautifulSoup(year_html.content, 'html.parser')
        movies = year_soup.find_all('a', class_= 'anchormob')     # Finds the top 50 movies for the year
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
        
        movie_dict = {'movie-id' : movie_id}
        
        try:
            title = movie_soup.find('a', href = re.compile('movieid')).text
            movie_dict['title'] = str(title)
            print('Now scraping data for %s....' % str(title))
        except AttributeError:
            movie_dict['title'] = None
        
        try:
            release_date = movie_soup.find('span', class_ = 'redtext').text
            release_date = str(release_date).strip()    # Stripping leading and trailing whitespaces
            movie_dict['release-date'] = release_date
        except AttributeError:
            movie_dict['release-date'] = None
                    
        try:
            runtime = movie_soup.find('a', href = re.compile('running-time.php')).next.next.next
            runtime = str(runtime)
            runtime = runtime.replace('min', '')
            movie_dict['runtime'] = int(runtime)
        except AttributeError:
            movie_dict['runtime'] = int(runtime)
        
        try:
            genre = movie_soup.find('a', href = re.compile('genre.php')).text
            movie_dict['genre'] = str(genre)
        except AttributeError:
            movie_dict['genre'] = None
        
        try:
            screens = movie_soup.find('a', href = re.compile('screens.php')).find_next('td', class_ = 'td_cst_wd').text
            movie_dict['screens'] = int(str(screens))
        except AttributeError:
            movie_dict['screens'] = None
        
        try:
            india_footfalls = movie_soup.find('a', href = re.compile('india-footfalls.php?')).find_next('td').find_next('td').text
            movie_dict['india-footfalls'] = tointeger(india_footfalls)
        except AttributeError:
            movie_dict['india-footfalls'] = None
            
        india_nett_gross = val_in_soup(movie_soup, 'net_box_office.php')
        movie_dict['india-nett-gross'] = tointeger(india_nett_gross)
        
        india_adj_nett_gross = val_in_soup(movie_soup, 'india-adjusted-nett-gross.php?fm=1')
        movie_dict['india-adjusted-nett-gross'] = tointeger(india_adj_nett_gross)
         
        fields = ['budget.php', 'india-first-day.php', 'india-first-weekend.php', 'india-first-week.php', 
                  'india-total-gross.php', 'india-distributor-share.php', 'worldwide-total-gross.php']
        for field in fields:
            field_val = val_in_soup(movie_soup, field)
            movie_dict[field[:-4]] = field_val
        
        movie_lst.append(movie_dict)
        
        # Getting Weekly Data
        weeklies = movie_soup.find_all('a', href = re.compile('weekly-movies'))
        weekly_soup_lst = []
        
        for weekly in weeklies:
            weekly_url = home + weekly.get('href')
            weekly_html = requests.get(weekly_url)
            weekly_soup = BeautifulSoup(weekly_html.content, 'html.parser')
            weekly_soup_lst.append(weekly_soup)
                
        # Territory Consolidated Revenue
        
        for regn in regn_lst:
            weekly_dict= {'movie-id' : movie_id}
            weekly_dict['region-id'] = int(regn_master.loc[regn_master['region'] == regn, 'region-id'])
            
            # Regional total revenue
            phrase = 'net_box_office.php?cityName=' + regn
            foo = weekly_soup.find('a', href = phrase).next.next # returns either a bs4.element.Tag or a string
            try:
                if foo[:1] == '-':
                    weekly_dict['net-gross'] = 0
            except TypeError:
                weekly_dict['net-gross'] = val_in_soup(movie_soup, phrase)
            
            # Regional total distributor reveune
            phrase = 'india-distributor-share.php?cityName=' + regn
            foo = weekly_soup.find('a', href = phrase).next.next # returns either a bs4.element.Tag or a string
            try:
                if foo[:1] == '-':
                    weekly_dict['dist-net-gross'] = 0
            except TypeError:
                weekly_dict['dist-net-gross'] = val_in_soup(movie_soup, phrase)
                
            # For weekly revenue for each region
            i = 4
            for w_soup in weekly_soup_lst:
                phrase = 'week_report_india.php?type_key=' + weekly_master.columns[i] + '&cityName=' + regn
                weekly_dict[weekly_master.columns[i]] = val_in_soup(w_soup, phrase, tag = 'td')
                i +=1
                      
            weekly_lst.append(weekly_dict)
            
    movie_master = movie_master.append(movie_lst)
    weekly_master = weekly_master.append(weekly_lst, ignore_index = True)
    
    movie_master.set_index(keys = ['movie-id'], drop = True, verify_integrity = True, inplace = True)
    weekly_master.set_index(keys = ['movie-id', 'region-id'], drop = True, verify_integrity = True, inplace = True)
    
    movie_master.to_hdf('./data/movie_master.h5', key = 'df', format = 'table', append = True)
    weekly_master.to_hdf('./data/weekly_master.h5', key = 'df', format = 'table', append = True)
    
 
    


#%% Scratch

movie_master.to_hdf('./data/movie_master.h5', key = 'df', format = 'table', append = True, data_columns = ['movie-id', 'genre'])

foo = pd.read_hdf('./data/weekly_master.h5')

