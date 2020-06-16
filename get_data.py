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

# regional_master dataframe
region_master_cols = ['mumbai', 'delhi_up', 'east_punjab', 'rajasthan', 'cp_berar', 'ci', 
                      'nizam', 'mysore', 'tn_kerla', 'bihar', 'west_bengal', 'assam', 
                      'orrisa']
d = {'region' : region_master_cols, 'region_id' : range(1, len(region_master_cols)+1)}
region_master = pd.DataFrame(data = d)

# Saving region_master
region_master.to_hdf('./data/region_master.h5', key = 'df', format = 'table', append = True)

# Columns for movie_master and weekly_master. The dataframe is setup later (repeatedly)
movie_master_cols = ['movie_id', 'title','release_date', 'runtime', 'genre', 'screens', 
                     'india-footfalls', 'budget', 'india-nett-gross', 
                     'india-adjusted-nett-gross', 'india-first-day', 'india-first-weekend', 
                     'india-first-week', 'india-total-gross', 'india-distributor-share', 
                     'worldwide-total-gross']

weekly_master_cols = ['movie_id', 'region_id', 'net-gross', 'dist-net-gross', 'week_1', 
                      'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'week_8', 
                      'week_9', 'week_ten', 'week_x11']

# Years for which scraping is undertaken
years = [x for x in range(2018, 2020, 1)]

# Unique movie_id seed for each run
movie_id = int(datetime.now().strftime('%y%m%d%H%M'))   


#%% Step 2: Scrapping Data

# Loop - I: For each year of the run
for year in years:
    
    # movie_master and weekly_master dataframes are "renewed" for each year
    movie_master = pd.DataFrame(data = None, columns = movie_master_cols, dtype = 'int')
    weekly_master = pd.DataFrame(data = None, columns = weekly_master_cols, dtype = 'int')
    
    print('\nNow scraping data for Year-%i....' % year)
    
    param = {'year' : year}
    try:
        year_html = requests.get(nett_gross_url, params = param)
        year_soup = BeautifulSoup(year_html.content, 'html.parser')
        
        # Get the movies of the year
        movies = year_soup.find_all('a', class_= 'anchormob')
    except ConnectionError:
        print('Connection Error. Going to next year')
        continue 
        
    movie_lst = []
    weekly_lst = []
    
    # Loop - II: For each year, for each movie
    for movie in movies:
        
        # Scrap details for movie_master                                             
        movie_href = movie.get('href')
        movie_url = home + movie_href
        try:
            movie_html = requests.get(movie_url)
            movie_soup = BeautifulSoup(movie_html.content, 'html.parser')
        except ConnectionError:
            print('Connection Error for %s. Trying the next movie' % movie_url)
            continue
        
        movie_id += 1
        movie_dict = {'movie_id' : movie_id}
        
        try:
            title = movie_soup.find('a', href = re.compile('movieid')).text
            movie_dict['title'] = str(title)
            print('Now scraping data for %s....' % str(title))
        except AttributeError:
            movie_dict['title'] = ''
        
        try:
            release_date = movie_soup.find('span', class_ = 'redtext').text
            # Stripping leading & trailing whitespaces
            release_date = str(release_date).strip()
            movie_dict['release_date'] = release_date
        except AttributeError:
            movie_dict['release_date'] = ''
                    
        try:
            runtime = movie_soup.find('a', href = re.compile('running-time.php')).next.next.next
            runtime = str(runtime)
            runtime = runtime.replace('min', '')
            movie_dict['runtime'] = int(runtime)
        except AttributeError:
            movie_dict['runtime'] = 0
        
        try:
            genre = movie_soup.find('a', href = re.compile('genre.php')).text
            movie_dict['genre'] = str(genre)
        except AttributeError:
            movie_dict['genre'] = ''
        
        try:
            screens = movie_soup.find('a', href = re.compile('screens.php')).find_next('td', class_ = 'td_cst_wd').text
            movie_dict['screens'] = int(str(screens))
        except AttributeError:
            movie_dict['screens'] = ''
        
        try:
            india_footfalls = movie_soup.find('a', href = re.compile('india-footfalls.php?')).find_next('td').find_next('td').text
            if india_footfalls == '--':
                movie_dict['india-footfalls'] = 0
            else:
                movie_dict['india-footfalls'] = tointeger(india_footfalls)
        except AttributeError:
            movie_dict['india-footfalls'] = 0
            
        india_nett_gross = val_in_soup(movie_soup, 'net_box_office.php')
        movie_dict['india-nett-gross'] = india_nett_gross
        #movie_dict['india-nett-gross'] = tointeger(india_nett_gross)
        
        india_adj_nett_gross = val_in_soup(movie_soup, 'india-adjusted-nett-gross.php?fm=1')
        movie_dict['india-adjusted-nett-gross'] = india_adj_nett_gross
        #movie_dict['india-adjusted-nett-gross'] = tointeger(india_adj_nett_gross)
         
        fields = ['budget.php', 'india-first-day.php', 'india-first-weekend.php', 
                  'india-first-week.php', 'india-total-gross.php', 
                  'india-distributor-share.php', 'worldwide-total-gross.php']
        for field in fields:
            field_val = val_in_soup(movie_soup, field)
            movie_dict[field[:-4]] = field_val
        
        # Completed scrapimg details for movie_master
        movie_lst.append(movie_dict)
        
        ####

        # Scrap details for weekly_master
        weeklies = movie_soup.find_all('a', href = re.compile('weekly-movies'))
        weekly_soup_lst = []
        
        # Scrap html for all the weeks
        for weekly in weeklies:
            weekly_url = home + weekly.get('href')
            try:
                weekly_html = requests.get(weekly_url)
                weekly_soup = BeautifulSoup(weekly_html.content, 'html.parser')
                weekly_soup_lst.append(weekly_soup)
            except ConnectionError:
                print('Connection Error for %s. Trying the next week' % weekly_url)
                continue
                
        # Loop - III: For each year, for each movie, for each region 
        for regn in region_master_cols:
            
            weekly_dict= {'movie_id' : movie_id}
            weekly_dict['region_id'] = int(region_master.loc[region_master['region'] == regn, 'region_id'])
            
            # Regional total revenue
            phrase = 'net_box_office.php?cityName=' + regn
            foo = weekly_soup.find('a', href = phrase).next.next 
            # returns either a bs4.element.Tag or a string
            try:
                if foo[:1] == '-':
                    weekly_dict['net-gross'] = 0
            except TypeError:
                weekly_dict['net-gross'] = val_in_soup(movie_soup, phrase)
            
            # Regional total distributor reveune
            phrase = 'india-distributor-share.php?cityName=' + regn
            foo = weekly_soup.find('a', href = phrase).next.next 
            # returns either a bs4.element.Tag or a string
            try:
                if foo[:1] == '-':
                    weekly_dict['dist-net-gross'] = 0
            except TypeError:
                weekly_dict['dist-net-gross'] = val_in_soup(movie_soup, phrase)
            
            i = 4   # Parameter for Loop IV
            
            # Loop - IV: For each year, for each movie, for each region, for each week soup
            for w_soup in weekly_soup_lst:
                
                phrase = 'week_report_india.php?type_key=' + weekly_master.columns[i] + '&cityName=' + regn
                weekly_dict[weekly_master.columns[i]] = val_in_soup(w_soup, phrase, tag = 'td')
                i +=1
                      
            # Completed scraping details for weekly_master
            weekly_lst.append(weekly_dict)
            
            ####
    
    # Forming the dataframes and writing to disk
    movie_master = movie_master.append(movie_lst, ignore_index = True)
    weekly_master = weekly_master.append(weekly_lst, ignore_index = True)
    
    '''When writing to hdf, it is important to set the width for columns containing strings to a value that will accomodate the largest possible size. Counter intutively the parameter is named min_itemsize''' 
       
    try:
        movie_master.to_hdf('./data/movie_master.h5', key = 'df', format = 'table', append = True, min_itemsize = {'title' : 100, 'release_date' : 11, 'genre' : 25})
        weekly_master.to_hdf('./data/weekly_master.h5', key = 'df', format = 'table', append = True)
    except ValueError:
        print('Encountered ValueError when writing data for %s to disk' % year)
        continue

print("\nFinished the run")
