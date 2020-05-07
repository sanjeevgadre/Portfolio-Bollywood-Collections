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
net_gross_url = home + 'india-total-nett-gross.php'

year = 2000                                                         # For now doing it for only one year

param = {'year' : year}
try:
    year_html = requests.get(net_gross_url, params = param)
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

def movie_val_in_soup(string):
    '''
    Finds the value associated with the string from the movie_soup 

    Parameters
    ----------
    string : str
        A string associated with a specific field.

    Returns
    -------
    field : TYPE
        Appropriately formatted integer.

    '''
    
    field = movie_soup.find('a', href = string).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    field = tointeger(field)
    return field

movie_id = 0
movie_lst = []
regn_cons_lst = []
regn_week_lst = []

for movie in movies[:1]:                                           # For now doing it for only one movie    
    movie_href = movie.get('href')
    movie_url = home + movie_href
    movie_html = requests.get(movie_url)
    movie_soup = BeautifulSoup(movie_html.content, 'html.parser')
    
    movie_id += 1 
    
    title = movie_soup.find('a', href = re.compile('movieid')).text
    title = str(title)
    
    release_date = movie_soup.find('span', class_ = 'redtext').text
    release_date = str(release_date)
    
    runtime = movie_soup.find('a', href = re.compile('running-time.php')).next.next.next
    runtime = str(runtime)
    runtime = runtime.replace('min', '')
    runtime = int(runtime)
    
    genre = movie_soup.find('a', href = re.compile('genre.php')).text
    genre = str(genre)
    
    screens = movie_soup.find('a', href = re.compile('screens.php')).find_next('td', class_ = 'td_cst_wd').text
    screens = int(str(screens))
    
    india_footfalls = movie_soup.find('a', href = re.compile('india-footfalls.php?')).find_next('td').find_next('td').text
    india_footfalls = tointeger(india_footfalls)
    
    budget = movie_val_in_soup('budget.php')
    
    india_first_day = movie_val_in_soup('india-first-day.php')
    
    india_first_weekend = movie_val_in_soup('india-first-weekend.php')
        
    india_first_week = movie_val_in_soup('india-first-week.php')
    
    india_nett_gross = movie_val_in_soup('net_box_office.php')
    
    india_gross = movie_val_in_soup('india-total-gross.php')
    
    india_disti_nett_gross = movie_val_in_soup('india-distributor-share.php')
    
    india_adj_nett_gross = movie_val_in_soup('india-adjusted-nett-gross.php?fm=1')
    
    world_gross = movie_val_in_soup('worldwide-total-gross.php')
    
    rec = [movie_id, title, release_date, runtime, genre, screens, india_footfalls, 
           budget, world_gross, india_gross, india_nett_gross, india_disti_nett_gross, india_adj_nett_gross, 
           india_first_day, india_first_weekend, india_first_week]
    movie_lst.append(rec)
        
    # Territory Consolidated Revenue Data
    mumbai_nett_gross = movie_val_in_soup('net_box_office.php?cityName=mumbai')
    
    mumbai_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=mumbai')

    delhi_nett_gross = movie_val_in_soup('net_box_office.php?cityName=delhi_up')
    
    delhi_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=delhi_up')

    epunjab_nett_gross = movie_val_in_soup('net_box_office.php?cityName=east_punjab')

    epunjab_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=east_punjab')

    rajasthan_nett_gross = movie_val_in_soup('net_box_office.php?cityName=rajasthan')
  
    rajasthan_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=rajasthan')
    
    cpberar_nett_gross = movie_val_in_soup('net_box_office.php?cityName=cp_berar')
 
    cpberar_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=cp_berar')
 
    ci_nett_gross = movie_val_in_soup('net_box_office.php?cityName=ci')
   
    ci_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=ci')
   
    nizam_nett_gross = movie_val_in_soup('net_box_office.php?cityName=nizam')
   
    nizam_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=nizam')
 
    mysore_nett_gross = movie_val_in_soup('net_box_office.php?cityName=mysore')
   
    mysore_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=mysore')
  
    tnkerala_nett_gross = movie_val_in_soup('net_box_office.php?cityName=tn_kerla')
 
    tnkerala_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=tn_kerla')

    bihar_nett_gross = movie_val_in_soup('net_box_office.php?cityName=bihar')
  
    bihar_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=bihar')
   
    wb_nett_gross = movie_val_in_soup('net_box_office.php?cityName=west_bengal')

    wb_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=west_bengal')

    assam_nett_gross = movie_val_in_soup('net_box_office.php?cityName=assam')

    assam_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=assam')
 
    orrisa_nett_gross = movie_val_in_soup('net_box_office.php?cityName=orrisa')
  
    orrisa_disti_nett_gross = movie_val_in_soup('india-distributor-share.php?cityName=orrisa')

    rec = [movie_id, mumbai_nett_gross, mumbai_disti_nett_gross, delhi_nett_gross, delhi_disti_nett_gross, 
           epunjab_nett_gross, epunjab_disti_nett_gross, rajasthan_nett_gross, rajasthan_disti_nett_gross, 
           cpberar_nett_gross, cpberar_disti_nett_gross, ci_nett_gross, ci_disti_nett_gross, 
           nizam_nett_gross, nizam_disti_nett_gross, mysore_nett_gross, mysore_disti_nett_gross, 
           tnkerala_nett_gross, tnkerala_disti_nett_gross, bihar_nett_gross, bihar_disti_nett_gross, 
           wb_nett_gross, wb_disti_nett_gross, assam_nett_gross, assam_disti_nett_gross, 
           orrisa_nett_gross, orrisa_disti_nett_gross]
    regn_cons_lst.append(rec)
    
#%% Scratch
