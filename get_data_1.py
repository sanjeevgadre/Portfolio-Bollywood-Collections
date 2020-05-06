#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 09:01:57 2020

@author: sanjeev
"""

# %% Libraries
import numpy as np
import requests
import re
from bs4 import BeautifulSoup

# %% Get Data
'''home = 'https://www.boxofficeindia.com/'
years = np.arange(2000, 2020, 1)
adj_net_gross_url = home + 'india-adjusted-nett-gross.php'

years = [2000]      # To test subsequent code. To be deleted later

for y in years:
    params = {'year' : y}
    try:
        year_html = requests.get(adj_net_gross_url, params = params)
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
        
# %% Step 1
home = 'https://www.boxofficeindia.com/'
net_gross_url = home + 'india-total-nett-gross.php'
year = 2000                                                 # For now doing it for only one year

param = {'year' : year}
try:
    year_html = requests.get(net_gross_url, params = param)
    soup = BeautifulSoup(year_html.content, 'html.parser')
    movies = soup.find_all('a', class_= 'anchormob')        # Finds the top 50 movies for the year
except ConnectionError:
    exit 

#%% Step 2
movie_rank = 0
for movie in movies[:5]:                                    # For now doing it for only one movie
    movie_href = movie.get('href')
    movie_url = home + movie_href
    movie_html = requests.get(movie_url)
    soup = BeautifulSoup(movie_html.content, 'html.parser')
    movie_rank += 1 
    
    key = int(str(year) + str(movie_rank).zfill(2))          # zfill() pads 0s to the left
    title = soup.find('a', href = re.compile('movieid')).text
    release_date = soup.find('span', class_ = 'redtext').text
    runtime = soup.find('a', href = re.compile('running-time.php')).next.next.next
    genre = soup.find('a', href = re.compile('genre.php')).text
    screens = soup.find('a', href = re.compile('screens.php')).find_next('td', class_ = 'td_cst_wd').text
    budget = soup.find('a', href = re.compile('budget.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_footfalls = soup.find('a', href = re.compile('india-footfalls.php?')).find_next('td').find_next('td').text
    india_first_day = soup.find('a', href = re.compile('india-first-day.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_first_weekend = soup.find('a', href = re.compile('india-first-weekend.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_first_week = soup.find('a', href = re.compile('india-first-week.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_nett_gross = soup.find('a', href = re.compile('net_box_office.php?')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_gross = soup.find('a', href = re.compile('india-total-gross.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_adj_net_gross = soup.find('a', href = re.compile('india-adjusted-nett-gross.php?')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    world_gross = soup.find('a', href = re.compile('worldwide-total-gross.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next

