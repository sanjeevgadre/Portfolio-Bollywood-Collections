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

#%% Change type from bs4.navigable.string to integers. Additionally strip the commas in the string. Used       in Step 2
def tointeger01(field):
    field = str(field)
    field = field.replace(',', '')
    field = int(field)
    return field

#%% Change type from bs4.navigable.string to integers. Additionally strip the commas and | in the string. Used in Step 2
def tointeger02(field):
    field = str(field)
    field = field.replace(',', '')
    field = field.replace('|', '')
    field = int(field)
    return field

#%% Step 2
movie_rank = 0
for movie in movies[:1]:                                    # For now doing it for only one movie
    movie_href = movie.get('href')
    movie_url = home + movie_href
    movie_html = requests.get(movie_url)
    soup = BeautifulSoup(movie_html.content, 'html.parser')
    movie_rank += 1 
    
    key = int(str(year) + str(movie_rank).zfill(2))          # zfill() pads 0s to the left

    title = soup.find('a', href = re.compile('movieid')).text
    title = str(title)

    release_date = soup.find('span', class_ = 'redtext').text
    release_date = str(release_date)
    
    runtime = soup.find('a', href = re.compile('running-time.php')).next.next.next
    runtime = str(runtime)
    runtime = runtime.replace('min', '')
    runtime = int(runtime)
    
    genre = soup.find('a', href = re.compile('genre.php')).text
    genre = str(genre)
    
    screens = soup.find('a', href = re.compile('screens.php')).find_next('td', class_ = 'td_cst_wd').text
    screens = int(str(screens))
    
    budget = soup.find('a', href = re.compile('budget.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    budget = tointeger01(budget)
    
    india_footfalls = soup.find('a', href = re.compile('india-footfalls.php?')).find_next('td').find_next('td').text
    india_footfalls = tointeger01(india_footfalls)
    
    india_first_day = soup.find('a', href = re.compile('india-first-day.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_first_day = tointeger01(india_first_day)
    
    india_first_weekend = soup.find('a', href = re.compile('india-first-weekend.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_first_weekend = tointeger01(india_first_weekend)
    
    india_first_week = soup.find('a', href = re.compile('india-first-week.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_first_week = tointeger01(india_first_week)
    
    india_nett_gross = soup.find('a', href = re.compile('net_box_office.php?')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_nett_gross = tointeger01(india_nett_gross)
    
    india_gross = soup.find('a', href = re.compile('india-total-gross.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_gross = tointeger01(india_gross)
    
    india_adj_net_gross = soup.find('a', href = re.compile('india-adjusted-nett-gross.php?')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_adj_net_gross = tointeger01(india_adj_net_gross)
    
    world_gross = soup.find('a', href = re.compile('worldwide-total-gross.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    world_gross = tointeger01(world_gross)
    
    india_disti_net_gross = soup.find('a', href = re.compile('india-distributor-share.php')).find_next('img', src = re.compile('images/rupeesim-black.png')).next
    india_disti_net_gross = tointeger01(india_disti_net_gross)
    
    mumbai_net_gross = soup.find('a', href = 'net_box_office.php?cityName=mumbai').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    mumbai_net_gross = tointeger02(mumbai_net_gross)
    
    mumbai_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=mumbai').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    mumbai_disti_net_gross = tointeger01(mumbai_disti_net_gross)
    
    delhi_net_gross = soup.find('a', href = 'net_box_office.php?cityName=delhi_up').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    delhi_net_gross = tointeger02(delhi_net_gross)
    
    delhi_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=delhi_up').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    delhi_disti_net_gross = tointeger01(delhi_disti_net_gross)
     
    epunjab_net_gross = soup.find('a', href = 'net_box_office.php?cityName=east_punjab').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    epunjab_net_gross = tointeger02(epunjab_net_gross)
    
    epunjab_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=east_punjab').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    epunjab_disti_net_gross = tointeger01(epunjab_disti_net_gross)
     
    rajasthan_net_gross = soup.find('a', href = 'net_box_office.php?cityName=rajasthan').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    rajasthan_net_gross = tointeger02(rajasthan_net_gross)
    
    rajasthan_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=rajasthan').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    rajasthan_disti_net_gross = tointeger01(rajasthan_disti_net_gross)
    
    cpberar_net_gross = soup.find('a', href = 'net_box_office.php?cityName=cp_berar').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    cpberar_net_gross = tointeger02(cpberar_net_gross)
    
    cpberar_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=cp_berar').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    cpberar_disti_net_gross = tointeger01(cpberar_disti_net_gross)
    
    ci_net_gross = soup.find('a', href = 'net_box_office.php?cityName=ci').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    ci_net_gross = tointeger02(ci_net_gross)
    
    ci_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=ci').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    ci_disti_net_gross = tointeger01(ci_disti_net_gross)
    
    nizam_net_gross = soup.find('a', href = 'net_box_office.php?cityName=nizam').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    nizam_net_gross = tointeger02(nizam_net_gross)
    
    nizam_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=nizam').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    nizam_disti_net_gross = tointeger01(nizam_disti_net_gross)
    
    mysore_net_gross = soup.find('a', href = 'net_box_office.php?cityName=mysore').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    mysore_net_gross = tointeger02(mysore_net_gross)
    
    mysore_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=mysore').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    mysore_disti_net_gross = tointeger01(mysore_disti_net_gross)
    
    tnkerala_net_gross = soup.find('a', href = 'net_box_office.php?cityName=tn_kerla').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    tnkerala_net_gross = tointeger02(tnkerala_net_gross)
    
    tnkerala_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=tn_kerla').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    tnkerala_disti_net_gross = tointeger01(tnkerala_disti_net_gross)
    
    bihar_net_gross = soup.find('a', href = 'net_box_office.php?cityName=bihar').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    bihar_net_gross = tointeger02(bihar_net_gross)
    
    bihar_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=bihar').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    bihar_disti_net_gross = tointeger01(bihar_disti_net_gross)
    
    wb_net_gross = soup.find('a', href = 'net_box_office.php?cityName=west_bengal').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    wb_net_gross = tointeger02(wb_net_gross)
    
    wb_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=west_bengal').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    wb_disti_net_gross = tointeger01(wb_disti_net_gross)
    
    assam_net_gross = soup.find('a', href = 'net_box_office.php?cityName=assam').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    assam_net_gross = tointeger02(assam_net_gross)
    
    assam_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=assam').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    assam_disti_net_gross = tointeger01(assam_disti_net_gross)
    
    orrisa_net_gross = soup.find('a', href = 'net_box_office.php?cityName=orrisa').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    orrisa_net_gross = tointeger02(orrisa_net_gross)
    
    orrisa_disti_net_gross = soup.find('a', href = 'india-distributor-share.php?cityName=orrisa').find_next('img', src = re.compile('images/rupeesim-black.png')).next
    orrisa_disti_net_gross = tointeger01(orrisa_disti_net_gross)

#%% Converting bs4 strings to sensible datatypes
    print(tointeger02(mumbai_net_gross))