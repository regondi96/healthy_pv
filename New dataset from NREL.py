#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import sys, os
from IPython.display import display

# Declare all variables as strings. Spaces must be replaced with '+'
# Define the lat, long of the location and the year, This case is Tuxtla Gutiérrez
# So, the lat and the lon are 16,-93.
lat, lon, year = 16.7577, -93.1299, 2017
# You must request an NSRDB api key from the link above
api_key = 'uvKAIKNaSEgiBRByQbdz6TQ2JszAVol3xcvPygtC'
# Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
attributes = 'clearsky_ghi,clearsky_dhi,clearsky_dni,cloud_type,ghi,dhi,dni,wind_speed,wind_direction,air_temperature,solar_zenith_angle'
# Choose year of data
year = '2017'
# Set leap year to true or false. True will return leap day data if present, false will not.
leap_year = 'false'
# Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
interval = '30'
# Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
# NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
# local time zone.
utc = 'false'
# Your full name, use '+' instead of spaces.
your_name = 'Aguilar+Argueta'
# Your reason for using the NSRDB.
reason_for_use = 'Academic+Purposes'
# Your affiliation
your_affiliation = 'Universidad+de+Ciencias+y+Artes+de+Chiapas'
# Your email address
your_email = 'argueta.auaj@gmail.com'
# Please join our mailing list so we can keep you up-to-date on new developments.
mailing_list = 'false'
# Declare url string
url = 'http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
# Return just the first 2 lines to get metadata:
info = pd.read_csv(url, nrows=1)
# See metadata for specified properties, e.g., timezone and elevation
timezone, elevation = info['Local Time Zone'], info['Elevation']
# Set the dataset
df = pd.read_csv('http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes), skiprows=2)
# Set the time index in the pandas dataframe:
df = df.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=525600/int(interval)))
# Drop the useless data
df.drop(("Year"), axis=1, inplace=True)
df.drop(("Month"), axis=1, inplace=True)
df.drop(("Day"), axis=1, inplace=True)
df.drop(("Hour"), axis=1, inplace=True)
df.drop(("Minute"), axis=1, inplace=True)
#Download the new dataframe
export_csv = df.to_csv ('C:\Users\Hogar\Desktop\dataset2017.csv', index = True, header = True) 
#Don't forget to add '.csv' at the end of the path





