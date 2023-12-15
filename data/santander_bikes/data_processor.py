#!/usr/bin/env python3
## Import files
import glob
import numpy as np
from datetime import datetime, timezone
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances as haversine
from math import radians
import pickle 

## Calculate Haversine distance matrix between stations
# - Import stations and locations (and sort in alphabetical order - the file is not sorted, even though it looks like it is!)
locations = pd.read_csv('santander_locations.csv',quotechar='"').sort_values('StationName')
# - Earth radius
re = 6365.079
# - Transform latitude and longitude in degrees to radians
locations['latitude_radians'] = locations['latitude'].transform(lambda x: radians(float(x)))
locations['longitude_radians'] = locations['longitude'].transform(lambda x: radians(float(x)))
# - Obtain the distance matrix
H = haversine(locations[['latitude_radians','longitude_radians']]) * re
# - Save resulting Numpy array
np.save('santander_distances.npy', arr=H)
# Find repeated stations (stations with the same latitude and longitude)
repeated_stations = np.where((H == 0.0).sum(axis=1) != 1) ## (array([ 62,  63, 175, 176, 347, 348, 487, 488, 601, 602, 655, 656, 768, 769, 770]),)
# Merge stations with identical locations
repeated_indices = [63, 176, 348, 488, 602, 656, 769, 770]
H = np.delete(H, repeated_indices, axis=0)
H = np.delete(H, repeated_indices, axis=1)
# - Create a mapping (and its inverse) from a station name to a number (in alphabetical order)
map_stations = {}
k = 0
for id, station in enumerate(sorted(set(locations['StationName']))):
	map_stations[station] = k 
	if id not in repeated_indices:
		map_stations[k] = station
		k += 1

# - Save resulting dictionary
with open('santander_dictionary.pkl', 'wb') as f:
    pickle.dump(map_stations, f)

## Start date in entire dataset is 02/03/2022
start_date = int(datetime(2022,3,2,0,0).replace(tzinfo=timezone.utc).timestamp())
## Function to transform a string x in datetime format with a specific start date and splitting delimiters
def datefy(x, start_date, date_split='/', time_split=':', invert_date=False):
	dt = x.rstrip('\r\n').split(' ')
	date = dt[0].split(date_split)
	if invert_date:
		date = date[::-1]
	time = dt[1].split(time_split)
	return int(datetime(int(date[2]),int(date[1]),int(date[0]),int(time[0]),int(time[1])).replace(tzinfo=timezone.utc).timestamp()) - start_date

## Preprocessing function for station names (some recording errors are present in the Santander Cycles data)
def docking_station(x):
	if 'Lansdown ' in x:
		x = x.replace('Lansdown ', 'Lansdowne ')
	while ' ,' in x:
		x = x.replace(' ,', ',')
	if '  ' in x:
		x = x.replace('  ', ' ')
	if '_OLD' in x:
		x = x.replace('_OLD', '')
	if "'" in x:
		x = x.replace("'", "'")
	if ' road' in x:
		x = x.replace(' road', ' Road')
	if 'Haggerston ' in x:
		x = x.replace(', Haggerston ', ', Haggerston')
	return x 

## Preprocess all training data files
# - Obtain all file names for training data
files = glob.glob('training/*.csv')
# - Concatenate in a unique Pandas DataFrame
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
# - Transform the stations according to the mapping (for simplicity of processing)
df['start_id'] = df['StartStation Name'].transform(lambda x: map_stations[docking_station(x)])
df['end_id'] = df['EndStation Name'].transform(lambda x: map_stations[docking_station(x)])
# - Transform times to minutes since start of the recording period, and add small noise
np.random.seed(1234567)
df['start_time'] = df['Start Date'].transform(lambda x: datefy(x, start_date=start_date) / 60 + np.random.uniform())
df['end_time'] = df['End Date'].transform(lambda x: datefy(x, start_date=start_date) / 60 + np.random.uniform())
# - Sort by start time
df = df.sort_values(by='start_time')
# - Save the resulting DataFrame
df[['start_id','end_id','start_time','end_time']].to_csv('santander_train.csv', sep=',', columns=None, header=True, index=False)

## Preprocess all test data files
# - Obtain all file names for test data
files = glob.glob('test/*.csv')
# - Obtain a DataFrame for each file in a loop
list_dfs = []
np.random.seed(171)
for file in files:
	# Read the file
	df_test = pd.read_csv(file)
	# Columns format changes from file 335JourneyDataExtract12Sep2022-18Sep2022.csv onwards
	if int(file.split('/')[-1].split('Journey')[0]) < 335:
		## FORMAT: Rental Id,Duration,Bike Id,End Date,EndStation Id,EndStation Name,Start Date,StartStation Id,StartStation Name
		# Transform the stations according to the mapping (for simplicity of processing)
		df_test['start_id'] = df_test['StartStation Name'].transform(lambda x: map_stations[docking_station(x)])
		df_test['end_id'] = df_test['EndStation Name'].transform(lambda x: map_stations[docking_station(x)])
		# Transform times to minutes since start of the recording period, and add small noise
		df_test['start_time'] = df_test['Start Date'].transform(lambda x: datefy(x, start_date=start_date) / 60 + np.random.uniform())
		df_test['end_time'] = df_test['End Date'].transform(lambda x: datefy(x, start_date=start_date) / 60 + np.random.uniform())
	else:
		## FORMAT: Number,Start date,Start station number,Start station,End date,End station number,End station,Bike number,Bike model,Total duration,Total duration (ms)
		# Remove repairs
		df_test = df_test[~df_test['Start station'].str.contains('Mechanical Workshop')]
		df_test = df_test[~df_test['End station'].str.contains('Mechanical Workshop')]
		# Transform the stations according to the mapping (for simplicity of processing)
		df_test['start_id'] = df_test['Start station'].transform(lambda x: map_stations[docking_station(x)])
		df_test['end_id'] = df_test['End station'].transform(lambda x: map_stations[docking_station(x)])
		# Transform times to minutes since start of the recording period, and add small noise to start time
		df_test['start_time'] = df_test['Start date'].transform(lambda x: datefy(x, start_date=start_date, date_split='-', invert_date=True) / 60 + np.random.uniform())
		# Use the exact duration in milliseconds to obtain the end time in minutes
		df_test['end_time'] = df_test['start_time'] + df_test['Total duration (ms)'] / 60 / 1000
	# Append all DataFrames in a list (only for selected columns)
	list_dfs.append(df_test[['start_id','end_id','start_time','end_time']])

# - Concatenate all DataFrames into a unique DF, and sort by start time
df_test = pd.concat(list_dfs, axis=0, ignore_index=True).sort_values(by='start_time')
# - Save the resulting DataFrame
df_test[['start_id','end_id','start_time','end_time']].to_csv('santander_test.csv', sep=',', columns=None, header=True, index=False)
# - Delete list of dataframes
del list_dfs
