## Import files
import glob
import numpy as np
from datetime import datetime, timezone
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances as haversine
from math import radians
import pickle 

import time

start = time.time()

print("Calculating Haversine distance matrix between stations...")
## Calculate Haversine distance matrix between stations
# - Import stations and locations (and sort in alphabetical order - the file is not sorted, even though it looks like it is!)
locations = pd.read_csv('santander_locations.csv', quotechar='"').sort_values('StationName')

# - Earth radius
re = 6365.079
# - Transform latitude and longitude in degrees to radians
locations['latitude_radians'] = locations['latitude'].transform(lambda x: radians(float(x)))
locations['longitude_radians'] = locations['longitude'].transform(lambda x: radians(float(x)))
# - Obtain the distance matrix
H = haversine(locations[['latitude_radians','longitude_radians']]) * re
# - Find repeated stations (stations with the same latitude and longitude)
repeated_stations = np.where((H == 0.0).sum(axis=1) != 1) ## (array([ 62,  63, 175, 176, 347, 348, 487, 488, 601, 602, 655, 656, 768, 769, 770]),)
repeated_indices = [62, 175, 347, 487, 601, 655, 768, 769] ## Each repeated_stations entry is paired

## Start date in entire dataset is 02/01/2019
start_date = int(datetime(2019,1,2,0,0).replace(tzinfo=timezone.utc).timestamp())
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

print("Preprocessing data files...")
## Preprocess all data files
files = glob.glob('../csv_files/*.csv')

## Concatenate in a unique Pandas DataFrame
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
# Map the station names to their correct strings
df['StartStation Name'] = df['StartStation Name'].apply(lambda x: docking_station(x))
df['EndStation Name'] = df['EndStation Name'].apply(lambda x: docking_station(x))

# Check if name has corresponding location 
df = (
	df[df['StartStation Name'].isin(
		locations['StationName']) 
		& 
		df['EndStation Name'].isin(locations['StationName'])]
) ## Every row now has a corresponding longitude and latitude

# Create a mapping from station name to a number
map_stations = {}
k = 0
for id, station in enumerate(sorted(set(locations['StationName']))):
	map_stations[station] = k 
	if id not in repeated_indices:
		k += 1
	
# Invert map_stations (will contain two values if more than one key)
map_stations_inverted = {}
for key, value in map_stations.items():
	if value not in map_stations_inverted:
		map_stations_inverted[value] = key

# Map station names to numbers - there are 791 unique values in start_id and end_id
# that completely overlap. The numbers cover integers from 0 to 802 inclusive.
df['start_id'] = df['StartStation Name'].apply(lambda x: map_stations[x])
df['end_id'] = df['EndStation Name'].apply(lambda x: map_stations[x])

# These integers are mapped to range(791). Create an inverse map from network indices
# to station names in map_stations_inverted. 
number_dict = {old_value: new_value for new_value, old_value in enumerate(df['start_id'].unique())}
number_dict_inverted = {value: key for key, value in number_dict.items()}
network_to_station = (
	{node_idx: map_stations_inverted[number_dict_inverted[node_idx]] 
  for node_idx in range(df['start_id'].nunique())}
) ## Maps network index to station name
network_to_longitude_latitude = (
	{node_idx: locations.loc[locations['StationName'] == network_to_station[node_idx], ['longitude', 'latitude']].values 
  	for node_idx in range(df['start_id'].nunique())}
) ## Maps network index to latitude and longitude

# Save dictionaries
with open(f'network_to_station_dict.pkl', 'wb') as file:
    pickle.dump(network_to_station, file)

with open(f'network_to_longitude_latitude_dict.pkl', 'wb') as file:
    pickle.dump(network_to_longitude_latitude, file)

# Map the start_id and end_id using the dictionaries
df['start_id'] = df['start_id'].map(number_dict)
df['end_id'] = df['end_id'].map(number_dict)

# - Transform times to minutes since start of the recording period, and add small noise
np.random.seed(1234567)
df['start_time'] = df['Start Date'].transform(lambda x: datefy(x, start_date=start_date) / 60 + np.random.uniform())
df['end_time'] = df['End Date'].transform(lambda x: datefy(x, start_date=start_date) / 60 + np.random.uniform())
# - Sort by start time
df = df.sort_values(by='start_time')
# - Compute the end week of the trip
df['end_week'] = (df['end_time'] / (24 * 60)) // 7 + 0.5

# - Turn into a network for processing
df_full_network = df[['start_id', 'end_id', 'end_week']].rename(
    columns={'start_id': 'i', 'end_id': 'j', 'end_week': 'time'}
)

print("Creating pivot table...")
# - Create a pivot table and fill missing values with None
pivot_table = (
    df_full_network
            .pivot_table(index='i', columns='j', values='time', 
                        aggfunc=list, fill_value=None))
pivot_table.replace({np.nan: None}, inplace=True)

# - Ensure indices run from 0, to max(max(i), max(j))
max_node = max(df_full_network.i.unique().max(), 
               df_full_network.j.unique().max())
new_index = range(max_node + 1)
new_columns = range(max_node + 1)
new_pivot_table = pd.DataFrame(index=new_index, 
                               columns=new_columns)
new_pivot_table.update(pivot_table)

# - Replace NaN with None
new_pivot_table.replace({np.nan: None}, inplace=True)       

# - Turn to dict of dict for variational Bayes analysis
network = new_pivot_table.to_dict(orient='index')

# - Pickle the dictionary 
with open(f'santander_network.pkl', 'wb') as file:
    pickle.dump(network, file)

end = time.time()
print(f"Time taken: {start} - {end}")
