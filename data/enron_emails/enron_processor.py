#%%
import numpy as np
import pandas as pd
import pickle

enron_data = pd.read_csv('data/enron_emails/enron_tuples_filter.txt',
            sep=",", header=None, 
            dtype= {0:'int32', 1:'int32', 2:'int32'})
enron_data.columns = ['time', 'i', 'j']

# Reset minimum time to 0
enron_data['time'] = enron_data['time'] - enron_data['time'].min()

# Convert 'time' column to datetime format
enron_data['time_date'] = pd.to_datetime(enron_data['time'], unit='s', 
                                         origin='unix')
# Calculate the difference in weeks
enron_data['week'] = (enron_data['time_date'] - enron_data['time_date'].min()).dt.days // 7
# enron_data = enron_data.drop(columns=['time', 'time_date'])

# Add to shift to within range
enron_data['week'] = enron_data['week'] + 0.5

# Create a pivot table and fill missing values with None
pivot_table = (
    enron_data.pivot_table(index='i', columns='j', values='week', 
                           aggfunc=list, fill_value=None))

# Replace NaN with None
pivot_table.replace({np.nan: None}, inplace=True)

## Need to ensure indices run from 0, to max(max(i), max(j))
# Determine the shape of the new DataFrame
max_i = max(pivot_table.index.max(), pivot_table.columns.max())
new_index = range(max_i + 1)
new_columns = range(max_i + 1)

# Create a DataFrame with the specified shape
new_pivot_table = pd.DataFrame(index=new_index, columns=new_columns)

# Update the new DataFrame with existing data
new_pivot_table.update(pivot_table)

# Replace NaN with None
new_pivot_table.replace({np.nan: None}, inplace=True)

# Turn to dict of dict
enron_network = new_pivot_table.to_dict(orient='index')

# Pickle the dictionary 
with open('data/enron_emails/enron_network.pkl', 'wb') as file:
    pickle.dump(enron_network, file)

# %%
