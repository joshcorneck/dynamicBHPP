#%%
import pandas as pd
import numpy as np
import pickle
#%%
#####################################################################
#########################  2011 DATA ################################
#####################################################################

df_hs = pd.read_csv('data/high_school/thiers_2011.csv',
                    names=['time', 'i', 'j', 'Ci', 'Cj'],
                    sep='\t')

# Map classes to integers
unique_elements = df_hs[['Ci', 'Cj']].stack().unique()
enumeration = {element: i for i, element in enumerate(unique_elements)}
df_hs['Ci'] = df_hs['Ci'].map(enumeration)
df_hs['Cj'] = df_hs['Cj'].map(enumeration)

# Shift time index to 10 (midpoint of first time bin)
df_hs['time'] = df_hs['time'] - (df_hs['time'].min() - 10)

# Index with a day and create sub dataframes
df_hs['day'] = 0
df_hs.loc[(df_hs['time'] >= 86430) & (df_hs['time'] < 172510), 'day'] = 1
df_hs.loc[(df_hs['time'] >= 172510) & (df_hs['time'] < 252770), 'day'] = 2
df_hs.loc[(df_hs['time'] >= 252770), 'day'] = 3

for day in [0,1,2,3]:
    # Select day and map time to 10 as minimum
    df_hs_temp = df_hs[df_hs.day == day]
    df_hs_temp.loc[:, 'time'] = (
        df_hs_temp.loc[:, 'time'] - df_hs_temp.loc[:, 'time'].min()
    )
    df_hs_temp.loc[:, 'time'] = df_hs_temp.loc[:, 'time'] / 20
    
    # Map to minute intervals (+0.5 to ensure counted correctly)
    df_hs_temp_copy = df_hs_temp.copy()
    df_hs_temp_copy.loc[:,'time_min'] = df_hs_temp_copy.loc[:, 'time'] // 3 + 0.5
    df_hs_temp = df_hs_temp_copy.copy()

    # Map to 10 minute intervals (+0.5 to ensure counted correctly)
    df_hs_temp_copy = df_hs_temp.copy()
    df_hs_temp_copy.loc[:,'time_min'] = df_hs_temp_copy.loc[:, 'time'] // 3 + 0.5
    df_hs_temp = df_hs_temp_copy.copy()

    # Create a pivot table and fill missing values with None
    pivot_table = (
        df_hs_temp.drop(columns=['Ci','Cj'])
                .pivot_table(index='i', columns='j', values='time', 
                            aggfunc=list, fill_value=None))
    pivot_table.replace({np.nan: None}, inplace=True)

    ## Need to ensure indices run from 0, to max(max(i), max(j))
    # Determine the shape of the new DataFrame
    max_i = max(pivot_table.index.max(), 
                pivot_table.columns.max())
    new_index = range(max_i + 1)
    new_columns = range(max_i + 1)

    # Create a DataFrame with the specified shape
    new_pivot_table = pd.DataFrame(index=new_index, columns=new_columns)

    # Update the new DataFrame with existing data
    new_pivot_table.update(pivot_table)

    # Replace NaN with None
    new_pivot_table.replace({np.nan: None}, inplace=True)       

    # Turn to dict of dict for variational Bayes analysis
    network = new_pivot_table.to_dict(orient='index')

    # Pickle the dictionary 
    with open(f'data/high_school/data_2011/high_school_network_2011_{day}.pkl', 'wb') as file:
        pickle.dump(network, file)    
# %%
#####################################################################
#########################  2012 DATA ################################
#####################################################################
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

df_hs = pd.read_csv('data/high_school/data_2012/thiers_2012.csv',
                    names=['time_UNIX', 'i', 'j', 'Ci', 'Cj'],
                    sep='\t')

# Read true groups
true_groups = pd.read_csv("data/high_school/data_2012/true_groups_2012.txt",
                          names=['i', 'Ci', 'gender'],
                          sep='\t')

# Mappings
node_mapping = {element: i for i, element in enumerate(true_groups.i)}
group_mapping = {element: i for i, element in enumerate(true_groups.Ci.unique())}
df_hs['Ci'] = df_hs['Ci'].map(group_mapping)
df_hs['Cj'] = df_hs['Cj'].map(group_mapping)
df_hs['i'] = df_hs['i'].map(node_mapping)
df_hs['j'] = df_hs['j'].map(node_mapping)

# Map UNIX ctime and get seconds elapsed since start of the day. Time corresponds
# to whether there was an observation on [t-20, t]
df_hs['time_real'] = df_hs['time_UNIX'].apply(lambda x: datetime.fromtimestamp(x))
df_hs['date'] = df_hs['time_real'].dt.date
date_mapping = {element: i for i, element in enumerate(df_hs.date.unique())}
start_time_day = df_hs.groupby('date')['time_real'].transform('min')
df_hs['time'] = (df_hs['time_real'] - start_time_day).dt.total_seconds()
df_hs['date'] = df_hs['date'].map(date_mapping)

df_hs_rest = df_hs[['date', 'time', 'i', 'j', 'Ci', 'Cj']]

for day in df_hs_rest.date.unique():
    # Select day
    df_hs_temp = df_hs_rest[df_hs_rest.date == day].drop(columns='date')

    # Map times to an interval of [0,100] 
    #  1 time step is then (max_time / 100)
    max_time = df_hs_temp['time'].max()
    df_hs_temp.loc[:,'time'] = df_hs_temp.loc[:,'time'] / max_time * 100

    # Create a pivot table and fill missing values with None
    pivot_table = (
        df_hs_temp.drop(columns=['Ci','Cj'])
                .pivot_table(index='i', columns='j', values='time', 
                            aggfunc=list, fill_value=None))
    pivot_table.replace({np.nan: None}, inplace=True)

    ## Need to ensure indices run from 0, to max(max(i), max(j))
    # Determine the shape of the new DataFrame
    max_i = max(node_mapping.values())
    new_index = range(max_i + 1)
    new_columns = range(max_i + 1)

    # Create a DataFrame with the specified shape
    new_pivot_table = pd.DataFrame(index=new_index, columns=new_columns)

    # Update the new DataFrame with existing data
    new_pivot_table.update(pivot_table)

    # Replace NaN with None
    new_pivot_table.replace({np.nan: None}, inplace=True)       

    # Turn to dict of dict for variational Bayes analysis
    network = new_pivot_table.to_dict(orient='index')

    # Pickle the dictionary 
    with open(f'data/high_school/data_2012/high_school_network_2012_{day}.pkl', 'wb') as file:
        pickle.dump(network, file)

# %%
