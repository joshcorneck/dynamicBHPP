#%%
import numpy as np
import pandas as pd
import pickle

df_full_test = pd.read_csv("data/santander_bikes/santander_test.csv")
df_full_train = pd.read_csv("data/santander_bikes/santander_train.csv")
df_full = pd.concat([df_full_train, df_full_test], axis=0, ignore_index=True).sort_values(by='end_time')

df_full['end_week'] = (df_full['end_time'] / (24 * 60)) // 7 + 0.5

#%%
df_full_network = df_full[['start_id', 'end_id', 'end_week']].rename(
    columns={'start_id': 'i', 'end_id': 'j', 'end_week': 'time'}
)

# Create a pivot table and fill missing values with None
pivot_table = (
    df_full_network
            .pivot_table(index='i', columns='j', values='time', 
                        aggfunc=list, fill_value=None))
pivot_table.replace({np.nan: None}, inplace=True)

## Need to ensure indices run from 0, to max(max(i), max(j))
# Determine the shape of the new DataFrame
max_node = max(df_full_network.i.unique().max(), 
               df_full_network.j.unique().max())
new_index = range(max_node + 1)
new_columns = range(max_node + 1)

# Create a DataFrame with the specified shape
new_pivot_table = pd.DataFrame(index=new_index, 
                               columns=new_columns)

# Update the new DataFrame with existing data
new_pivot_table.update(pivot_table)

# Replace NaN with None
new_pivot_table.replace({np.nan: None}, inplace=True)       

# Turn to dict of dict for variational Bayes analysis
network = new_pivot_table.to_dict(orient='index')

# Pickle the dictionary 
with open(f'data/santander_bikes/santander_network.pkl', 'wb') as file:
    pickle.dump(network, file)
# %%
