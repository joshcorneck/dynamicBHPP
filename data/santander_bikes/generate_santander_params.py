#%%
import pandas as pd
import itertools

num_nodes_set = [803]
num_groups_set = [2,3,4,5,6]
n_cavi_set = [3]
delta_z_set = [1, 0.1]
delta_pi_set = [1, 0.1]
delta_lam_set = [1, 0.1]

all_combinations = list(
    itertools.product(num_nodes_set, num_groups_set,
                      n_cavi_set, delta_z_set,  
                      delta_pi_set, delta_lam_set)
)

with open('data/santander_bikes/santander_params.txt', 'w') as file:
    for combination in all_combinations:
        line = ' '.join(map(str, combination)) + '\n'
        file.write(line)

print(len(all_combinations))

df_combs = pd.DataFrame(all_combinations, 
                        columns=['num_nodes', 'num_groups', 'n_cavi',
                                'delta_z', 'delta_pi', 'delta_lam'])
df_combs.to_pickle('data/santander_bikes/df_high_school_2012_params.pkl')

# %%
