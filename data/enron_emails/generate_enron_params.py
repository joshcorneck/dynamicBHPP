#%%
import itertools

num_nodes_set = [184]
num_groups_set = [2, 3, 4, 5, 6]
n_cavi_set = [2]
delta_pi_set = [1]
delta_rho_set = [1]
delta_lam_set = [1]

all_combinations = list(
    itertools.product(num_nodes_set, num_groups_set,
                      n_cavi_set, delta_pi_set, delta_rho_set, 
                      delta_lam_set)
)

with open('data/enron_emails/enron_params.txt', 'w') as file:
    for combination in all_combinations:
        line = ' '.join(map(str, combination)) + '\n'
        file.write(line)

print(len(all_combinations))

# %%
