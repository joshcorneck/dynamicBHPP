import numpy as np
from more_itertools import flatten
from abc import ABC, abstractmethod
from typing import Dict, List

class BaseNetwork(ABC):

    @abstractmethod
    def _create_change_point_times(self):
        pass

    @abstractmethod
    def _create_node_memberships(self):
        pass

    @abstractmethod
    def _create_rate_matrices(self):
        pass

    @abstractmethod
    def _sample_adjancency_matrix(self):
        pass
    
    @abstractmethod
    def _sample_single_edge(self):
        pass


class PoissonNetwork(BaseNetwork):
    """
    The base class for networks (complete or incomplete) with 
    Poisson processes living on the edges. The num_nodes are assumed to 
    be partitioned into groups, with the Poisson process on each edge
    being dependent upon the group identity of the num_nodes it joins.

    Parameters:
        - num_nodes: an integer for the number of num_nodes in the network.
        - num_groups: an integer for the number of latent groups.
        - T_max: the upper limit of the time interval from which we
                 sample.
        - lam_matrix: array for the rate parameters.
        - rho_matrix: array for edge connection probabilities. For a complete
                      network, don't supply one.
    """

    def __init__(self, num_nodes: int, num_groups: int, T_max: float,
                 lam_matrix: np.array, rho_matrix: np.array = None, 
                 sparsity: float = None) -> None:
        super().__init__()
        self.num_nodes = num_nodes; self.num_groups = num_groups; self.T_max = T_max
        self.lam_matrix = lam_matrix.astype(float); self.rho_matrix = rho_matrix
        self.sparsity = sparsity

        # Flags for method calling
        self._create_change_point_times_flag = False

    def _create_change_point_times(self, mem_change: bool, rate_change: bool, group_num_change:bool,
                                   mem_change_times: list, rate_change_times: list, 
                                   group_num_change_times: list, num_mem_cps: int, 
                                   num_rate_cps: int):
        """
        Create a list of the relevant change point times. The user can supply the change
        times, or they can be randomly sampled.
        Parameters:
            - mem_change: Boolean for whether we see a change in memberships.
            - rate_change: Boolean for whether we see rate matrix change.
            - mem_change_times: list of change point times for membership changes,
                                if this isn't given, it's sampled.
            - rate_change_times: list of change point times for rate matrix changes,
                                 if this isn't given, it's sampled.
            - num_mem_cps: the number of membership change points (for sampling).
            - num_rate_cps: the number of rate change points (for sampling).
        """
        self._create_change_point_times_flag = True
        self.mem_change = mem_change
        self.rate_change = rate_change
        self.group_num_change = group_num_change

        ## CASE 1: rate and mem changes
        if mem_change & rate_change:
            ## Membership change times  
            # If not supplied, change times are randomly sampled.
            if mem_change_times is None:
                if num_mem_cps == 0:
                    raise ValueError("Please supply a non-zero integer for num_mem_cps")
                else:
                    self.num_mem_cps = num_mem_cps

                    self.mem_change_point_times = np.random.uniform(low=0, high=self.T_max, 
                                                    size=num_mem_cps)
            else:
                self.num_mem_cps = num_mem_cps
                self.mem_change_point_times = mem_change_times
            self.mem_change_point_times.sort()
            
            ## Rate change times
            # If not supplied, change times are randomly sampled.
            if rate_change_times is None:
                if num_rate_cps == 0:
                    raise ValueError("Please supply a non-zero integer for num_rate_cps")
                else:
                    self.num_rate_cps = num_rate_cps
                    self.rate_change_point_times = np.random.uniform(low=0, high=self.T_max, 
                                                    size=num_rate_cps)
            else:
                self.num_rate_cps = num_rate_cps
                self.rate_change_point_times = rate_change_times
            self.rate_change_point_times.sort()

            # Create an array of complete change times. Contains an additional row
            # with 0 for mem and 1 for rate.
            source_indicator = np.concatenate((
                np.zeros_like(self.mem_change_point_times),
                np.ones_like(self.rate_change_point_times)
                )
            )
            full_cp_times = np.concatenate(
                (self.mem_change_point_times,
                 self.rate_change_point_times
                )
            )
            sort_indices = np.argsort(full_cp_times)
            sorted_full_cp_times = full_cp_times[sort_indices]
            sorted_source_indicator = source_indicator[sort_indices]

            full_cp_times_indicator = np.vstack(
                (sorted_full_cp_times,
                 sorted_source_indicator)
            )
            self.full_cp_times = full_cp_times_indicator
            

        ## CASE 2: mem but not rate changes
        elif (mem_change & (not rate_change)):
            # If not supplied, change times are randomly sampled.
            if mem_change_times is None:
                if num_mem_cps == 0:
                    raise ValueError("Please supply a non-zero integer for num_mem_cps")
                else:
                    self.num_mem_cps = num_mem_cps

                    self.mem_change_point_times = np.random.uniform(low=0, high=self.T_max, 
                                                    size=num_mem_cps)
            else:
                self.num_mem_cps = num_mem_cps
                self.mem_change_point_times = mem_change_times
            self.mem_change_point_times.sort()
        
        ## CASE 3: rate but not mem changes
        elif (rate_change & (not mem_change)):
            # If not supplied, change times are randomly sampled.
            if rate_change_times is None:
                if num_rate_cps == 0:
                    raise ValueError("Please supply a non-zero integer for num_rate_cps")
                else:
                    self.num_rate_cps = num_rate_cps
                    self.rate_change_point_times = np.random.uniform(low=0, high=self.T_max, 
                                                    size=num_rate_cps)
            else:
                self.num_rate_cps = num_rate_cps
                self.rate_change_point_times = rate_change_times
            self.rate_change_point_times.sort()

        # CASE 4: group merger
        elif (group_num_change):
            # If not supplied, change times are randomly sampled.
            if group_num_change_times is None:
                raise ValueError("Supply times for changes to group number.")
            self.group_num_change_times = group_num_change_times
            self.group_num_change_times.sort()
        
    def _create_node_memberships(self, group_sizes: list, mem_change_nodes: np.array):
        """
        Method to create a list of numpy arrays that give the membership of each node.
        This will be of length num_mem_cps + 1 (so it is of length 1 if no changes).
        There are different methods for assigning group memberships.
        Parameters:
            - group_sizes: the number of number of nodes in each group. The length must
                            be the same as the number of groups.
            # - group_assignment_type: how we split the group memberships over the nodes. This
            #                             can be any of "sequential",  "alternate" or "random".
        """
        if group_sizes is None:
            raise ValueError("Supply group sizes.")
        if not self._create_change_point_times_flag:
            raise ValueError("You must call _create_change_point_times first.")
        if self.group_num_change:
            for i in range(len(group_sizes)):
                if np.array(group_sizes[i]).sum() != self.num_nodes:
                    raise ValueError("Ensure the all group sizes sum to the total number of nodes.")
                if group_sizes[i].shape[0] != self.num_groups:
                    raise ValueError("Ensure that group_sizes shape matches the number of groups.")
        else:
            if np.array(group_sizes).sum() != self.num_nodes:
                raise ValueError("Ensure the group sizes sum to the total number of nodes.")
            if group_sizes.shape[0] != self.num_groups:
                raise ValueError("Ensure that group_sizes shape matches the number of groups.")

        # # Order the nodes to match the desired split of group assignments
        # if group_assignment_type == 'sequential':
        #     pass

        # elif group_assignment_type == 'alternate':
        #     # Get the number of unique elements and the counts
        #     unique_elements, counts = np.unique(groups, return_counts=True)

        #     # Create a repeating array, where we repeat the minimum number of times
        #     min_count = np.min(counts)
        #     repeating_array = np.tile(unique_elements, min_count)

        #     # Append the remaining counts of the other groups
        #     remaining_elements = []
        #     for i in range(len(unique_elements)):
        #         num_to_append = counts[i] - min_count
        #         remaining_elements.append([i] * num_to_append)
        #     remaining_elements = np.array(list(flatten(remaining_elements)))

        #     groups = np.concatenate((repeating_array, remaining_elements)).astype(int)

        # elif group_assignment_type == 'random':
        #     np.random.shuffle(groups)
        
        # else:
        #     raise ValueError("""Please supply group_assignment_type from the
        #                     list ['sequential', 'alternate', 'random']""")
        
        ## Create assignments for change points
        if self.mem_change:
            if isinstance(group_sizes, list):
                raise ValueError("Ensure only one matrix of group_sizes is supplied.")
            
            # Create initial group assignments
            initial_groups = np.array(
                list(flatten([[i]*j for i,j in enumerate(group_sizes)]))
            )

            # Create a list of lists in which we have the group assignments in each region.
            groups_in_regions = []
            groups_in_regions.append(initial_groups)
            # Copy orginal group array
            groups_cps = initial_groups.copy()
            # Iterate over change points
            for cp in range(self.num_mem_cps):
                old_group = groups_cps[mem_change_nodes[cp]]
                new_group = old_group
                while new_group == old_group:
                    new_group = np.random.randint(0, self.num_groups)
                groups_cps[mem_change_nodes[cp]] = new_group 
                groups_in_regions.append(groups_cps.copy())

            return groups_in_regions
        
        ## Create assignments for changing number of groups
        elif self.group_num_change:
            if len(group_sizes) != (len(self.group_num_change_times) + 1):
                raise ValueError("""Supplied number of group sizes matrices
                                 must match the number of change times""")
            
            # Create a list of lists in which we have the group assignments in each region.
            groups_in_regions = []

            # Initial group assignments
            initial_groups = np.array(
                list(flatten([[i]*j for i,j in enumerate(group_sizes[0])]))
            )
            groups_in_regions.append(initial_groups)
            # Iterate over change points
            for cp in range(len(self.group_num_change_times)):
                new_groups = np.array(
                    list(flatten([[i]*j for i,j in enumerate(group_sizes[cp + 1])]))
                )
                groups_in_regions.append(new_groups.copy())

            return groups_in_regions
        
        else:
            return initial_groups
        
    def _create_rate_matrices(self, sigma: float, entries_to_change: list):
        """
        Currently this only has functionality to randomly change one of the entries 
        according to a normal distribution centred at that value with s.d. sigma.
        """
        if self.rate_change:
            self.lam_matrices = []
            self.lam_matrices.append(self.lam_matrix)

            for i in range(self.num_rate_cps):
                if entries_to_change is None:
                    changing_idx = np.random.randint(low=0, high=self.num_groups, size=2)
                else: 
                    changing_idx = entries_to_change[i]
                new_rate = np.abs(np.random.normal(
                    loc=self.lam_matrix[changing_idx[0], changing_idx[1]],
                    scale=sigma))
                new_rate_matrix = self.lam_matrices[i].copy()
                new_rate_matrix[changing_idx[0], changing_idx[1]] = new_rate
                self.lam_matrices.append(new_rate_matrix)
        else:
            pass

    def _sample_adjancency_matrix(self, groups: int, sparsity: float):
        """
        Samples an adjacency matrix from the supplied rho_matrix.
        Parameters:
            - groups: the node memberships in each region.
        """

        # Extract the groups for each node. If there is a membership change point, we 
        # will have a list not numpy array.
        if (not self.mem_change) & (not self.group_num_change):
            group_assignments = groups
        else:
            group_assignments = groups[0]

        ## If a sparsity is supplied, randomly assign edges to that proportion
        if sparsity is not None:
            adjacency_matrix = np.ones((self.num_nodes, self.num_nodes))
            np.fill_diagonal(adjacency_matrix, 0)

            # Randomly set '1 - sparsity' proportion of entries to zero
            num_nonzero_entries = int((1 - sparsity) * (self.num_nodes ** 2 - self.num_nodes))
            indices_to_zero = np.random.choice(range(self.num_nodes ** 2), num_nonzero_entries, replace=False)
            adjacency_matrix_flat = adjacency_matrix.flatten()
            adjacency_matrix_flat[indices_to_zero] = 0
            adjacency_matrix = adjacency_matrix_flat.reshape((self.num_nodes, self.num_nodes))
        
        ## Otherwise use the supplied rho_matrix, and set to fully-connected if rho_matrix is None
        else:
            if self.rho_matrix is not None:
                temp_mat = self.rho_matrix[group_assignments]
                edge_probs = temp_mat[:, group_assignments]
                np.fill_diagonal(edge_probs, 0)
                adjacency_matrix = (
                    (np.random.rand(self.num_nodes, self.num_nodes) < edge_probs).astype(int)
                )
            else:
                adjacency_matrix = np.ones((self.num_nodes, self.num_nodes))
                np.fill_diagonal(adjacency_matrix, 0)

        return adjacency_matrix
    
    def _sample_single_edge(self, lam: float, t_start: float, t_end: float) -> list:
        """
        A method to sample the Poisson process on a given edge between
        two time points. The two time points are variables to allow
        for simulation of change points.
        Parameters: 
            - lam: the intensity value of the given Poisson process.
            - t_start: start of the simulation window.
            - t_end: end of the simulation window.
        """
        num_arrivals = np.random.poisson(lam * (t_end - t_start))
        arrivals = np.random.uniform(t_start, t_end, size=(num_arrivals, ))
        arrivals.sort()

        return arrivals.tolist()
    
    def sample_network(self, rate_change: bool = False, mem_change: bool = False, 
                        group_num_change: bool = False,
                        num_rate_cps: int = 0, num_mem_cps: int = 0, 
                        group_sizes: np.array = None, mem_change_times: np.array = None, 
                        mem_change_nodes: np.array = None, rate_change_times: np.array = None, 
                        entries_to_change: list = None, rate_matrices: List[np.array] = None,
                        group_num_change_times: np.array = None) -> Dict[int, Dict[int, list]]:
        """
        A method to sample the full network.
        Parameters:
            - rate_change: bool for whether we have a change in the rate matrix.
            - mem_change: bool for whether we have a change in the group memberships.
            - num_mem_cps: the number of membership change points we observe.
            - num_rate_cps: the number of rate change points we observe.
            # - group_assignment_type: how we split the group memberships over the nodes. This
            #                         can be any of "sequential",  "alternate" or "random".
            - group_sizes: the number of nodes in each group (must sum to num_nodes).
            - mem_change_point_times: the times of the changes (must have length equal to num_mem_cps).
            - mem_change_nodes: the nodes that change at each change point (must have length equal to num_mem_cps).
            - entries_to_change: the entry of the rate matrix that changes at each change point.
            - rate_matrices: list of rate matrices.
        """
        ###
        # STEP 1
        # Create change points and relevant matrices and/or changing nodes.
        ###
        ## Checks
        if mem_change:
            if (mem_change_nodes is not None) & (mem_change_times is not None):
                    if len(mem_change_nodes) != len(mem_change_times):
                        raise ValueError("""mem_change_nodes and mem_change_times must have
                                         the same length.""")
            # if num_mem_cps supplied, check that it agrees
            if num_mem_cps != 0:
                if (mem_change_nodes is not None):
                    if num_mem_cps != len(mem_change_nodes):
                        num_mem_cps = len(mem_change_nodes)
                if (mem_change_times is not None):
                    if num_mem_cps != len(mem_change_times):
                        num_mem_cps = mem_change_times
            # if num_mem_cps is not supplied, then it is inferred 
            if num_mem_cps == 0:
                if (mem_change_times is not None) & (mem_change_nodes is not None):
                    num_mem_cps = len(mem_change_times)
                elif (mem_change_times is not None):
                    num_mem_cps = len(mem_change_times)
                elif (mem_change_nodes is not None):
                    num_mem_cps = len(mem_change_nodes)
                else:
                    raise ValueError("""Please supply a non-zero value of num_mem_cps""")
            # if mem_change_nodes not supplied, it is created randomly. mem_change_times
            # are created later if not supplied.
            if mem_change_nodes is None:
                mem_change_nodes = np.random.choice(np.arange(self.num_nodes),
                                                    size=num_mem_cps,
                                                    replace=True)
        if rate_change:
            if (rate_change_times is not None) & (entries_to_change is not None):
                    if len(rate_change_times) != len(entries_to_change):
                        raise ValueError("""rate_change_times and entries_to_change must have
                                         the same length.""")
            # if num_rate_cps supplied, check that it agrees
            if num_rate_cps != 0:
                if (entries_to_change is not None):
                    if num_rate_cps != len(entries_to_change):
                        num_rate_cps = len(entries_to_change)
                if (rate_change_times is not None):
                    if num_rate_cps != len(rate_change_times):
                        num_rate_cps = rate_change_times
            # num_rate_cps is inferred if rate_change_times are supplied
            if num_rate_cps == 0:
                if (rate_change_times is not None) & (entries_to_change is not None):
                    num_rate_cps = len(rate_change_times)
                elif (rate_change_times is not None):
                    num_rate_cps = len(rate_change_times)
                elif (entries_to_change is not None):
                    num_rate_cps = len(entries_to_change)
                else:
                    raise ValueError("""Please supply a non-zero value of num_mem_cps""")
            # if rate matrices is supplied, it must agree with num_rate_cps
            if rate_matrices is not None:
                if len(rate_matrices) != (num_rate_cps + 1):
                    raise ValueError("""Must supply a number of rate matrices to agree with 
                                     num_rate_cps""")
        
        num_cps = num_mem_cps + num_rate_cps

        # Create change point times
        self._create_change_point_times(mem_change, rate_change, group_num_change, mem_change_times, 
                                        rate_change_times, group_num_change_times,
                                        num_mem_cps, num_rate_cps)

        groups_in_regions = self._create_node_memberships(group_sizes, mem_change_nodes)
        # Create rate matrices
        if rate_matrices is None:
            self._create_rate_matrices(sigma=2, entries_to_change=entries_to_change)
        else:
            self.lam_matrices = rate_matrices

        ###
        # STEP 2
        ###
        adjacency_matrix = self._sample_adjancency_matrix(groups_in_regions, self.sparsity)
        self.adjacency_matrix = adjacency_matrix

        ###
        # STEP 3
        ###
        # Sample the network
        network = dict()
        for i in range(self.num_nodes):
            # Dictionary to store arrivals relating to node i
            network[i] = dict()
            for j in range(self.num_nodes):
                network[i][j] = []
                if self.adjacency_matrix[i,j] == 0:
                    network[i][j] = None
                else:
                    ###
                    # MEMBERSHIP CHANGE
                    ###
                    if (mem_change & (not rate_change)):
                        # Run from start to first CP
                        # Map node i and j to their groups
                        group_i = groups_in_regions[0][i]
                        group_j = groups_in_regions[0][j]

                        network[i][j] += (
                            self._sample_single_edge(self.lam_matrix[group_i,group_j],
                                    t_start=0, t_end=self.mem_change_point_times[0])
                        )
                        # Iterate over CPs
                        for cp in range(1, num_mem_cps):
                            # Map node i and j to their groups
                            group_i = groups_in_regions[cp][i]
                            group_j = groups_in_regions[cp][j]
                            if j == i:
                                network[i][j] = None
                            else:

                                network[i][j] += (
                                    self._sample_single_edge(self.lam_matrix[group_i,group_j],
                                            t_start=self.mem_change_point_times[cp-1], 
                                            t_end=self.mem_change_point_times[cp])
                                )
                        # From final CP to the end
                        if num_mem_cps == 1:
                            cp = 0
                        group_i = groups_in_regions[num_mem_cps][i]
                        group_j = groups_in_regions[num_mem_cps][j]

                        network[i][j] += (
                            self._sample_single_edge(self.lam_matrix[group_i,group_j],
                                    t_start=self.mem_change_point_times[num_mem_cps-1], 
                                    t_end=self.T_max)
                        )
                    ###
                    # VARYING NUMBER OF GROUPS
                    ###
                    elif (group_num_change):
                        num_group_changes = len(group_num_change_times)
                        # Run from start to first CP
                        # Map node i and j to their groups
                        group_i = groups_in_regions[0][i]
                        group_j = groups_in_regions[0][j]

                        network[i][j] += (
                            self._sample_single_edge(self.lam_matrix[group_i,group_j],
                                    t_start=0, t_end=self.group_num_change_times[0])
                        )
                        # Iterate over CPs
                        for cp in range(1, num_group_changes):
                            # Map node i and j to their groups
                            group_i = groups_in_regions[cp][i]
                            group_j = groups_in_regions[cp][j]
                            if j == i:
                                network[i][j] = None
                            else:
                                network[i][j] += (
                                    self._sample_single_edge(self.lam_matrix[group_i,group_j],
                                            t_start=self.group_num_change_times[cp-1], 
                                            t_end=self.group_num_change_times[cp])
                                )
                        # From final CP to the end
                        if num_group_changes == 1:
                            cp = 0
                        group_i = groups_in_regions[num_group_changes][i]
                        group_j = groups_in_regions[num_group_changes][j]

                        network[i][j] += (
                            self._sample_single_edge(self.lam_matrix[group_i,group_j],
                                    t_start=self.group_num_change_times[num_group_changes-1], 
                                    t_end=self.T_max)
                        )
                    ###
                    # RATE CHANGE
                    ###
                    elif (rate_change & (not mem_change)):
                        # Run from start to first CP
                        # Map node i and j to their groups
                        group_i = groups_in_regions[i]
                        group_j = groups_in_regions[j]

                        network[i][j] += (
                            self._sample_single_edge(self.lam_matrices[0][group_i,group_j],
                                    t_start=0, t_end=self.rate_change_point_times[0])
                        )
                        # Iterate over CPs
                        for cp in range(1, num_rate_cps):
                            if j == i:
                                network[i][j] = None
                            else:

                                network[i][j] += (
                                    self._sample_single_edge(self.lam_matrices[cp][group_i,group_j],
                                            t_start=self.rate_change_point_times[cp-1], 
                                            t_end=self.rate_change_point_times[cp])
                                )
                        # From final CP to the end
                        if num_rate_cps == 1:
                            cp = 0

                        network[i][j] += (
                            self._sample_single_edge(self.lam_matrices[num_rate_cps][group_i,group_j],
                                    t_start=self.rate_change_point_times[num_rate_cps-1], 
                                    t_end=self.T_max)
                        )

                    ###
                    # MEMBERSHIP & RATE CHANGE
                    ###
                    elif (rate_change & mem_change):
                        # Run from start to first CP
                        # Map node i and j to their initial groups
                        group_i = groups_in_regions[0][i]
                        group_j = groups_in_regions[0][j]

                        # Get current (initial) lam_matrix
                        lam_matrix_curr = self.lam_matrices[0]

                        network[i][j] += (
                            self._sample_single_edge(lam_matrix_curr[group_i,group_j],
                                    t_start=0, t_end=self.full_cp_times[0,0])
                        )
                        # Iterate over CPs
                        cp_mem_track = 0; cp_rate_track = 0
                        for cp in range(1, num_cps):
                            cp_type = self.full_cp_times[1,cp]

                            # Get current memberships or current 
                            if cp_type == 0:
                                cp_mem_track += 1
                                group_i = groups_in_regions[cp_mem_track][i]
                                group_j = groups_in_regions[cp_mem_track][j]
                            elif cp_type == 1:
                                cp_rate_track += 1
                                lam_matrix_curr = self.lam_matrices[cp_rate_track]
                            else:
                                # ADD FUNCTIONALITY FOR SIMULTANEOUS SWITCHES
                                cp_mem_track += 1
                                cp_rate_track += 1

                            if j == i:
                                network[i][j] = None
                            else:
                                network[i][j] += (
                                    self._sample_single_edge(lam_matrix_curr[group_i,group_j],
                                            t_start=self.full_cp_times[0,cp-1], 
                                            t_end=self.full_cp_times[0,cp])
                                )

                        # From the final change point to the end.
                        if self.full_cp_times[1,-1] == 0:
                            group_i = groups_in_regions[-1][i]
                            group_j = groups_in_regions[-1][j]
                        elif self.full_cp_times[1,-1] == 1:
                            lam_matrix_curr = self.lam_matrices[-1]
                        else:
                            # ADD FUNCTIONALITY FOR SIMULTANEOUS SWITCHES
                            pass
                            
                        network[i][j] += (
                            self._sample_single_edge(lam_matrix_curr[group_i,group_j],
                                    t_start=self.full_cp_times[0,-1], 
                                    t_end=self.T_max)
                        )

                    else:
                        # Map node i and j to their groups
                        group_i = groups_in_regions[i]
                        group_j = groups_in_regions[j]

                        network[i][j] += (
                            self._sample_single_edge(self.lam_matrix[group_i,group_j],
                                            t_start=0, t_end=self.T_max)
                        )

        return (network, groups_in_regions)

