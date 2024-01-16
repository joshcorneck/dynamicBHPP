#%%
import numpy as np
from more_itertools import flatten
from abc import ABC, abstractmethod

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
                 lam_matrix: np.array, rho_matrix: np.array = None) -> None:
        super().__init__()
        self.num_nodes = num_nodes; self.num_groups = num_groups; self.T_max = T_max
        self.lam_matrix = lam_matrix.astype(float); self.rho_matrix = rho_matrix

        # Flags for method calling
        self._create_change_point_times_flag = False

    def _create_change_point_times(self, mem_change: bool, rate_change: bool, 
                                   mem_change_times: list, rate_change_times: list,
                                   num_mem_cps: int, num_rate_cps: int):
        """
        Create a list of the relevant change point times.
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

        if mem_change & rate_change:
            raise ValueError("Currently there is only functionality for one type of change.")

        # Ensure we have a list of memership/rate change point times if needed.
        if mem_change:
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
        
        if rate_change:
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

    def _create_node_memberships(self, group_sizes: list, group_assignment_type: str,
                                 changing_nodes: np.array):
        """
        Method to create a list of numpy arrays that give the membership of each node.
        This will be of length num_mem_cps + 1.
        Parameters:
            - group_sizes: the number of number of nodes in each group. The length must
                            be the same as the number of groups.
            - group_assignment_type: how we split the group memberships over the nodes. This
                                        can be any of "sequential",  "alternate" or "random".
        """
        if group_sizes is None:
            raise ValueError("Supply group sizes.")
        if not self._create_change_point_times_flag:
            raise ValueError("You must call _create_change_point_times first.")
        if len(group_sizes) != self.num_groups:
            raise ValueError("Ensure that len(group_sizes) matches the number of groups.")
        if np.array(group_sizes).sum() != self.num_nodes:
            raise ValueError("Ensure the group sizes sum to the total number of nodes.")
        
        groups = np.array(
            list(flatten([[i]*j for i,j in enumerate(group_sizes)]))
        )

        # Order the nodes to match the desired split of group assignments
        if group_assignment_type == 'sequential':
            pass

        elif group_assignment_type == 'alternate':
            # Get the number of unique elements and the counts
            unique_elements, counts = np.unique(groups, return_counts=True)

            # Create a repeating array, where we repeat the minimum number of times
            min_count = np.min(counts)
            repeating_array = np.tile(unique_elements, min_count)

            # Append the remaining counts of the other groups
            remaining_elements = []
            for i in range(len(unique_elements)):
                num_to_append = counts[i] - min_count
                remaining_elements.append([i] * num_to_append)
            remaining_elements = np.array(list(flatten(remaining_elements)))

            groups = np.concatenate((repeating_array, remaining_elements)).astype(int)

        elif group_assignment_type == 'random':
            np.random.shuffle(groups)
        
        else:
            raise ValueError("""Please supply group_assignment_type from the
                            list ['sequential', 'alternate', 'random']""")
        
        # If a membership change point(s) is specified, then we sample the group assignments
        # in each region around the change points. Otherwise, simply return original group
        # assignments.
        if self.mem_change:
            # Create a list of lists in which we have the group assignments in each region.
            groups_in_regions = []
            groups_in_regions.append(groups)
            # Copy orginal group array
            groups_cps = groups.copy()
            # Iterate over change points
            for cp in range(self.num_mem_cps):
                old_group = groups_cps[changing_nodes[cp]]
                new_group = old_group
                while new_group == old_group:
                    new_group = np.random.randint(0,self.num_groups)
                groups_cps[changing_nodes[cp]] = new_group 
                groups_in_regions.append(groups_cps.copy())

            return groups_in_regions
        
        else:
            return groups
        
    def _create_rate_matrices(self, sigma: float):
        """
        Currently this only has functionality to randomly change one of the entries 
        according to a normal distribution centred at that value with s.d. sigma.
        """
        ### TO DELETE ###
        #               #

        
        self.lam_matrices = []
        self.lam_matrices.append(self.lam_matrix)
        
        #               #
        ### TO DELETE ###
        if self.rate_change:
            self.lam_matrices = []
            self.lam_matrices.append(self.lam_matrix)

            for i in range(self.num_rate_cps):
                entry_to_change = np.random.randint(low=0, high=self.num_groups, size=2)
                new_rate = np.abs(np.random.normal(
                    loc=self.lam_matrix[entry_to_change[0], entry_to_change[1]],
                    scale=sigma))
                new_rate_matrix = self.lam_matrices[i].copy()
                new_rate_matrix[entry_to_change[0], entry_to_change[1]] = new_rate
                self.lam_matrices.append(new_rate_matrix)
        else:
            pass

    def _sample_adjancency_matrix(self, groups: int):
        """
        Samples an adjacency matrix from the supplied rho_matrix.
        Parameters:
            - groups: the node memberships in each region.
        """

        # Extract the groups for each node. If there is a membership change point, we 
        # will have a list not numpy array.
        if not self.mem_change:
            group_assignments = groups
        else:
            group_assignments = groups[0]

        # Create an n x n matrix where the entries are the relevant 
        # rho_matrix values and sample the adjacency matrix
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
                        num_rate_cps: int = 0, num_mem_cps: int = 0,
                        group_assignment_type: str = 'sequential', group_sizes: np.array = None, 
                        mem_change_times: np.array = None, rate_change_times: np.array = None, 
                        changing_nodes: np.array = None) -> dict(dict()):
        """
        A method to sample the full network.
        Parameters:
            - rate_change: bool for whether we have a change in the rate matrix.
            - mem_change: bool for whether we have a change in the group memberships.
            - num_mem_cps: the number of membership change points we observe.
            - num_rate_cps: the number of rate change points we observe.
            - group_assignment_type: how we split the group memberships over the nodes. This
                                    can be any of "sequential",  "alternate" or "random".
            - group_sizes: the number of nodes in each group (must sum to num_nodes).
            - mem_change_point_times: the times of the changes (must have length equal to num_mem_cps).
            - changing_nodes: the nodes that change at each change point (must have length equal to num_mem_cps).
        """
        ###
        # STEP 1
        ###

        # Create necessary node memberships and rate matrices.
        self._create_change_point_times(mem_change, rate_change, mem_change_times, rate_change_times,
                                        num_mem_cps, num_rate_cps)
        groups_in_regions = self._create_node_memberships(group_sizes, group_assignment_type,
                                                          changing_nodes)
        self._create_rate_matrices(sigma=2)


        ###
        # STEP 2
        ###
        adjacency_matrix = self._sample_adjancency_matrix(groups_in_regions)
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
                    if self.mem_change:
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
                    elif self.rate_change:
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
                    else:
                        # Map node i and j to their groups
                        group_i = groups_in_regions[i]
                        group_j = groups_in_regions[j]

                        network[i][j] += (
                            self._sample_single_edge(self.lam_matrix[group_i,group_j],
                                            t_start=0, t_end=self.T_max)
                        )

        return (network, groups_in_regions)

