# import numpy as np
# # from ripser import Rips # No longer directly used for clustering logic

# class TopologyGroups:
#     """
#     Identifies groups of agents based on their proximity in the state space.
#     This simplified version uses a direct distance threshold (eps) for clustering.
#     """
#     def __init__(self, eps: float = 0.3):
#         """
#         Initializes the TopologyGroups component.

#         Args:
#             eps (float): The maximum distance between two samples for one to be considered
#                          as in the neighborhood of the other. Agents within this distance
#                          will be considered part of the same group.
#         """
#         self.eps = eps
#         # print(f"Rips(maxdim=1, thresh=inf, coeff=2, do_cocycles=False, n_perm = None, verbose=True)")
#         # The above line was from the original code using ripser.
#         # It's commented out as we are using a simpler clustering for demo purposes.

#     def cluster(self, states: np.ndarray) -> list[list[int]]:
#         """
#         Clusters agents into groups based on their proximity using the 'eps' threshold.

#         Args:
#             states (np.ndarray): A NumPy array of agent states.
#                                  Expected shape: (num_agents, state_dim).

#         Returns:
#             list[list[int]]: A list of lists, where each inner list contains the indices
#                              of agents belonging to the same group.
#         """
#         num_agents = states.shape[0]
#         if num_agents == 0:
#             return []

#         # Create an adjacency matrix based on epsilon distance
#         adj_matrix = np.zeros((num_agents, num_agents), dtype=bool)
#         for i in range(num_agents):
#             for j in range(num_agents):
#                 if i == j:
#                     adj_matrix[i, j] = True # An agent is always connected to itself
#                 else:
#                     distance = np.linalg.norm(states[i] - states[j])
#                     if distance <= self.eps:
#                         adj_matrix[i, j] = True

#         # Find connected components (groups) using depth-first search
#         visited = [False] * num_agents
#         groups = []

#         for i in range(num_agents):
#             if not visited[i]:
#                 current_group = []
#                 stack = [i]
#                 visited[i] = True
#                 while stack:
#                     agent_idx = stack.pop()
#                     current_group.append(agent_idx)
#                     for neighbor_idx in range(num_agents):
#                         if adj_matrix[agent_idx, neighbor_idx] and not visited[neighbor_idx]:
#                             visited[neighbor_idx] = True
#                             stack.append(neighbor_idx)
#                 # Only add groups with more than one agent (to represent collaboration)
#                 if len(current_group) > 1:
#                     groups.append(sorted(current_group)) # Sort for consistent output

#         return groups
import numpy as np
# from ripser import Rips # Not directly used for simple clustering, but kept for context if needed for higher homology

class TopologyGroups:
    """
    Identifies topological groups (connected components) among agents based on
    their proximity in the state space using a direct distance threshold (eps).
    This simulates the 'Topological Group Formation' part of the CISO theory,
    focusing on H_0 (connected components).
    """
    def __init__(self, eps: float = 0.3):
        """
        Initializes the TopologyGroups component.

        Args:
            eps (float): The maximum Euclidean distance between two agents for them
                         to be considered connected and potentially part of the same group.
        """
        self.eps = eps
        # In a full ripser integration for persistent homology, you might initialize:
        # self.rips = Rips(maxdim=0, coeff=2, thresh=self.eps)
        # However, for simply extracting epsilon-connected components, direct DSU is clearer.

    def cluster(self, states: np.ndarray) -> list[list[int]]:
        """
        Clusters agents into groups based on their proximity using the 'eps' threshold.
        This method uses a Disjoint Set Union (DSU) approach to find connected components.

        Args:
            states (np.ndarray): A NumPy array of agent states.
                                 Expected shape: (num_agents, state_dim).

        Returns:
            list[list[int]]: A list of lists, where each inner list contains the indices
                             of agents belonging to the same connected group.
                             Only groups with more than one agent are returned to represent
                             meaningful collaboration.
        """
        num_agents = states.shape[0]
        if num_agents == 0:
            return []

        # Disjoint Set Union (DSU) data structure for finding connected components
        parent = list(range(num_agents)) # Each agent is initially its own parent (own set)

        def find(i):
            """Finds the representative (root) of the set containing element i."""
            if parent[i] == i:
                return i
            # Path compression: make the current node's parent the root
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            """Unites the sets containing elements i and j."""
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j # Merge set of i into set of j
                return True
            return False

        # Build connections based on the 'eps' threshold
        # Iterate through all unique pairs of agents
        for i in range(num_agents):
            for j in range(i + 1, num_agents): # Ensure unique pairs (i, j) and avoid (i, i)
                # Calculate the Euclidean distance between agent i's state and agent j's state
                distance = np.linalg.norm(states[i] - states[j])
                # If the distance is within the epsilon threshold, union their sets
                if distance <= self.eps:
                    union(i, j)

        # Extract groups from the DSU structure
        groups_map = {}
        for i in range(num_agents):
            root = find(i) # Find the root for each agent
            if root not in groups_map:
                groups_map[root] = []
            groups_map[root].append(i) # Add agent to its corresponding group

        # Filter for groups that contain more than one agent
        # Sort each group internally and sort the list of groups for consistent output
        final_groups = [sorted(group) for group in groups_map.values() if len(group) > 1]

        return final_groups
