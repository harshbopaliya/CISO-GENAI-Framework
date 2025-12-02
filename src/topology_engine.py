"""
Agent Clustering Engine

Groups agents based on their interactions and proximity in state space.
Useful for:
1. Discovering which agents naturally work together
2. Reducing computation by treating clusters as single units
3. Hierarchical credit assignment (credit to groups, then within groups)
"""

import numpy as np
from typing import List, Dict, Tuple


class AgentClusterer:
    """
    Clusters agents into groups based on state-space proximity.
    
    Uses Disjoint Set Union (DSU) for efficient connected component detection.
    """
    
    def __init__(self, eps: float = 1.0):
        """
        Initialize the clusterer.
        
        Args:
            eps: Maximum distance for two agents to be considered "connected"
        """
        self.eps = eps
    
    def cluster(self, states: np.ndarray) -> List[List[int]]:
        """
        Cluster agents based on proximity.
        
        Args:
            states: Array of shape (num_agents, state_dim)
        
        Returns:
            List of groups, where each group is a list of agent indices.
            Only groups with 2+ agents are returned.
        """
        num_agents = states.shape[0]
        if num_agents == 0:
            return []
        
        # DSU (Disjoint Set Union) data structure
        parent = list(range(num_agents))
        
        def find(i: int) -> int:
            if parent[i] != i:
                parent[i] = find(parent[i])  # Path compression
            return parent[i]
        
        def union(i: int, j: int):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j
        
        # Connect agents within epsilon distance
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                distance = np.linalg.norm(states[i] - states[j])
                if distance <= self.eps:
                    union(i, j)
        
        # Extract groups
        groups_map: Dict[int, List[int]] = {}
        for i in range(num_agents):
            root = find(i)
            if root not in groups_map:
                groups_map[root] = []
            groups_map[root].append(i)
        
        # Return only multi-agent groups
        return [sorted(g) for g in groups_map.values() if len(g) > 1]
    
    def cluster_by_agent_ids(
        self, 
        agent_states: Dict[str, np.ndarray]
    ) -> List[List[str]]:
        """
        Cluster using agent IDs instead of indices.
        
        Args:
            agent_states: Dict mapping agent_id -> state array
        
        Returns:
            List of groups, each group is a list of agent IDs
        """
        agent_ids = list(agent_states.keys())
        states = np.array([agent_states[aid] for aid in agent_ids])
        
        index_groups = self.cluster(states)
        
        # Convert indices to agent IDs
        return [[agent_ids[i] for i in group] for group in index_groups]


class InteractionTracker:
    """
    Track agent interactions over time to discover stable groups.
    """
    
    def __init__(self, eps: float = 1.0, history_length: int = 100):
        """
        Args:
            eps: Distance threshold for interaction
            history_length: Number of steps to track
        """
        self.eps = eps
        self.history_length = history_length
        self.interaction_counts: Dict[Tuple[str, str], int] = {}
        self.step_count = 0
    
    def record_step(self, agent_states: Dict[str, np.ndarray]):
        """Record interactions at current step."""
        agent_ids = list(agent_states.keys())
        
        for i, aid_i in enumerate(agent_ids):
            for aid_j in agent_ids[i+1:]:
                distance = np.linalg.norm(
                    agent_states[aid_i] - agent_states[aid_j]
                )
                if distance <= self.eps:
                    pair = tuple(sorted([aid_i, aid_j]))
                    self.interaction_counts[pair] = \
                        self.interaction_counts.get(pair, 0) + 1
        
        self.step_count += 1
    
    def get_frequent_pairs(self, min_frequency: float = 0.5) -> List[Tuple[str, str]]:
        """
        Get agent pairs that interact frequently.
        
        Args:
            min_frequency: Minimum fraction of steps with interaction
        
        Returns:
            List of (agent_i, agent_j) tuples
        """
        threshold = self.step_count * min_frequency
        return [
            pair for pair, count in self.interaction_counts.items()
            if count >= threshold
        ]
    
    def reset(self):
        """Clear interaction history."""
        self.interaction_counts = {}
        self.step_count = 0


# Backward compatibility alias
TopologyGroups = AgentClusterer
