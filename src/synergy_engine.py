"""
Synergy Detection Engine

Measures whether agents working together produce better results than
the sum of their individual contributions.

Synergy = R(coalition) - sum(R(individual_i))

If Synergy > 0: Agents complement each other
If Synergy < 0: Agents interfere with each other
If Synergy = 0: Agents are independent
"""

import numpy as np
from typing import Callable, Dict, List, Tuple
from itertools import combinations


class SynergyDetector:
    """
    Detects synergistic effects between agents.
    
    Key question: Do agents perform better together than alone?
    """
    
    def __init__(self, baseline_policy: Callable):
        """
        Initialize synergy detector.
        
        Args:
            baseline_policy: Policy used when an agent is "absent" from coalition
        """
        self.baseline_policy = baseline_policy
    
    def compute_synergy(
        self,
        env,
        starts: Dict[str, np.ndarray],
        policy_map: Dict[str, Callable],
        rollout_fn: Callable,
        coalition: List[str]
    ) -> float:
        """
        Compute synergy for a specific coalition of agents.
        
        Synergy = R(coalition together) - sum(R(each agent alone))
        
        Args:
            env: The environment
            starts: Starting states
            policy_map: Agent policies
            rollout_fn: Rollout function
            coalition: List of agent IDs in the coalition
        
        Returns:
            float: Synergy value (positive = complementary, negative = interference)
        """
        agents = list(policy_map.keys())
        
        # Reward when coalition works together (others use baseline)
        coalition_policy_map = {}
        for a in agents:
            if a in coalition:
                coalition_policy_map[a] = policy_map[a]
            else:
                coalition_policy_map[a] = self.baseline_policy
        R_together, _ = rollout_fn(env, starts, coalition_policy_map)
        
        # Sum of individual rewards (each agent alone)
        sum_individual = 0.0
        for agent in coalition:
            individual_policy_map = {}
            for a in agents:
                if a == agent:
                    individual_policy_map[a] = policy_map[a]
                else:
                    individual_policy_map[a] = self.baseline_policy
            R_individual, _ = rollout_fn(env, starts, individual_policy_map)
            sum_individual += R_individual
        
        synergy = R_together - sum_individual
        return synergy
    
    def detect_all_pairwise_synergies(
        self,
        env,
        starts: Dict[str, np.ndarray],
        policy_map: Dict[str, Callable],
        rollout_fn: Callable
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute synergy for all pairs of agents.
        
        Returns:
            Dict mapping (agent_i, agent_j) -> synergy value
        """
        agents = list(policy_map.keys())
        synergies = {}
        
        for pair in combinations(agents, 2):
            synergy = self.compute_synergy(env, starts, policy_map, rollout_fn, list(pair))
            synergies[pair] = synergy
        
        return synergies
    
    def find_best_coalition(
        self,
        env,
        starts: Dict[str, np.ndarray],
        policy_map: Dict[str, Callable],
        rollout_fn: Callable,
        min_size: int = 2
    ) -> Tuple[List[str], float]:
        """
        Find the coalition with highest synergy.
        
        Args:
            min_size: Minimum coalition size to consider
        
        Returns:
            Tuple of (best_coalition, best_synergy)
        """
        agents = list(policy_map.keys())
        best_coalition = []
        best_synergy = float('-inf')
        
        # Try all possible coalitions of size >= min_size
        for size in range(min_size, len(agents) + 1):
            for coalition in combinations(agents, size):
                synergy = self.compute_synergy(
                    env, starts, policy_map, rollout_fn, list(coalition)
                )
                if synergy > best_synergy:
                    best_synergy = synergy
                    best_coalition = list(coalition)
        
        return best_coalition, best_synergy


class InterferenceDetector:
    """
    Detects when agents interfere with (hurt) each other's performance.
    """
    
    def __init__(self, baseline_policy: Callable):
        self.baseline_policy = baseline_policy
    
    def detect_interference(
        self,
        env,
        starts: Dict[str, np.ndarray],
        policy_map: Dict[str, Callable],
        rollout_fn: Callable,
        agent_a: str,
        agent_b: str
    ) -> Dict[str, float]:
        """
        Check if agent_b interferes with agent_a's performance.
        
        Returns dict with:
        - 'a_alone': Reward when A acts alone
        - 'a_with_b': Reward when A acts with B
        - 'interference': a_alone - a_with_b (positive = B hurts A)
        """
        agents = list(policy_map.keys())
        
        # A alone
        a_alone_map = {a: self.baseline_policy for a in agents}
        a_alone_map[agent_a] = policy_map[agent_a]
        R_a_alone, _ = rollout_fn(env, starts, a_alone_map)
        
        # A with B
        a_with_b_map = {a: self.baseline_policy for a in agents}
        a_with_b_map[agent_a] = policy_map[agent_a]
        a_with_b_map[agent_b] = policy_map[agent_b]
        R_a_with_b, _ = rollout_fn(env, starts, a_with_b_map)
        
        return {
            'a_alone': R_a_alone,
            'a_with_b': R_a_with_b,
            'interference': R_a_alone - R_a_with_b
        }


# Backward compatibility alias
HJBSolver = SynergyDetector
