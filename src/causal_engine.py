"""
Causal Credit Assignment Engine

Implements do-intervention based causal credit attribution for multi-agent systems.
Given a shared reward, determines each agent's causal contribution using:
1. Marginal Contribution: R_with_agent - R_without_agent
2. Monte-Carlo Shapley Values: Fair distribution across all agent orderings
"""

import random
import numpy as np
from typing import Callable, Dict, List, Tuple, Any


class CausalCreditAssignment:
    """
    Core engine for attributing shared rewards to individual agents
    using causal interventions (do-calculus).
    
    The key insight: Instead of just observing "Agent A was present when reward happened",
    we ask "What would have happened if Agent A acted differently (or did nothing)?"
    
    This is the do-calculus approach: do(agent_k = baseline) vs observe(agent_k = action)
    """
    
    def __init__(self, baseline_policy: Callable = None):
        """
        Initialize the causal credit assignment engine.
        
        Args:
            baseline_policy: The "null" policy used for counterfactual comparison.
                           If None, defaults to random action selection.
        """
        self.baseline_policy = baseline_policy
    
    def set_baseline_policy(self, policy: Callable):
        """Set the baseline policy for counterfactual comparisons."""
        self.baseline_policy = policy
    
    def marginal_contribution(
        self,
        env,
        starts: Dict[str, np.ndarray],
        policy_map: Dict[str, Callable],
        target_agent: str,
        rollout_fn: Callable
    ) -> float:
        """
        Compute marginal contribution of target_agent using do-intervention.
        
        marginal = R_full - R_without_target
        
        Where R_without_target is computed by replacing target_agent with baseline_policy.
        This implements: do(agent_k = baseline) to measure counterfactual outcome.
        
        Args:
            env: The environment to run rollouts in
            starts: Starting positions/states for each agent
            policy_map: Maps agent_id -> policy function
            target_agent: The agent whose contribution we're measuring
            rollout_fn: Function to execute a rollout: (env, starts, policy_map) -> (reward, trajectory)
        
        Returns:
            float: The marginal causal contribution of target_agent
        """
        if self.baseline_policy is None:
            raise ValueError("Baseline policy not set. Call set_baseline_policy() first.")
        
        # Full reward with all agents using their real policies
        R_full, _ = rollout_fn(env, starts, policy_map)
        
        # Counterfactual: Replace target agent with baseline policy
        counterfactual_policy_map = dict(policy_map)
        counterfactual_policy_map[target_agent] = self.baseline_policy
        R_without, _ = rollout_fn(env, starts, counterfactual_policy_map)
        
        # Marginal contribution = difference
        marginal = R_full - R_without
        return marginal
    
    def compute_all_marginals(
        self,
        env,
        starts: Dict[str, np.ndarray],
        policy_map: Dict[str, Callable],
        rollout_fn: Callable
    ) -> Dict[str, float]:
        """
        Compute marginal contributions for all agents.
        
        Args:
            env: The environment
            starts: Starting states
            policy_map: Agent policies
            rollout_fn: Rollout function
        
        Returns:
            Dict mapping agent_id -> marginal contribution
        """
        marginals = {}
        for agent_id in policy_map.keys():
            marginals[agent_id] = self.marginal_contribution(
                env, starts, policy_map, agent_id, rollout_fn
            )
        return marginals


class ShapleyValueEstimator:
    """
    Monte-Carlo Shapley value estimation for fair credit assignment.
    
    Shapley values consider all possible orderings of agents joining a coalition,
    providing a theoretically fair way to distribute shared rewards.
    
    Properties of Shapley values:
    1. Efficiency: Sum of all Shapley values = Total reward
    2. Symmetry: Identical agents get identical values
    3. Null player: Agent with zero contribution gets zero
    4. Additivity: Values can be combined across multiple games
    """
    
    def __init__(self, baseline_policy: Callable, num_samples: int = 100):
        """
        Initialize Shapley value estimator.
        
        Args:
            baseline_policy: Policy used for agents not in the coalition
            num_samples: Number of random permutations to sample for Monte-Carlo estimation
        """
        self.baseline_policy = baseline_policy
        self.num_samples = num_samples
    
    def estimate(
        self,
        env,
        starts: Dict[str, np.ndarray],
        policy_map: Dict[str, Callable],
        rollout_fn: Callable
    ) -> Dict[str, float]:
        """
        Estimate Shapley values using Monte-Carlo sampling over permutations.
        
        For each sampled permutation, we compute the marginal contribution of each
        agent when they "join" the coalition in that order.
        
        Args:
            env: The environment
            starts: Starting states for agents
            policy_map: Maps agent_id -> policy function (their "real" policy)
            rollout_fn: Function to execute rollout: (env, starts, policy_map) -> (reward, traj)
        
        Returns:
            Dict mapping agent_id -> estimated Shapley value
        """
        agents = list(policy_map.keys())
        shapley_values = {agent: 0.0 for agent in agents}
        
        for _ in range(self.num_samples):
            # Random permutation of agents
            perm = agents.copy()
            random.shuffle(perm)
            
            # Build coalition incrementally
            coalition = []
            R_prev = None
            
            for agent in perm:
                coalition.append(agent)
                
                # Build policy map: coalition members use real policies, others use baseline
                current_policy_map = {}
                for a in agents:
                    if a in coalition:
                        current_policy_map[a] = policy_map[a]
                    else:
                        current_policy_map[a] = self.baseline_policy
                
                # Run rollout with current coalition
                R_coalition, _ = rollout_fn(env, starts, current_policy_map)
                
                # Compute marginal contribution
                if R_prev is None:
                    # First agent in permutation - compare against all-baseline
                    marginal = R_coalition
                else:
                    marginal = R_coalition - R_prev
                
                R_prev = R_coalition
                shapley_values[agent] += marginal
        
        # Average over all samples
        for agent in shapley_values:
            shapley_values[agent] /= self.num_samples
        
        return shapley_values


class CounterfactualBaseline:
    """
    Defines different baseline policies for counterfactual comparisons.
    
    The choice of baseline matters:
    - Random: "What if agent acted randomly?"
    - Null: "What if agent did nothing?"
    - Fixed: "What if agent always did action X?"
    """
    
    @staticmethod
    def random_policy(state: np.ndarray, action_space: int = 5) -> int:
        """Random action baseline - agent acts randomly."""
        return random.randint(0, action_space - 1)
    
    @staticmethod
    def null_policy(state: np.ndarray) -> int:
        """Null/stay action - agent does nothing (action 4 = stay)."""
        return 4
    
    @staticmethod
    def create_fixed_action_policy(action: int) -> Callable:
        """Create a policy that always returns a fixed action."""
        def fixed_policy(state: np.ndarray) -> int:
            return action
        return fixed_policy


# Backward compatibility alias
CausalAdvantage = CausalCreditAssignment
