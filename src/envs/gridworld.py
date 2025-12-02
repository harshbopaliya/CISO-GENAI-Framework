"""
GridWorld Environment for Multi-Agent Credit Assignment

A simple grid environment where multiple agents navigate.
Designed to test credit assignment: when agents contribute to shared rewards,
who actually caused the outcome?
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Manhattan distance between two points."""
    return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))


class GridWorld:
    """
    Simple multi-agent GridWorld environment.
    
    Actions:
        0: up    (y -= 1)
        1: right (x += 1)
        2: down  (y += 1)
        3: left  (x -= 1)
        4: stay  (no movement)
    
    Reward: Negative sum of Manhattan distances to center (shared/team reward)
    """
    
    def __init__(self, grid_size: int = 5, max_steps: int = 10):
        """
        Initialize GridWorld.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            max_steps: Maximum steps per episode
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.center = np.array([grid_size // 2, grid_size // 2])
        self.positions: Dict[str, np.ndarray] = {}
        self.t = 0
    
    def reset(self, starts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Reset environment with given starting positions.
        
        Args:
            starts: Dict mapping agent_id -> starting position [x, y]
        
        Returns:
            Current positions of all agents
        """
        self.positions = {k: v.copy() for k, v in starts.items()}
        self.t = 0
        return self.get_state()
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current positions of all agents."""
        return {k: v.copy() for k, v in self.positions.items()}
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool]:
        """
        Execute actions for all agents.
        
        Args:
            actions: Dict mapping agent_id -> action (0-4)
        
        Returns:
            Tuple of (new_positions, shared_reward, done)
        """
        # Apply actions
        for agent_id, action in actions.items():
            pos = self.positions[agent_id]
            
            if action == 0:  # up
                pos[1] = max(0, pos[1] - 1)
            elif action == 1:  # right
                pos[0] = min(self.grid_size - 1, pos[0] + 1)
            elif action == 2:  # down
                pos[1] = min(self.grid_size - 1, pos[1] + 1)
            elif action == 3:  # left
                pos[0] = max(0, pos[0] - 1)
            # action == 4: stay (no change)
        
        self.t += 1
        done = self.t >= self.max_steps
        
        # Shared/team reward: negative total Manhattan distance to center
        total_distance = sum(
            manhattan_distance(pos, self.center) 
            for pos in self.positions.values()
        )
        reward = -float(total_distance)
        
        return self.get_state(), reward, done
    
    def get_individual_rewards(self) -> Dict[str, float]:
        """Get individual rewards (for comparison, not used in credit assignment)."""
        return {
            agent_id: -float(manhattan_distance(pos, self.center))
            for agent_id, pos in self.positions.items()
        }


def rollout(
    env: GridWorld,
    starts: Dict[str, np.ndarray],
    policy_map: Dict[str, Callable]
) -> Tuple[float, List[Dict[str, np.ndarray]]]:
    """
    Execute a full episode rollout.
    
    Args:
        env: The GridWorld environment
        starts: Starting positions for each agent
        policy_map: Dict mapping agent_id -> policy function (state -> action)
    
    Returns:
        Tuple of (cumulative_reward, trajectory)
        - cumulative_reward: Total reward over episode
        - trajectory: List of state dicts at each step
    """
    state = env.reset(starts)
    trajectory = [dict(state)]
    cumulative_reward = 0.0
    done = False
    
    while not done:
        # Get actions from policies
        actions = {}
        for agent_id, pos in state.items():
            actions[agent_id] = policy_map[agent_id](np.array(pos))
        
        # Step environment
        state, reward, done = env.step(actions)
        trajectory.append(dict(state))
        cumulative_reward += reward
    
    return cumulative_reward, trajectory


# Backward compatibility with Gymnasium-style API
class GridWorldEnv:
    """Gymnasium-compatible wrapper for GridWorld."""
    
    def __init__(self, num_agents: int = 3, grid_size: int = 5, max_steps: int = 100):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.env = GridWorld(grid_size=grid_size, max_steps=max_steps)
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self._rng = np.random.default_rng()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Random starting positions
        starts = {
            aid: self._rng.integers(0, self.grid_size, size=2)
            for aid in self.agent_ids
        }
        state = self.env.reset(starts)
        return state, {}
    
    def step(self, actions: Dict[str, int]):
        state, reward, done = self.env.step(actions)
        
        # Return per-agent rewards (same shared reward for backward compat)
        rewards = {aid: reward / self.num_agents for aid in self.agent_ids}
        
        return state, rewards, done, False, {}
