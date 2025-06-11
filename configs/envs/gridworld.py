import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    """
    A simple GridWorld environment using Gymnasium for the CISO demo.
    Agents move on a grid, and their reward is based on their distance to a central point.
    """
    def __init__(self, num_agents=3):
        self.num_agents = num_agents
        self.grid_size = 5 # Defines the size of the square grid (e.g., 5x5)
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(0, self.grid_size, shape=(2,), dtype=np.int32)
            for i in range(num_agents)
        })
        # Action space: 0: up, 1: right, 2: down, 3: left
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(4)
            for i in range(num_agents)
        })
        self.agent_pos = {} # Stores the current position of each agent
        self.target_pos = np.array([self.grid_size / 2 - 0.5, self.grid_size / 2 - 0.5]) # Center of the grid
        self.current_step = 0
        self.max_steps = 100 # Maximum steps per episode

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state, placing agents randomly.
        This method is compliant with Gymnasium API, returning (observation, info).
        """
        super().reset(seed=seed) # Call parent reset for Gymnasium compatibility
        self.agent_pos = {f"agent_{i}": self.np_random.integers(0, self.grid_size, size=2, dtype=np.int32)
                          for i in range(self.num_agents)}
        self.current_step = 0
        # print(f"\n--- Environment Reset --- Initial Agent Positions: {self.agent_pos}") # Removed print for cleaner training output
        return self._get_obs(), {} # Return observations and an empty info dictionary

    def _get_obs(self):
        """
        Returns the current observations for all agents.
        Each agent observes its own (x, y) coordinates.
        """
        return {k: v.copy() for k, v in self.agent_pos.items()}

    def step(self, actions):
        """
        Applies agent actions to the environment and calculates new state, rewards.
        Each agent moves based on its discrete action (0-3).
        """
        rewards = {}
        for agent_id, action in actions.items():
            # Update agent position based on action
            if action == 0:  # Up
                self.agent_pos[agent_id][1] += 1
            elif action == 1:  # Right
                self.agent_pos[agent_id][0] += 1
            elif action == 2:  # Down
                self.agent_pos[agent_id][1] -= 1
            elif action == 3:  # Left
                self.agent_pos[agent_id][0] -= 1

            # Clip positions to stay within grid boundaries
            self.agent_pos[agent_id] = np.clip(self.agent_pos[agent_id], 0, self.grid_size - 1)

            # Calculate reward: negative Euclidean distance to the center target
            self.target_pos = np.array([self.grid_size / 2 - 0.5, self.grid_size / 2 - 0.5]) # Ensure target_pos is accessible
            rewards[agent_id] = -np.linalg.norm(self.agent_pos[agent_id] - self.target_pos)

        self.current_step += 1
        done = self.current_step >= self.max_steps # Episode ends after max_steps
        truncated = False # For simplicity, not using truncation for this demo

        # print(f"  Step {self.current_step} - Agent Positions: {self.agent_pos} - Rewards: {rewards}") # Removed print for cleaner training output

        return self._get_obs(), rewards, done, truncated, {} # obs, rewards, done, truncated, info
