import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(self, num_agents=3):
        self.num_agents = num_agents
        self.grid_size = 5
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(0, self.grid_size, shape=(2,))
            for i in range(num_agents)
        })
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(4)
            for i in range(num_agents)
        })

    def reset(self):
        self.agent_pos = {f"agent_{i}": np.random.randint(0, self.grid_size, size=2)
                         for i in range(self.num_agents)}
        return self._get_obs()

    def _get_obs(self):
        return {k: v.copy() for k, v in self.agent_pos.items()}

    def step(self, actions):
        rewards = {}
        for agent_id, action in actions.items():
            if action == 0: self.agent_pos[agent_id][1] += 1
            elif action == 1: self.agent_pos[agent_id][0] += 1
            elif action == 2: self.agent_pos[agent_id][1] -= 1
            elif action == 3: self.agent_pos[agent_id][0] -= 1
            self.agent_pos[agent_id] = np.clip(self.agent_pos[agent_id], 0, self.grid_size - 1)
            rewards[agent_id] = -np.linalg.norm(self.agent_pos[agent_id] - np.array([2.5, 2.5]))
        return self._get_obs(), rewards, False, {}
