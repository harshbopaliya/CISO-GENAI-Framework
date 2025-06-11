# agent_demo/agents.py

import torch
import torch.nn.functional as F # Import F for softmax
import torch.distributions as distributions # Import distributions for Categorical
import numpy as np
from src.policies import AgentPolicy # Assuming AgentPolicy takes state and returns logits

class BaseAgent:
    """Base class for all agents in the demo."""
    def __init__(self, agent_id: str, state_dim: int, action_dim: int):
        self.agent_id = agent_id
        # Initialize the policy network for this agent
        # GridWorld observation is 2D (x, y), action is discrete (0-3)
        self.policy = AgentPolicy(state_dim=state_dim, action_dim=action_dim)

    def act(self, observation: np.ndarray) -> int:
        """
        Takes an observation (agent's position) and returns a discrete action (0-3).
        Actions are sampled stochastically from the policy's output.
        """
        # Convert numpy observation (e.g., [x, y]) to torch tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        # Get logits from the policy network
        logits = self.policy(obs_tensor)
        # Apply softmax to get probabilities
        action_probs = F.softmax(logits, dim=-1)
        # Create a categorical distribution and sample an action
        dist = distributions.Categorical(action_probs)
        action = dist.sample().item() # Sample an action from the distribution
        return action

    def learn(self, *args, **kwargs):
        """
        Placeholder for learning logic. In a full RL setup, this would update policy weights.
        For this demo, we'll keep it simple and not implement policy updates here.
        """
        pass

# Example specialized agents (conceptual, their behavior is driven by BaseAgent.act)
class PlannerAgent(BaseAgent):
    def __init__(self, agent_id: str, state_dim: int = 2, action_dim: int = 4):
        super().__init__(agent_id, state_dim, action_dim)
        print(f"  {self.agent_id} (Planner) initialized.")

class CoderAgent(BaseAgent):
    def __init__(self, agent_id: str, state_dim: int = 2, action_dim: int = 4):
        super().__init__(agent_id, state_dim, action_dim)
        print(f"  {self.agent_id} (Coder) initialized.")

class DebaterAgent(BaseAgent):
    def __init__(self, agent_id: str, state_dim: int = 2, action_dim: int = 4):
        super().__init__(agent_id, state_dim, action_dim)
        print(f"  {self.agent_id} (Debater) initialized.")
