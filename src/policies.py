"""
Policy Definitions for Multi-Agent Systems

Provides different policy types:
1. Deterministic policies (greedy, fixed)
2. Stochastic policies (random, epsilon-greedy)
3. Learnable policies (neural network based)
"""

import random
import numpy as np
from typing import Callable, Dict


# ============== Deterministic Policies ==============

def create_greedy_policy(target: np.ndarray) -> Callable:
    """
    Create a greedy policy that moves towards a target position.
    
    Args:
        target: Target position [x, y] to move towards
    
    Returns:
        Policy function: state -> action
    """
    def greedy_policy(state: np.ndarray) -> int:
        dx = target[0] - state[0]
        dy = target[1] - state[1]
        
        if abs(dx) >= abs(dy):
            if dx > 0:
                return 1  # right
            elif dx < 0:
                return 3  # left
            else:
                return 4  # stay
        else:
            if dy > 0:
                return 2  # down
            elif dy < 0:
                return 0  # up
            else:
                return 4  # stay
    
    return greedy_policy


def create_center_seeking_policy(grid_size: int = 5) -> Callable:
    """Create a policy that moves towards the grid center."""
    center = np.array([grid_size // 2, grid_size // 2])
    return create_greedy_policy(center)


# ============== Stochastic Policies ==============

def random_policy(state: np.ndarray) -> int:
    """Random action selection (uniform over action space)."""
    return random.choice([0, 1, 2, 3, 4])


def stay_policy(state: np.ndarray) -> int:
    """Always stay in place (null action)."""
    return 4


def create_epsilon_greedy_policy(
    base_policy: Callable, 
    epsilon: float = 0.1
) -> Callable:
    """
    Wrap a policy with epsilon-greedy exploration.
    
    Args:
        base_policy: The underlying policy
        epsilon: Probability of random action
    
    Returns:
        Epsilon-greedy policy
    """
    def epsilon_greedy(state: np.ndarray) -> int:
        if random.random() < epsilon:
            return random.choice([0, 1, 2, 3, 4])
        return base_policy(state)
    
    return epsilon_greedy


# ============== Policy Registry ==============

class PolicyRegistry:
    """
    Registry for managing named policies.
    """
    
    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        self.policies: Dict[str, Callable] = {
            "random": random_policy,
            "stay": stay_policy,
            "center": create_center_seeking_policy(grid_size),
        }
    
    def register(self, name: str, policy: Callable):
        """Register a new policy."""
        self.policies[name] = policy
    
    def get(self, name: str) -> Callable:
        """Get a policy by name."""
        if name not in self.policies:
            raise KeyError(f"Policy '{name}' not found. Available: {list(self.policies.keys())}")
        return self.policies[name]
    
    def list_policies(self) -> list:
        """List all available policy names."""
        return list(self.policies.keys())


# ============== Learnable Policies (Optional PyTorch) ==============

try:
    import torch
    import torch.nn as nn
    
    class NeuralPolicy(nn.Module):
        """Neural network policy for learning."""
        
        def __init__(self, state_dim: int = 2, action_dim: int = 5, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """Returns action logits."""
            return self.net(state)
        
        def act(self, state: np.ndarray) -> int:
            """Select action from state (numpy input)."""
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                logits = self.forward(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            return action
    
    # Backward compatibility
    AgentPolicy = NeuralPolicy
    
except ImportError:
    # PyTorch not available - provide dummy
    class AgentPolicy:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for NeuralPolicy. Install with: pip install torch")
