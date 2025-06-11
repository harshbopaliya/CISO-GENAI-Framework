import torch
import torch.nn as nn

class AgentPolicy(nn.Module):
    def __init__(self, state_dim=2, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim))

    def forward(self, x):
        return self.net(x)
