# import torch
# import torch.nn as nn

# class CausalAdvantage(nn.Module):
#     """
#     Estimates the causal advantage (or impact) of agents in a multi-agent system.
#     The network takes the concatenated states of all agents and outputs a single value
#     representing the overall causal impact or advantage.
#     """
#     def __init__(self, state_dim: int):
#         """
#         Initializes the CausalAdvantage network.

#         Args:
#             state_dim (int): The total flattened dimension of all agent states
#                              (num_agents * single_agent_state_dim).
#         """
#         super().__init__()
#         # The input layer should match the 'state_dim' directly,
#         # which is already calculated as the combined flattened state of all agents.
#         self.global_net = nn.Sequential(
#             nn.Linear(state_dim, 64), # FIX: Changed from state_dim * 3 to state_dim
#             nn.ReLU(),
#             nn.Linear(64, 1) # Output a single scalar advantage score
#         )

#     def forward(self, states: torch.Tensor) -> torch.Tensor:
#         """
#         Computes the causal advantage.

#         Args:
#             states (torch.Tensor): A tensor representing the combined states of all agents.
#                                    Expected shape: (batch_size, total_state_dim),
#                                    where total_state_dim is num_agents * single_agent_state_dim.
#                                    In our demo, batch_size is 1, so (1, num_agents * single_agent_state_dim).

#         Returns:
#             torch.Tensor: A tensor containing the causal advantage score.
#         """
#         # The states are expected to be already flattened and batched by the caller (demo_app.py)
#         # So, no need for .flatten().unsqueeze(0) here if the input is already (1, total_state_dim)
#         # If 'states' is (num_agents, single_agent_state_dim), then flatten it here:
#         if states.dim() == 2 and states.shape[0] > 1: # If it's (num_agents, single_agent_state_dim)
#              global_state = states.flatten().unsqueeze(0) # Flatten and add batch dimension
#         else: # Assumed to be already (1, total_state_dim)
#              global_state = states
             
#         return self.global_net(global_state)
import torch
import torch.nn as nn

class CausalAdvantage(nn.Module):
    """
    Estimates the global causal advantage following the theoretical decomposition.
    For this demo, Q and V functions are simplified linear networks,
    and causal core/synergy calculations are illustrative.
    """
    def __init__(self, state_dim: int):
        """
        Initializes the CausalAdvantage network components.

        Args:
            state_dim (int): The total flattened dimension of all agent states
                             (num_agents * single_agent_state_dim).
        """
        super().__init__()
        # Q-network: Approximates Q(s, a) for the global state-action.
        # Takes flattened state and conceptually combined action.
        # For simplicity, we'll only pass the state here.
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs a single Q-value
        )
        # V-network: Approximates V(s) for the global state.
        self.v_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs a single V-value
        )

        # Placeholder for causal core network (e.g., modeling confounders)
        # In a real CISO, this would involve learning P(z|s_C)
        self.causal_core_net = nn.Sequential(
            nn.Linear(state_dim, 32), # Simplified: takes global state
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs a scalar representing causal core contribution
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Computes the global causal advantage.

        Args:
            states (torch.Tensor): A tensor representing the combined states of all agents.
                                   Expected shape: (batch_size, total_state_dim),
                                   e.g., (1, num_agents * single_agent_state_dim).

        Returns:
            torch.Tensor: A tensor containing the global causal advantage score.
        """
        # Ensure states is in the correct batched format (1, total_state_dim)
        if states.dim() == 1:
            states = states.unsqueeze(0) # Add batch dimension if missing

        # Conceptual Q-value (Q(s, a) - we don't have joint actions here, so just Q(s))
        # In a full impl, this would take (s, a) where 'a' is the joint action.
        q_value = self.q_net(states)

        # Conceptual V-value
        v_value = self.v_net(states)

        # Causal Core (A_do_C): Simplified from (Q - V) * P(z|s_C)
        # Here, it's just a separate network's output to represent its contribution.
        # In a real implementation, 'P(z|s_C)' would be learned.
        causal_core_component = self.causal_core_net(states)

        # Interventional Synergy (A_syn_k): This is more complex in theory,
        # involves counterfactuals. For this demo, we'll represent it as a
        # simple difference from Q-value based on 'prior' (e.g., V-value).
        # The theoretical A_syn_k is related to HJB, but for global advantage,
        # it's a component. We'll simulate its contribution simply.
        # This is a highly simplified proxy for E_do(a_k)[A_syn_k]
        interventional_synergy_component = (q_value - v_value) * 0.5 # A placeholder multiplier

        # Causal Shapley Weight (gamma_k): In theory, this uses mutual information.
        # Here, we'll use a fixed conceptual weight for simplicity for the demo.
        # In a real system, gamma_k would be learned and per-agent/group.
        gamma_k_concept = 0.5 # A fixed conceptual weight for demo purposes

        # Global Advantage Decomposition:
        # A_global = A_do_C + sum(gamma_k * E_do(a_k)[A_syn_k])
        # Since we have a single global state, we sum up conceptual components.
        global_advantage = causal_core_component + gamma_k_concept * interventional_synergy_component

        return global_advantage
