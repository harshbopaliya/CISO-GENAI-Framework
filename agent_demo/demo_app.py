import yaml
import torch
import numpy as np
import sys
import os

# Add the parent directory (CISO-GENAI) to sys.path to enable imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import CISO framework components from src
from src.causal_engine import CausalAdvantage
from src.synergy_engine import HJBSolver
from src.topology_engine import TopologyGroups
# from src.policies import AgentPolicy # Policies are used inside agents.py

# Import demo-specific components
from agent_demo.demo_env import GridWorldEnv
from agent_demo.agents import PlannerAgent, CoderAgent, DebaterAgent

def run_demo():
    """
    Runs a multi-agent CISO demo using the GridWorld environment,
    integrating updated CISO components reflecting theoretical structure.
    """
    print("🚀 Starting CISO Multi-Agent GridWorld Demo (Theory-Aligned) 🚀")

    # 1. Load configuration
    try:
        with open("agent_demo/demo_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print(f"  Loaded config: {config}")
    except FileNotFoundError:
        print("Error: demo_config.yaml not found. Make sure it's in the agent_demo folder.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing demo_config.yaml: {e}")
        return

    # Extract configuration parameters
    num_agents = config["env"]["num_agents"]
    max_steps_per_episode = config["env"]["max_steps"]
    hjb_lambda = config["ciso"]["hjb_lambda"]
    topology_eps = config["ciso"]["topology_eps"]
    # gamma = config["ciso"]["gamma"]
    # causal_shapley_beta = config["ciso"]["causal_shapley_beta"]

    # The state dimension for a single agent in GridWorldEnv is 2 (x, y coordinates)
    single_agent_state_dim = 2
    # The action dimension for a single agent in GridWorldEnv is 4 (discrete movements)
    single_agent_action_dim = 4

    # 2. Initialize Environment
    env = GridWorldEnv(num_agents=num_agents)
    env.max_steps = max_steps_per_episode
    print(f"  Environment initialized: GridWorldEnv with {num_agents} agents.")
    print(f"  Max steps per episode: {env.max_steps}")

    # 3. Initialize Agents
    agents = {
        "agent_0": PlannerAgent("agent_0", state_dim=single_agent_state_dim, action_dim=single_agent_action_dim),
        "agent_1": CoderAgent("agent_1", state_dim=single_agent_state_dim, action_dim=single_agent_action_dim),
        "agent_2": DebaterAgent("agent_2", state_dim=single_agent_state_dim, action_dim=single_agent_action_dim),
    }
    print("  Agents initialized: Planner, Coder, Debater.")

    # 4. Initialize CISO Framework Components
    # CausalAdvantage expects a flattened state of all agents (num_agents * single_agent_state_dim)
    total_state_dim_for_causal_net = num_agents * single_agent_state_dim
    causal_net = CausalAdvantage(state_dim=total_state_dim_for_causal_net)

    # HJBSolver expects single agent state dim as part of its internal network
    hjb_solver = HJBSolver(lambda_reg=hjb_lambda, state_dim=single_agent_state_dim)

    # TopologyGroups expects an epsilon for clustering
    topo_engine = TopologyGroups(eps=topology_eps)

    print("\n--- Running Simulation Episode ---")
    observations, _ = env.reset() # Get initial observations from gymnasium reset

    done = False
    truncated = False
    step_count = 0

    while not done and not truncated and step_count < max_steps_per_episode:
        print(f"\n--- Simulation Step {step_count + 1}/{max_steps_per_episode} ---")

        # Agents generate actions based on their current observations
        actions_to_env = {}
        # Prepare individual agent states for CISO components, ensuring consistent order
        all_agent_states_for_ciso = []
        for agent_id in sorted(observations.keys()):
            # Each agent's observation is its (x,y) position
            obs_for_agent = observations[agent_id]
            action = agents[agent_id].act(obs_for_agent)
            actions_to_env[agent_id] = action
            all_agent_states_for_ciso.append(torch.tensor(obs_for_agent, dtype=torch.float32))
            print(f"  {agent_id} chose action: {action}")

        # Stack all agent observations into a single tensor for CISO components that take combined state
        # This will be (num_agents, single_agent_state_dim) e.g., (3, 2)
        states_for_ciso_components = torch.stack(all_agent_states_for_ciso)

        # Apply actions to the environment
        next_observations, rewards, done, truncated, info = env.step(actions_to_env)

        # 5. Apply CISO Framework Components (Conceptual)
        # Causal Advantage (Global Advantage Decomposition)
        try:
            # CausalAdvantage's forward expects (batch_size, total_state_dim)
            # So, flatten states_for_ciso_components (3,2) to (1,6)
            flattened_states_for_causal = states_for_ciso_components.flatten().unsqueeze(0)
            causal_advantage_global = causal_net(flattened_states_for_causal)
            print(f"  Causal Advantage (Global): {causal_advantage_global.item():.3f} "
                  f"(Conceptually A_do_C + sum(gamma_k * E_do(a_k)[A_syn_k]))")
        except Exception as e:
            print(f"  Error calculating Causal Advantage: {e}")

        # Emergent Synergy (HJB Manifold Approximation)
        try:
            # HJBSolver's forward expects (num_agents, single_agent_state_dim)
            synergy_scores_per_agent = hjb_solver(states_for_ciso_components)
            # Take the mean to get an overall synergy value for display
            mean_synergy = synergy_scores_per_agent.mean().item()
            print(f"  Emergent Synergy (Mean HJB Value): {mean_synergy:.3f} "
                  f"(Approximation of HJB PDE solution for synergy)")
        except Exception as e:
            print(f"  Error calculating Emergent Synergy: {e}")

        # Topological Groups (Persistent Homology Clustering)
        try:
            # TopologyGroups' cluster expects a numpy array
            agent_groups = topo_engine.cluster(states_for_ciso_components.detach().numpy())
            print(f"  Topological Groups Discovered: {agent_groups} "
                  f"(H_0 connected components at eps={topology_eps})")
        except Exception as e:
            print(f"  Error discovering Topological Groups: {e}")

        observations = next_observations # Update observations for the next step
        step_count += 1

    if done or truncated:
        print(f"\n--- Simulation Episode Finished (Done: {done}, Truncated: {truncated}) ---")
    else:
        print("\n--- Simulation Concluded (Max steps reached) ---")

    print("\n--- Demo Concluded ---")

if __name__ == "__main__":
    run_demo()
