import yaml
import torch

# Correct imports for your installed ciso-genai package:
# The 'src.' prefix is removed because 'ciso_genai' is now the top-level package after installation.
# from ciso_genai.causal_engine import CausalAdvantage
# from ciso_genai.synergy_engine import HJBSolver
# from ciso_genai.topology_engine import TopologyGroups
# from ciso_genai.police import AgentPolicy # Assuming AgentPolicy is located in police.py
# from ciso_genai.envs.gridworld import GridWorldEnv # Corrected import path for GridWorldEnv
from src.causal_engine import CausalAdvantage
from src.synergy_engine import HJBSolver
from src.topology_engine import TopologyGroups
from src.policies import AgentPolicy
from src.envs.gridworld import GridWorldEnv

def train():
    # Load configuration from the 'configs' directory.
    # 'train.py' lives in the project root, so 'configs/ciso_default.yaml'
    # is relative to 'train.py' itself.
    with open("configs/ciso_default.yaml") as f:
        config = yaml.safe_load(f)

    # Environment and agent parameters
    num_agents = config["env"]["num_agents"]
    # Assuming each agent's state is (x, y), so state_dim = 2
    single_agent_state_dim = 2 
    
    # Initialize the GridWorld environment
    env = GridWorldEnv(num_agents=num_agents)
    # Ensure env's max_steps matches config for consistency
    env.max_steps = config["env"]["max_steps"]
    
    # Initialize policies for each agent
    policies = {f"agent_{i}": AgentPolicy() for i in range(num_agents)} 
    
    # Initialize the optimizer for training policies
    optimizer = torch.optim.Adam([p for policy in policies.values() for p in policy.parameters()])
    
    # Initialize CISO analytical components
    # CausalAdvantage takes the flattened total state dimension (num_agents * single_agent_state_dim)
    causal_net = CausalAdvantage(state_dim=num_agents * single_agent_state_dim)
    # HJBSolver takes the individual agent state dimension
    hjb_solver = HJBSolver(lambda_reg=config["ciso"]["hjb_lambda"], state_dim=single_agent_state_dim)
    # TopologyGroups takes the epsilon threshold for clustering
    topo = TopologyGroups(eps=config["ciso"]["topology_eps"])

    print(f"Starting training for {num_agents} agents over {config['training']['num_episodes']} episodes...")

    # Main training loop
    for episode in range(config["training"]["num_episodes"]):
        # FIX: Ensure env.reset() returns (observation, info)
        # The GridWorldEnv from demo_env_py should already do this.
        obs, info = env.reset() 
        done = False
        truncated = False
        step_count = 0
        
        # Episode loop
        # Loop until episode is done (e.g., all agents reached goal) or truncated (e.g., max steps reached)
        while not done and not truncated and step_count < config["env"]["max_steps"]: 
            actions = {}
            # Agents choose actions based on their current policy and observation
            for agent_id, policy in policies.items():
                # Ensure observation is a tensor before passing to policy network
                logits = policy(torch.tensor(obs[agent_id], dtype=torch.float32))
                actions[agent_id] = torch.argmax(logits).item() # Assuming discrete actions (0,1,2,3 for GridWorld)

            # Step the environment with chosen actions
            # Gymnasium step returns (observations, rewards, done, truncated, info)
            next_obs, rewards, episode_done, truncated, info = env.step(actions) 
            
            # Check if any agent's 'done' status signals global episode termination
            # Assuming 'episode_done' is a dict of {agent_id: bool} or a single bool
            if isinstance(episode_done, dict):
                # If all agents are individually done, the episode is done
                done = all(episode_done.values())
            else:
                # If it's a single boolean, use it directly
                done = episode_done


            # Prepare current states for CISO component analysis
            # Stack agent observations into a single PyTorch tensor (num_agents, single_agent_state_dim)
            current_states_tensor = torch.stack([
                torch.tensor(obs[k], dtype=torch.float32) for k in obs.keys()
            ])

            # Apply CISO components for analysis
            # Causal Advantage requires a flattened state tensor (batch_size=1, total_state_dim)
            adv = causal_net(current_states_tensor.flatten().unsqueeze(0)) 
            synergy = hjb_solver(current_states_tensor)
            groups = topo.cluster(current_states_tensor.detach().numpy()) # Convert to NumPy for ripser

            # Print current step's analysis results
            # FIX: Use .mean().item() for Causal Advantage if it's a multi-element tensor
            print(f"Episode {episode}, Step {step_count} | Groups: {groups} | Advantage: {adv.mean().item():.3f} | Synergy: {synergy.mean().item():.3f}")

            # Update observation for the next loop iteration
            obs = next_obs 
            step_count += 1
            
        print(f"Episode {episode} finished. Done: {done}, Truncated: {truncated}")

    print("\n--- Training Concluded ---")

if __name__ == "__main__":
    train()

