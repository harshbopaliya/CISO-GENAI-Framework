import yaml
import torch
from src.causal_engine import CausalAdvantage
from src.synergy_engine import HJBSolver
from src.topology_engine import TopologyGroups
from src.policies import AgentPolicy
from configs.envs.gridworld import GridWorldEnv

def train():
    with open("configs/ciso_default.yaml") as f:
        config = yaml.safe_load(f)

    env = GridWorldEnv(num_agents=config["env"]["num_agents"])
    policies = {f"agent_{i}": AgentPolicy() for i in range(3)}
    optimizer = torch.optim.Adam([p for policy in policies.values() for p in policy.parameters()])
    causal_net = CausalAdvantage(state_dim=2)
    hjb_solver = HJBSolver(lambda_reg=config["ciso"]["hjb_lambda"])
    topo = TopologyGroups(eps=config["ciso"]["topology_eps"])

    for episode in range(100):
        obs = env.reset()
        while True:
            actions = {}
            for agent_id, policy in policies.items():
                logits = policy(torch.tensor(obs[agent_id], dtype=torch.float32))
                actions[agent_id] = torch.argmax(logits).item()

            states = torch.stack([torch.tensor(obs[k], dtype=torch.float32) for k in obs.keys()])
            adv = causal_net(states)
            synergy = hjb_solver(states)
            groups = topo.cluster(states.numpy())
            print(f"Episode {episode} | Groups: {groups} | Advantage: {adv.item():.3f}")
            break

if __name__ == "__main__":
    train()
