# 🚀 CISO-GENAI-Framework: Causal Credit Assignment for Multi-Agent AI

## The Problem We Solve

### Multi-Agent Credit Assignment: Who Actually Caused the Reward?

When multiple agents act in an environment and receive a **shared reward**, how do you know which agent actually caused it?

```
┌─────────────┐     ┌─────────────┐
│   Agent A   │     │   Agent B   │
│  (Policy πA)│     │  (Policy πB)│
└──────┬──────┘     └──────┬──────┘
       │                   │
       │   Both act in     │
       │   environment     │
       ▼                   ▼
┌─────────────────────────────────┐
│         Environment             │
└───────────────┬─────────────────┘
                │
                ▼
         ┌──────────────┐
         │ Team Reward  │
         │    = +10     │
         └──────────────┘
                │
                ▼
    ❓ WHO CAUSED THIS REWARD? ❓
```

**This framework answers that question using causal interventions.**

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Example Output](#example-output)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
# Install from PyPI
pip install ciso-genai

# Or install from source
git clone https://github.com/harshbopaliya/CISO-GENAI-Framework.git
cd CISO-GENAI-Framework
pip install -e .
```

## Quick Start

```python
from src import (
    CausalCreditAssignment, 
    ShapleyValueEstimator,
    GridWorld, 
    rollout,
    PolicyRegistry,
    random_policy
)
import numpy as np

# Setup environment and policies
env = GridWorld(grid_size=5, max_steps=10)
policies = PolicyRegistry(grid_size=5)

agent_policies = {
    "agent_A": policies.get("center"),  # Smart: moves to center
    "agent_B": policies.get("center"),  # Smart: moves to center
}

starts = {
    "agent_A": np.array([0, 0]),
    "agent_B": np.array([4, 4]),
}

# Run episode and get shared reward
reward, trajectory = rollout(env, starts, agent_policies)
print(f"Team Reward: {reward}")

# CREDIT ASSIGNMENT: Who caused this reward?
credit_engine = CausalCreditAssignment(baseline_policy=random_policy)
marginals = credit_engine.compute_all_marginals(env, starts, agent_policies, rollout)

for agent, contribution in marginals.items():
    print(f"{agent} contributed: {contribution:+.2f}")
```

## Key Features

| Feature | Description | Method |
|---------|-------------|--------|
| **Marginal Contribution** | How much does each agent add? | `R_with_agent - R_without_agent` |
| **Shapley Values** | Fair credit distribution | Monte-Carlo sampling over orderings |
| **Synergy Detection** | Do agents help or hurt each other? | `R_together - sum(R_individual)` |
| **Counterfactual Baselines** | What-if analysis | Replace agent with null/random policy |

## How It Works

### 1. Do-Intervention (Causal Credit)

Instead of just observing correlations, we **intervene**:

```
Traditional: "Agent A was present when reward happened" (correlation)

CISO: "If we REPLACE Agent A with a random policy,
       reward drops from +10 to +3.
       Therefore Agent A CAUSED +7 of the reward." (causation)
```

### 2. Shapley Values (Fair Credit)

For fair distribution, we consider all possible orderings:

```python
# Shapley value for agent_i = average marginal contribution
# across all possible orderings of agents joining the "coalition"

shapley_estimator = ShapleyValueEstimator(baseline_policy=random_policy, num_samples=100)
shapley_values = shapley_estimator.estimate(env, starts, agent_policies, rollout)
# {'agent_A': 5.2, 'agent_B': 4.8}  # Fair split of reward
```

### 3. Synergy Detection

Check if agents work better together:

```python
synergy_detector = SynergyDetector(baseline_policy=random_policy)
synergy = synergy_detector.compute_synergy(env, starts, policies, rollout, ["agent_A", "agent_B"])

# synergy > 0: Agents COMPLEMENT each other
# synergy < 0: Agents INTERFERE with each other
# synergy = 0: Agents are INDEPENDENT
```

## API Reference

### CausalCreditAssignment

```python
class CausalCreditAssignment:
    def __init__(self, baseline_policy: Callable)
    
    def marginal_contribution(self, env, starts, policy_map, target_agent, rollout_fn) -> float
        """Compute: R_full - R_without_target_agent"""
    
    def compute_all_marginals(self, env, starts, policy_map, rollout_fn) -> Dict[str, float]
        """Compute marginal contributions for all agents"""
```

### ShapleyValueEstimator

```python
class ShapleyValueEstimator:
    def __init__(self, baseline_policy: Callable, num_samples: int = 100)
    
    def estimate(self, env, starts, policy_map, rollout_fn) -> Dict[str, float]
        """Estimate Shapley values via Monte-Carlo sampling"""
```

### SynergyDetector

```python
class SynergyDetector:
    def __init__(self, baseline_policy: Callable)
    
    def compute_synergy(self, env, starts, policy_map, rollout_fn, coalition) -> float
        """Synergy = R(together) - sum(R(individual))"""
    
    def detect_all_pairwise_synergies(self, ...) -> Dict[Tuple[str, str], float]
        """Compute synergy for all agent pairs"""
```

## Example Output

```bash
$ python train.py

============================================================
🚀 CISO-GENAI: Causal Credit Assignment Demo
============================================================

📍 Setup:
   Grid size: 5x5
   Max steps: 10
   Agent A starts at: [0 0] (policy: center-seeking)
   Agent B starts at: [4 4] (policy: center-seeking)

============================================================
📊 STEP 1: Full Episode Rollout (Both Agents Active)
============================================================
   Cumulative Team Reward: -12.00

============================================================
🔬 STEP 2: Marginal Contributions (Do-Intervention)
============================================================
   Question: What if each agent acted randomly instead?

   Results:
      agent_A: +8.00
         → agent_A HELPED (reward dropped by 8.00 without them)
      agent_B: +6.00
         → agent_B HELPED (reward dropped by 6.00 without them)

============================================================
⚖️  STEP 3: Shapley Values (Fair Credit Assignment)
============================================================
   Shapley Values (fair distribution):
      agent_A: +5.20
      agent_B: +4.80

============================================================
🤝 STEP 4: Synergy Detection
============================================================
   Synergy(A, B) = +2.00
   → Agents COMPLEMENT each other (better together)

✅ Demo Complete!
```

## Project Structure

```
CISO-GENAI-Framework/
├── src/
│   ├── __init__.py           # Package exports
│   ├── causal_engine.py      # CausalCreditAssignment, ShapleyValueEstimator
│   ├── synergy_engine.py     # SynergyDetector, InterferenceDetector
│   ├── topology_engine.py    # AgentClusterer, InteractionTracker
│   ├── policies.py           # PolicyRegistry, predefined policies
│   └── envs/
│       └── gridworld.py      # GridWorld environment, rollout function
├── configs/
│   └── ciso_default.yaml     # Default configuration
├── train.py                  # Main demo script
├── demo.py                   # Minimal sanity test
└── README.md
```

## Contributing

Contributions welcome! Areas to explore:

- [ ] **Temporal Credit Assignment** - Handle delayed rewards (action at t=0 causes reward at t=100)
- [ ] **Learning Integration** - Use causal credits to train agent policies
- [ ] **LLM Agent Support** - Apply to multi-LLM systems (Planner, Coder, Reviewer)
- [ ] **Richer Environments** - Beyond GridWorld

## License

MIT License - see [LICENSE](LICENSE)

## Contact

**Harsh Bopaliya**
- GitHub: [harshbopaliya](https://github.com/harshbopaliya)
- LinkedIn: [harshbopaliya2003](https://www.linkedin.com/in/harshbopaliya2003/)
- Email: bopaliyaharsh7@gmail.com

---
⭐ If you find this useful, please star the repo!