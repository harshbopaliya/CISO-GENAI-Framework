# 🚀 CISO-GENAI-Framework: Causal Intelligence for Multi-Agent Generative AI

## Table of Contents

- [The Problem We Solve](#the-problem-we-solve)
- [About CISO-GENAI-Framework](#about-ciso-genai-framework)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Demo](#running-the-demo)
- [Understanding the Demo Output](#understanding-the-demo-output)
- [Agents in the Demo](#agents-in-the-demo)
- [Roadmap: What We Need to Build](#roadmap-what-we-need-to-build)
- [Future Work & Contribution](#future-work--contribution)
- [License](#license)
- [Contact](#contact)

## The Problem We Solve

### Multi-Agent Credit Assignment: Who Actually Caused the Reward?

In multi-agent reinforcement learning, there's a fundamental unsolved problem:

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
│    (Shared state, actions)      │
└───────────────┬─────────────────┘
                │
                ▼
         ┌──────────────┐
         │   Reward: +10 │
         └──────────────┘
                │
                ▼
    ❓ WHO CAUSED THIS REWARD? ❓
```

**The Core Problem:**
- Agent A has its own policy and takes actions
- Agent B has its own policy and takes actions  
- A reward of +10 is received
- **There is NO standard way to determine which agent's actions actually caused that reward**

**Why This Matters for GenAI:**
When you have multiple LLM agents (a Planner, a Coder, a Reviewer), and the final output is good/bad:
- Did the Planner's strategy cause the success?
- Or was it the Coder's implementation?
- Or did the Reviewer catch a critical bug?

**Current approaches fail because:**
1. **Correlation ≠ Causation** - Just because Agent A acted before success doesn't mean it *caused* success
2. **Shared rewards hide individual contributions** - Team reward doesn't tell you who contributed
3. **Counterfactual reasoning is missing** - We need to ask "What if Agent A did something different?"

## About CISO-GENAI-Framework

The CISO-GENAI-Framework implements **Causal Invariant Synergy Optimization (CISO)**, a framework designed to solve the multi-agent credit assignment problem using:

1. **Causal Interventions** - Use do-calculus to determine TRUE causal impact
2. **Counterfactual Reasoning** - "What would have happened if Agent A acted differently?"
3. **Reward Attribution** - Decompose shared rewards into individual causal contributions

This repository provides a foundational architecture for building causal credit assignment into multi-agent RL systems.

## Key Features

- **Causal Advantage Interventions**: Determines which agent's actions actually CAUSED the reward using do-calculus interventions, not just correlations.
- **Counterfactual Credit Assignment**: Asks "What reward would we get if Agent A did nothing?" to isolate individual contributions.
- **Reward Decomposition**: Breaks down a shared team reward (e.g., +10) into per-agent causal contributions (e.g., Agent A: +7, Agent B: +3).
- **Topological Group Formation**: Dynamically discovers which agents are working together effectively based on their state-space interactions.

## Project Structure

```
CISO-GENAI/
├── configs/
│   └── ciso_default.yaml         # Default configurations for CISO components
├── src/
│   ├── envs/
│   │   └── gridworld.py          # Gymnasium GridWorld environment setup
│   ├── causal_engine.py          # Implementation of Causal Advantage
│   ├── policies.py               # Base policy network for agents
│   ├── synergy_engine.py         # Approximation of Emergent Synergy Manifolds
│   ├── topology_engine.py        # Implementation of Topological Group Formation
│   └── ...
├── agent_demo/                   # Directory for the multi-agent CISO
│   ├── demo_config.yaml          # Configurations specific to the demo
│   ├── demo_env.py               # The GridWorldEnv used in the demo
│   ├── agents.py                 # Defines the agent classes (Planner, Coder, Debater)
│   └── demo_app.py               # Main script to run the CISO multi-agent demo
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── setup.py                      # (Optional) For packaging the framework
└── train.py                      # (Example) Script for full training (not demo focused)
```

## Setup and Installation

To set up and run the CISO-GENAI demo, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/harshbopaliya/CISO-GENAI-Framework.git
cd CISO-GENAI-Framework
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Since the ciso-genai framework is now published on PyPI, you can install it directly.

First, install PyTorch, as it's a core dependency and requires specific installation based on your system (CPU/GPU).

For CPU-only PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then, install the ciso-genai framework and its remaining dependencies:
```bash
pip install ciso-genai
```

> **Note**: This command will automatically install dependencies like numpy, gymnasium, ripser, and PyYAML as specified in ciso-genai's setup.py. If you want to use the local requirements.txt for development or specific version pinning, you can run `pip install -r requirements.txt` after installing PyTorch.

## Running the Demo

Navigate to the root directory of the CISO-GENAI-Framework and run the demo application:

```bash
python -m agent_demo.demo_app
```

This will start a simulation in the GridWorld environment with three agents. The CISO framework components will analyze their interactions at each step.

### Running the Training Script

You can also run the training script to see the framework in a learning context:

```bash
python train.py
```

This will show output like:
```
Episode 21, Step 328 | Groups: [] | Advantage: -0.356 | Synergy: -0.082
```

## Understanding the Demo Output

The demo output will display information at each simulation step:

- **Agent Positions**: The (x, y) coordinates of each agent on the grid
- **Rewards**: The individual reward each agent receives (negative Euclidean distance to the grid center)
- **Causal Advantage (Global)**: A scalar value indicating the estimated global causal impact of the current state/actions
- **Emergent Synergy (Mean HJB Value)**: A scalar value approximating the overall synergy or fluidity of collaboration
- **Topological Groups Discovered**: A list of lists, where each inner list contains the numerical indices of agents identified as a cohesive group based on their proximity

### Sample Demo Output

```
--- Simulation Step 999/1000 ---
  agent_0 chose action: 1
  agent_1 chose action: 2
  agent_2 chose action: 2
  Step 999 - Agent Positions: {'agent_0': array([4, 3], dtype=int32), 'agent_1': array([4, 0], dtype=int32), 'agent_2': array([4, 0], dtype=int32)} - Rewards: {'agent_0': np.float64(-2.23606797749979), 'agent_1': np.float64(-2.8284271247461903), 'agent_2': np.float64(-2.8284271247461903)}
  Causal Advantage (Global): -0.866 (Conceptually A_do_C + sum(gamma_k * E_do(a_k)[A_syn_k]))
  Emergent Synergy (Mean HJB Value): 0.039 (Approximation of HJB PDE solution for synergy)
  Topological Groups Discovered: [[1, 2]] (H_0 connected components at eps=0.3)
--- Simulation Step 1000/1000 ---
  agent_0 chose action: 1
  agent_1 chose action: 2
  agent_2 chose action: 1
  Step 1000 - Agent Positions: {'agent_0': array([4, 3], dtype=int32), 'agent_1': array([4, 0], dtype=int32), 'agent_2': array([4, 0], dtype=int32)} - Rewards: {'agent_0': np.float64(-2.23606797749979), 'agent_1': np.float64(-2.8284271247461903), 'agent_2': np.float64(-2.8284271247461903)}
  Causal Advantage (Global): -0.828 (Conceptually A_do_C + sum(gamma_k * E_do(a_k)[A_syn_k]))
  Emergent Synergy (Mean HJB Value): 0.033 (Approximation of HJB PDE solution for synergy)
  Topological Groups Discovered: [[1, 2]] (H_0 connected components at eps=0.3)
--- Simulation Episode Finished (Done: True, Truncated: False) ---
--- Demo Concluded ---
```

> **Note**: `[]` (empty list) means no two agents are within the `topology_eps` threshold defined in `agent_demo/demo_config.yaml`. When agents are close enough, you'll see groups like `[[1, 2]]` indicating agents 1 and 2 form a topological group.

## Agents in the Demo

The demo features three agents with conceptual roles:

- **agent_0 (PlannerAgent)**: Conceptually for high-level strategy
- **agent_1 (CoderAgent)**: Conceptually for implementation logic
- **agent_2 (DebaterAgent)**: Conceptually for communication/conflict resolution

> **Important**: In this current demo, these agents are functionally identical. They all inherit from `BaseAgent` and use a stochastic policy to move randomly within the GridWorld, aiming for individual rewards based on proximity to the center. They do not perform specific "planning," "coding," or "debating" tasks. The names serve as a conceptual scaffold for future, more complex CISO implementations.

## Future Work & Contribution

This framework is a starting point for exploring CISO. Future work could include:

- Implementing actual learning algorithms (`learn` method) for agents
- Developing richer environments with more complex, collaborative tasks
- Introducing explicit communication channels and role-specific behaviors for agents
- Full implementation of the advanced mathematical formulations of CISO

**Contributions are welcome!** Feel free to fork the repository, make improvements, and submit pull requests.

## Roadmap: What We Need to Build

### Phase 1: Core Credit Assignment Engine (Priority: HIGH)

The fundamental problem: Given a shared reward, determine each agent's causal contribution.

```python
# What we need to build:
class CausalCreditAssignment:
    def attribute_reward(self, 
                         shared_reward: float,
                         agent_actions: Dict[str, Action],
                         agent_states: Dict[str, State]) -> Dict[str, float]:
        """
        Input:  shared_reward = 10.0
                agent_actions = {"agent_A": action_1, "agent_B": action_2}
        
        Output: {"agent_A": 7.0, "agent_B": 3.0}  # Causal contributions
        """
        pass
```

**Implementation Requirements:**
- [ ] **Counterfactual Baseline**: Compute "What if agent_k did nothing (null action)?"
- [ ] **Marginal Contribution**: `reward_with_agent - reward_without_agent`
- [ ] **Shapley Value Integration**: Fair distribution based on all possible agent coalitions
- [ ] **Temporal Credit**: Handle delayed rewards (action at t=0 causes reward at t=100)

### Phase 2: Intervention Mechanism

```python
class InterventionEngine:
    def do_intervention(self, agent_id: str, fixed_action: Action):
        """
        Implements do(agent_k = action) from causal inference.
        Fixes one agent's action and observes counterfactual outcome.
        """
        pass
    
    def compute_causal_effect(self, agent_id: str) -> float:
        """
        Causal Effect = E[Reward | do(agent_k = actual_action)] 
                      - E[Reward | do(agent_k = null_action)]
        """
        pass
```

### Phase 3: Real Learning Integration

Current agents don't learn. We need:

```python
class LearningAgent:
    def learn(self, 
              state: State, 
              action: Action, 
              causal_reward: float,  # NOT shared reward, but ATTRIBUTED reward
              next_state: State):
        """
        Key insight: Agent learns from its CAUSAL contribution,
        not the shared team reward.
        """
        # Policy gradient with causal reward
        loss = -log_prob(action) * causal_reward
        loss.backward()
        self.optimizer.step()
```

### Phase 4: Multi-Agent Scenarios to Test

| Scenario | Description | Credit Assignment Challenge |
|----------|-------------|----------------------------|
| **Helper vs. Hinderer** | Agent A helps, Agent B hurts | Must give A positive, B negative credit |
| **Sequential Dependency** | A's output is B's input | Must trace causal chain |
| **Redundant Actions** | A and B do the same thing | Must split credit fairly |
| **Free Rider** | A does work, B does nothing | Must give A full credit, B zero |

### Phase 5: GenAI-Specific Extensions

For LLM agents specifically:

```python
class LLMCreditAssignment:
    def attribute_to_llm_agent(self,
                                final_output_quality: float,
                                agent_contributions: List[LLMOutput]) -> Dict[str, float]:
        """
        Given: Final code quality score = 0.85
        Agents: Planner (prompt), Coder (implementation), Reviewer (fixes)
        
        Output: Who caused the quality?
        - Planner: 0.3 (good spec)
        - Coder: 0.4 (solid implementation)  
        - Reviewer: 0.15 (caught bugs)
        """
        pass
```

### Architecture Diagram: What We're Building

```
┌────────────────────────────────────────────────────────────────┐
│                    CISO-GENAI Framework                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Agent A    │  │   Agent B    │  │   Agent C    │         │
│  │  (Policy πA) │  │  (Policy πB) │  │  (Policy πC) │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              Environment (Shared State)              │      │
│  └──────────────────────┬──────────────────────────────┘      │
│                         │                                      │
│                         ▼                                      │
│                  ┌──────────────┐                              │
│                  │ Shared Reward │                              │
│                  │    R = +10    │                              │
│                  └──────┬───────┘                              │
│                         │                                      │
│         ┌───────────────┼───────────────┐                      │
│         ▼               ▼               ▼                      │
│  ┌─────────────────────────────────────────────────────┐      │
│  │         CAUSAL CREDIT ASSIGNMENT ENGINE             │      │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │      │
│  │  │Counterfactual│ │  Shapley   │ │  Temporal   │   │      │
│  │  │  Baseline   │ │   Values   │ │   Credit    │   │      │
│  │  └─────────────┘ └─────────────┘ └─────────────┘   │      │
│  └──────────────────────┬──────────────────────────────┘      │
│                         │                                      │
│         ┌───────────────┼───────────────┐                      │
│         ▼               ▼               ▼                      │
│     ┌───────┐       ┌───────┐       ┌───────┐                 │
│     │ rA=+6 │       │ rB=+3 │       │ rC=+1 │                 │
│     └───┬───┘       └───┬───┘       └───┬───┘                 │
│         │               │               │                      │
│         ▼               ▼               ▼                      │
│      Agent A         Agent B         Agent C                   │
│      learns          learns          learns                    │
│      from rA         from rB         from rC                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Why This Matters: The Research Gap

### Current State of Multi-Agent RL Credit Assignment

| Approach | How it Works | Limitation |
|----------|--------------|------------|
| **Shared Reward** | All agents get same reward | No individual accountability |
| **Individual Reward** | Each agent has own reward function | Doesn't capture team dynamics |
| **Difference Rewards** | `r_i = R_team - R_team_without_i` | Computationally expensive, needs simulator |
| **COMA** | Counterfactual baseline per agent | Still correlational, not truly causal |
| **QMIX/VDN** | Decompose Q-values | Assumes monotonic decomposition |

### What CISO Adds: True Causal Attribution

```
Traditional: "Agent A was present when reward happened" (correlation)

CISO: "If we intervene and remove Agent A's action, 
       reward drops from +10 to +3, 
       therefore Agent A CAUSED +7 of the reward" (causation)
```

This is the **do-calculus** approach from Judea Pearl's causal inference, applied to MARL.

## Contact

**Harsh Bopaliya**

- GitHub: [https://github.com/harshbopaliya]
- LinkedIn: [https://www.linkedin.com/in/harshbopaliya2003/]
- Email: [bopaliyaharsh7@gmail.com]

---

⭐ If you find this project helpful, please consider giving it a star!