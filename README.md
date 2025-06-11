# 🚀 CISO-GENAI-Framework: Causal Intelligence for Multi-Agent Generative AI

## Table of Contents

- [About CISO-GENAI-Framework](#about-ciso-genai-framework)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Demo](#running-the-demo)
- [Understanding the Demo Output](#understanding-the-demo-output)
- [Agents in the Demo](#agents-in-the-demo)
- [Future Work & Contribution](#future-work--contribution)
- [License](#license)
- [Contact](#contact)

## About CISO-GENAI-Framework

The CISO-GENAI-Framework implements core concepts from **Causal Invariant Synergy Optimization (CISO)**, a novel approach designed to bring causal reasoning, geometric cooperation, and topological group discovery to multi-agent reinforcement learning (MARL). As Generative AI systems scale to many interacting LLMs and agents, coordinating their efforts becomes a significant challenge. CISO aims to cut through the noise, identify true causal contributions, foster intelligent collaboration, and dynamically discover effective agent teams.

This repository provides a foundational demo illustrating how the CISO components analyze agent interactions within a simple environment.

## Key Features

- **Causal Advantage Interventions**: Understands what truly made the difference in multi-agent outcomes, moving beyond mere correlation to pinpoint causal credit.
- **Emergent Synergy Manifolds**: Models the optimal collaboration surface among agents, aiming to unlock fluid and intelligent teamwork.
- **Topological Group Formation**: Dynamically discovers natural, high-performing agent groups based on their interactions and proximity in state space.

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Harsh Bopaliya**

- GitHub: [https://github.com/harshbopaliya]
- LinkedIn: [https://www.linkedin.com/in/harshbopaliya2003/]
- Email: [bopaliyaharsh7@gmail.com]

---

⭐ If you find this project helpful, please consider giving it a star!