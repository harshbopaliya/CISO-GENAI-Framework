"""
ciso_sanity_test.py

Minimal demo: 2-agent GridWorld + causal attribution via do-intervention and
Monte-Carlo Shapley sampling.

Run: python ciso_sanity_test.py
"""

import numpy as np
import random
from itertools import permutations, combinations
from typing import Callable, Dict, List, Tuple

# ---------- simple gridworld ----------
GRID_SIZE = 5
CENTER = np.array([GRID_SIZE // 2, GRID_SIZE // 2])

def manhattan(a, b):
    return int(abs(a[0]-b[0]) + abs(a[1]-b[1]))

class GridWorld:
    def __init__(self, grid_size=GRID_SIZE, max_steps=5):
        self.grid_size = grid_size
        self.max_steps = max_steps

    def reset(self, starts: Dict[str, np.ndarray]):
        self.positions = {k: v.copy() for k, v in starts.items()}
        self.t = 0
        return dict(self.positions)

    def step(self, actions: Dict[str, int]):
        # actions: 0=up,1=right,2=down,3=left,4=stay
        for aid, a in actions.items():
            if a == 0: self.positions[aid][1] = max(0, self.positions[aid][1]-1)
            if a == 1: self.positions[aid][0] = min(self.grid_size-1, self.positions[aid][0]+1)
            if a == 2: self.positions[aid][1] = min(self.grid_size-1, self.positions[aid][1]+1)
            if a == 3: self.positions[aid][0] = max(0, self.positions[aid][0]-1)
            # 4 stay -> no change
        self.t += 1
        done = (self.t >= self.max_steps)
        # team/shared reward: negative sum of Manhattan distances to center (higher is better when closer)
        total_dist = sum(manhattan(pos, CENTER) for pos in self.positions.values())
        reward = -float(total_dist)
        return dict(self.positions), reward, done

# ---------- policies ----------
def planner_policy(state_pos: np.ndarray) -> int:
    # Simple greedy move towards center (deterministic)
    dx = CENTER[0] - state_pos[0]
    dy = CENTER[1] - state_pos[1]
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 3 if dx < 0 else 4
    else:
        return 2 if dy > 0 else 0 if dy < 0 else 4

def random_policy(_: np.ndarray) -> int:
    return random.choice([0,1,2,3,4])

# wrapper to call policy by name
POLICIES = {
    "planner": planner_policy,
    "random": random_policy
}

# ---------- rollout helper ----------
def rollout(env: GridWorld, starts: Dict[str, np.ndarray], policy_map: Dict[str, Callable[[np.ndarray], int]]) -> Tuple[float, List[Dict[str, np.ndarray]]]:
    state = env.reset(starts)
    traj = [dict(state)]
    cumulative_reward = 0.0
    done = False
    while not done:
        actions = {}
        for aid, pos in state.items():
            actions[aid] = policy_map[aid](np.array(pos))
        state, r, done = env.step(actions)
        traj.append(dict(state))
        cumulative_reward += r
    return cumulative_reward, traj

# ---------- causal attribution via do-intervention ----------
def marginal_contribution_by_intervention(env: GridWorld, starts: Dict[str, np.ndarray],
                                           base_policy_map: Dict[str, Callable],
                                           target_agent: str,
                                           baseline_policy: Callable) -> float:
    """
    Compute marginal contribution of target_agent using intervention:
      marginal = R_full - R_without_target
    where R_without_target is computed by replacing target_agent with baseline_policy.
    """
    # Full reward
    R_full, _ = rollout(env, starts, base_policy_map)
    # Create counterfactual policy map
    cf_map = dict(base_policy_map)
    cf_map[target_agent] = baseline_policy
    R_without, _ = rollout(env, starts, cf_map)
    marginal = R_full - R_without
    return marginal

# ---------- Monte Carlo Shapley (approx) ----------
def mc_shapley(env: GridWorld, starts: Dict[str, np.ndarray],
               policy_names: Dict[str, str], baseline_policy: Callable,
               num_samples: int = 100) -> Dict[str, float]:
    """
    Approximate Shapley values by sampling permutations.
    policy_names: map agent_id -> policy_name (e.g., "planner" or "random")
    """
    agents = list(policy_names.keys())
    n = len(agents)
    shapley = {a: 0.0 for a in agents}
    for _ in range(num_samples):
        perm = agents.copy()
        random.shuffle(perm)
        coalition = []
        R_prev = None
        # We'll build coalition incrementally; members in coalition use their real policies,
        # non-members use baseline.
        for i, agent in enumerate(perm):
            coalition.append(agent)
            # Construct policy map: coalition agents use real policies, others baseline
            pm = {}
            for a in agents:
                if a in coalition:
                    pm[a] = POLICIES[policy_names[a]]
                else:
                    pm[a] = baseline_policy
            R_coal, _ = rollout(env, starts, pm)
            if R_prev is None:
                marginal = R_coal  # coalition previously empty -> previous R was with all baseline: compute baseline R0
            else:
                marginal = R_coal - R_prev
            R_prev = R_coal
            shapley[agent] += marginal
    # average over samples
    for a in shapley:
        shapley[a] /= num_samples
    return shapley

# ---------- demo/test ----------
def demo_seeded():
    random.seed(0)
    np.random.seed(0)

    env = GridWorld(grid_size=GRID_SIZE, max_steps=5)
    # start positions
    starts = {
        "agent_A": np.array([0, 0]),  # top-left
        "agent_B": np.array([4, 4])   # bottom-right
    }
    # define base policy map: Agent A = planner, Agent B = planner (both good)
    base_policy_map = {
        "agent_A": POLICIES["planner"],
        "agent_B": POLICIES["planner"]
    }

    print("=== Full rollout (both planners) ===")
    R_full, traj = rollout(env, starts, base_policy_map)
    print("Cumulative team reward:", R_full)
    print("Trajectory (positions per step):")
    for step, s in enumerate(traj):
        print(f"  step {step}: {s}")

    print("\n=== Marginal contributions by simple intervention (replace target with RANDOM) ===")
    marg_A = marginal_contribution_by_intervention(env, starts, base_policy_map, "agent_A", POLICIES["random"])
    marg_B = marginal_contribution_by_intervention(env, starts, base_policy_map, "agent_B", POLICIES["random"])
    print(f"marginal(agent_A) = {marg_A:.3f}")
    print(f"marginal(agent_B) = {marg_B:.3f}")
    print(f"Sum of marginals = {marg_A + marg_B:.3f} (should be approx R_full - R_baseline_all)")

    print("\n=== Monte-Carlo Shapley approx (agents -> policies: planner/planner) ===")
    policy_names = {"agent_A": "planner", "agent_B": "planner"}
    shap = mc_shapley(env, starts, policy_names, POLICIES["random"], num_samples=50)
    for a, v in shap.items():
        print(f"Shapley({a}) ≈ {v:.3f}")
    print("Note: Shapley values approximate fair marginal contributions across permutations.")

if __name__ == "__main__":
    demo_seeded()
