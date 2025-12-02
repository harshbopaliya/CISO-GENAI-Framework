"""
CISO Credit Assignment Training Demo

Demonstrates causal credit assignment in a multi-agent GridWorld:
1. Multiple agents act with different policies
2. Environment returns a SHARED reward
3. Framework attributes the reward to individual agents causally

This answers: "Which agent actually CAUSED the reward?"
"""

import yaml
import random
import numpy as np
from typing import Dict, Callable

from src.envs.gridworld import GridWorld, rollout
from src.causal_engine import CausalCreditAssignment, ShapleyValueEstimator, CounterfactualBaseline
from src.synergy_engine import SynergyDetector
from src.topology_engine import AgentClusterer
from src.policies import PolicyRegistry, random_policy


def main():
    """Run the credit assignment demo."""
    
    print("=" * 60)
    print("🚀 CISO-GENAI: Causal Credit Assignment Demo")
    print("=" * 60)
    
    # Load configuration
    with open("configs/ciso_default.yaml") as f:
        config = yaml.safe_load(f)
    
    # Setup
    grid_size = config["env"]["grid_size"]
    max_steps = config["env"]["max_steps"]
    num_samples = config["credit_assignment"]["shapley_samples"]
    
    # Initialize environment
    env = GridWorld(grid_size=grid_size, max_steps=max_steps)
    
    # Initialize policy registry
    policies = PolicyRegistry(grid_size=grid_size)
    
    # Define agents and their policies
    # Agent A: Smart (moves to center)
    # Agent B: Smart (also moves to center)
    agent_policies = {
        "agent_A": policies.get("center"),  # Smart policy
        "agent_B": policies.get("center"),  # Smart policy
    }
    
    # Starting positions
    starts = {
        "agent_A": np.array([0, 0]),    # Top-left corner
        "agent_B": np.array([4, 4]),    # Bottom-right corner
    }
    
    # Baseline policy for counterfactuals
    baseline = random_policy
    
    print("\n📍 Setup:")
    print(f"   Grid size: {grid_size}x{grid_size}")
    print(f"   Max steps: {max_steps}")
    print(f"   Agent A starts at: {starts['agent_A']} (policy: center-seeking)")
    print(f"   Agent B starts at: {starts['agent_B']} (policy: center-seeking)")
    print(f"   Baseline for counterfactuals: random")
    
    # ============== 1. Run full episode ==============
    print("\n" + "=" * 60)
    print("📊 STEP 1: Full Episode Rollout (Both Agents Active)")
    print("=" * 60)
    
    random.seed(42)
    np.random.seed(42)
    
    R_full, trajectory = rollout(env, starts, agent_policies)
    
    print(f"\n   Cumulative Team Reward: {R_full:.2f}")
    print("\n   Trajectory:")
    for step, state in enumerate(trajectory):
        print(f"      Step {step}: A={state['agent_A']} B={state['agent_B']}")
    
    # ============== 2. Marginal Contributions ==============
    print("\n" + "=" * 60)
    print("🔬 STEP 2: Marginal Contributions (Do-Intervention)")
    print("=" * 60)
    print("\n   Question: What if each agent acted randomly instead?")
    
    credit_engine = CausalCreditAssignment(baseline_policy=baseline)
    
    marginals = credit_engine.compute_all_marginals(
        env, starts, agent_policies, rollout
    )
    
    print("\n   Results:")
    for agent, marginal in marginals.items():
        print(f"      {agent}: {marginal:+.2f}")
        if marginal > 0:
            print(f"         → {agent} HELPED (reward dropped by {marginal:.2f} without them)")
        elif marginal < 0:
            print(f"         → {agent} HURT (reward increased by {-marginal:.2f} without them)")
        else:
            print(f"         → {agent} had NO EFFECT")
    
    print(f"\n   Sum of marginals: {sum(marginals.values()):.2f}")
    
    # ============== 3. Shapley Values ==============
    print("\n" + "=" * 60)
    print("⚖️  STEP 3: Shapley Values (Fair Credit Assignment)")
    print("=" * 60)
    print(f"\n   Monte-Carlo estimation with {num_samples} samples...")
    
    shapley_estimator = ShapleyValueEstimator(
        baseline_policy=baseline, 
        num_samples=num_samples
    )
    
    shapley_values = shapley_estimator.estimate(
        env, starts, agent_policies, rollout
    )
    
    print("\n   Shapley Values (fair distribution):")
    for agent, value in shapley_values.items():
        print(f"      {agent}: {value:+.2f}")
    
    print(f"\n   Sum of Shapley values: {sum(shapley_values.values()):.2f}")
    print(f"   (Should approximately equal full reward: {R_full:.2f})")
    
    # ============== 4. Synergy Detection ==============
    print("\n" + "=" * 60)
    print("🤝 STEP 4: Synergy Detection")
    print("=" * 60)
    print("\n   Question: Do agents perform better TOGETHER than alone?")
    
    synergy_detector = SynergyDetector(baseline_policy=baseline)
    
    coalition = list(agent_policies.keys())
    synergy = synergy_detector.compute_synergy(
        env, starts, agent_policies, rollout, coalition
    )
    
    print(f"\n   Synergy(A, B) = {synergy:+.2f}")
    if synergy > 0:
        print("   → Agents COMPLEMENT each other (better together)")
    elif synergy < 0:
        print("   → Agents INTERFERE with each other (worse together)")
    else:
        print("   → Agents are INDEPENDENT (no synergy)")
    
    # ============== 5. Different Policy Scenario ==============
    print("\n" + "=" * 60)
    print("🔄 STEP 5: Asymmetric Policies (A=smart, B=random)")
    print("=" * 60)
    
    asymmetric_policies = {
        "agent_A": policies.get("center"),  # Smart
        "agent_B": random_policy,            # Random (bad)
    }
    
    R_asymmetric, _ = rollout(env, starts, asymmetric_policies)
    print(f"\n   Team reward with A=smart, B=random: {R_asymmetric:.2f}")
    
    # Credit assignment for asymmetric case
    asymmetric_marginals = credit_engine.compute_all_marginals(
        env, starts, asymmetric_policies, rollout
    )
    
    print("\n   Marginal contributions:")
    for agent, marginal in asymmetric_marginals.items():
        print(f"      {agent}: {marginal:+.2f}")
    
    print("\n   Note: When B is already random, replacing B with random")
    print("   should give ~0 marginal (B contributes nothing extra)")
    
    # ============== Summary ==============
    print("\n" + "=" * 60)
    print("📋 SUMMARY: Causal Credit Assignment")
    print("=" * 60)
    print("""
   The CISO framework answers the fundamental question:
   
   "Given a shared reward, which agent CAUSED it?"
   
   Methods demonstrated:
   1. Marginal Contribution: R_with_agent - R_without_agent
   2. Shapley Values: Fair attribution considering all orderings
   3. Synergy Detection: Do agents help or hurt each other?
   
   Key insight: We use INTERVENTIONS (do-calculus), not just
   correlations, to determine true causal credit.
    """)
    
    print("=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

