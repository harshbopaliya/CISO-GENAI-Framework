"""
CISO-GENAI: Causal Intelligence for Multi-Agent Systems

A framework for solving the multi-agent credit assignment problem:
Given a shared reward, determine each agent's causal contribution.

Main Components:
- CausalCreditAssignment: Do-intervention based credit attribution
- ShapleyValueEstimator: Fair credit distribution using Shapley values
- SynergyDetector: Detect complementary/interfering agent interactions
- AgentClusterer: Group agents by interaction patterns
"""

from .causal_engine import (
    CausalCreditAssignment,
    ShapleyValueEstimator,
    CounterfactualBaseline,
)
from .synergy_engine import SynergyDetector, InterferenceDetector
from .topology_engine import AgentClusterer, InteractionTracker
from .policies import PolicyRegistry, random_policy, stay_policy
from .envs.gridworld import GridWorld, rollout

__version__ = "0.1.0"

__all__ = [
    # Credit Assignment
    "CausalCreditAssignment",
    "ShapleyValueEstimator", 
    "CounterfactualBaseline",
    # Synergy
    "SynergyDetector",
    "InterferenceDetector",
    # Clustering
    "AgentClusterer",
    "InteractionTracker",
    # Policies
    "PolicyRegistry",
    "random_policy",
    "stay_policy",
    # Environment
    "GridWorld",
    "rollout",
]

