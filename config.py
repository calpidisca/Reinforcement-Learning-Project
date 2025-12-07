"""Experiment configuration for superpermutation RL research."""

from typing import Dict, List, Tuple

# Permutation sizes to test
N_LIST: List[int] = [3, 4, 5, 6]

# Environment types
ENV_TYPES: List[str] = ["symbol", "word_cost"]

# Algorithms to test
ALGOS: List[str] = ["ppo", "a2c",  "random", "greedy"] #"maskable_ppo", "dqn", "dqn_dueling",

# Random seeds for experiments
SEEDS: List[int] = [0, 1, 2, 42, 100, 123, 999]

# Hyperparameter grids (for learning algorithms only)
HYPER_GRIDS: Dict[str, Dict[str, List[float]]] = {
    "ppo": {
        "learning_rate": [3e-4, 1e-4],
        "gamma": [0.99, 0.995],
    },
    "maskable_ppo": {
        "learning_rate": [3e-4],
        "gamma": [0.99],
    },
    "a2c": {
        "learning_rate": [7e-4],
        "gamma": [0.99, 0.995],
    },
    "dqn": {
        "learning_rate": [1e-4],
        "gamma": [0.99],
    },
    "dqn_dueling": {
        "learning_rate": [1e-4],
        "gamma": [0.99],
    },
}

# Total timesteps per algorithm
TIMESTEPS: Dict[str, int] = {
    "ppo": 300_000,
    "maskable_ppo": 300_000,
    "a2c": 300_000,
    "dqn": 300_000,
    "dqn_dueling": 300_000,
}

# Episodes for non-learning baselines
BASELINE_EPISODES: Dict[str, int] = {
    "random": 100,
    "greedy": 100,
}

# Conditions for skipping certain algorithm/environment combinations
# Format: (algo_name, env_type, n) -> skip
SKIP_CONDITIONS: List[Tuple[str, str, int]] = [
    # Skip DQN variants for word_cost and n=6 (too heavy)
    ("dqn", "word_cost", 6),
    ("dqn_dueling", "word_cost", 6),
]


def should_skip(algo_name: str, env_type: str, n: int) -> bool:
    """
    Check if a combination should be skipped.
    
    Args:
        algo_name: Algorithm name
        env_type: Environment type
        n: Permutation size
        
    Returns:
        True if should skip, False otherwise
    """
    return (algo_name, env_type, n) in SKIP_CONDITIONS

