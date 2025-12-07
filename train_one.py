"""Train a single RL run."""

import os
import time
import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.logger import configure
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gymnasium import spaces

from envs.symbol_env import SymbolSuperpermEnv
from envs.word_env_cost import WordCostSuperpermEnv
from callbacks import SuperpermEpisodeCallback
from utils import (
    generate_run_name,
    create_metadata,
    save_metadata,
    set_all_seeds,
    EnvType,
    AlgoName,
)


ALGO_MAP = {
    "ppo": PPO,
    "maskable_ppo": MaskablePPO,
    "a2c": A2C,
    "dqn": DQN,  # basic: dueling=False
    "dqn_dueling": DQN,  # with dueling=True
}


def word_env_mask_fn(obs) -> np.ndarray:
    """
    Action mask function for WordCostSuperpermEnv.
    
    Masks out actions where:
    - The permutation is already covered AND
    - The cost equals the maximum cost (n, meaning zero overlap)
    
    Args:
        obs: Observation dict from WordCostSuperpermEnv
        
    Returns:
        Boolean mask of shape (m,) where False means invalid action
    """
    coverage = obs["coverage"]
    costs = obs["costs"]
    
    # Maximum cost should be n (alphabet size), which occurs when overlap is 0
    # We can infer n from the maximum cost value
    max_cost = np.max(costs)
    
    # Mask: allow all actions by default
    mask = np.ones(len(coverage), dtype=bool)
    
    # Mask out covered permutations with maximum cost (zero overlap, already covered)
    # This prevents adding a permutation that's already covered with no benefit
    mask = ~((coverage == 1) & (np.abs(costs - max_cost) < 0.01))
    
    return mask


def make_env(env_type: EnvType, n: int, use_mask: bool = False):
    """
    Create an environment.
    
    Args:
        env_type: "symbol" or "word_cost"
        n: Alphabet size
        use_mask: Whether to apply action masking (for MaskablePPO on WordEnv)
        
    Returns:
        Gymnasium environment
    """
    if env_type == "symbol":
        env = SymbolSuperpermEnv(n=n)
        return env
    elif env_type == "word_cost":
        env = WordCostSuperpermEnv(n=n)
        if use_mask:
            env = ActionMasker(env, word_env_mask_fn)
        return env
    else:
        raise ValueError(f"Unknown env_type: {env_type}")


def train_one_run(
    env_type: EnvType,
    n: int,
    algo_name: AlgoName,
    hyperparams: dict,
    seed: int,
    total_timesteps: int,
    log_root: str = "logs",
) -> str:
    """
    Train a single RL run.
    
    Args:
        env_type: "symbol" or "word_cost"
        n: Alphabet size
        algo_name: Algorithm name ("ppo", "maskable_ppo", "a2c", "dqn", "dqn_dueling")
        hyperparams: Hyperparameters dict
        seed: Random seed
        total_timesteps: Total training timesteps
        log_root: Root directory for logs
    """
    # Create run directory
    run_name = generate_run_name(env_type, n, algo_name, seed, hyperparams)
    run_dir = os.path.join(log_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create environment
    use_mask = (algo_name == "maskable_ppo" and env_type == "word_cost")
    env = make_env(env_type, n, use_mask=use_mask)
    
    # Set seeds
    env.reset(seed=seed)
    set_all_seeds(seed)
    
    # Get algorithm class
    AlgoClass = ALGO_MAP.get(algo_name)
    if AlgoClass is None:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Determine device (GPU if available, else CPU)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
        print("Note: To use GPU, ensure PyTorch with CUDA support is installed")
    
    # Check observation space type (handle wrapped environments)
    obs_space = env.observation_space
    # If wrapped, get the actual observation space
    while hasattr(obs_space, "observation_space"):
        obs_space = obs_space.observation_space
    
    # Determine policy type
    # Check if observation space is a Dict (not a Python dict, but gymnasium.spaces.Dict)
    if isinstance(obs_space, spaces.Dict):
        policy = "MultiInputPolicy"
    else:
        policy = "MlpPolicy"
    
    # Configure policy kwargs
    policy_kwargs = {}
    if algo_name == "dqn":
        policy_kwargs = {"policy_kwargs": {"dueling": False}}
    elif algo_name == "dqn_dueling":
        policy_kwargs = {"policy_kwargs": {"dueling": True}}
    
    # Configure SB3 logger
    logger = configure(run_dir, ["csv", "tensorboard"])
    
    # Create model
    model = AlgoClass(
        policy,
        env,
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
        **hyperparams,
        **policy_kwargs,
    )
    model.set_logger(logger)
    
    # Create callback (save episodes for successful runs)
    episodes_dir = os.path.join(run_dir, "episodes")
    callback = SuperpermEpisodeCallback(verbose=1, save_episodes_dir=episodes_dir)
    
    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
    )
    wall_time = time.time() - start_time
    
    # Save model
    model.save(os.path.join(run_dir, "model"))
    
    # Save metadata
    metadata = create_metadata(
        env_type=env_type,
        n=n,
        algo_name=algo_name,
        hyperparams=hyperparams,
        seed=seed,
        total_timesteps=total_timesteps,
        wall_time=wall_time,
    )
    save_metadata(run_dir, metadata)
    
    return run_dir

