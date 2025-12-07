"""Random policy baseline."""

import os
import csv
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from utils import detect_env_type, create_metadata, save_metadata


def run_random(
    env: gym.Env,
    total_episodes: int,
    max_steps_per_episode: int,
    log_dir: str,
    seed: int,
) -> None:
    """
    Run random policy baseline.
    
    Args:
        env: Gymnasium environment
        total_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        log_dir: Directory to save logs
        seed: Random seed
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Set seed
    np.random.seed(seed)
    
    # CSV file for episode metrics
    csv_path = os.path.join(log_dir, "progress.csv")
    
    # Field names matching SB3 format
    fieldnames = [
        "episode",
        "final_length",
        "success",
        "coverage_ratio",
        "episode_steps",
        "episode_return",
        "covered_permutations",
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for episode in tqdm(range(total_episodes), desc="Random policy episodes"):
            # Reset with seed offset
            obs, info = env.reset(seed=seed + episode)
            
            episode_return = 0.0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < max_steps_per_episode:
                # Sample random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_return += reward
                episode_steps += 1
                done = terminated or truncated
            
            # Write episode metrics
            writer.writerow({
                "episode": episode,
                "final_length": info.get("final_length", len(env.string) if hasattr(env, "string") else 0),
                "success": 1 if info.get("success", False) else 0,
                "coverage_ratio": info.get("coverage_ratio", 0.0),
                "episode_steps": episode_steps,
                "episode_return": episode_return,
                "covered_permutations": info.get("covered_permutations", 0),
            })
            f.flush()
    
    # Save metadata
    env_type = detect_env_type(env)
    n = env.n if hasattr(env, "n") else 0
    metadata = create_metadata(
        env_type=env_type,
        n=n,
        algo_name="random",
        hyperparams={},
        seed=seed,
        total_episodes=total_episodes,
    )
    save_metadata(log_dir, metadata)


