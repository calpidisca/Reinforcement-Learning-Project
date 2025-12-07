"""Greedy policy baseline."""

import os
import csv
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from utils import detect_env_type, create_metadata, save_metadata


def run_greedy(
    env: gym.Env,
    total_episodes: int,
    max_steps_per_episode: int,
    log_dir: str,
    seed: int,
) -> None:
    """
    Run greedy policy baseline.
    
    For SymbolEnv: selects action that maximizes expected new permutations.
    For WordEnv: selects action with minimum cost among uncovered permutations.
    
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
    
    # Detect environment type
    detected_env_type = detect_env_type(env)
    # Map to internal representation for greedy action selection
    env_type = "word" if detected_env_type == "word_cost" else "symbol"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for episode in tqdm(range(total_episodes), desc="Greedy policy episodes"):
            # Reset with seed offset
            obs, info = env.reset(seed=seed + episode)
            
            episode_return = 0.0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < max_steps_per_episode:
                if env_type == "symbol":
                    action = _greedy_action_symbol(env, obs)
                elif env_type == "word":
                    action = _greedy_action_word(env, obs)
                else:
                    # Fallback to random
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
    n = env.n if hasattr(env, "n") else 0
    metadata = create_metadata(
        env_type=detected_env_type,
        n=n,
        algo_name="greedy",
        hyperparams={},
        seed=seed,
        total_episodes=total_episodes,
    )
    save_metadata(log_dir, metadata)


def _greedy_action_symbol(env: gym.Env, obs: dict) -> int:
    """
    Greedy action selection for symbol environment.
    
    For each possible action, estimate the number of new permutations
    that would be discovered, and select the action with the highest score.
    """
    best_action = 0
    best_score = float("-inf")
    
    # Create a temporary copy of coverage for simulation
    temp_coverage = env.coverage.copy()
    temp_string = env.string.copy()
    
    for action in range(env.action_space.n):
        # Simulate adding this symbol
        symbol = action + 1
        test_string = temp_string + [symbol]
        
        # Estimate new permutations (simplified: check if this creates new length-n substrings)
        # We'll use a heuristic: count how many new permutations might be formed
        # by checking the last n symbols
        new_perms_estimate = 0
        if len(test_string) >= env.n:
            # Check the last n symbols
            last_n = tuple(test_string[-env.n:])
            if last_n in env.perms:
                idx = env.perms.index(last_n)
                if not temp_coverage[idx]:
                    new_perms_estimate = 1
        
        # Score: new_perms - small length penalty
        score = new_perms_estimate - 0.01  # Small penalty for length
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action


def _greedy_action_word(env: gym.Env, obs: dict) -> int:
    """
    Greedy action selection for word environment.
    
    Select the action (permutation) with minimum cost among uncovered ones.
    If all are covered, select the one with minimum cost.
    """
    coverage = obs["coverage"]
    costs = obs["costs"]
    
    # Prefer uncovered permutations
    uncovered_mask = coverage == 0
    uncovered_costs = costs.copy()
    uncovered_costs[~uncovered_mask] = float("inf")  # Mask covered ones
    
    if np.any(uncovered_mask):
        # Select uncovered permutation with minimum cost
        best_action = int(np.argmin(uncovered_costs))
    else:
        # All covered, select minimum cost overall
        best_action = int(np.argmin(costs))
    
    return best_action


