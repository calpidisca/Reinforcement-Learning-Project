"""Run full experiment sweep over all combinations."""

import os
import itertools
from train_one import train_one_run, make_env
from algorithms.random_policy import run_random
from algorithms.greedy_policy import run_greedy
from metrics import aggregate_metrics
from config import (
    N_LIST,
    ENV_TYPES,
    ALGOS,
    SEEDS,
    HYPER_GRIDS,
    TIMESTEPS,
    BASELINE_EPISODES,
    should_skip,
)
from utils import generate_run_name


def generate_hyperparam_combinations(hyper_grid: dict) -> list[dict]:
    """Generate all combinations of hyperparameters."""
    if not hyper_grid:
        return [{}]
    
    keys = list(hyper_grid.keys())
    values = list(hyper_grid.values())
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def run_sweep(log_root: str = "logs") -> None:
    """Run the full experiment sweep."""
    total_runs = 0
    
    for n in N_LIST:
        for env_type in ENV_TYPES:
            for algo_name in ALGOS:
                # Check if should skip this combination
                if should_skip(algo_name, env_type, n):
                    print(f"Skipping {algo_name} for {env_type} n={n} (skip condition)")
                    continue
                
                if algo_name in ["random", "greedy"]:
                    # Non-learning baselines
                    for seed in SEEDS:
                        run_name = generate_run_name(env_type, n, algo_name, seed, None)
                        run_dir = os.path.join(log_root, run_name)
                        os.makedirs(run_dir, exist_ok=True)
                        
                        # Create environment using common function
                        env = make_env(env_type, n, use_mask=False)
                        
                        # Run baseline
                        if algo_name == "random":
                            run_random(
                                env=env,
                                total_episodes=BASELINE_EPISODES["random"],
                                max_steps_per_episode=env.max_steps,
                                log_dir=run_dir,
                                seed=seed,
                            )
                        elif algo_name == "greedy":
                            run_greedy(
                                env=env,
                                total_episodes=BASELINE_EPISODES["greedy"],
                                max_steps_per_episode=env.max_steps,
                                log_dir=run_dir,
                                seed=seed,
                            )
                        
                        total_runs += 1
                        print(f"Completed: {run_name} ({total_runs} total runs)")
                
                else:
                    # RL algorithms
                    # Get hyperparameter combinations
                    hyper_grid = HYPER_GRIDS.get(algo_name, {})
                    hyper_combos = generate_hyperparam_combinations(hyper_grid)
                    
                    # Get timesteps for this algorithm
                    total_timesteps = TIMESTEPS.get(algo_name)
                    if total_timesteps is None:
                        print(f"Warning: No timesteps defined for {algo_name}, skipping")
                        continue
                    
                    for hyperparams in hyper_combos:
                        for seed in SEEDS:
                            try:
                                run_dir = train_one_run(
                                    env_type=env_type,
                                    n=n,
                                    algo_name=algo_name,
                                    hyperparams=hyperparams,
                                    seed=seed,
                                    total_timesteps=total_timesteps,
                                    log_root=log_root,
                                )
                                total_runs += 1
                                print(f"Completed: {env_type}_n{n}_{algo_name}_seed{seed} ({total_runs} total runs)")
                            except Exception as e:
                                print(f"Error in {env_type}_n{n}_{algo_name}_seed{seed}: {e}")
                                continue
    
    print(f"\nSweep complete! Total runs: {total_runs}")
    
    # Aggregate metrics after all runs are complete
    print("\nAggregating metrics...")
    try:
        aggregate_metrics(log_root=log_root)
        print("Metrics aggregation complete!")
    except Exception as e:
        print(f"Error aggregating metrics: {e}")


if __name__ == "__main__":
    run_sweep()


