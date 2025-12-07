"""Common utility functions for superpermutation RL research."""

import os
import json
import numpy as np
import torch
from typing import Dict, Any, Optional
import gymnasium as gym

from .types import EnvType, AlgoName


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, (np.integer, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8, np.uint16,
                        np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def generate_run_name(
    env_type: EnvType,
    n: int,
    algo_name: AlgoName,
    seed: int,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a run name from parameters.
    
    Args:
        env_type: Environment type
        n: Permutation size
        algo_name: Algorithm name
        seed: Random seed
        hyperparams: Optional hyperparameters dict
        
    Returns:
        Run name string
    """
    run_name = f"{env_type}_n{n}_{algo_name}_seed{seed}"
    if hyperparams:
        # Add hyperparam signature to run name
        hp_sig = "_".join(f"{k}{v}" for k, v in sorted(hyperparams.items()))
        run_name = f"{run_name}_{hp_sig}"
    return run_name


def create_metadata(
    env_type: EnvType,
    n: int,
    algo_name: AlgoName,
    hyperparams: Dict[str, Any],
    seed: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Create metadata dictionary for a run.
    
    Args:
        env_type: Environment type
        n: Permutation size
        algo_name: Algorithm name
        hyperparams: Hyperparameters dict
        seed: Random seed
        **kwargs: Additional metadata fields
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        "env_type": env_type,
        "n": n,
        "algo_name": algo_name,
        "hyperparams": hyperparams,
        "seed": seed,
    }
    metadata.update(kwargs)
    return metadata


def save_metadata(run_dir: str, metadata: Dict[str, Any]) -> None:
    """
    Save metadata to a JSON file.
    
    Args:
        run_dir: Run directory path
        metadata: Metadata dictionary
    """
    # Convert numpy types to Python native types for JSON serialization
    metadata_serializable = convert_numpy_types(metadata)
    
    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_serializable, f, indent=2)


def detect_env_type(env: gym.Env) -> EnvType:
    """
    Detect environment type from an environment instance.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        Environment type ("symbol" or "word_cost")
    """
    if hasattr(env, "costs"):
        return "word_cost"
    elif hasattr(env, "n") and hasattr(env, "perms"):
        return "symbol"
    else:
        # Fallback: check class name
        class_name = env.__class__.__name__
        if "Word" in class_name or "word" in class_name.lower():
            return "word_cost"
        else:
            return "symbol"


def set_all_seeds(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

