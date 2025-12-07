"""Stable-Baselines3 callbacks for superpermutation training."""

import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any, Optional

from utils.common import convert_numpy_types


class SuperpermEpisodeCallback(BaseCallback):
    """
    Callback to log episode-level metrics for superpermutation environments.
    
    Logs:
    - superperm/final_length
    - superperm/success (0/1)
    - superperm/coverage_ratio
    - superperm/episode_steps
    - superperm/new_perms_total (if available)
    - superperm/duplicate_action_ratio
    - superperm/episode_return
    - superperm/best_final_length_so_far
    """
    
    def __init__(self, verbose: int = 0, save_episodes_dir: Optional[str] = None, top_k: int = 5):
        super().__init__(verbose)
        self.best_final_length_so_far = float("inf")
        self.episode_count = 0
        self.episode_actions = []  # Track actions for duplicate ratio
        self.episode_return = 0.0
        self.episode_steps = 0
        self.save_episodes_dir = save_episodes_dir
        self.top_k = top_k  # Number of top episodes to save
        self.top_episodes = []  # List of (final_length, episode_data) tuples, sorted by length
        if self.save_episodes_dir:
            os.makedirs(self.save_episodes_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Track actions for duplicate ratio
        if self.locals.get("actions") is not None:
            action = self.locals["actions"][0] if isinstance(self.locals["actions"], (list, np.ndarray)) else self.locals["actions"]
            self.episode_actions.append(int(action))
        
        # Accumulate reward
        if "rewards" in self.locals:
            reward = self.locals["rewards"][0] if isinstance(self.locals["rewards"], (list, np.ndarray)) else self.locals["rewards"]
            self.episode_return += float(reward)
        
        self.episode_steps += 1
        
        # Check if episode is done
        dones = self.locals.get("dones", [False])
        terminated = self.locals.get("terminated", [False])
        truncated = self.locals.get("truncated", [False])
        
        # Check if any episode ended
        if isinstance(dones, (list, np.ndarray)):
            done = bool(dones[0]) if len(dones) > 0 else False
        else:
            done = bool(dones)
        
        if isinstance(terminated, (list, np.ndarray)):
            term = bool(terminated[0]) if len(terminated) > 0 else False
        else:
            term = bool(terminated)
        
        if isinstance(truncated, (list, np.ndarray)):
            trunc = bool(truncated[0]) if len(truncated) > 0 else False
        else:
            trunc = bool(truncated)
        
        if done or term or trunc:
            self._on_episode_end()
        
        return True
    
    def _on_episode_end(self) -> None:
        """Called when an episode ends."""
        # Get info from the environment
        infos = self.locals.get("infos", [{}])
        if isinstance(infos, (list, np.ndarray)) and len(infos) > 0:
            info = infos[0]
        else:
            info = {}
        
        # Extract metrics from info
        final_length = info.get("final_length", self.episode_steps)
        success = 1 if info.get("success", False) else 0
        coverage_ratio = info.get("coverage_ratio", 0.0)
        sequence = info.get("sequence", [])
        
        # Compute duplicate action ratio
        if len(self.episode_actions) > 1:
            unique_actions = len(set(self.episode_actions))
            duplicate_ratio = 1.0 - (unique_actions / len(self.episode_actions))
        else:
            duplicate_ratio = 0.0
        
        # Update best final length
        if success and final_length < self.best_final_length_so_far:
            self.best_final_length_so_far = final_length
        
        # Log metrics
        self.logger.record("superperm/final_length", float(final_length))
        self.logger.record("superperm/success", float(success))
        self.logger.record("superperm/coverage_ratio", float(coverage_ratio))
        self.logger.record("superperm/episode_steps", float(self.episode_steps))
        self.logger.record("superperm/episode_return", float(self.episode_return))
        self.logger.record("superperm/duplicate_action_ratio", float(duplicate_ratio))
        self.logger.record("superperm/best_final_length_so_far", float(self.best_final_length_so_far))
        
        # New perms total (if available in info)
        if "new_perms_total" in info:
            self.logger.record("superperm/new_perms_total", float(info["new_perms_total"]))
        
        # Save episode info if directory is provided (for successful episodes)
        # Only keep top-k shortest episodes
        if self.save_episodes_dir and success and sequence:
            episode_data = {
                "episode": self.episode_count,
                "final_length": int(final_length),
                "episode_return": float(self.episode_return),
                "sequence": list(sequence) if isinstance(sequence, (list, tuple)) else sequence,
                "success": bool(success),
                "coverage_ratio": float(coverage_ratio),
            }
            
            # Add to top episodes list
            self.top_episodes.append((final_length, episode_data))
            
            # Sort by final_length (ascending), then by episode_return (descending)
            self.top_episodes.sort(key=lambda x: (x[0], -x[1]["episode_return"]))
            
            # Keep only top-k
            if len(self.top_episodes) > self.top_k:
                self.top_episodes = self.top_episodes[:self.top_k]
            
            # Save all top-k episodes
            self._save_top_episodes()
        
        # Reset episode tracking
        self.episode_count += 1
        self.episode_actions = []
        self.episode_return = 0.0
        self.episode_steps = 0
    
    def _save_top_episodes(self) -> None:
        """Save the top-k episodes to disk."""
        if not self.save_episodes_dir:
            return
        
        # Remove old episode files
        try:
            for file in os.listdir(self.save_episodes_dir):
                if file.startswith("episode_") and file.endswith(".json"):
                    os.remove(os.path.join(self.save_episodes_dir, file))
        except (OSError, FileNotFoundError):
            pass  # Directory might not exist or be empty
        
        # Save current top-k episodes
        for rank, (final_length, episode_data) in enumerate(self.top_episodes, 1):
            episode_data["rank"] = rank
            # Convert numpy types to Python native types for JSON serialization
            episode_data_serializable = convert_numpy_types(episode_data)
            episode_path = os.path.join(self.save_episodes_dir, f"episode_{episode_data['episode']}.json")
            with open(episode_path, "w") as f:
                json.dump(episode_data_serializable, f, indent=2)

