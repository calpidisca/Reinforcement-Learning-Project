"""Word-based superpermutation environment with cost state."""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional

from .utils import (
    get_all_permutations,
    max_overlap,
    merge_with_overlap,
    update_coverage_for_string,
)


class WordCostSuperpermEnv(gym.Env):
    """
    Word-based superpermutation environment with cost state.
    
    At each step, the agent selects one permutation to add to the sequence.
    The cost of adding a permutation is (n - overlap), where overlap is
    the maximum overlap between the current sequence and the selected permutation.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        n: int = 4,
        max_length: int | None = None,
        max_steps: int | None = None,
        alpha_new_perm_reward: float = 1.0,
        goal_bonus: float = 50.0,
        length_penalty_at_goal: float = 0.0,
    ):
        """
        Initialize the word-based superpermutation environment.
        
        Args:
            n: Alphabet size (permutations of {1, 2, ..., n})
            max_length: Maximum allowed sequence length (default: (n/2) * n!)
            max_steps: Maximum number of steps (default: 3 * n!)
            alpha_new_perm_reward: Reward per newly discovered permutation
            goal_bonus: Bonus reward when all permutations are covered
            length_penalty_at_goal: Additional penalty based on final length at goal
        """
        super().__init__()
        
        self.n = n
        self.perms = get_all_permutations(n)
        self.m = len(self.perms)  # n!
        self.perms_as_lists = [list(p) for p in self.perms]
        
        # Set defaults
        if max_length is None:
            max_length = int((n / 2) * math.factorial(n))
        if max_steps is None:
            max_steps = 3 * self.m
        
        self.max_length = max_length
        self.max_steps = max_steps
        self.alpha_new_perm_reward = alpha_new_perm_reward
        self.goal_bonus = goal_bonus
        self.length_penalty_at_goal = length_penalty_at_goal
        
        # Internal state
        self.string: list[int] = []
        self.coverage: np.ndarray = np.zeros(self.m, dtype=bool)
        self.step_count: int = 0
        self.costs: np.ndarray = np.zeros(self.m, dtype=np.float32)
        
        # Action space: choose one permutation from {0, ..., m-1}
        self.action_space = spaces.Discrete(self.m)
        
        # Observation space
        self.observation_space = spaces.Dict({
            "coverage": spaces.MultiBinary(self.m),
            "costs": spaces.Box(low=0.0, high=float(self.n), shape=(self.m,), dtype=np.float32),
            "length": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
    
    def _compute_cost_vector(self) -> np.ndarray:
        """
        Compute the cost vector for all permutations.
        
        Cost[i] = n - overlap(string, perms[i])
        where overlap is the maximum overlap length.
        
        Returns:
            Cost array of shape (m,)
        """
        costs = np.zeros(self.m, dtype=np.float32)
        for i, word in enumerate(self.perms_as_lists):
            if len(self.string) == 0:
                overlap = 0
            else:
                overlap = max_overlap(self.string, word)
            costs[i] = float(self.n - overlap)
        return costs
    
    def _get_observation(self) -> Dict[str, Any]:
        """Build the current observation."""
        # Coverage: boolean array
        coverage = self.coverage.copy().astype(np.int8)
        
        # Costs: already computed in self.costs
        costs = self.costs.copy()
        
        # Length: normalized to [0, 1]
        length_norm = np.array([min(1.0, len(self.string) / self.max_length)], dtype=np.float32)
        
        return {
            "coverage": coverage,
            "costs": costs,
            "length": length_norm,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.string = []
        self.coverage[:] = False
        self.step_count = 0
        self.costs = self._compute_cost_vector()
        
        obs = self._get_observation()
        info = {"n": self.n}
        
        return obs, info
    
    def step(self, action: int) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Args:
            action: Index of permutation to add (0 to m-1)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        i = int(action)
        word = self.perms_as_lists[i]
        
        # Compute overlap and cost
        if len(self.string) == 0:
            overlap = 0
        else:
            overlap = max_overlap(self.string, word)
        cost = self.n - overlap
        
        # Merge word into string
        self.string = merge_with_overlap(self.string, word)
        self.step_count += 1
        
        # Update coverage
        delta_new_perms = update_coverage_for_string(
            string=self.string,
            n=self.n,
            perms=self.perms,
            coverage=self.coverage,
            search_window=2 * self.n,  # Only check recent suffix
        )
        
        # Compute reward
        step_length_penalty = -float(cost)
        new_perm_reward = self.alpha_new_perm_reward * float(delta_new_perms)
        reward = step_length_penalty + new_perm_reward
        
        # Check termination conditions
        success = False
        truncated = False
        terminated = False
        
        if self.coverage.sum() == self.m:
            # All permutations covered
            success = True
            terminated = True
            final_length = len(self.string)
            reward += self.goal_bonus - self.length_penalty_at_goal * final_length
        elif len(self.string) > self.max_length or self.step_count >= self.max_steps:
            truncated = True
            terminated = True
        else:
            terminated = False
        
        # Recompute cost vector for next step
        self.costs = self._compute_cost_vector()
        
        # Build observation
        obs = self._get_observation()
        
        # Build info
        info = {
            "n": self.n,
            "step_count": self.step_count,
            "current_length": len(self.string),
            "coverage_ratio": float(self.coverage.sum()) / float(self.m),
            "success": success,
            "truncated": truncated,
        }
        
        if terminated or truncated:
            info["final_length"] = len(self.string)
            info["sequence"] = list(self.string)
            info["covered_permutations"] = int(self.coverage.sum())
        
        return obs, reward, terminated, truncated, info


