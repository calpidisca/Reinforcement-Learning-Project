"""Symbol-based superpermutation environment."""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional

from .utils import get_all_permutations, update_coverage_for_string


class SymbolSuperpermEnv(gym.Env):
    """
    Symbol-based superpermutation environment.
    
    At each step, the agent adds one symbol (1 to n) to the sequence.
    The goal is to construct a sequence that contains all n! permutations
    as contiguous substrings while minimizing the total length.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        n: int = 4,
        max_length: int | None = None,
        max_steps: int | None = None,
        lambda_length_penalty: float = 0.1,
        alpha_new_perm_reward: float = 1.0,
        goal_bonus: float = 50.0,
        length_penalty_at_goal: float = 0.0,
    ):
        """
        Initialize the symbol-based superpermutation environment.
        
        Args:
            n: Alphabet size (permutations of {1, 2, ..., n})
            max_length: Maximum allowed sequence length (default: (n/2) * n!)
            max_steps: Maximum number of steps (default: max_length)
            lambda_length_penalty: Penalty per step
            alpha_new_perm_reward: Reward per newly discovered permutation
            goal_bonus: Bonus reward when all permutations are covered
            length_penalty_at_goal: Additional penalty based on final length at goal
        """
        super().__init__()
        
        self.n = n
        self.alphabet = list(range(1, n + 1))
        self.perms = get_all_permutations(n)
        self.m = len(self.perms)  # n!
        
        # Set defaults
        if max_length is None:
            max_length = int((n / 2) * math.factorial(n))
        if max_steps is None:
            max_steps = max_length
        
        self.max_length = max_length
        self.max_steps = max_steps
        self.lambda_length_penalty = lambda_length_penalty
        self.alpha_new_perm_reward = alpha_new_perm_reward
        self.goal_bonus = goal_bonus
        self.length_penalty_at_goal = length_penalty_at_goal
        
        # Internal state
        self.string: list[int] = []
        self.coverage: np.ndarray = np.zeros(self.m, dtype=bool)
        self.step_count: int = 0
        
        # Action space: choose one symbol from {1, ..., n}
        self.action_space = spaces.Discrete(n)
        
        # Observation space
        self.observation_space = spaces.Dict({
            "suffix": spaces.MultiDiscrete([n + 1] * (n - 1)),  # 0-padded, values in {0,...,n}
            "coverage": spaces.MultiBinary(self.m),
            "length": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
    
    def _get_observation(self) -> Dict[str, Any]:
        """Build the current observation."""
        # Suffix: last (n-1) symbols, 0-padded
        suffix = np.zeros(self.n - 1, dtype=np.int32)
        if len(self.string) > 0:
            suffix_start = max(0, len(self.string) - (self.n - 1))
            suffix_len = len(self.string) - suffix_start
            suffix[-suffix_len:] = self.string[suffix_start:]
        
        # Coverage: boolean array
        coverage = self.coverage.copy().astype(np.int8)
        
        # Length: normalized to [0, 1]
        length_norm = np.array([min(1.0, len(self.string) / self.max_length)], dtype=np.float32)
        
        return {
            "suffix": suffix,
            "coverage": coverage,
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
        
        obs = self._get_observation()
        info = {"n": self.n}
        
        return obs, info
    
    def step(self, action: int) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Map action to symbol: action ∈ {0,...,n-1} -> symbol ∈ {1,...,n}
        symbol = action + 1
        
        # Append symbol to string
        self.string.append(symbol)
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
        new_perms_reward = self.alpha_new_perm_reward * delta_new_perms
        step_length_penalty = -self.lambda_length_penalty
        reward = new_perms_reward + step_length_penalty
        
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


