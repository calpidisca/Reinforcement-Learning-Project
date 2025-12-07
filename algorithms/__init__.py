"""Non-learning baseline policies."""

from .random_policy import run_random
from .greedy_policy import run_greedy

__all__ = ["run_random", "run_greedy"]


