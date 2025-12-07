"""Type definitions for superpermutation RL research."""

from typing import Literal

# Environment type
EnvType = Literal["symbol", "word_cost"]

# Algorithm name
AlgoName = Literal["ppo", "maskable_ppo", "a2c", "dqn", "dqn_dueling", "random", "greedy"]

