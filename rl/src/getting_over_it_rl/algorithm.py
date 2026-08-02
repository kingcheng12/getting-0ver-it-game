from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np


class RLAlgorithm(ABC):
    """Framework-neutral contract for a future continuous-control agent."""

    @abstractmethod
    def learn(self, environment: gym.Env, total_steps: int) -> None:
        """Update the policy using transitions from the environment."""

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Return one action for one observation."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save all state needed to continue training or evaluate."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "RLAlgorithm":
        """Restore an algorithm from a checkpoint."""


def create_algorithm(
    checkpoint_path: Optional[Path] = None,
) -> RLAlgorithm:
    """Construct or load the algorithm selected for this project."""
    checkpoint_hint = (
        f" The requested checkpoint was: {checkpoint_path}."
        if checkpoint_path is not None
        else ""
    )
    raise NotImplementedError(
        "No reinforcement-learning algorithm is configured. Implement "
        "create_algorithm() in getting_over_it_rl/algorithm.py and choose "
        "a continuous-control algorithm such as PPO or SAC."
        + checkpoint_hint
    )
