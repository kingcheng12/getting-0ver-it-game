from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from .config import SACConfig


class RLAlgorithm(ABC):
    """Framework-neutral contract for a future continuous-control agent."""

    @abstractmethod
    def learn(
        self,
        environment: gym.Env,
        total_steps: int,
        checkpoint_path: Optional[Path] = None,
    ) -> object:
        """Update the policy using transitions from the environment."""

    def ensure_training_ready(self) -> None:
        """Fail before environment creation if learning is unavailable."""

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
    def load(
        cls, path: Path, device: Optional[str] = None
    ) -> "RLAlgorithm":
        """Restore an algorithm from a checkpoint."""


def create_algorithm(
    checkpoint_path: Optional[Path] = None,
    *,
    config: Optional[SACConfig] = None,
    seed: int = 0,
    device: Optional[str] = None,
) -> RLAlgorithm:
    """Construct a new SAC agent or restore one from a checkpoint."""
    from .sac_algorithm import SACAlgorithm

    if config is not None and not isinstance(config, SACConfig):
        raise TypeError("config must be an SACConfig")
    if checkpoint_path is not None:
        if config is not None:
            raise ValueError(
                "config cannot be supplied when loading a checkpoint"
            )
        return SACAlgorithm.load(checkpoint_path, device=device)
    if device is not None:
        if config is not None:
            raise ValueError(
                "device must be set through config for a new algorithm"
            )
        config = SACConfig(device=device)
    return SACAlgorithm(config=config or SACConfig(), seed=seed)
