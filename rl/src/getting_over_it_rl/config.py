from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Optional, Tuple

import torch


def checkpoint_root() -> Path:
    """Return the project-local directory reserved for model artifacts."""
    return Path(__file__).resolve().parents[2] / "checkpoints"


def resolve_torch_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Check the NVIDIA "
            "driver and install the project's CUDA-enabled PyTorch build, "
            "or use device='cpu'."
        )
    return device


@dataclass(frozen=True)
class EnvironmentConfig:
    file_name: Optional[str] = None
    worker_id: int = 0
    timeout_wait: int = 300
    max_episode_steps: int = 1250
    render_mode: str = "human"

    def __post_init__(self) -> None:
        if self.worker_id < 0:
            raise ValueError("worker_id must be non-negative")
        if self.timeout_wait <= 0:
            raise ValueError("timeout_wait must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.render_mode != "human":
            raise ValueError("render_mode must be 'human'")


@dataclass(frozen=True)
class SACConfig:
    hidden_sizes: Tuple[int, ...] = (256, 256)
    replay_capacity: int = 200_000
    batch_size: int = 256
    warmup_steps: int = 5_000
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_learning_rate: float = 3e-4
    automatic_entropy_tuning: bool = True
    initial_entropy_coefficient: float = 0.2
    target_entropy: Optional[float] = None
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    checkpoint_interval: int = 10_000
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(
            size <= 0 for size in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive values")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.batch_size > self.replay_capacity:
            raise ValueError(
                "batch_size cannot exceed replay_capacity"
            )
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.updates_per_step <= 0:
            raise ValueError("updates_per_step must be positive")
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not math.isfinite(self.tau) or not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        for name, value in (
            ("actor_learning_rate", self.actor_learning_rate),
            ("critic_learning_rate", self.critic_learning_rate),
            ("entropy_learning_rate", self.entropy_learning_rate),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            not math.isfinite(self.log_std_min)
            or not math.isfinite(self.log_std_max)
            or self.log_std_min >= self.log_std_max
        ):
            raise ValueError("log_std_min must be less than log_std_max")
        if self.target_entropy is not None and not math.isfinite(
            self.target_entropy
        ):
            raise ValueError("target_entropy must be finite")
        if (
            not math.isfinite(self.initial_entropy_coefficient)
            or self.initial_entropy_coefficient <= 0.0
        ):
            raise ValueError(
                "initial_entropy_coefficient must be positive"
            )
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        try:
            torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                f"Invalid PyTorch device: {self.device}"
            ) from error

    def resolve_device(self) -> torch.device:
        return resolve_torch_device(self.device)


@dataclass(frozen=True)
class TrainingConfig:
    environment: EnvironmentConfig = field(
        default_factory=EnvironmentConfig
    )
    seed: int = 0
    total_steps: int = 100_000
    sac: SACConfig = field(default_factory=SACConfig)
    checkpoint_path: Path = field(
        default_factory=lambda: checkpoint_root() / "latest"
    )

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    environment: EnvironmentConfig = field(
        default_factory=EnvironmentConfig
    )
    seed: int = 0
    episodes: int = 10
    device: str = "cpu"
    checkpoint_path: Path = field(
        default_factory=lambda: checkpoint_root() / "latest"
    )

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        try:
            torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                f"Invalid PyTorch device: {self.device}"
            ) from error

    def resolve_device(self) -> torch.device:
        return resolve_torch_device(self.device)
