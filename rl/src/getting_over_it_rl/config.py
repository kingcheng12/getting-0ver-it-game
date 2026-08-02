from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def checkpoint_root() -> Path:
    """Return the project-local directory reserved for model artifacts."""
    return Path(__file__).resolve().parents[2] / "checkpoints"


@dataclass(frozen=True)
class EnvironmentConfig:
    file_name: Optional[str] = None
    worker_id: int = 0
    timeout_wait: int = 300
    max_episode_steps: int = 1250
    render_mode: str = "human"


@dataclass(frozen=True)
class TrainingConfig:
    environment: EnvironmentConfig = field(
        default_factory=EnvironmentConfig
    )
    seed: int = 0
    total_steps: int = 100_000
    checkpoint_path: Path = field(
        default_factory=lambda: checkpoint_root() / "latest"
    )


@dataclass(frozen=True)
class EvaluationConfig:
    environment: EnvironmentConfig = field(
        default_factory=EnvironmentConfig
    )
    seed: int = 0
    episodes: int = 10
    checkpoint_path: Path = field(
        default_factory=lambda: checkpoint_root() / "latest"
    )
