from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class EvaluationEpisodeMetrics:
    episode: int
    episode_return: float
    episode_length: int
    maximum_height: float
    terminated: bool
    truncated: bool
    outcome: str


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: Tuple[EvaluationEpisodeMetrics, ...]
    mean_return: float
    return_standard_deviation: float
    mean_episode_length: float
    mean_maximum_height: float
    best_maximum_height: float
    success_rate: float
    fall_rate: float
    truncation_rate: float
