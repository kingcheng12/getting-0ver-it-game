import argparse
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import gymnasium as gym
import numpy as np

import getting_over_it_env  # noqa: F401 - registers the environment
from getting_over_it_env import ENVIRONMENT_ID

from .algorithm import RLAlgorithm, create_algorithm
from .config import EnvironmentConfig, EvaluationConfig
from .evaluation_metrics import (
    EvaluationEpisodeMetrics,
    EvaluationSummary,
)

AlgorithmLoader = Callable[..., RLAlgorithm]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved algorithm in the Unity environment."
    )
    parser.add_argument("--file-name")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--timeout-wait", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
    )
    return parser


def parse_config(argv: Optional[Sequence[str]] = None) -> EvaluationConfig:
    args = build_parser().parse_args(argv)
    defaults = EvaluationConfig()
    return EvaluationConfig(
        environment=EnvironmentConfig(
            file_name=args.file_name,
            worker_id=args.worker_id,
            timeout_wait=args.timeout_wait,
            max_episode_steps=args.max_episode_steps,
        ),
        seed=args.seed,
        episodes=args.episodes,
        device=args.device,
        checkpoint_path=(
            args.checkpoint_path
            if args.checkpoint_path is not None
            else defaults.checkpoint_path
        ),
    )


def run_evaluation(
    config: EvaluationConfig,
    algorithm_loader: AlgorithmLoader = create_algorithm,
) -> EvaluationSummary:
    algorithm = algorithm_loader(
        config.checkpoint_path, device=config.device
    )

    environment: Optional[gym.Env] = None
    try:
        environment = gym.make(
            ENVIRONMENT_ID,
            file_name=config.environment.file_name,
            worker_id=config.environment.worker_id,
            timeout_wait=config.environment.timeout_wait,
            max_episode_steps=config.environment.max_episode_steps,
            render_mode=config.environment.render_mode,
        )

        episode_metrics: List[EvaluationEpisodeMetrics] = []
        for episode_index in range(config.episodes):
            observation, info = environment.reset(
                seed=config.seed + episode_index
            )
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            maximum_height = _maximum_height(info, 0.0)
            final_reward = 0.0
            while not (terminated or truncated):
                action = algorithm.predict(
                    observation, deterministic=True
                )
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = (
                    environment.step(action)
                )
                final_reward = _finite_reward(reward)
                terminated = _boolean_flag(terminated, "terminated")
                truncated = _boolean_flag(truncated, "truncated")
                if terminated and truncated:
                    raise ValueError(
                        "environment returned both terminated and truncated"
                    )
                episode_return += final_reward
                episode_length += 1
                maximum_height = max(
                    maximum_height,
                    _maximum_height(info, maximum_height),
                )

            metrics = EvaluationEpisodeMetrics(
                episode=episode_index + 1,
                episode_return=episode_return,
                episode_length=episode_length,
                maximum_height=maximum_height,
                terminated=terminated,
                truncated=truncated,
                outcome=_episode_outcome(
                    terminated, truncated, final_reward
                ),
            )
            episode_metrics.append(metrics)
            _print_episode(metrics)

        summary = _summarize(episode_metrics)
        _print_summary(summary)
        return summary
    finally:
        if environment is not None:
            environment.close()


def _finite_reward(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError("environment reward must be a scalar")
    reward = float(value)
    if not np.isfinite(reward):
        raise ValueError("environment returned non-finite reward")
    return reward


def _boolean_flag(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"environment {name} must be a boolean")
    return bool(value)


def _maximum_height(info: Any, fallback: float) -> float:
    if not isinstance(info, dict):
        return fallback
    value = info.get("maximum_height", fallback)
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        return fallback
    height = float(value)
    return height if np.isfinite(height) else fallback


def _episode_outcome(
    terminated: bool, truncated: bool, final_reward: float
) -> str:
    if truncated:
        return "truncated"
    if terminated:
        return "success" if np.isclose(final_reward, 10.0) else "fall"
    return "incomplete"


def _summarize(
    episodes: List[EvaluationEpisodeMetrics],
) -> EvaluationSummary:
    if not episodes:
        raise ValueError("evaluation requires at least one episode")
    returns = np.asarray(
        [episode.episode_return for episode in episodes],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [episode.episode_length for episode in episodes],
        dtype=np.float64,
    )
    heights = np.asarray(
        [episode.maximum_height for episode in episodes],
        dtype=np.float64,
    )
    count = float(len(episodes))
    return EvaluationSummary(
        episodes=tuple(episodes),
        mean_return=float(returns.mean()),
        return_standard_deviation=float(returns.std()),
        mean_episode_length=float(lengths.mean()),
        mean_maximum_height=float(heights.mean()),
        best_maximum_height=float(heights.max()),
        success_rate=sum(
            episode.outcome == "success" for episode in episodes
        )
        / count,
        fall_rate=sum(
            episode.outcome == "fall" for episode in episodes
        )
        / count,
        truncation_rate=sum(
            episode.outcome == "truncated" for episode in episodes
        )
        / count,
    )


def _print_episode(metrics: EvaluationEpisodeMetrics) -> None:
    print(
        f"evaluation_episode={metrics.episode} "
        f"return={metrics.episode_return:.4f} "
        f"length={metrics.episode_length} "
        f"max_height={metrics.maximum_height:.4f} "
        f"outcome={metrics.outcome}",
        flush=True,
    )


def _print_summary(summary: EvaluationSummary) -> None:
    print(
        f"evaluation_summary episodes={len(summary.episodes)} "
        f"mean_return={summary.mean_return:.4f} "
        f"return_std={summary.return_standard_deviation:.4f} "
        f"mean_length={summary.mean_episode_length:.2f} "
        f"mean_max_height={summary.mean_maximum_height:.4f} "
        f"best_max_height={summary.best_maximum_height:.4f} "
        f"success_rate={summary.success_rate:.3f} "
        f"fall_rate={summary.fall_rate:.3f} "
        f"truncation_rate={summary.truncation_rate:.3f}",
        flush=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_config(argv)
    try:
        run_evaluation(config)
    except FileNotFoundError as error:
        raise SystemExit(
            f"No SAC checkpoint is available at "
            f"'{config.checkpoint_path}'. Train the agent first with "
            f"'goi-train --checkpoint-path "
            f"{config.checkpoint_path}', or pass a valid "
            "--checkpoint-path."
        ) from None


if __name__ == "__main__":
    main()
