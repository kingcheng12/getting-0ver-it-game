import argparse
from pathlib import Path
from typing import Callable, Optional, Sequence

import gymnasium as gym

import getting_over_it_env  # noqa: F401 - registers the environment
from getting_over_it_env import ENVIRONMENT_ID

from .algorithm import RLAlgorithm, create_algorithm
from .config import EnvironmentConfig, TrainingConfig

AlgorithmFactory = Callable[[], RLAlgorithm]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an algorithm on the Unity environment."
    )
    parser.add_argument("--file-name")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--timeout-wait", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
    )
    return parser


def parse_config(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    args = build_parser().parse_args(argv)
    defaults = TrainingConfig()
    return TrainingConfig(
        environment=EnvironmentConfig(
            file_name=args.file_name,
            worker_id=args.worker_id,
            timeout_wait=args.timeout_wait,
            max_episode_steps=args.max_episode_steps,
        ),
        seed=args.seed,
        total_steps=args.total_steps,
        checkpoint_path=(
            args.checkpoint_path
            if args.checkpoint_path is not None
            else defaults.checkpoint_path
        ),
    )


def run_training(
    config: TrainingConfig,
    algorithm_factory: AlgorithmFactory = create_algorithm,
) -> None:
    # This deliberately fails before opening Unity until a factory exists.
    algorithm = algorithm_factory()

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
        environment.reset(seed=config.seed)
        algorithm.learn(environment, config.total_steps)
        config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        algorithm.save(config.checkpoint_path)
    finally:
        if environment is not None:
            environment.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_training(parse_config(argv))


if __name__ == "__main__":
    main()
