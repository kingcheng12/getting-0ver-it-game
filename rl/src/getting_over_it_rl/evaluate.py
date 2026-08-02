import argparse
from pathlib import Path
from typing import Callable, Optional, Sequence

import gymnasium as gym

import getting_over_it_env  # noqa: F401 - registers the environment
from getting_over_it_env import ENVIRONMENT_ID

from .algorithm import RLAlgorithm, create_algorithm
from .config import EnvironmentConfig, EvaluationConfig

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
) -> None:
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

        for episode in range(config.episodes):
            observation, _ = environment.reset(
                seed=config.seed + episode
            )
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = algorithm.predict(
                    observation, deterministic=True
                )
                observation, _, terminated, truncated, _ = (
                    environment.step(action)
                )
    finally:
        if environment is not None:
            environment.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_evaluation(parse_config(argv))


if __name__ == "__main__":
    main()
