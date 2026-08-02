import argparse
from pathlib import Path
import sys
from typing import Callable, Optional, Sequence

import gymnasium as gym

import getting_over_it_env  # noqa: F401 - registers the environment
from getting_over_it_env import ENVIRONMENT_ID

from .algorithm import RLAlgorithm, create_algorithm
from .config import EnvironmentConfig, SACConfig, TrainingConfig

AlgorithmFactory = Callable[..., RLAlgorithm]


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
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--resume-from", type=Path, default=None)
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
        sac=SACConfig(device=args.device),
        checkpoint_path=(
            args.checkpoint_path
            if args.checkpoint_path is not None
            else defaults.checkpoint_path
        ),
        resume_from=args.resume_from,
    )


def run_training(
    config: TrainingConfig,
    algorithm_factory: AlgorithmFactory = create_algorithm,
) -> None:
    print(
        f"Initializing SAC on {config.sac.device}. Keep Unity out of "
        "Play mode until Python prints the waiting-for-Unity message.",
        flush=True,
    )
    if config.resume_from is None:
        algorithm = algorithm_factory(config=config.sac, seed=config.seed)
    else:
        algorithm = algorithm_factory(
            config.resume_from, device=config.sac.device
        )
    algorithm.ensure_training_ready()
    print("SAC initialized; opening the Unity connection.", flush=True)

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
        algorithm.learn(
            environment,
            config.total_steps,
            checkpoint_path=config.checkpoint_path,
        )
    except BaseException:
        try:
            algorithm.save(config.checkpoint_path)
        except Exception as checkpoint_error:
            print(
                "Training failed and the emergency checkpoint could not "
                f"be saved: {checkpoint_error}",
                file=sys.stderr,
                flush=True,
            )
        raise
    else:
        algorithm.save(config.checkpoint_path)
    finally:
        if environment is not None:
            environment.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_training(parse_config(argv))


if __name__ == "__main__":
    main()
