import argparse
from pathlib import Path
import sys
from typing import Callable, Optional, Sequence

import gymnasium as gym

import getting_over_it_env  # noqa: F401 - registers the environment
from getting_over_it_env import ENVIRONMENT_ID

from .algorithm import RLAlgorithm, create_algorithm
from .config import EnvironmentConfig, SACConfig, TrainingConfig
from .demonstrations import select_demonstrations
from .sac_algorithm import SACAlgorithm

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
    parser.add_argument(
        "--demonstration", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--demo-episode", nargs=2, action="append", default=[],
        metavar=("FILE", "INDEX"),
    )
    parser.add_argument(
        "--demo-filter",
        choices=("successful", "non-fall", "all"),
        default="successful",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=None,
        help=(
            "Reuse network weights from a checkpoint while resetting "
            "replay, optimizers, entropy state, RNG state, and counters."
        ),
    )
    return parser


def parse_config(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    args = build_parser().parse_args(argv)
    defaults = TrainingConfig()
    episode_selections = []
    for path, index_text in args.demo_episode:
        try:
            index = int(index_text)
        except ValueError as error:
            raise ValueError(
                "--demo-episode INDEX must be an integer"
            ) from error
        episode_selections.append((Path(path), index))
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
        initialize_from=args.initialize_from,
        demonstrations=tuple(args.demonstration),
        demonstration_episodes=tuple(episode_selections),
        demonstration_filter=args.demo_filter,
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
    if config.initialize_from is not None:
        algorithm = algorithm_factory(
            config.initialize_from,
            config=config.sac,
            seed=config.seed,
            weights_only=True,
        )
    elif config.resume_from is None:
        algorithm = algorithm_factory(config=config.sac, seed=config.seed)
    else:
        algorithm = algorithm_factory(
            config.resume_from, device=config.sac.device
        )
    algorithm.ensure_training_ready()
    explicit_demonstrations = bool(
        config.demonstrations or config.demonstration_episodes
    )
    if explicit_demonstrations:
        if not isinstance(algorithm, SACAlgorithm):
            raise TypeError("Demonstration replay requires SACAlgorithm")
        required = algorithm.configured_demonstration_batch_size
        buffer = select_demonstrations(
            config.demonstrations,
            config.demonstration_episodes,
            config.demonstration_filter,
            seed=config.seed,
            minimum_transitions=max(64, required),
        )
        algorithm.set_demonstrations(buffer)
        print(
            f"Loaded {len(buffer)} immutable demonstration transitions "
            f"from {len(buffer.episodes)} episodes.",
            flush=True,
        )
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
