import argparse

import gymnasium as gym
import numpy as np

import getting_over_it_env  # noqa: F401 - registers the environment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send random Gymnasium actions to Unity MainScene."
    )
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    print(
        "Start this script first, then press Play in Unity when prompted.",
        flush=True,
    )
    env = gym.make(
        "GettingOverItUnity-v0",
        timeout_wait=args.timeout,
    )
    try:
        observation, info = env.reset(seed=0)
        print(
            f"Connected. Reset observation ({observation.size} values):\n"
            f"{np.array2string(observation, precision=3)}\n"
            f"info={info}",
            flush=True,
        )

        for index in range(args.steps):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print(
                    f"Episode {reason} at step {index + 1}; "
                    f"reward={reward:.4f}. Resetting.",
                    flush=True,
                )
                observation, info = env.reset()

        print(
            f"Completed {args.steps} random actions; final info={info}",
            flush=True,
        )
    finally:
        env.close()
        print("Unity communicator closed.", flush=True)


if __name__ == "__main__":
    main()
