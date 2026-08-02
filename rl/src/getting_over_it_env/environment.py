from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment


class GettingOverItEnv(gym.Env):
    """A single-agent Gymnasium adapter for the Unity Editor environment."""

    metadata = {"render_modes": ["human"], "render_fps": 12}

    behavior_name = "GettingOverIt"
    observation_size = 20
    action_size = 2

    def __init__(
        self,
        file_name: Optional[str] = None,
        worker_id: int = 0,
        timeout_wait: int = 300,
        max_episode_steps: int = 1250,
        render_mode: str = "human",
        _unity_env: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if render_mode != "human":
            raise ValueError("render_mode must be 'human'")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        self.render_mode = render_mode
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_size,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_size,),
            dtype=np.float32,
        )

        self._max_episode_steps = int(max_episode_steps)
        self._episode_steps = 0
        self._agent_id: Optional[int] = None
        self._selected_behavior: Optional[str] = None
        self._closed = False

        if _unity_env is not None:
            self._unity = _unity_env
        else:
            if file_name is None:
                print(
                    "Waiting for Unity Editor on localhost:5004. "
                    "Open MainScene and press Play within "
                    f"{timeout_wait} seconds.",
                    flush=True,
                )
            self._unity = UnityEnvironment(
                file_name=file_name,
                worker_id=worker_id,
                timeout_wait=timeout_wait,
                seed=0,
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._require_open()
        self._unity.reset()
        self._selected_behavior = self._select_behavior()
        self._validate_behavior_spec(self._selected_behavior)

        decision_steps, terminal_steps = self._unity.get_steps(
            self._selected_behavior
        )
        if len(terminal_steps) != 0 or len(decision_steps) != 1:
            raise RuntimeError(
                "Expected exactly one decision agent after reset; "
                f"got {len(decision_steps)} decision and "
                f"{len(terminal_steps)} terminal agents."
            )

        self._agent_id = next(iter(decision_steps))
        self._episode_steps = 0
        step = decision_steps[self._agent_id]
        observation = self._flatten_observations(step.obs)
        return observation, self._info(observation)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._require_open()
        if self._agent_id is None or self._selected_behavior is None:
            raise RuntimeError("Call reset() before step().")

        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, "
                f"got {action_array.shape}."
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("Action values must be finite.")

        clipped = np.clip(action_array, -1.0, 1.0).astype(
            np.float32, copy=False
        )
        action_tuple = ActionTuple(continuous=clipped.reshape(1, -1))
        self._unity.set_action_for_agent(
            self._selected_behavior, self._agent_id, action_tuple
        )
        self._unity.step()
        self._episode_steps += 1

        decision_steps, terminal_steps = self._unity.get_steps(
            self._selected_behavior
        )
        terminated = False
        truncated = False

        if self._agent_id in terminal_steps:
            result = terminal_steps[self._agent_id]
            observation = self._flatten_observations(result.obs)
            reward = float(result.reward)
            truncated = bool(result.interrupted)
            terminated = not truncated
        elif self._agent_id in decision_steps:
            result = decision_steps[self._agent_id]
            observation = self._flatten_observations(result.obs)
            reward = float(result.reward)
        else:
            raise RuntimeError(
                f"Agent {self._agent_id} was missing after Unity stepped."
            )

        if (
            not terminated
            and not truncated
            and self._episode_steps >= self._max_episode_steps
        ):
            truncated = True

        return (
            observation,
            reward,
            terminated,
            truncated,
            self._info(observation),
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._agent_id = None
        self._selected_behavior = None
        if self._unity is not None:
            self._unity.close()

    def _select_behavior(self) -> str:
        names = [
            name
            for name in self._unity.behavior_specs
            if name.split("?")[0] == self.behavior_name
        ]
        if len(names) != 1:
            available = ", ".join(self._unity.behavior_specs) or "<none>"
            raise RuntimeError(
                f"Expected one '{self.behavior_name}' behavior, found "
                f"{len(names)}. Available: {available}."
            )
        return names[0]

    def _validate_behavior_spec(self, behavior_name: str) -> None:
        spec = self._unity.behavior_specs[behavior_name]
        observation_count = sum(
            int(np.prod(observation.shape))
            for observation in spec.observation_specs
        )
        if observation_count != self.observation_size:
            raise RuntimeError(
                f"Behavior exposes {observation_count} observation values; "
                f"expected {self.observation_size}."
            )
        if (
            spec.action_spec.continuous_size != self.action_size
            or spec.action_spec.discrete_size != 0
        ):
            raise RuntimeError(
                "Behavior must expose exactly two continuous actions and "
                "no discrete actions."
            )

    def _flatten_observations(self, observations: Any) -> np.ndarray:
        flattened = np.concatenate(
            [np.asarray(value, dtype=np.float32).reshape(-1)
             for value in observations]
        ).astype(np.float32, copy=False)
        if flattened.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Unity returned observation shape {flattened.shape}; "
                f"expected {self.observation_space.shape}."
            )
        if not np.all(np.isfinite(flattened)):
            raise RuntimeError("Unity returned non-finite observations.")
        return np.clip(flattened, -1.0, 1.0).astype(
            np.float32, copy=False
        )

    def _info(self, observation: np.ndarray) -> Dict[str, Any]:
        return {
            "agent_id": self._agent_id,
            "episode_steps": self._episode_steps,
            "body_relative_y": float(observation[1]),
            "maximum_height": float(observation[9]),
        }

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The environment is closed.")
