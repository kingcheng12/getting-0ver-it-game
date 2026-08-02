from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch


Shape = Tuple[int, ...]
Device = Union[str, torch.device]


@dataclass(frozen=True)
class ReplayBatch:
    """A sampled batch of SAC transitions."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    bootstrap_mask: torch.Tensor
    episode_end: torch.Tensor


class ReplayBuffer:
    """Fixed-size, uniformly sampled replay memory for SAC."""

    STATE_VERSION = 1

    def __init__(
        self,
        capacity: int,
        observation_shape: Sequence[int] = (20,),
        action_shape: Sequence[int] = (2,),
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(
            capacity, (int, np.integer)
        ):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self._capacity = int(capacity)
        self._observation_shape = self._validate_shape(
            observation_shape, "observation_shape"
        )
        self._action_shape = self._validate_shape(
            action_shape, "action_shape"
        )
        self._observations = np.empty(
            (self._capacity,) + self._observation_shape,
            dtype=np.float32,
        )
        self._actions = np.empty(
            (self._capacity,) + self._action_shape,
            dtype=np.float32,
        )
        self._rewards = np.empty((self._capacity, 1), dtype=np.float32)
        self._next_observations = np.empty_like(self._observations)
        self._terminated = np.empty((self._capacity, 1), dtype=np.bool_)
        self._truncated = np.empty((self._capacity, 1), dtype=np.bool_)
        self._write_position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def observation_shape(self) -> Shape:
        return self._observation_shape

    @property
    def action_shape(self) -> Shape:
        return self._action_shape

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Copy one transition into the next ring-buffer slot."""
        observation_array = self._validate_array(
            observation, self._observation_shape, "observation"
        )
        action_array = self._validate_array(
            action, self._action_shape, "action"
        )
        action_array = np.clip(action_array, -1.0, 1.0)
        next_observation_array = self._validate_array(
            next_observation,
            self._observation_shape,
            "next_observation",
        )
        reward_value = self._validate_reward(reward)
        terminated_value = self._validate_flag(terminated, "terminated")
        truncated_value = self._validate_flag(truncated, "truncated")
        if terminated_value and truncated_value:
            raise ValueError(
                "terminated and truncated cannot both be true"
            )

        index = self._write_position
        self._observations[index] = observation_array
        self._actions[index] = action_array
        self._rewards[index, 0] = reward_value
        self._next_observations[index] = next_observation_array
        self._terminated[index, 0] = terminated_value
        self._truncated[index, 0] = truncated_value

        self._write_position = (index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, device: Device = "cpu") -> ReplayBatch:
        """Uniformly sample distinct stored transitions."""
        if isinstance(batch_size, bool) or not isinstance(
            batch_size, (int, np.integer)
        ):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from "
                f"a buffer containing {self._size}."
            )

        target_device = torch.device(device)
        indices = self._rng.choice(
            self._size, size=int(batch_size), replace=False
        )
        observations = self._tensor(
            self._observations[indices], target_device
        )
        actions = self._tensor(self._actions[indices], target_device)
        rewards = self._tensor(self._rewards[indices], target_device)
        next_observations = self._tensor(
            self._next_observations[indices], target_device
        )
        terminated = self._tensor(
            self._terminated[indices], target_device
        )
        truncated = self._tensor(
            self._truncated[indices], target_device
        )
        bootstrap_mask = (~terminated).to(dtype=torch.float32)
        episode_end = terminated | truncated

        return ReplayBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminated=terminated,
            truncated=truncated,
            bootstrap_mask=bootstrap_mask,
            episode_end=episode_end,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return a self-contained checkpoint of data and sampling state."""
        return {
            "version": self.STATE_VERSION,
            "capacity": self._capacity,
            "observation_shape": self._observation_shape,
            "action_shape": self._action_shape,
            "size": self._size,
            "write_position": self._write_position,
            "observations": self._observations[: self._size].copy(),
            "actions": self._actions[: self._size].copy(),
            "rewards": self._rewards[: self._size].copy(),
            "next_observations": self._next_observations[
                : self._size
            ].copy(),
            "terminated": self._terminated[: self._size].copy(),
            "truncated": self._truncated[: self._size].copy(),
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a compatible replay checkpoint after strict validation."""
        if not isinstance(state, dict):
            raise TypeError("Replay buffer state must be a dictionary")

        expected_keys = {
            "version",
            "capacity",
            "observation_shape",
            "action_shape",
            "size",
            "write_position",
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminated",
            "truncated",
            "rng_state",
        }
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))
            extra = sorted(set(state) - expected_keys)
            raise ValueError(
                f"Invalid replay state keys; missing={missing}, extra={extra}"
            )
        if state["version"] != self.STATE_VERSION:
            raise ValueError(
                f"Unsupported replay state version: {state['version']}"
            )
        if state["capacity"] != self._capacity:
            raise ValueError("Replay state capacity is incompatible")
        if tuple(state["observation_shape"]) != self._observation_shape:
            raise ValueError("Replay state observation_shape is incompatible")
        if tuple(state["action_shape"]) != self._action_shape:
            raise ValueError("Replay state action_shape is incompatible")

        size = self._validate_index(state["size"], "size", self._capacity)
        write_position = self._validate_index(
            state["write_position"],
            "write_position",
            self._capacity - 1,
        )
        if size < self._capacity and write_position != size:
            raise ValueError(
                "Replay state write_position must equal size before rollover"
            )

        arrays = (
            ("observations", self._observations),
            ("actions", self._actions),
            ("rewards", self._rewards),
            ("next_observations", self._next_observations),
            ("terminated", self._terminated),
            ("truncated", self._truncated),
        )
        validated_arrays = {}
        for name, destination in arrays:
            value = state[name]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Replay state {name} must be a NumPy array")
            expected_shape = (size,) + destination.shape[1:]
            if (
                value.shape != expected_shape
                or value.dtype != destination.dtype
            ):
                raise ValueError(
                    f"Replay state {name} must have shape "
                    f"{expected_shape} and dtype {destination.dtype}"
                )
            if np.issubdtype(value.dtype, np.floating) and not np.all(
                np.isfinite(value)
            ):
                raise ValueError(f"Replay state {name} contains non-finite data")
            validated_arrays[name] = value

        candidate_rng = np.random.default_rng()
        try:
            candidate_rng.bit_generator.state = deepcopy(state["rng_state"])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("Replay state has an invalid rng_state") from error

        for name, destination in arrays:
            np.copyto(destination[:size], validated_arrays[name])
        self._size = size
        self._write_position = write_position
        self._rng = candidate_rng

    @staticmethod
    def _validate_shape(shape: Sequence[int], name: str) -> Shape:
        try:
            result = tuple(shape)
        except TypeError as error:
            raise TypeError(f"{name} must be a sequence of integers") from error
        if not result:
            raise ValueError(f"{name} cannot be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in result
        ):
            raise ValueError(f"{name} must contain positive integers")
        return tuple(int(value) for value in result)

    @staticmethod
    def _validate_array(
        value: np.ndarray, expected_shape: Shape, name: str
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    @staticmethod
    def _validate_reward(value: float) -> float:
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise TypeError("reward must be a numeric scalar")
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("reward must be a numeric scalar") from error
        if not np.isfinite(result):
            raise ValueError("reward must be finite")
        return result

    @staticmethod
    def _validate_flag(value: bool, name: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
        return bool(value)

    @staticmethod
    def _validate_index(value: Any, name: str, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"Replay state {name} must be an integer")
        result = int(value)
        if not 0 <= result <= maximum:
            raise ValueError(
                f"Replay state {name} must be between 0 and {maximum}"
            )
        return result

    @staticmethod
    def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(array, device=device)
