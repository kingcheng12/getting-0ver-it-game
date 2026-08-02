from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, replace
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch

from .algorithm import RLAlgorithm
from .config import SACConfig
from .networks import create_sac_networks
from .replay_buffer import ReplayBuffer
from .sac_update import SACUpdater


class SACAlgorithm(RLAlgorithm):
    """Concrete SAC state, inference, and checkpoint boundary."""

    CHECKPOINT_VERSION = 1
    DEFAULT_OBSERVATION_DIMENSION = 20
    DEFAULT_ACTION_DIMENSION = 2

    def __init__(
        self,
        config: SACConfig = SACConfig(),
        observation_dimension: int = DEFAULT_OBSERVATION_DIMENSION,
        action_dimension: int = DEFAULT_ACTION_DIMENSION,
        seed: int = 0,
    ) -> None:
        if not isinstance(config, SACConfig):
            raise TypeError("config must be an SACConfig")
        self.observation_dimension = self._positive_dimension(
            observation_dimension, "observation_dimension"
        )
        self.action_dimension = self._positive_dimension(
            action_dimension, "action_dimension"
        )
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        self.config = config
        self.seed = int(seed)
        self.device = config.resolve_device()
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        self.networks = create_sac_networks(
            config,
            observation_dimension=self.observation_dimension,
            action_dimension=self.action_dimension,
            device=self.device,
        )
        self.updater = SACUpdater(self.networks, config)
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_capacity,
            observation_shape=(self.observation_dimension,),
            action_shape=(self.action_dimension,),
            seed=self.seed,
        )
        self.rng = np.random.default_rng(self.seed)
        self.environment_steps = 0
        self.gradient_updates = 0
        self.episodes_completed = 0

    def ensure_training_ready(self) -> None:
        raise NotImplementedError(
            "The SAC algorithm components are ready, but the environment "
            "training loop is not implemented yet. Implement roadmap "
            "stage 7 before opening Unity for training."
        )

    def learn(self, environment: gym.Env, total_steps: int) -> None:
        self.ensure_training_ready()

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        if not isinstance(observation, np.ndarray):
            raise TypeError("observation must be a NumPy array")
        if observation.dtype != np.float32:
            raise TypeError("observation must have dtype float32")
        expected_shape = (self.observation_dimension,)
        if observation.shape != expected_shape:
            raise ValueError(
                f"observation must have shape {expected_shape}, "
                f"got {observation.shape}"
            )
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation must contain only finite values")
        if not isinstance(deterministic, (bool, np.bool_)):
            raise TypeError("deterministic must be a boolean")

        self.networks.eval()
        observation_tensor = torch.as_tensor(
            observation, device=self.device
        )
        with torch.no_grad():
            if deterministic:
                action_tensor = self.networks.actor.deterministic(
                    observation_tensor
                )
            else:
                action_tensor, _ = self.networks.actor.sample(
                    observation_tensor
                )
        action = action_tensor.cpu().numpy().astype(np.float32, copy=True)
        if action.shape != (self.action_dimension,) or not np.all(
            np.isfinite(action)
        ):
            raise RuntimeError("Actor returned an invalid action")
        return np.clip(action, -1.0, 1.0).astype(
            np.float32, copy=False
        )

    def save(self, path: Path) -> None:
        checkpoint_path = Path(path)
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            raise IsADirectoryError(
                f"Checkpoint path is a directory: {checkpoint_path}"
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = self._checkpoint_state()

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{checkpoint_path.name}.",
                suffix=".tmp",
                dir=str(checkpoint_path.parent),
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            torch.save(state, temporary_path)
            os.replace(str(temporary_path), str(checkpoint_path))
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(
        cls, path: Path, device: Optional[str] = None
    ) -> "SACAlgorithm":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAC checkpoint does not exist: {checkpoint_path}"
            )
        if not checkpoint_path.is_file():
            raise IsADirectoryError(
                f"SAC checkpoint is not a file: {checkpoint_path}"
            )
        try:
            state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as error:
            raise ValueError(
                f"Unable to read SAC checkpoint: {checkpoint_path}"
            ) from error
        cls._validate_checkpoint_keys(state)
        if state["version"] != cls.CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported SAC checkpoint version: "
                f"{state['version']}"
            )

        config = cls._restore_config(state["config"], device)
        observation_dimension = cls._positive_dimension(
            state["observation_dimension"], "observation_dimension"
        )
        action_dimension = cls._positive_dimension(
            state["action_dimension"], "action_dimension"
        )
        seed = state["seed"]
        algorithm = cls(
            config=config,
            observation_dimension=observation_dimension,
            action_dimension=action_dimension,
            seed=seed,
        )
        algorithm._validate_replay_dimensions(state["replay_buffer"])
        try:
            algorithm.networks.load_state_dict(
                state["networks"], strict=True
            )
            algorithm.updater.load_state_dict(state["updater"])
            algorithm.replay_buffer.load_state_dict(
                state["replay_buffer"]
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError("SAC checkpoint state is incompatible") from error

        algorithm.environment_steps = cls._counter(
            state["environment_steps"], "environment_steps"
        )
        algorithm.gradient_updates = cls._counter(
            state["gradient_updates"], "gradient_updates"
        )
        algorithm.episodes_completed = cls._counter(
            state["episodes_completed"], "episodes_completed"
        )
        try:
            algorithm.rng.bit_generator.state = deepcopy(
                state["numpy_rng_state"]
            )
            torch_rng_state = state["torch_rng_state"]
            if not isinstance(torch_rng_state, torch.Tensor):
                raise TypeError("torch_rng_state must be a tensor")
            torch.set_rng_state(torch_rng_state.cpu())
            algorithm._restore_cuda_rng(state["cuda_rng_states"])
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError("SAC checkpoint RNG state is invalid") from error
        return algorithm

    def _checkpoint_state(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "observation_dimension": self.observation_dimension,
            "action_dimension": self.action_dimension,
            "seed": self.seed,
            "networks": self.networks.state_dict(),
            "updater": self.updater.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "environment_steps": self._counter(
                self.environment_steps, "environment_steps"
            ),
            "gradient_updates": self._counter(
                self.gradient_updates, "gradient_updates"
            ),
            "episodes_completed": self._counter(
                self.episodes_completed, "episodes_completed"
            ),
            "numpy_rng_state": deepcopy(self.rng.bit_generator.state),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
        }

    @classmethod
    def _validate_checkpoint_keys(cls, state: Any) -> None:
        if not isinstance(state, dict):
            raise ValueError("SAC checkpoint must contain a dictionary")
        expected_keys = {
            "version",
            "config",
            "observation_dimension",
            "action_dimension",
            "seed",
            "networks",
            "updater",
            "replay_buffer",
            "environment_steps",
            "gradient_updates",
            "episodes_completed",
            "numpy_rng_state",
            "torch_rng_state",
            "cuda_rng_states",
        }
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))
            extra = sorted(set(state) - expected_keys)
            raise ValueError(
                "Invalid SAC checkpoint keys; "
                f"missing={missing}, extra={extra}"
            )

    @staticmethod
    def _restore_config(
        state: Any, device: Optional[str]
    ) -> SACConfig:
        if not isinstance(state, dict):
            raise ValueError("Checkpoint config must be a dictionary")
        expected_fields = {field.name for field in fields(SACConfig)}
        if set(state) != expected_fields:
            raise ValueError("Checkpoint SACConfig fields are incompatible")
        try:
            config = SACConfig(**state)
            if device is not None:
                config = replace(config, device=device)
            config.resolve_device()
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("Checkpoint SACConfig is invalid") from error
        return config

    def _validate_replay_dimensions(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise ValueError("Checkpoint replay state must be a dictionary")
        if tuple(state.get("observation_shape", ())) != (
            self.observation_dimension,
        ):
            raise ValueError(
                "Checkpoint replay observation dimension is incompatible"
            )
        if tuple(state.get("action_shape", ())) != (
            self.action_dimension,
        ):
            raise ValueError(
                "Checkpoint replay action dimension is incompatible"
            )

    def _restore_cuda_rng(self, states: Any) -> None:
        if not isinstance(states, list):
            raise TypeError("cuda_rng_states must be a list")
        if self.device.type != "cuda":
            return
        if not states:
            return
        device_index = (
            torch.cuda.current_device()
            if self.device.index is None
            else self.device.index
        )
        state = states[
            device_index if device_index < len(states) else 0
        ]
        if not isinstance(state, torch.Tensor):
            raise TypeError("CUDA RNG state must be a tensor")
        torch.cuda.set_rng_state(state.cpu(), self.device)

    @staticmethod
    def _positive_dimension(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return int(value)

    @staticmethod
    def _counter(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"Checkpoint {name} must be an integer")
        if value < 0:
            raise ValueError(f"Checkpoint {name} must be non-negative")
        return int(value)
