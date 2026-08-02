from __future__ import annotations

from copy import deepcopy
import math
from typing import Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from .config import SACConfig, resolve_torch_device


Device = Union[str, torch.device]


def _validate_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validate_hidden_sizes(hidden_sizes: Sequence[int]) -> Tuple[int, ...]:
    try:
        sizes = tuple(hidden_sizes)
    except TypeError as error:
        raise TypeError("hidden_sizes must be a sequence of integers") from error
    if not sizes:
        raise ValueError("hidden_sizes cannot be empty")
    if any(
        isinstance(size, bool)
        or not isinstance(size, (int, np.integer))
        or size <= 0
        for size in sizes
    ):
        raise ValueError("hidden_sizes must contain positive integers")
    return tuple(int(size) for size in sizes)


def _build_mlp(
    input_dimension: int,
    hidden_sizes: Sequence[int],
) -> Tuple[nn.Sequential, int]:
    layers = []
    previous_size = input_dimension
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(previous_size, hidden_size), nn.ReLU()))
        previous_size = hidden_size
    return nn.Sequential(*layers), previous_size


def _validate_feature_tensor(
    tensor: torch.Tensor,
    expected_dimension: int,
    name: str,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim == 0 or tensor.shape[-1] != expected_dimension:
        raise ValueError(
            f"{name} must have final dimension {expected_dimension}, "
            f"got shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy for continuous hammer commands."""

    def __init__(
        self,
        observation_dimension: int,
        action_dimension: int,
        hidden_sizes: Sequence[int] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.observation_dimension = _validate_dimension(
            observation_dimension, "observation_dimension"
        )
        self.action_dimension = _validate_dimension(
            action_dimension, "action_dimension"
        )
        sizes = _validate_hidden_sizes(hidden_sizes)
        if (
            not math.isfinite(log_std_min)
            or not math.isfinite(log_std_max)
            or log_std_min >= log_std_max
        ):
            raise ValueError("log_std_min must be less than log_std_max")
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.trunk, output_dimension = _build_mlp(
            self.observation_dimension, sizes
        )
        self.mean_head = nn.Linear(output_dimension, self.action_dimension)
        self.log_std_head = nn.Linear(
            output_dimension, self.action_dimension
        )

    def forward(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_feature_tensor(
            observations, self.observation_dimension, "observations"
        )
        features = self.trunk(observations)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(
            min=self.log_std_min, max=self.log_std_max
        )
        return mean, log_std

    def sample(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return their corrected log probabilities."""
        mean, log_std = self(observations)
        distribution = Normal(mean, log_std.exp())
        pre_tanh_action = distribution.rsample()
        action = torch.tanh(pre_tanh_action)

        log_probability = distribution.log_prob(pre_tanh_action)
        correction = 2.0 * (
            math.log(2.0)
            - pre_tanh_action
            - F.softplus(-2.0 * pre_tanh_action)
        )
        log_probability = (log_probability - correction).sum(
            dim=-1, keepdim=True
        )
        return action, log_probability

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = self(observations)
        return torch.tanh(mean)


class QCritic(nn.Module):
    """State-action value network returning one scalar per transition."""

    def __init__(
        self,
        observation_dimension: int,
        action_dimension: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.observation_dimension = _validate_dimension(
            observation_dimension, "observation_dimension"
        )
        self.action_dimension = _validate_dimension(
            action_dimension, "action_dimension"
        )
        sizes = _validate_hidden_sizes(hidden_sizes)
        self.trunk, output_dimension = _build_mlp(
            self.observation_dimension + self.action_dimension,
            sizes,
        )
        self.value_head = nn.Linear(output_dimension, 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        _validate_feature_tensor(
            observations, self.observation_dimension, "observations"
        )
        _validate_feature_tensor(actions, self.action_dimension, "actions")
        if observations.shape[:-1] != actions.shape[:-1]:
            raise ValueError(
                "observations and actions must have identical leading "
                f"dimensions, got {tuple(observations.shape)} and "
                f"{tuple(actions.shape)}"
            )
        if observations.device != actions.device:
            raise ValueError("observations and actions must be on one device")
        if observations.dtype != actions.dtype:
            raise TypeError("observations and actions must have one dtype")

        features = self.trunk(torch.cat((observations, actions), dim=-1))
        return self.value_head(features)


class SACNetworks(nn.Module):
    """Registered collection of online and target SAC networks."""

    def __init__(
        self,
        actor: GaussianActor,
        critic_1: QCritic,
        critic_2: QCritic,
        target_critic_1: QCritic,
        target_critic_2: QCritic,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = target_critic_1
        self.target_critic_2 = target_critic_2
        self._freeze_targets()

    def _freeze_targets(self) -> None:
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)
        self.target_critic_1.eval()
        self.target_critic_2.eval()

    def train(self, mode: bool = True) -> "SACNetworks":
        super().train(mode)
        self._freeze_targets()
        return self


def create_sac_networks(
    config: SACConfig,
    observation_dimension: int = 20,
    action_dimension: int = 2,
    device: Device = None,
) -> SACNetworks:
    """Construct independent online critics and frozen target copies."""
    if not isinstance(config, SACConfig):
        raise TypeError("config must be an SACConfig")
    observation_dimension = _validate_dimension(
        observation_dimension, "observation_dimension"
    )
    action_dimension = _validate_dimension(
        action_dimension, "action_dimension"
    )
    target_device = (
        config.resolve_device()
        if device is None
        else resolve_torch_device(str(device))
    )

    actor = GaussianActor(
        observation_dimension,
        action_dimension,
        config.hidden_sizes,
        config.log_std_min,
        config.log_std_max,
    )
    critic_1 = QCritic(
        observation_dimension, action_dimension, config.hidden_sizes
    )
    critic_2 = QCritic(
        observation_dimension, action_dimension, config.hidden_sizes
    )
    networks = SACNetworks(
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        target_critic_1=deepcopy(critic_1),
        target_critic_2=deepcopy(critic_2),
    )
    return networks.to(target_device)
