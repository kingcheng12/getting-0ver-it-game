from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from .config import SACConfig
from .networks import QCritic, SACNetworks
from .replay_buffer import ReplayBatch


@dataclass(frozen=True)
class SACUpdateMetrics:
    actor_loss: float
    critic_1_loss: float
    critic_2_loss: float
    entropy_loss: Optional[float]
    entropy_coefficient: float
    policy_entropy: float
    mean_bellman_target: float


class SACUpdater:
    """Perform isolated Soft Actor-Critic parameter updates."""

    def __init__(self, networks: SACNetworks, config: SACConfig) -> None:
        if not isinstance(networks, SACNetworks):
            raise TypeError("networks must be an SACNetworks instance")
        if not isinstance(config, SACConfig):
            raise TypeError("config must be an SACConfig")

        self.networks = networks
        self.config = config
        self.device = self._network_device(networks)
        self.target_entropy = (
            -float(networks.actor.action_dimension)
            if config.target_entropy is None
            else float(config.target_entropy)
        )

        self.actor_optimizer = Adam(
            networks.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_1_optimizer = Adam(
            networks.critic_1.parameters(), lr=config.critic_learning_rate
        )
        self.critic_2_optimizer = Adam(
            networks.critic_2.parameters(), lr=config.critic_learning_rate
        )

        initial_log_alpha = math.log(config.initial_entropy_coefficient)
        if config.automatic_entropy_tuning:
            self.log_alpha = nn.Parameter(
                torch.tensor(initial_log_alpha, device=self.device)
            )
            self.entropy_optimizer: Optional[Adam] = Adam(
                (self.log_alpha,), lr=config.entropy_learning_rate
            )
            self._fixed_alpha = None
        else:
            self.log_alpha = None
            self.entropy_optimizer = None
            self._fixed_alpha = torch.tensor(
                config.initial_entropy_coefficient, device=self.device
            )

    @property
    def alpha(self) -> torch.Tensor:
        if self.log_alpha is not None:
            return self.log_alpha.exp()
        if self._fixed_alpha is None:
            raise RuntimeError("Entropy coefficient is not initialized")
        return self._fixed_alpha

    def update(self, batch: ReplayBatch) -> SACUpdateMetrics:
        self._validate_batch(batch)
        self.networks.train()

        bellman_target = self._compute_bellman_target(batch)
        critic_1_loss, critic_2_loss = self._update_critics(
            batch, bellman_target
        )
        actor_loss, log_probability = self._update_actor(
            batch.observations
        )
        entropy_loss = self._update_entropy(log_probability)
        self.soft_update_targets()

        return SACUpdateMetrics(
            actor_loss=float(actor_loss.detach().item()),
            critic_1_loss=float(critic_1_loss.detach().item()),
            critic_2_loss=float(critic_2_loss.detach().item()),
            entropy_loss=(
                None
                if entropy_loss is None
                else float(entropy_loss.detach().item())
            ),
            entropy_coefficient=float(self.alpha.detach().item()),
            policy_entropy=float(
                -log_probability.detach().mean().item()
            ),
            mean_bellman_target=float(bellman_target.mean().item()),
        )

    @torch.no_grad()
    def _compute_bellman_target(
        self, batch: ReplayBatch
    ) -> torch.Tensor:
        next_actions, next_log_probability = self.networks.actor.sample(
            batch.next_observations
        )
        next_q_1 = self.networks.target_critic_1(
            batch.next_observations, next_actions
        )
        next_q_2 = self.networks.target_critic_2(
            batch.next_observations, next_actions
        )
        minimum_next_q = torch.minimum(next_q_1, next_q_2)
        future_value = (
            minimum_next_q
            - self.alpha.detach() * next_log_probability
        )
        return (
            batch.rewards
            + self.config.gamma * batch.bootstrap_mask * future_value
        )

    def _update_critics(
        self, batch: ReplayBatch, bellman_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_q_1 = self.networks.critic_1(
            batch.observations, batch.actions
        )
        critic_1_loss = F.mse_loss(predicted_q_1, bellman_target)
        self._ensure_finite(critic_1_loss, "critic_1_loss")
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_1_optimizer.zero_grad(set_to_none=True)

        predicted_q_2 = self.networks.critic_2(
            batch.observations, batch.actions
        )
        critic_2_loss = F.mse_loss(predicted_q_2, bellman_target)
        self._ensure_finite(critic_2_loss, "critic_2_loss")
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        return critic_1_loss, critic_2_loss

    def _update_actor(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        critics = (self.networks.critic_1, self.networks.critic_2)
        previous_gradient_states = [
            [parameter.requires_grad for parameter in critic.parameters()]
            for critic in critics
        ]
        for critic in critics:
            critic.requires_grad_(False)

        try:
            actions, log_probability = self.networks.actor.sample(
                observations
            )
            q_1 = self.networks.critic_1(observations, actions)
            q_2 = self.networks.critic_2(observations, actions)
            actor_loss = (
                self.alpha.detach() * log_probability
                - torch.minimum(q_1, q_2)
            ).mean()
            self._ensure_finite(actor_loss, "actor_loss")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            for critic, states in zip(critics, previous_gradient_states):
                for parameter, requires_grad in zip(
                    critic.parameters(), states
                ):
                    parameter.requires_grad_(requires_grad)

        return actor_loss, log_probability

    def _update_entropy(
        self, log_probability: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.log_alpha is None or self.entropy_optimizer is None:
            return None
        entropy_loss = -(
            self.log_alpha
            * (log_probability.detach() + self.target_entropy)
        ).mean()
        self._ensure_finite(entropy_loss, "entropy_loss")
        self.entropy_optimizer.zero_grad(set_to_none=True)
        entropy_loss.backward()
        self.entropy_optimizer.step()
        return entropy_loss

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        self._soft_update(
            self.networks.critic_1, self.networks.target_critic_1
        )
        self._soft_update(
            self.networks.critic_2, self.networks.target_critic_2
        )

    def _soft_update(self, online: QCritic, target: QCritic) -> None:
        tau = self.config.tau
        for online_parameter, target_parameter in zip(
            online.parameters(), target.parameters()
        ):
            target_parameter.mul_(1.0 - tau)
            target_parameter.add_(online_parameter, alpha=tau)

    def _validate_batch(self, batch: ReplayBatch) -> None:
        if not isinstance(batch, ReplayBatch):
            raise TypeError("batch must be a ReplayBatch")

        observation_dimension = self.networks.actor.observation_dimension
        action_dimension = self.networks.actor.action_dimension
        expected = {
            "observations": (None, observation_dimension, torch.float32),
            "actions": (None, action_dimension, torch.float32),
            "rewards": (None, 1, torch.float32),
            "next_observations": (
                None,
                observation_dimension,
                torch.float32,
            ),
            "terminated": (None, 1, torch.bool),
            "truncated": (None, 1, torch.bool),
            "bootstrap_mask": (None, 1, torch.float32),
            "episode_end": (None, 1, torch.bool),
        }
        batch_size = None
        for name, (_, feature_size, dtype) in expected.items():
            tensor = getattr(batch, name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"batch.{name} must be a torch.Tensor")
            if tensor.ndim != 2 or tensor.shape[1] != feature_size:
                raise ValueError(
                    f"batch.{name} must have shape (batch, "
                    f"{feature_size}), got {tuple(tensor.shape)}"
                )
            if batch_size is None:
                batch_size = tensor.shape[0]
                if batch_size == 0:
                    raise ValueError("batch cannot be empty")
            elif tensor.shape[0] != batch_size:
                raise ValueError("all batch tensors must have one batch size")
            if tensor.dtype != dtype:
                raise TypeError(f"batch.{name} must have dtype {dtype}")
            if tensor.device != self.device:
                raise ValueError(
                    f"batch.{name} must be on device {self.device}"
                )
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"batch.{name} contains non-finite values")

        if torch.any(batch.terminated & batch.truncated):
            raise ValueError(
                "a transition cannot be both terminated and truncated"
            )
        expected_bootstrap = (~batch.terminated).to(torch.float32)
        if not torch.equal(batch.bootstrap_mask, expected_bootstrap):
            raise ValueError("bootstrap_mask must equal 1 - terminated")
        expected_episode_end = batch.terminated | batch.truncated
        if not torch.equal(batch.episode_end, expected_episode_end):
            raise ValueError(
                "episode_end must equal terminated | truncated"
            )
        if torch.any(batch.actions < -1.0) or torch.any(batch.actions > 1.0):
            raise ValueError("batch.actions must remain in [-1, 1]")

    @staticmethod
    def _network_device(networks: SACNetworks) -> torch.device:
        parameters = list(networks.parameters())
        if not parameters:
            raise ValueError("networks must contain parameters")
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise ValueError("all networks must be on one device")
        return parameters[0].device

    @staticmethod
    def _ensure_finite(value: torch.Tensor, name: str) -> None:
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} became non-finite")
