from dataclasses import replace
import math
from types import MethodType

import pytest
import torch

from getting_over_it_rl import (
    ReplayBatch,
    SACConfig,
    SACUpdateMetrics,
    SACUpdater,
    create_sac_networks,
)


def make_batch(batch_size=8, *, terminated=None, truncated=None):
    observations = torch.randn(batch_size, 20)
    actions = torch.empty(batch_size, 2).uniform_(-1.0, 1.0)
    rewards = torch.randn(batch_size, 1)
    next_observations = torch.randn(batch_size, 20)
    if terminated is None:
        terminated = torch.zeros(batch_size, 1, dtype=torch.bool)
    if truncated is None:
        truncated = torch.zeros(batch_size, 1, dtype=torch.bool)
    return ReplayBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=(~terminated).to(torch.float32),
        episode_end=terminated | truncated,
    )


def clone_parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def parameters_changed(before, module):
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters())
    )


def small_config(**overrides):
    values = {
        "hidden_sizes": (16, 16),
        "actor_learning_rate": 1e-2,
        "critic_learning_rate": 1e-2,
        "entropy_learning_rate": 1e-2,
    }
    values.update(overrides)
    return SACConfig(**values)


def test_complete_update_returns_finite_metrics_and_updates_online_networks():
    config = small_config()
    networks = create_sac_networks(config)
    updater = SACUpdater(networks, config)
    actor_before = clone_parameters(networks.actor)
    critic_1_before = clone_parameters(networks.critic_1)
    critic_2_before = clone_parameters(networks.critic_2)

    metrics = updater.update(make_batch())

    assert isinstance(metrics, SACUpdateMetrics)
    for value in (
        metrics.actor_loss,
        metrics.critic_1_loss,
        metrics.critic_2_loss,
        metrics.entropy_loss,
        metrics.entropy_coefficient,
        metrics.policy_entropy,
        metrics.mean_bellman_target,
    ):
        assert value is not None and math.isfinite(value)
    assert parameters_changed(actor_before, networks.actor)
    assert parameters_changed(critic_1_before, networks.critic_1)
    assert parameters_changed(critic_2_before, networks.critic_2)
    assert not any(
        parameter.grad is not None
        for parameter in networks.target_critic_1.parameters()
    )
    assert not any(
        parameter.grad is not None
        for parameter in networks.target_critic_2.parameters()
    )


def test_actor_update_does_not_change_or_accumulate_gradients_in_critics():
    config = small_config()
    networks = create_sac_networks(config)
    updater = SACUpdater(networks, config)
    critic_1_before = clone_parameters(networks.critic_1)
    critic_2_before = clone_parameters(networks.critic_2)

    updater._update_actor(torch.randn(8, 20))

    assert not parameters_changed(critic_1_before, networks.critic_1)
    assert not parameters_changed(critic_2_before, networks.critic_2)
    assert all(
        parameter.grad is None for parameter in networks.critic_1.parameters()
    )
    assert all(
        parameter.grad is None for parameter in networks.critic_2.parameters()
    )


def test_bellman_target_uses_twin_minimum_entropy_and_terminal_mask():
    config = small_config(
        gamma=0.9,
        automatic_entropy_tuning=False,
        initial_entropy_coefficient=0.2,
    )
    networks = create_sac_networks(config)
    updater = SACUpdater(networks, config)

    def fixed_sample(self, observations):
        actions = torch.zeros(observations.shape[0], 2)
        log_probability = torch.full((observations.shape[0], 1), -0.5)
        return actions, log_probability

    def q_value(value):
        def forward(self, observations, actions):
            return torch.full((observations.shape[0], 1), value)

        return forward

    networks.actor.sample = MethodType(fixed_sample, networks.actor)
    networks.target_critic_1.forward = MethodType(
        q_value(3.0), networks.target_critic_1
    )
    networks.target_critic_2.forward = MethodType(
        q_value(5.0), networks.target_critic_2
    )
    terminated = torch.tensor([[True], [False]])
    truncated = torch.tensor([[False], [True]])
    batch = make_batch(
        2, terminated=terminated, truncated=truncated
    )
    batch.rewards.fill_(1.0)

    target = updater._compute_bellman_target(batch)

    expected = torch.tensor([[1.0], [1.0 + 0.9 * (3.0 + 0.1)]])
    torch.testing.assert_close(target, expected)
    assert not target.requires_grad
    assert all(
        parameter.grad is None for parameter in networks.actor.parameters()
    )


def test_automatic_entropy_tuning_uses_default_target_and_changes_alpha():
    config = small_config()
    networks = create_sac_networks(config)
    updater = SACUpdater(networks, config)
    alpha_before = updater.alpha.detach().clone()

    metrics = updater.update(make_batch())

    assert updater.target_entropy == -2.0
    assert metrics.entropy_loss is not None
    assert not torch.equal(alpha_before, updater.alpha.detach())


def test_fixed_entropy_coefficient_has_no_entropy_loss():
    config = small_config(
        automatic_entropy_tuning=False,
        initial_entropy_coefficient=0.35,
        target_entropy=-1.5,
    )
    updater = SACUpdater(create_sac_networks(config), config)

    metrics = updater.update(make_batch())

    assert updater.target_entropy == -1.5
    assert updater.alpha.item() == pytest.approx(0.35)
    assert metrics.entropy_loss is None
    assert metrics.entropy_coefficient == pytest.approx(0.35)


def test_soft_update_matches_polyak_equation_exactly():
    config = small_config(tau=0.25)
    networks = create_sac_networks(config)
    updater = SACUpdater(networks, config)
    target_1_before = clone_parameters(networks.target_critic_1)
    target_2_before = clone_parameters(networks.target_critic_2)
    with torch.no_grad():
        for parameter in networks.critic_1.parameters():
            parameter.add_(1.0)
        for parameter in networks.critic_2.parameters():
            parameter.sub_(1.0)
    online_1_after = clone_parameters(networks.critic_1)
    online_2_after = clone_parameters(networks.critic_2)

    updater.soft_update_targets()

    for old_target, online, updated_target in zip(
        target_1_before,
        online_1_after,
        networks.target_critic_1.parameters(),
    ):
        expected = 0.75 * old_target + 0.25 * online
        torch.testing.assert_close(updated_target, expected)
    for old_target, online, updated_target in zip(
        target_2_before,
        online_2_after,
        networks.target_critic_2.parameters(),
    ):
        expected = 0.75 * old_target + 0.25 * online
        torch.testing.assert_close(updated_target, expected)


@pytest.mark.parametrize(
    ("field", "replacement", "error", "message"),
    [
        (
            "observations",
            torch.zeros(8, 19),
            ValueError,
            "observations",
        ),
        (
            "actions",
            torch.full((8, 2), 2.0),
            ValueError,
            "actions",
        ),
        (
            "rewards",
            torch.full((8, 1), float("nan")),
            ValueError,
            "non-finite",
        ),
        (
            "terminated",
            torch.zeros(8, 1),
            TypeError,
            "dtype",
        ),
    ],
)
def test_updater_rejects_invalid_batches(
    field, replacement, error, message
):
    batch = replace(make_batch(), **{field: replacement})
    updater = SACUpdater(
        create_sac_networks(small_config()), small_config()
    )

    with pytest.raises(error, match=message):
        updater.update(batch)


def test_updater_rejects_inconsistent_terminal_masks():
    batch = make_batch()
    batch = replace(batch, bootstrap_mask=torch.zeros(8, 1))
    config = small_config()
    updater = SACUpdater(create_sac_networks(config), config)

    with pytest.raises(ValueError, match="bootstrap_mask"):
        updater.update(batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_updater_rejects_batch_on_different_device():
    config = small_config(device="cuda")
    updater = SACUpdater(create_sac_networks(config), config)

    with pytest.raises(ValueError, match="device"):
        updater.update(make_batch())
