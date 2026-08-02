import numpy as np
import pytest
import torch

from getting_over_it_rl import (
    GaussianActor,
    QCritic,
    SACConfig,
    SACNetworks,
    create_sac_networks,
)


def test_actor_shapes_for_single_and_batched_observations():
    actor = GaussianActor(20, 2, hidden_sizes=(32, 16))

    single_mean, single_log_std = actor(torch.zeros(20))
    batch_mean, batch_log_std = actor(torch.zeros(5, 20))

    assert single_mean.shape == (2,)
    assert single_log_std.shape == (2,)
    assert batch_mean.shape == (5, 2)
    assert batch_log_std.shape == (5, 2)


def test_actor_clamps_log_standard_deviation():
    actor = GaussianActor(
        20, 2, hidden_sizes=(16,), log_std_min=-3.0, log_std_max=1.0
    )
    with torch.no_grad():
        actor.log_std_head.weight.fill_(100.0)
        actor.log_std_head.bias.fill_(100.0)

    _, upper = actor(torch.ones(4, 20))
    _, lower = actor(-torch.ones(4, 20))

    assert torch.all(upper <= 1.0)
    assert torch.all(upper >= -3.0)
    assert torch.all(lower <= 1.0)
    assert torch.all(lower >= -3.0)


def test_sampled_actions_and_log_probabilities_are_bounded_and_finite():
    actor = GaussianActor(20, 2, hidden_sizes=(32, 32))
    observations = torch.full((64, 20), 1_000.0)

    actions, log_probabilities = actor.sample(observations)

    assert actions.shape == (64, 2)
    assert log_probabilities.shape == (64, 1)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probabilities).all()


def test_deterministic_action_is_tanh_of_mean_and_repeatable():
    actor = GaussianActor(20, 2, hidden_sizes=(16,))
    observations = torch.randn(3, 20)

    mean, _ = actor(observations)
    first = actor.deterministic(observations)
    second = actor.deterministic(observations)

    torch.testing.assert_close(first, torch.tanh(mean))
    torch.testing.assert_close(first, second)


def test_reparameterized_sample_propagates_actor_gradients():
    actor = GaussianActor(20, 2, hidden_sizes=(16,))
    actions, log_probabilities = actor.sample(torch.randn(8, 20))

    (actions.mean() + log_probabilities.mean()).backward()

    assert all(
        parameter.grad is not None for parameter in actor.parameters()
    )
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )


def test_critic_shapes_and_gradients_for_single_and_batch_inputs():
    critic = QCritic(20, 2, hidden_sizes=(32, 16))
    single_value = critic(torch.zeros(20), torch.zeros(2))
    batch_value = critic(torch.randn(7, 20), torch.randn(7, 2))

    assert single_value.shape == (1,)
    assert batch_value.shape == (7, 1)

    batch_value.mean().backward()
    assert all(
        parameter.grad is not None for parameter in critic.parameters()
    )


def test_factory_creates_independent_critics_and_matching_targets():
    networks = create_sac_networks(
        SACConfig(hidden_sizes=(16, 16)), device="cpu"
    )

    critic_1_parameters = list(networks.critic_1.parameters())
    critic_2_parameters = list(networks.critic_2.parameters())
    target_1_parameters = list(networks.target_critic_1.parameters())
    target_2_parameters = list(networks.target_critic_2.parameters())

    assert not any(
        first.data_ptr() == second.data_ptr()
        for first, second in zip(critic_1_parameters, critic_2_parameters)
    )
    for online, target in zip(critic_1_parameters, target_1_parameters):
        torch.testing.assert_close(online, target)
        assert online.data_ptr() != target.data_ptr()
    for online, target in zip(critic_2_parameters, target_2_parameters):
        torch.testing.assert_close(online, target)
        assert online.data_ptr() != target.data_ptr()


def test_targets_remain_frozen_and_in_evaluation_mode():
    networks = create_sac_networks(
        SACConfig(hidden_sizes=(16,)), device="cpu"
    )

    networks.train()

    assert networks.actor.training
    assert networks.critic_1.training
    assert not networks.target_critic_1.training
    assert not networks.target_critic_2.training
    assert not any(
        parameter.requires_grad
        for parameter in networks.target_critic_1.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in networks.target_critic_2.parameters()
    )


def test_network_collection_state_dict_round_trip():
    source = create_sac_networks(
        SACConfig(hidden_sizes=(16,)), device="cpu"
    )
    restored = create_sac_networks(
        SACConfig(hidden_sizes=(16,)), device="cpu"
    )

    restored.load_state_dict(source.state_dict())
    observation = torch.randn(20)

    torch.testing.assert_close(
        source.actor.deterministic(observation),
        restored.actor.deterministic(observation),
    )


@pytest.mark.parametrize(
    ("observations", "actions", "message"),
    [
        (torch.zeros(19), torch.zeros(2), "observations"),
        (torch.zeros(20), torch.zeros(3), "actions"),
        (torch.zeros(4, 20), torch.zeros(3, 2), "leading"),
    ],
)
def test_critic_rejects_incompatible_inputs(
    observations, actions, message
):
    with pytest.raises(ValueError, match=message):
        QCritic(20, 2, hidden_sizes=(16,))(observations, actions)


def test_networks_reject_invalid_dimensions_and_tensor_types():
    with pytest.raises(ValueError, match="observation_dimension"):
        GaussianActor(0, 2)
    with pytest.raises(ValueError, match="action_dimension"):
        QCritic(20, -1)
    with pytest.raises(TypeError, match="torch.Tensor"):
        GaussianActor(20, 2)(np.zeros(20, dtype=np.float32))
    with pytest.raises(TypeError, match="floating-point"):
        GaussianActor(20, 2)(torch.zeros(20, dtype=torch.int64))


def test_factory_returns_registered_collection_on_cpu():
    networks = create_sac_networks(SACConfig(hidden_sizes=(16,)))

    assert isinstance(networks, SACNetworks)
    assert all(
        parameter.device.type == "cpu" for parameter in networks.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_factory_can_place_all_networks_on_gpu():
    networks = create_sac_networks(
        SACConfig(hidden_sizes=(16,), device="cuda")
    )

    assert all(
        parameter.device.type == "cuda" for parameter in networks.parameters()
    )
