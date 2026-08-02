from copy import deepcopy

import numpy as np
import pytest
import torch

from getting_over_it_rl import ReplayBatch, ReplayBuffer


def transition(index, *, terminated=False, truncated=False):
    observation = np.full((20,), index, dtype=np.float32)
    action = np.array([index, -index], dtype=np.float32)
    return (
        observation,
        action,
        float(index),
        observation + 0.5,
        terminated,
        truncated,
    )


def test_add_and_sample_return_sac_tensor_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=8, seed=7)
    for index in range(4):
        buffer.add(*transition(index))

    batch = buffer.sample(3, device="cpu")

    assert isinstance(batch, ReplayBatch)
    assert batch.observations.shape == (3, 20)
    assert batch.actions.shape == (3, 2)
    assert batch.rewards.shape == (3, 1)
    assert batch.next_observations.shape == (3, 20)
    assert batch.observations.dtype == torch.float32
    assert batch.actions.dtype == torch.float32
    assert batch.rewards.dtype == torch.float32
    assert batch.terminated.dtype == torch.bool
    assert batch.truncated.dtype == torch.bool
    assert batch.bootstrap_mask.dtype == torch.float32
    assert batch.episode_end.dtype == torch.bool
    assert batch.observations.device.type == "cpu"


def test_add_copies_inputs_and_ring_buffer_replaces_oldest_entries():
    buffer = ReplayBuffer(capacity=3, seed=0)
    observation, action, reward, next_observation, terminated, truncated = (
        transition(0)
    )
    buffer.add(
        observation,
        action,
        reward,
        next_observation,
        terminated,
        truncated,
    )
    observation[:] = 99
    action[:] = 99
    next_observation[:] = 99
    for index in range(1, 5):
        buffer.add(*transition(index))

    batch = buffer.sample(3)
    stored_ids = set(batch.rewards.squeeze(1).tolist())

    assert len(buffer) == 3
    assert stored_ids == {2.0, 3.0, 4.0}


def test_add_stores_the_clipped_action_sent_to_unity():
    buffer = ReplayBuffer(capacity=1, seed=0)
    values = list(transition(0))
    values[1] = np.array([2.0, -3.0], dtype=np.float32)

    buffer.add(*values)

    torch.testing.assert_close(
        buffer.sample(1).actions,
        torch.tensor([[1.0, -1.0]], dtype=torch.float32),
    )


def test_sampling_is_seeded_uniform_and_without_replacement():
    first = ReplayBuffer(capacity=10, seed=123)
    second = ReplayBuffer(capacity=10, seed=123)
    for index in range(10):
        first.add(*transition(index))
        second.add(*transition(index))

    first_rewards = first.sample(6).rewards
    second_rewards = second.sample(6).rewards

    torch.testing.assert_close(first_rewards, second_rewards)
    assert torch.unique(first_rewards).numel() == 6


def test_terminal_and_truncation_masks_have_sac_semantics():
    buffer = ReplayBuffer(capacity=3, seed=0)
    buffer.add(*transition(0))
    buffer.add(*transition(1, terminated=True))
    buffer.add(*transition(2, truncated=True))

    batch = buffer.sample(3)
    by_reward = {
        int(reward.item()): index
        for index, reward in enumerate(batch.rewards)
    }

    normal = by_reward[0]
    terminal = by_reward[1]
    timeout = by_reward[2]
    assert batch.bootstrap_mask[normal].item() == 1.0
    assert not batch.episode_end[normal].item()
    assert batch.bootstrap_mask[terminal].item() == 0.0
    assert batch.episode_end[terminal].item()
    assert batch.bootstrap_mask[timeout].item() == 1.0
    assert batch.episode_end[timeout].item()


def test_state_round_trip_restores_data_position_and_rng():
    original = ReplayBuffer(capacity=4, seed=42)
    for index in range(6):
        original.add(*transition(index))
    state = original.state_dict()
    restored = ReplayBuffer(capacity=4, seed=999)
    restored.load_state_dict(state)

    assert len(restored) == len(original)
    torch.testing.assert_close(
        original.sample(4).rewards, restored.sample(4).rewards
    )

    original.add(*transition(10))
    restored.add(*transition(10))
    torch.testing.assert_close(
        original.sample(4).rewards, restored.sample(4).rewards
    )


def test_state_dict_does_not_alias_buffer_storage():
    buffer = ReplayBuffer(capacity=2, seed=0)
    buffer.add(*transition(1))
    state = buffer.state_dict()
    state["observations"][0] = 99

    assert not torch.all(buffer.sample(1).observations == 99)


def test_state_dict_only_contains_initialized_transitions():
    buffer = ReplayBuffer(capacity=100, seed=0)
    buffer.add(*transition(1))

    state = buffer.state_dict()

    assert state["observations"].shape == (1, 20)
    assert state["actions"].shape == (1, 2)
    assert state["rewards"].shape == (1, 1)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("observation", np.zeros((19,), dtype=np.float32), ValueError),
        ("action", np.zeros((3,), dtype=np.float32), ValueError),
        (
            "next_observation",
            np.full((20,), np.nan, dtype=np.float32),
            ValueError,
        ),
        ("reward", float("inf"), ValueError),
        ("terminated", 1, TypeError),
        ("truncated", 0, TypeError),
    ],
)
def test_add_rejects_malformed_transitions(field, value, error):
    names = [
        "observation",
        "action",
        "reward",
        "next_observation",
        "terminated",
        "truncated",
    ]
    values = list(transition(0))
    values[names.index(field)] = value

    with pytest.raises(error):
        ReplayBuffer(capacity=2).add(*values)


def test_add_rejects_transition_that_is_both_terminal_and_truncated():
    with pytest.raises(ValueError, match="cannot both be true"):
        ReplayBuffer(capacity=2).add(
            *transition(0, terminated=True, truncated=True)
        )


@pytest.mark.parametrize("batch_size", [0, -1, 3])
def test_sample_rejects_invalid_batch_size(batch_size):
    buffer = ReplayBuffer(capacity=2)
    buffer.add(*transition(0))
    buffer.add(*transition(1))

    with pytest.raises(ValueError):
        buffer.sample(batch_size)


def test_load_rejects_incompatible_or_corrupt_state():
    buffer = ReplayBuffer(capacity=2)
    buffer.add(*transition(0))
    state = buffer.state_dict()

    incompatible = deepcopy(state)
    incompatible["capacity"] = 3
    with pytest.raises(ValueError, match="capacity"):
        buffer.load_state_dict(incompatible)

    corrupt = deepcopy(state)
    corrupt["observations"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        buffer.load_state_dict(corrupt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_sample_can_transfer_batch_to_gpu():
    buffer = ReplayBuffer(capacity=2, seed=0)
    buffer.add(*transition(0))

    batch = buffer.sample(1, device="cuda")

    assert batch.observations.device.type == "cuda"
    assert batch.bootstrap_mask.device.type == "cuda"
