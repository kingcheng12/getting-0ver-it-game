from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from getting_over_it_rl import (
    ReplayBatch,
    SACAlgorithm,
    SACConfig,
    create_algorithm,
)
from getting_over_it_rl import evaluate
from getting_over_it_rl.config import EnvironmentConfig, EvaluationConfig


def small_config(**overrides):
    values = {
        "hidden_sizes": (16, 16),
        "replay_capacity": 32,
        "batch_size": 4,
        "warmup_steps": 4,
    }
    values.update(overrides)
    return SACConfig(**values)


def make_batch(batch_size=4):
    terminated = torch.zeros(batch_size, 1, dtype=torch.bool)
    truncated = torch.zeros(batch_size, 1, dtype=torch.bool)
    return ReplayBatch(
        observations=torch.randn(batch_size, 20),
        actions=torch.empty(batch_size, 2).uniform_(-1.0, 1.0),
        rewards=torch.randn(batch_size, 1),
        next_observations=torch.randn(batch_size, 20),
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=torch.ones(batch_size, 1),
        episode_end=terminated | truncated,
    )


def populate_algorithm(algorithm):
    for index in range(6):
        observation = np.full((20,), index, dtype=np.float32)
        algorithm.replay_buffer.add(
            observation,
            np.array([0.25, -0.5], dtype=np.float32),
            float(index),
            observation + 0.5,
            terminated=index == 5,
            truncated=False,
        )
    algorithm.updater.update(make_batch())
    algorithm.environment_steps = 17
    algorithm.gradient_updates = 3
    algorithm.episodes_completed = 2


def assert_module_equal(first, second):
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert set(first_state) == set(second_state)
    for key in first_state:
        torch.testing.assert_close(first_state[key], second_state[key])


def test_algorithm_constructs_all_components_and_zeroed_counters():
    algorithm = SACAlgorithm(config=small_config(), seed=7)

    assert algorithm.observation_dimension == 20
    assert algorithm.action_dimension == 2
    assert algorithm.networks.actor.observation_dimension == 20
    assert algorithm.networks.actor.action_dimension == 2
    assert algorithm.updater.networks is algorithm.networks
    assert algorithm.replay_buffer.capacity == 32
    assert len(algorithm.replay_buffer) == 0
    assert algorithm.environment_steps == 0
    assert algorithm.gradient_updates == 0
    assert algorithm.episodes_completed == 0


def test_predict_supports_deterministic_and_stochastic_actions():
    algorithm = SACAlgorithm(config=small_config(), seed=1)
    observation = np.zeros((20,), dtype=np.float32)

    first = algorithm.predict(observation, deterministic=True)
    second = algorithm.predict(observation, deterministic=True)
    stochastic = algorithm.predict(observation, deterministic=False)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2,)
    assert first.dtype == np.float32
    assert stochastic.shape == (2,)
    assert stochastic.dtype == np.float32
    assert np.all(stochastic >= -1.0)
    assert np.all(stochastic <= 1.0)


@pytest.mark.parametrize(
    ("observation", "error", "message"),
    [
        ([0.0] * 20, TypeError, "NumPy"),
        (np.zeros(20, dtype=np.float64), TypeError, "float32"),
        (np.zeros(19, dtype=np.float32), ValueError, "shape"),
        (
            np.full(20, np.nan, dtype=np.float32),
            ValueError,
            "finite",
        ),
    ],
)
def test_predict_rejects_invalid_observations(
    observation, error, message
):
    algorithm = SACAlgorithm(config=small_config())

    with pytest.raises(error, match=message):
        algorithm.predict(observation)


def test_checkpoint_round_trip_preserves_complete_training_state(tmp_path):
    algorithm = SACAlgorithm(config=small_config(), seed=11)
    populate_algorithm(algorithm)
    checkpoint = tmp_path / "latest"
    expected_prediction = algorithm.predict(
        np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    )

    algorithm.save(checkpoint)
    expected_stochastic_prediction = algorithm.predict(
        np.zeros(20, dtype=np.float32), deterministic=False
    )
    restored = SACAlgorithm.load(checkpoint, device="cpu")

    np.testing.assert_array_equal(
        restored.predict(
            np.linspace(-1.0, 1.0, 20, dtype=np.float32)
        ),
        expected_prediction,
    )
    np.testing.assert_array_equal(
        restored.predict(
            np.zeros(20, dtype=np.float32), deterministic=False
        ),
        expected_stochastic_prediction,
    )
    assert_module_equal(algorithm.networks, restored.networks)
    assert restored.updater.alpha.item() == pytest.approx(
        algorithm.updater.alpha.item()
    )
    assert restored.updater.actor_optimizer.state_dict()["state"]
    assert restored.updater.critic_1_optimizer.state_dict()["state"]
    assert restored.updater.critic_2_optimizer.state_dict()["state"]
    assert restored.updater.entropy_optimizer is not None
    assert restored.updater.entropy_optimizer.state_dict()["state"]
    assert len(restored.replay_buffer) == len(algorithm.replay_buffer)
    assert restored.environment_steps == 17
    assert restored.gradient_updates == 3
    assert restored.episodes_completed == 2

    original_sample = algorithm.replay_buffer.sample(4).rewards
    restored_sample = restored.replay_buffer.sample(4).rewards
    torch.testing.assert_close(original_sample, restored_sample)
    np.testing.assert_array_equal(
        algorithm.rng.integers(0, 10_000, size=5),
        restored.rng.integers(0, 10_000, size=5),
    )


def test_checkpoint_save_replaces_existing_file_atomically(tmp_path):
    checkpoint = tmp_path / "latest"
    checkpoint.write_bytes(b"old checkpoint")
    algorithm = SACAlgorithm(config=small_config())

    algorithm.save(checkpoint)

    restored = SACAlgorithm.load(checkpoint)
    assert isinstance(restored, SACAlgorithm)
    assert not list(tmp_path.glob(".latest.*.tmp"))


def test_fixed_entropy_checkpoint_round_trip(tmp_path):
    config = small_config(
        automatic_entropy_tuning=False,
        initial_entropy_coefficient=0.35,
    )
    algorithm = SACAlgorithm(config=config)
    algorithm.updater.update(make_batch())
    checkpoint = tmp_path / "fixed-alpha"

    algorithm.save(checkpoint)
    restored = SACAlgorithm.load(checkpoint)

    assert restored.updater.log_alpha is None
    assert restored.updater.entropy_optimizer is None
    assert restored.updater.alpha.item() == pytest.approx(0.35)


def test_factory_constructs_and_loads_sac_algorithm(tmp_path):
    config = small_config()
    created = create_algorithm(config=config, seed=4)
    checkpoint = tmp_path / "agent"
    created.save(checkpoint)

    loaded = create_algorithm(checkpoint, device="cpu")

    assert isinstance(created, SACAlgorithm)
    assert isinstance(loaded, SACAlgorithm)
    assert loaded.seed == 4


@pytest.mark.parametrize("corruption", ["version", "dimension", "keys"])
def test_load_rejects_incompatible_checkpoints(tmp_path, corruption):
    algorithm = SACAlgorithm(config=small_config())
    state = algorithm._checkpoint_state()
    if corruption == "version":
        state["version"] = 999
    elif corruption == "dimension":
        state["replay_buffer"]["observation_shape"] = (19,)
    else:
        del state["networks"]
    checkpoint = tmp_path / corruption
    torch.save(state, checkpoint)

    with pytest.raises(ValueError):
        SACAlgorithm.load(checkpoint)


def test_load_rejects_missing_and_unreadable_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        SACAlgorithm.load(tmp_path / "missing")

    corrupt = tmp_path / "corrupt"
    corrupt.write_text("not a PyTorch checkpoint", encoding="utf-8")
    with pytest.raises(ValueError, match="Unable to read"):
        SACAlgorithm.load(corrupt)


class OneStepEnvironment(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, (20,), np.float32
        )
        self.closed = False
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(20, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action)
        return np.zeros(20, dtype=np.float32), 0.0, True, False, {}

    def close(self):
        self.closed = True


def test_evaluation_loads_device_and_uses_deterministic_prediction(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "agent"
    SACAlgorithm(config=small_config()).save(checkpoint)
    environment = OneStepEnvironment()
    monkeypatch.setattr(evaluate.gym, "make", lambda *args, **kwargs: environment)
    observed = {}

    def loader(path: Path, device=None):
        observed["path"] = path
        observed["device"] = device
        return SACAlgorithm.load(path, device=device)

    config = EvaluationConfig(
        environment=EnvironmentConfig(),
        episodes=2,
        device="cpu",
        checkpoint_path=checkpoint,
    )
    evaluate.run_evaluation(config, algorithm_loader=loader)

    assert observed == {"path": checkpoint, "device": "cpu"}
    assert len(environment.actions) == 2
    np.testing.assert_array_equal(
        environment.actions[0], environment.actions[1]
    )
    assert environment.closed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_checkpoint_can_load_with_cuda_override(tmp_path):
    checkpoint = tmp_path / "agent"
    SACAlgorithm(config=small_config()).save(checkpoint)

    restored = SACAlgorithm.load(checkpoint, device="cuda")

    assert restored.device.type == "cuda"
    assert all(
        parameter.device.type == "cuda"
        for parameter in restored.networks.parameters()
    )
