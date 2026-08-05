from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from getting_over_it_rl import (
    DemonstrationBuffer,
    DemonstrationEpisode,
    DemonstrationSource,
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


def make_demo_buffer(size=4, reward=99.0, seed=3):
    observations = np.zeros((size, 20), np.float32)
    return DemonstrationBuffer(
        (
            DemonstrationEpisode(
                source=DemonstrationSource("synthetic.demo", "a" * 64),
                index=0,
                observations=observations,
                actions=np.zeros((size, 2), np.float32),
                rewards=np.full((size, 1), reward, np.float32),
                next_observations=observations.copy(),
                terminated=np.zeros((size, 1), np.bool_),
                truncated=np.zeros((size, 1), np.bool_),
                outcome="incomplete",
            ),
        ),
        seed=seed,
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


def test_warmup_override_updates_updater_and_checkpoint(tmp_path):
    algorithm = SACAlgorithm(config=small_config(warmup_steps=4))

    algorithm.set_warmup_steps(0)

    assert algorithm.config.warmup_steps == 0
    assert algorithm.updater.config.warmup_steps == 0
    checkpoint = tmp_path / "warmup"
    algorithm.save(checkpoint)
    assert SACAlgorithm.load(checkpoint).config.warmup_steps == 0


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_warmup_override_rejects_invalid_values(value):
    algorithm = SACAlgorithm(config=small_config())

    with pytest.raises((TypeError, ValueError), match="warmup_steps"):
        algorithm.set_warmup_steps(value)


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
    algorithm.set_demonstrations(make_demo_buffer())
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
    assert len(restored.demonstration_buffer) == 4
    assert (
        restored.demonstration_buffer.episodes
        == algorithm.demonstration_buffer.episodes
    )
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


def test_mixed_batch_has_exact_default_25_75_composition():
    config = small_config(
        replay_capacity=300,
        batch_size=256,
        demonstration_batch_fraction=0.25,
    )
    algorithm = SACAlgorithm(config=config)
    algorithm.set_demonstrations(make_demo_buffer(size=64))
    for _ in range(192):
        zeros = np.zeros(20, np.float32)
        algorithm.replay_buffer.add(
            zeros, np.zeros(2, np.float32), 0.0, zeros, False, False
        )

    batch = algorithm._sample_training_batch()

    assert batch.rewards.shape == (256, 1)
    assert torch.count_nonzero(batch.rewards == 99.0).item() == 64
    assert torch.count_nonzero(batch.rewards == 0.0).item() == 192


def test_schema_one_checkpoint_migrates_with_empty_demonstrations(tmp_path):
    algorithm = SACAlgorithm(config=small_config())
    state = algorithm._checkpoint_state()
    state["version"] = 1
    state.pop("demonstration_buffer")
    state["config"].pop("demonstration_batch_fraction")
    checkpoint = tmp_path / "v1"
    torch.save(state, checkpoint)

    restored = SACAlgorithm.load(checkpoint)

    assert len(restored.demonstration_buffer) == 0
    assert restored.config.demonstration_batch_fraction == pytest.approx(0.25)


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


def test_weights_only_initialization_preserves_networks_and_resets_state(
    tmp_path,
):
    source = SACAlgorithm(config=small_config(), seed=11)
    populate_algorithm(source)
    checkpoint = tmp_path / "old-reward"
    source.save(checkpoint)

    transferred = SACAlgorithm.initialize_from_checkpoint(
        checkpoint,
        config=small_config(),
        seed=23,
    )

    assert_module_equal(source.networks.actor, transferred.networks.actor)
    assert_module_equal(source.networks.critic_1, transferred.networks.critic_1)
    assert_module_equal(source.networks.critic_2, transferred.networks.critic_2)
    assert_module_equal(
        transferred.networks.critic_1,
        transferred.networks.target_critic_1,
    )
    assert_module_equal(
        transferred.networks.critic_2,
        transferred.networks.target_critic_2,
    )
    assert len(transferred.replay_buffer) == 0
    assert transferred.environment_steps == 0
    assert transferred.gradient_updates == 0
    assert transferred.episodes_completed == 0
    assert transferred.seed == 23
    assert not transferred.updater.actor_optimizer.state_dict()["state"]
    assert not transferred.updater.critic_1_optimizer.state_dict()["state"]
    assert not transferred.updater.critic_2_optimizer.state_dict()["state"]
    assert transferred.updater.alpha.item() == pytest.approx(0.2)


def test_factory_supports_weights_only_initialization(tmp_path):
    source = SACAlgorithm(config=small_config(), seed=3)
    populate_algorithm(source)
    checkpoint = tmp_path / "source"
    source.save(checkpoint)

    transferred = create_algorithm(
        checkpoint,
        config=small_config(),
        seed=8,
        weights_only=True,
    )

    assert isinstance(transferred, SACAlgorithm)
    assert transferred.seed == 8
    assert len(transferred.replay_buffer) == 0
    assert transferred.environment_steps == 0


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
