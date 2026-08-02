from pathlib import Path
import math

import gymnasium as gym
import numpy as np
import pytest

from getting_over_it_rl import (
    SACAlgorithm,
    SACConfig,
    SACUpdateMetrics,
    TrainingConfig,
)
from getting_over_it_rl import train


def small_config(**overrides):
    values = {
        "hidden_sizes": (16,),
        "replay_capacity": 32,
        "batch_size": 2,
        "warmup_steps": 3,
        "updates_per_step": 1,
        "checkpoint_interval": 100,
    }
    values.update(overrides)
    return SACConfig(**values)


def update_metrics():
    return SACUpdateMetrics(
        actor_loss=1.0,
        critic_1_loss=2.0,
        critic_2_loss=3.0,
        entropy_loss=4.0,
        entropy_coefficient=0.2,
        policy_entropy=0.5,
        mean_bellman_target=0.25,
    )


class ScriptedEnvironment(gym.Env):
    def __init__(self, endings=None, fail_at_step=None):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(20,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.endings = endings or {}
        self.fail_at_step = fail_at_step
        self.global_step = 0
        self.episode_step = 0
        self.reset_count = 0
        self.closed = False
        self.actions = []
        self.reset_seeds = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_count += 1
        self.episode_step = 0
        self.reset_seeds.append(seed)
        observation = np.full(20, -self.reset_count / 10, np.float32)
        return observation, {"maximum_height": 0.0}

    def step(self, action):
        self.global_step += 1
        self.episode_step += 1
        if self.fail_at_step == self.global_step:
            raise RuntimeError("scripted environment failure")
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        terminated, truncated, reward = self.endings.get(
            self.global_step, (False, False, 0.1)
        )
        observation = np.full(
            20, self.global_step / 10, dtype=np.float32
        )
        info = {"maximum_height": self.global_step / 20}
        return observation, reward, terminated, truncated, info

    def close(self):
        self.closed = True


def test_learn_uses_warmup_then_policy_and_runs_configured_updates():
    algorithm = SACAlgorithm(
        config=small_config(updates_per_step=2), seed=5
    )
    environment = ScriptedEnvironment()
    predicted_observations = []
    update_batches = []

    def predict(observation, deterministic=True):
        predicted_observations.append(observation.copy())
        assert not deterministic
        return np.array([0.75, -0.25], dtype=np.float32)

    algorithm.predict = predict
    algorithm.updater.update = lambda batch: (
        update_batches.append(batch) or update_metrics()
    )

    summary = algorithm.learn(environment, total_steps=5)

    assert len(environment.actions) == 5
    assert len(predicted_observations) == 2
    np.testing.assert_array_equal(
        environment.actions[3], np.array([0.75, -0.25], np.float32)
    )
    np.testing.assert_array_equal(
        environment.actions[4], np.array([0.75, -0.25], np.float32)
    )
    assert len(update_batches) == 6
    assert algorithm.environment_steps == 5
    assert algorithm.gradient_updates == 6
    assert len(algorithm.replay_buffer) == 5
    assert summary.steps_completed == 5
    assert summary.last_update == update_metrics()


def test_training_loop_runs_real_sac_updates_end_to_end():
    algorithm = SACAlgorithm(
        config=small_config(warmup_steps=2, batch_size=2), seed=2
    )

    summary = algorithm.learn(ScriptedEnvironment(), total_steps=3)

    assert summary.gradient_updates == 2
    assert summary.last_update is not None
    assert math.isfinite(summary.last_update.actor_loss)
    assert math.isfinite(summary.last_update.critic_1_loss)
    assert math.isfinite(summary.last_update.critic_2_loss)


def test_learn_preserves_terminal_observations_and_resets_both_endings():
    endings = {
        2: (True, False, -1.0),
        4: (False, True, 0.25),
    }
    algorithm = SACAlgorithm(
        config=small_config(warmup_steps=10), seed=3
    )
    environment = ScriptedEnvironment(endings=endings)

    summary = algorithm.learn(environment, total_steps=4)
    state = algorithm.replay_buffer.state_dict()

    assert environment.reset_count == 3
    assert environment.reset_seeds == [3, 4, 5]
    assert state["terminated"].reshape(-1).tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert state["truncated"].reshape(-1).tolist() == [
        False,
        False,
        False,
        True,
    ]
    np.testing.assert_array_equal(
        state["next_observations"][1],
        np.full(20, 0.2, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        state["next_observations"][3],
        np.full(20, 0.4, dtype=np.float32),
    )
    assert algorithm.episodes_completed == 2
    assert [item.outcome for item in summary.episode_metrics] == [
        "fall",
        "truncated",
    ]
    assert [item.episode_length for item in summary.episode_metrics] == [2, 2]


def test_positive_terminal_reward_is_tracked_as_success():
    environment = ScriptedEnvironment(
        endings={1: (True, False, 10.0)}
    )
    algorithm = SACAlgorithm(
        config=small_config(warmup_steps=10)
    )

    summary = algorithm.learn(environment, total_steps=1)

    assert summary.episode_metrics[0].outcome == "success"
    assert summary.episode_metrics[0].maximum_height == pytest.approx(0.05)


def test_periodic_checkpoints_follow_global_step_when_resuming(tmp_path):
    checkpoint = tmp_path / "latest"
    algorithm = SACAlgorithm(
        config=small_config(
            warmup_steps=10,
            checkpoint_interval=2,
        )
    )
    algorithm.environment_steps = 1
    saved_at = []
    algorithm.save = lambda path: saved_at.append(
        (Path(path), algorithm.environment_steps)
    )

    algorithm.learn(
        ScriptedEnvironment(),
        total_steps=4,
        checkpoint_path=checkpoint,
    )

    assert saved_at == [(checkpoint, 2), (checkpoint, 4)]


def test_run_training_closes_and_saves_after_environment_failure(
    tmp_path, monkeypatch
):
    environment = ScriptedEnvironment(fail_at_step=2)
    monkeypatch.setattr(
        train.gym, "make", lambda *args, **kwargs: environment
    )
    checkpoint = tmp_path / "latest"
    config = TrainingConfig(
        total_steps=3,
        sac=small_config(warmup_steps=10),
        checkpoint_path=checkpoint,
    )

    with pytest.raises(RuntimeError, match="scripted environment failure"):
        train.run_training(config)

    assert checkpoint.is_file()
    restored = SACAlgorithm.load(checkpoint)
    assert restored.environment_steps == 1
    assert len(restored.replay_buffer) == 1
    assert environment.closed


def test_run_training_loads_resume_checkpoint_and_saves_output(
    tmp_path, monkeypatch
):
    resume_from = tmp_path / "old"
    output = tmp_path / "new"
    source = SACAlgorithm(config=small_config(warmup_steps=10), seed=9)
    source.environment_steps = 4
    source.save(resume_from)
    environment = ScriptedEnvironment()
    monkeypatch.setattr(
        train.gym, "make", lambda *args, **kwargs: environment
    )
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return SACAlgorithm.load(args[0], device=kwargs["device"])

    config = TrainingConfig(
        total_steps=1,
        sac=small_config(warmup_steps=10),
        checkpoint_path=output,
        resume_from=resume_from,
    )
    train.run_training(config, algorithm_factory=factory)

    assert calls == [((resume_from,), {"device": "cpu"})]
    assert SACAlgorithm.load(output).environment_steps == 5
    assert environment.closed


@pytest.mark.parametrize("total_steps", [0, -1, 1.5, True])
def test_learn_rejects_invalid_step_counts(total_steps):
    algorithm = SACAlgorithm(config=small_config())

    with pytest.raises((TypeError, ValueError), match="total_steps"):
        algorithm.learn(ScriptedEnvironment(), total_steps)
