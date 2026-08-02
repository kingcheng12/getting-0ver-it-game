from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import torch

from getting_over_it_rl import (
    EvaluationConfig,
    EvaluationSummary,
    SACAlgorithm,
    SACConfig,
)
from getting_over_it_rl import evaluate


def small_config():
    return SACConfig(
        hidden_sizes=(16,), replay_capacity=16, batch_size=2
    )


class EvaluationEnvironment(gym.Env):
    def __init__(self, scripts):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, (20,), np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (2,), np.float32
        )
        self.scripts = scripts
        self.episode = -1
        self.episode_step = 0
        self.seeds = []
        self.actions = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.episode_step = 0
        self.seeds.append(seed)
        return np.zeros(20, np.float32), {"maximum_height": 0.0}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        transition = self.scripts[self.episode][self.episode_step]
        self.episode_step += 1
        reward, terminated, truncated, height = transition
        observation = np.full(
            20, self.episode_step / 10, dtype=np.float32
        )
        return (
            observation,
            reward,
            terminated,
            truncated,
            {"maximum_height": height},
        )

    def close(self):
        self.closed = True


def snapshot_parameters(algorithm):
    return {
        key: value.detach().clone()
        for key, value in algorithm.networks.state_dict().items()
    }


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert set(first) == set(second)
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, list):
        assert len(first) == len(second)
        for first_item, second_item in zip(first, second):
            assert_nested_equal(first_item, second_item)
    else:
        assert first == second


def test_evaluation_reports_episode_and_aggregate_metrics(
    tmp_path, monkeypatch, capsys
):
    scripts = [
        [(0.5, False, False, 0.1), (10.0, True, False, 0.8)],
        [(0.25, False, False, 0.2), (-1.0, True, False, 0.3)],
        [(0.1, False, True, 0.4)],
    ]
    environment = EvaluationEnvironment(scripts)
    monkeypatch.setattr(
        evaluate.gym, "make", lambda *args, **kwargs: environment
    )
    checkpoint = tmp_path / "agent"
    SACAlgorithm(config=small_config()).save(checkpoint)
    config = EvaluationConfig(
        episodes=3, seed=7, checkpoint_path=checkpoint
    )

    summary = evaluate.run_evaluation(config)

    assert isinstance(summary, EvaluationSummary)
    assert environment.seeds == [7, 8, 9]
    assert [episode.outcome for episode in summary.episodes] == [
        "success",
        "fall",
        "truncated",
    ]
    assert [episode.episode_length for episode in summary.episodes] == [
        2,
        2,
        1,
    ]
    assert summary.mean_return == pytest.approx((10.5 - 0.75 + 0.1) / 3)
    assert summary.mean_episode_length == pytest.approx(5 / 3)
    assert summary.mean_maximum_height == pytest.approx(0.5)
    assert summary.best_maximum_height == pytest.approx(0.8)
    assert summary.success_rate == pytest.approx(1 / 3)
    assert summary.fall_rate == pytest.approx(1 / 3)
    assert summary.truncation_rate == pytest.approx(1 / 3)
    assert environment.closed
    output = capsys.readouterr().out
    assert "evaluation_episode=1" in output
    assert "outcome=success" in output
    assert "evaluation_summary episodes=3" in output


def test_evaluation_does_not_modify_training_state(tmp_path, monkeypatch):
    algorithm = SACAlgorithm(config=small_config(), seed=3)
    observation = np.zeros(20, np.float32)
    algorithm.replay_buffer.add(
        observation,
        np.zeros(2, np.float32),
        0.0,
        observation,
        False,
        False,
    )
    algorithm.environment_steps = 9
    algorithm.gradient_updates = 4
    algorithm.episodes_completed = 2
    checkpoint = tmp_path / "agent"
    algorithm.save(checkpoint)
    loaded = SACAlgorithm.load(checkpoint)
    parameters_before = snapshot_parameters(loaded)
    updater_before = deepcopy(loaded.updater.state_dict())
    replay_before = deepcopy(loaded.replay_buffer.state_dict())
    counters_before = (
        loaded.environment_steps,
        loaded.gradient_updates,
        loaded.episodes_completed,
    )
    environment = EvaluationEnvironment(
        [[(1.0, True, False, 0.5)], [(1.0, True, False, 0.5)]]
    )
    monkeypatch.setattr(
        evaluate.gym, "make", lambda *args, **kwargs: environment
    )
    evaluate.run_evaluation(
        EvaluationConfig(episodes=2, checkpoint_path=checkpoint),
        algorithm_loader=lambda *args, **kwargs: loaded,
    )

    for key, value in loaded.networks.state_dict().items():
        torch.testing.assert_close(value, parameters_before[key])
    assert_nested_equal(loaded.updater.state_dict(), updater_before)
    assert_nested_equal(loaded.replay_buffer.state_dict(), replay_before)
    assert counters_before == (
        loaded.environment_steps,
        loaded.gradient_updates,
        loaded.episodes_completed,
    )
    np.testing.assert_array_equal(
        environment.actions[0], environment.actions[1]
    )


def test_evaluation_closes_environment_after_step_failure(
    tmp_path, monkeypatch
):
    class FailingEnvironment(EvaluationEnvironment):
        def step(self, action):
            raise RuntimeError("evaluation failure")

    environment = FailingEnvironment([[(0.0, True, False, 0.0)]])
    monkeypatch.setattr(
        evaluate.gym, "make", lambda *args, **kwargs: environment
    )
    checkpoint = tmp_path / "agent"
    SACAlgorithm(config=small_config()).save(checkpoint)

    with pytest.raises(RuntimeError, match="evaluation failure"):
        evaluate.run_evaluation(
            EvaluationConfig(episodes=1, checkpoint_path=checkpoint)
        )

    assert environment.closed


@pytest.mark.parametrize(
    "transition",
    [
        (float("nan"), True, False, 0.0),
        (0.0, 1, False, 0.0),
        (0.0, True, True, 0.0),
    ],
)
def test_evaluation_rejects_invalid_environment_results(
    tmp_path, monkeypatch, transition
):
    environment = EvaluationEnvironment([[transition]])
    monkeypatch.setattr(
        evaluate.gym, "make", lambda *args, **kwargs: environment
    )
    checkpoint = tmp_path / "agent"
    SACAlgorithm(config=small_config()).save(checkpoint)

    with pytest.raises((TypeError, ValueError)):
        evaluate.run_evaluation(
            EvaluationConfig(episodes=1, checkpoint_path=checkpoint)
        )

    assert environment.closed


def test_evaluation_cli_explains_missing_checkpoint(tmp_path):
    checkpoint = tmp_path / "missing"

    with pytest.raises(SystemExit, match="Train the agent first"):
        evaluate.main(
            ["--checkpoint-path", str(checkpoint), "--episodes", "1"]
        )
