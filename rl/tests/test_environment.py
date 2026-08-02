from dataclasses import dataclass
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import getting_over_it_env  # noqa: F401
from getting_over_it_env import GettingOverItEnv


@dataclass
class FakeStep:
    obs: list
    reward: float = 0.0
    interrupted: bool = False


class FakeSteps:
    def __init__(self, values=None):
        self.values = values or {}

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __contains__(self, agent_id):
        return agent_id in self.values

    def __getitem__(self, agent_id):
        return self.values[agent_id]


class FakeUnityEnvironment:
    behavior_name = "GettingOverIt?team=0"

    def __init__(self):
        self.behavior_specs = {
            self.behavior_name: SimpleNamespace(
                observation_specs=[SimpleNamespace(shape=(20,))],
                action_spec=SimpleNamespace(
                    continuous_size=2, discrete_size=0
                ),
            )
        }
        self.observation = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
        self.reward = 0.25
        self.terminal = False
        self.interrupted = False
        self.closed = False
        self.reset_count = 0
        self.last_action = None

    def reset(self):
        self.reset_count += 1
        self.terminal = False
        self.interrupted = False

    def set_action_for_agent(self, behavior_name, agent_id, action):
        assert behavior_name == self.behavior_name
        assert agent_id == 17
        self.last_action = action

    def step(self):
        pass

    def get_steps(self, behavior_name):
        assert behavior_name == self.behavior_name
        result = FakeStep(
            obs=[self.observation.copy()],
            reward=self.reward,
            interrupted=self.interrupted,
        )
        if self.terminal:
            return FakeSteps(), FakeSteps({17: result})
        return FakeSteps({17: result}), FakeSteps()

    def close(self):
        self.closed = True


def make_env(**kwargs):
    backend = FakeUnityEnvironment()
    env = GettingOverItEnv(_unity_env=backend, **kwargs)
    return env, backend


def test_reset_flattens_observation_and_seeds_python():
    env, backend = make_env()
    observation, info = env.reset(seed=123)

    assert observation.shape == (20,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["agent_id"] == 17
    assert info["episode_steps"] == 0
    assert backend.reset_count == 1
    env.close()


def test_step_clips_action_and_maps_normal_step():
    env, backend = make_env()
    env.reset()
    observation, reward, terminated, truncated, info = env.step(
        np.array([4.0, -3.0], dtype=np.float32)
    )

    np.testing.assert_array_equal(
        backend.last_action.continuous,
        np.array([[1.0, -1.0]], dtype=np.float32),
    )
    assert env.observation_space.contains(observation)
    assert reward == pytest.approx(0.25)
    assert not terminated
    assert not truncated
    assert info["episode_steps"] == 1
    env.close()


@pytest.mark.parametrize(
    ("interrupted", "expected"),
    [(False, (True, False)), (True, (False, True))],
)
def test_terminal_mapping(interrupted, expected):
    env, backend = make_env()
    env.reset()
    backend.terminal = True
    backend.interrupted = interrupted

    _, _, terminated, truncated, _ = env.step(
        np.zeros(2, dtype=np.float32)
    )
    assert (terminated, truncated) == expected
    env.close()


def test_python_step_limit_maps_to_truncation():
    env, _ = make_env(max_episode_steps=2)
    env.reset()
    assert env.step(np.zeros(2, dtype=np.float32))[3] is False
    assert env.step(np.zeros(2, dtype=np.float32))[3] is True
    env.close()


def test_close_is_idempotent_and_prevents_reuse():
    env, backend = make_env()
    env.close()
    env.close()
    assert backend.closed
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_wrong_observation_size_is_rejected():
    env, backend = make_env()
    backend.behavior_specs[backend.behavior_name].observation_specs = [
        SimpleNamespace(shape=(19,))
    ]
    with pytest.raises(RuntimeError, match="19 observation"):
        env.reset()
    env.close()


def test_registered_environment_and_gymnasium_checker():
    backend = FakeUnityEnvironment()
    env = gym.make(
        "GettingOverItUnity-v0",
        _unity_env=backend,
        disable_env_checker=True,
    ).unwrapped
    check_env(env, skip_render_check=True)
    env.close()
