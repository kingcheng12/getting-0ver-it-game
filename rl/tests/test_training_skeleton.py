from pathlib import Path

import pytest

from getting_over_it_rl import (
    EnvironmentConfig,
    EvaluationConfig,
    TrainingConfig,
    create_algorithm,
)
from getting_over_it_rl import evaluate, train


def test_environment_defaults_match_gymnasium_wrapper():
    config = EnvironmentConfig()
    assert config.file_name is None
    assert config.worker_id == 0
    assert config.timeout_wait == 300
    assert config.max_episode_steps == 1250
    assert config.render_mode == "human"


def test_checkpoint_defaults_stay_under_rl_directory():
    training_path = TrainingConfig().checkpoint_path
    evaluation_path = EvaluationConfig().checkpoint_path

    assert training_path == evaluation_path
    assert training_path.name == "latest"
    assert training_path.parent.name == "checkpoints"
    assert training_path.parent.parent.name == "rl"


def test_cli_overrides_create_typed_configs(tmp_path: Path):
    training = train.parse_config(
        [
            "--worker-id",
            "2",
            "--total-steps",
            "42",
            "--checkpoint-path",
            str(tmp_path / "model"),
        ]
    )
    evaluation = evaluate.parse_config(["--episodes", "3"])

    assert training.environment.worker_id == 2
    assert training.total_steps == 42
    assert training.checkpoint_path == tmp_path / "model"
    assert evaluation.episodes == 3


def test_algorithm_factory_explains_next_step():
    with pytest.raises(NotImplementedError, match="PPO or SAC"):
        create_algorithm()


def test_training_fails_before_unity_is_opened(monkeypatch):
    monkeypatch.setattr(
        train.gym,
        "make",
        lambda *args, **kwargs: pytest.fail("Unity was opened"),
    )
    with pytest.raises(NotImplementedError, match="create_algorithm"):
        train.main([])


def test_evaluation_fails_before_unity_is_opened(monkeypatch):
    monkeypatch.setattr(
        evaluate.gym,
        "make",
        lambda *args, **kwargs: pytest.fail("Unity was opened"),
    )
    with pytest.raises(NotImplementedError, match="create_algorithm"):
        evaluate.main([])
