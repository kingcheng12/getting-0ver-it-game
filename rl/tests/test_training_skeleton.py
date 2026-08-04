from pathlib import Path

import pytest

from getting_over_it_rl import (
    EnvironmentConfig,
    EvaluationConfig,
    SACConfig,
    SACAlgorithm,
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


def test_sac_foundation_defaults_to_cpu():
    config = SACConfig()

    assert config.hidden_sizes == (256, 256)
    assert config.replay_capacity == 200_000
    assert config.batch_size == 256
    assert config.initial_entropy_coefficient == 0.2
    assert config.device == "cpu"
    assert config.resolve_device().type == "cpu"


def test_cuda_request_fails_clearly_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        SACConfig(device="cuda").resolve_device()


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
    evaluation = evaluate.parse_config(
        ["--episodes", "3", "--device", "cpu"]
    )

    assert training.environment.worker_id == 2
    assert training.total_steps == 42
    assert training.sac.device == "cpu"
    assert training.checkpoint_path == tmp_path / "model"
    assert evaluation.episodes == 3
    assert evaluation.device == "cpu"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_sizes": ()}, "hidden_sizes"),
        ({"replay_capacity": 0}, "replay_capacity"),
        (
            {"replay_capacity": 10, "batch_size": 11},
            "batch_size",
        ),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"gamma": 1.1}, "gamma"),
        ({"gamma": float("nan")}, "gamma"),
        ({"tau": 0.0}, "tau"),
        ({"actor_learning_rate": 0.0}, "actor_learning_rate"),
        ({"log_std_min": 2.0, "log_std_max": 2.0}, "log_std"),
        ({"target_entropy": float("inf")}, "target_entropy"),
        (
            {"initial_entropy_coefficient": 0.0},
            "initial_entropy_coefficient",
        ),
        ({"checkpoint_interval": 0}, "checkpoint_interval"),
        ({"device": "not-a-device"}, "device"),
    ],
)
def test_sac_configuration_rejects_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        SACConfig(**overrides)


def test_algorithm_factory_constructs_sac():
    algorithm = create_algorithm(
        config=SACConfig(
            hidden_sizes=(8,), replay_capacity=10, batch_size=4
        )
    )

    assert isinstance(algorithm, SACAlgorithm)


def test_training_cli_parses_resume_checkpoint():
    config = train.parse_config(
        ["--resume-from", "old-checkpoint", "--total-steps", "5"]
    )

    assert config.resume_from == Path("old-checkpoint")
    assert config.total_steps == 5


def test_training_cli_parses_weights_only_checkpoint():
    config = train.parse_config(
        ["--initialize-from", "old-checkpoint", "--total-steps", "5"]
    )

    assert config.initialize_from == Path("old-checkpoint")
    assert config.resume_from is None


def test_training_config_rejects_two_checkpoint_modes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingConfig(
            resume_from=Path("resume"),
            initialize_from=Path("initialize"),
        )


def test_evaluation_fails_before_unity_is_opened(monkeypatch):
    monkeypatch.setattr(
        evaluate.gym,
        "make",
        lambda *args, **kwargs: pytest.fail("Unity was opened"),
    )
    def missing_checkpoint(*args, **kwargs):
        raise FileNotFoundError("missing checkpoint")

    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        evaluate.run_evaluation(
            EvaluationConfig(), algorithm_loader=missing_checkpoint
        )
