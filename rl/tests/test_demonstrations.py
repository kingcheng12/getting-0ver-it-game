from pathlib import Path

import numpy as np
import pytest

from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import (
    BrainParametersProto,
)
from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)

from getting_over_it_rl.demonstrations import (
    DemonstrationBuffer,
    load_demonstration_file,
    select_demonstrations,
)


def _varint(value):
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _delimited(message):
    payload = message.SerializeToString()
    return _varint(len(payload)) + payload


def write_demo(path: Path, episodes=((70, "success"),)):
    records = []
    agent_id = 0
    for length, outcome in episodes:
        for step in range(length + 1):
            pair = AgentInfoActionPairProto()
            pair.agent_info.id = agent_id
            observation = pair.agent_info.observations.add()
            observation.shape.append(20)
            observation.float_data.data.extend(
                np.full(20, step / 100.0, dtype=np.float32)
            )
            if step > 0:
                pair.action_info.continuous_actions.extend(
                    [step / 100.0, -step / 100.0]
                )
                pair.agent_info.reward = 0.1
            if step == length:
                if outcome == "truncated":
                    pair.agent_info.max_step_reached = True
                    pair.agent_info.done = True
                elif outcome != "incomplete":
                    pair.agent_info.done = True
                    pair.agent_info.reward = 10.0 if outcome == "success" else -1.0
            records.append(pair)
        agent_id += 1

    metadata = DemonstrationMetaProto(
        api_version=1,
        demonstration_name="GOIHuman",
        number_steps=len(records),
        number_episodes=len(episodes),
    )
    brain = BrainParametersProto(brain_name="GettingOverIt?team=0")
    brain.action_spec.num_continuous_actions = 2
    header = _delimited(metadata)
    assert len(header) <= 33
    path.write_bytes(
        header
        + bytes(33 - len(header))
        + _delimited(brain)
        + b"".join(_delimited(record) for record in records)
        + metadata.SerializeToString()
    )
    return path


def test_release_17_temporal_alignment_and_terminal_mapping(tmp_path):
    demo = load_demonstration_file(
        write_demo(tmp_path / "sample.demo", ((3, "success"), (2, "truncated")))
    )
    first, second = demo.episodes
    assert first.length == 3
    np.testing.assert_allclose(first.observations[0], 0.0)
    np.testing.assert_allclose(first.next_observations[0], 0.01)
    np.testing.assert_allclose(first.actions[0], [0.01, -0.01])
    assert first.outcome == "successful"
    assert first.terminated[-1, 0] and not first.truncated[-1, 0]
    assert second.outcome == "truncated"
    assert second.truncated[-1, 0] and not second.terminated[-1, 0]


def test_selection_filter_explicit_override_and_deduplication(tmp_path):
    path = write_demo(
        tmp_path / "selection.demo",
        ((70, "success"), (70, "fall")),
    )
    successful = select_demonstrations((path,), outcome_filter="successful")
    assert len(successful) == 70
    loaded = load_demonstration_file(path)
    assert [episode.outcome for episode in loaded.episodes] == [
        "successful",
        "fall",
    ]
    all_selected = select_demonstrations(
        (path,), ((path, 1), (path, 1)), outcome_filter="successful"
    )
    assert len(all_selected) == 140
    with pytest.raises(ValueError, match="at least 100"):
        select_demonstrations(
            episode_selections=((path, 1),), minimum_transitions=100
        )


def test_incomplete_episode_is_preserved(tmp_path):
    demo = load_demonstration_file(
        write_demo(tmp_path / "partial.demo", ((5, "incomplete"),))
    )
    assert demo.episodes[0].outcome == "incomplete"
    assert demo.episodes[0].length == 5
    assert not demo.episodes[0].terminated.any()
    assert not demo.episodes[0].truncated.any()


def test_buffer_checkpoint_preserves_metadata_and_sampling_rng(tmp_path):
    demo = load_demonstration_file(write_demo(tmp_path / "state.demo"))
    buffer = DemonstrationBuffer(demo.episodes, seed=4)
    state = buffer.state_dict()
    restored = DemonstrationBuffer.from_state_dict(state)
    assert restored.episodes == buffer.episodes
    np.testing.assert_array_equal(
        buffer.sample(8).actions.numpy(), restored.sample(8).actions.numpy()
    )


def test_invalid_demo_is_rejected(tmp_path):
    path = tmp_path / "bad.demo"
    path.write_bytes(b"not a demonstration")
    with pytest.raises(ValueError, match="Invalid Release 17"):
        load_demonstration_file(path)
