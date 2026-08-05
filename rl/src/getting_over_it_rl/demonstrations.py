from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np

from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import (
    BrainParametersProto,
)
from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from google.protobuf.message import DecodeError

from .replay_buffer import Device, ReplayBatch, ReplayBuffer


DEMONSTRATION_API_VERSION = 1
DEMONSTRATION_METADATA_BYTES = 32
OBSERVATION_DIMENSION = 20
ACTION_DIMENSION = 2
BEHAVIOR_NAME = "GettingOverIt"


@dataclass(frozen=True)
class DemonstrationSource:
    path: str
    sha256: str


@dataclass(frozen=True)
class DemonstrationEpisode:
    source: DemonstrationSource
    index: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    outcome: str

    @property
    def length(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def episode_return(self) -> float:
        return float(self.rewards.sum(dtype=np.float64))


@dataclass(frozen=True)
class DemonstrationEpisodeSummary:
    source: DemonstrationSource
    index: int
    length: int
    episode_return: float
    outcome: str


@dataclass(frozen=True)
class DemonstrationFile:
    source: DemonstrationSource
    name: str
    episodes: Tuple[DemonstrationEpisode, ...]
    recorded_steps: int


class DemonstrationBuffer:
    """Immutable replay data loaded from selected human demonstrations."""

    STATE_VERSION = 1

    def __init__(
        self,
        episodes: Sequence[DemonstrationEpisode] = (),
        seed: Optional[int] = None,
    ) -> None:
        episodes = tuple(episodes)
        self._episodes = tuple(
            DemonstrationEpisodeSummary(
                episode.source,
                episode.index,
                episode.length,
                episode.episode_return,
                episode.outcome,
            )
            for episode in episodes
        )
        transition_count = sum(episode.length for episode in episodes)
        self._replay = ReplayBuffer(
            capacity=max(1, transition_count),
            observation_shape=(OBSERVATION_DIMENSION,),
            action_shape=(ACTION_DIMENSION,),
            seed=seed,
        )
        for episode in episodes:
            for index in range(episode.length):
                self._replay.add(
                    episode.observations[index],
                    episode.actions[index],
                    float(episode.rewards[index, 0]),
                    episode.next_observations[index],
                    bool(episode.terminated[index, 0]),
                    bool(episode.truncated[index, 0]),
                )

    @property
    def episodes(self) -> Tuple[DemonstrationEpisodeSummary, ...]:
        return self._episodes

    def __len__(self) -> int:
        return len(self._replay)

    def sample(self, batch_size: int, device: Device = "cpu") -> ReplayBatch:
        return self._replay.sample(batch_size, device=device)

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "replay": self._replay.state_dict(),
            "episodes": tuple(
                {
                    "source": {
                        "path": episode.source.path,
                        "sha256": episode.source.sha256,
                    },
                    "index": episode.index,
                    "length": episode.length,
                    "return": episode.episode_return,
                    "outcome": episode.outcome,
                }
                for episode in self._episodes
            ),
        }

    @classmethod
    def from_state_dict(cls, state: object) -> "DemonstrationBuffer":
        if not isinstance(state, dict) or set(state) != {
            "version",
            "replay",
            "episodes",
        }:
            raise ValueError("Invalid demonstration-buffer checkpoint state")
        if state["version"] != cls.STATE_VERSION:
            raise ValueError("Unsupported demonstration-buffer version")
        replay_state = state["replay"]
        if not isinstance(replay_state, dict):
            raise ValueError("Invalid demonstration replay state")
        metadata = state["episodes"]
        if not isinstance(metadata, tuple):
            raise ValueError("Invalid demonstration episode metadata")

        buffer = cls()
        capacity = replay_state.get("capacity")
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Invalid demonstration replay capacity")
        buffer._replay = ReplayBuffer(
            capacity=capacity,
            observation_shape=(OBSERVATION_DIMENSION,),
            action_shape=(ACTION_DIMENSION,),
        )
        buffer._replay.load_state_dict(replay_state)
        buffer._episodes = tuple(_metadata_summary(item) for item in metadata)
        if sum(episode.length for episode in buffer._episodes) != len(buffer):
            raise ValueError("Demonstration metadata length is incompatible")
        return buffer


def load_demonstration_file(path: Path) -> DemonstrationFile:
    demo_path = Path(path)
    if not demo_path.is_file():
        raise FileNotFoundError(f"Demonstration file does not exist: {demo_path}")
    source = DemonstrationSource(
        path=str(demo_path.resolve()),
        sha256=hashlib.sha256(demo_path.read_bytes()).hexdigest(),
    )
    try:
        with demo_path.open("rb") as stream:
            metadata = _read_delimited(stream, DemonstrationMetaProto)
            if metadata is None:
                raise ValueError("missing metadata")
            _validate_metadata(metadata)
            stream.seek(DEMONSTRATION_METADATA_BYTES + 1)
            brain = _read_delimited(stream, BrainParametersProto)
            if brain is None:
                raise ValueError("missing behavior parameters")
            _validate_brain(brain)
            pairs = _read_pairs(stream, metadata.number_steps)
            trailing_data = stream.read()
            if trailing_data not in (b"", metadata.SerializeToString()):
                raise ValueError("unexpected data after demonstration records")
    except (OSError, ValueError, DecodeError) as error:
        raise ValueError(f"Invalid Release 17 demonstration: {demo_path}") from error

    if metadata.number_steps != len(pairs):
        raise ValueError(
            f"Demonstration step count mismatch in {demo_path}: "
            f"metadata={metadata.number_steps}, records={len(pairs)}"
        )
    episodes = _build_episodes(source, pairs)
    return DemonstrationFile(
        source=source,
        name=metadata.demonstration_name,
        episodes=episodes,
        recorded_steps=len(pairs),
    )


def select_demonstrations(
    files: Sequence[Path] = (),
    episode_selections: Sequence[Tuple[Path, int]] = (),
    outcome_filter: str = "successful",
    seed: Optional[int] = None,
    minimum_transitions: int = 64,
) -> DemonstrationBuffer:
    if outcome_filter not in {"successful", "non-fall", "all"}:
        raise ValueError("demo filter must be successful, non-fall, or all")
    loaded: Dict[str, DemonstrationFile] = {}

    def load(path: Path) -> DemonstrationFile:
        resolved = str(Path(path).resolve())
        if resolved not in loaded:
            loaded[resolved] = load_demonstration_file(Path(path))
        return loaded[resolved]

    selected: Dict[Tuple[str, int], DemonstrationEpisode] = {}
    for path in files:
        demonstration = load(path)
        for episode in demonstration.episodes:
            if _matches_filter(episode.outcome, outcome_filter):
                selected[(episode.source.path, episode.index)] = episode
    for path, episode_index in episode_selections:
        demonstration = load(path)
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise TypeError("demonstration episode index must be an integer")
        if episode_index < 0 or episode_index >= len(demonstration.episodes):
            raise ValueError(
                f"Demonstration episode {episode_index} does not exist in {path}"
            )
        episode = demonstration.episodes[episode_index]
        selected[(episode.source.path, episode.index)] = episode

    buffer = DemonstrationBuffer(tuple(selected.values()), seed=seed)
    if len(buffer) < minimum_transitions:
        raise ValueError(
            f"Selected demonstrations contain {len(buffer)} transitions; "
            f"at least {minimum_transitions} are required"
        )
    return buffer


def _read_delimited(stream: BinaryIO, message_type: Type[object]):
    size = _read_varint(stream)
    if size is None:
        return None
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError("truncated protobuf message")
    message = message_type()
    message.ParseFromString(payload)
    return message


def _read_varint(stream: BinaryIO) -> Optional[int]:
    value = 0
    shift = 0
    while shift < 64:
        byte = stream.read(1)
        if not byte:
            return None if shift == 0 else _raise_truncated_varint()
        current = byte[0]
        value |= (current & 0x7F) << shift
        if not current & 0x80:
            return value
        shift += 7
    raise ValueError("protobuf length varint is too long")


def _raise_truncated_varint():
    raise ValueError("truncated protobuf length")


def _validate_metadata(metadata: DemonstrationMetaProto) -> None:
    if metadata.api_version != DEMONSTRATION_API_VERSION:
        raise ValueError("unsupported demonstration API version")
    if not metadata.demonstration_name:
        raise ValueError("demonstration name is empty")
    if metadata.number_steps <= 0:
        raise ValueError("demonstration contains no recorded steps")


def _validate_brain(brain: BrainParametersProto) -> None:
    if brain.brain_name.split("?")[0] != BEHAVIOR_NAME:
        raise ValueError("demonstration behavior is incompatible")
    if (
        brain.action_spec.num_continuous_actions != ACTION_DIMENSION
        or brain.action_spec.num_discrete_actions != 0
        or len(brain.action_spec.discrete_branch_sizes) != 0
    ):
        raise ValueError("demonstration action specification is incompatible")


def _read_pairs(
    stream: BinaryIO, number_steps: int
) -> List[AgentInfoActionPairProto]:
    pairs = []
    for _ in range(number_steps):
        pair = _read_delimited(stream, AgentInfoActionPairProto)
        if pair is None:
            raise ValueError("demonstration records are truncated")
        pairs.append(pair)
    return pairs


def _observation(pair: AgentInfoActionPairProto) -> np.ndarray:
    values = []
    for observation in pair.agent_info.observations:
        if observation.compression_type != 0 or observation.compressed_data:
            raise ValueError("compressed demonstration observations are unsupported")
        shape_size = int(np.prod(tuple(observation.shape), dtype=np.int64))
        data = np.asarray(observation.float_data.data, dtype=np.float32)
        if shape_size != data.size:
            raise ValueError("demonstration observation shape is invalid")
        values.append(data.reshape(-1))
    if not values:
        raise ValueError("demonstration record has no observations")
    flattened = np.concatenate(values).astype(np.float32, copy=False)
    if flattened.shape != (OBSERVATION_DIMENSION,):
        raise ValueError("demonstration observation dimension is incompatible")
    if not np.all(np.isfinite(flattened)):
        raise ValueError("demonstration observation contains non-finite values")
    if np.any(flattened < -1.0) or np.any(flattened > 1.0):
        raise ValueError("demonstration observation is outside [-1, 1]")
    return flattened.copy()


def _action(pair: AgentInfoActionPairProto) -> np.ndarray:
    action = np.asarray(pair.action_info.continuous_actions, dtype=np.float32)
    if action.shape != (ACTION_DIMENSION,):
        raise ValueError("demonstration action dimension is incompatible")
    if not np.all(np.isfinite(action)):
        raise ValueError("demonstration action contains non-finite values")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError("demonstration action is outside [-1, 1]")
    if len(pair.action_info.discrete_actions) != 0:
        raise ValueError("demonstration contains discrete actions")
    return action.copy()


def _build_episodes(
    source: DemonstrationSource,
    pairs: Sequence[AgentInfoActionPairProto],
) -> Tuple[DemonstrationEpisode, ...]:
    episodes = []
    transitions = []
    previous_observation = None
    previous_id = None
    episode_index = 0

    for pair in pairs:
        current_observation = _observation(pair)
        current_id = int(pair.agent_info.id)
        if previous_observation is not None and current_id == previous_id:
            reward = float(pair.agent_info.reward + pair.agent_info.group_reward)
            if not np.isfinite(reward):
                raise ValueError("demonstration reward is non-finite")
            truncated = bool(pair.agent_info.max_step_reached)
            terminated = bool(pair.agent_info.done) and not truncated
            transitions.append(
                (
                    previous_observation,
                    _action(pair),
                    reward,
                    current_observation,
                    terminated,
                    truncated,
                )
            )

        if pair.agent_info.done:
            if transitions:
                episodes.append(
                    _episode_from_transitions(
                        source, episode_index, transitions
                    )
                )
                episode_index += 1
            transitions = []
            previous_observation = None
            previous_id = None
        else:
            if previous_id is not None and current_id != previous_id:
                if transitions:
                    episodes.append(
                        _episode_from_transitions(
                            source, episode_index, transitions
                        )
                    )
                    episode_index += 1
                transitions = []
            previous_observation = current_observation
            previous_id = current_id

    if transitions:
        episodes.append(
            _episode_from_transitions(source, episode_index, transitions)
        )
    return tuple(episodes)


def _episode_from_transitions(
    source: DemonstrationSource,
    index: int,
    transitions: Sequence[
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, bool]
    ],
) -> DemonstrationEpisode:
    observations = np.stack([item[0] for item in transitions]).astype(np.float32)
    actions = np.stack([item[1] for item in transitions]).astype(np.float32)
    rewards = np.asarray(
        [item[2] for item in transitions], dtype=np.float32
    ).reshape(-1, 1)
    next_observations = np.stack([item[3] for item in transitions]).astype(np.float32)
    terminated = np.asarray(
        [item[4] for item in transitions], dtype=np.bool_
    ).reshape(-1, 1)
    truncated = np.asarray(
        [item[5] for item in transitions], dtype=np.bool_
    ).reshape(-1, 1)
    if bool(truncated[-1, 0]):
        outcome = "truncated"
    elif bool(terminated[-1, 0]):
        outcome = "successful" if float(rewards[-1, 0]) > 0.0 else "fall"
    else:
        outcome = "incomplete"
    return DemonstrationEpisode(
        source=source,
        index=index,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminated=terminated,
        truncated=truncated,
        outcome=outcome,
    )


def _matches_filter(outcome: str, outcome_filter: str) -> bool:
    if outcome_filter == "all":
        return True
    if outcome_filter == "successful":
        return outcome == "successful"
    return outcome != "fall"


def _metadata_summary(item: object) -> DemonstrationEpisodeSummary:
    if not isinstance(item, dict) or set(item) != {
        "source",
        "index",
        "length",
        "return",
        "outcome",
    }:
        raise ValueError("Invalid demonstration episode metadata")
    source = item["source"]
    if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
        raise ValueError("Invalid demonstration source metadata")
    if not isinstance(source["path"], str) or not source["path"]:
        raise ValueError("Invalid demonstration source path")
    if (
        not isinstance(source["sha256"], str)
        or len(source["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in source["sha256"])
    ):
        raise ValueError("Invalid demonstration source hash")
    index = item["index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Invalid demonstration episode index")
    length = item["length"]
    if not isinstance(length, int) or length < 0:
        raise ValueError("Invalid demonstration episode length")
    episode_return = item["return"]
    if isinstance(episode_return, bool) or not np.isscalar(episode_return):
        raise ValueError("Invalid demonstration episode return")
    episode_return = float(episode_return)
    if not np.isfinite(episode_return):
        raise ValueError("Invalid demonstration episode return")
    outcome = item["outcome"]
    if outcome not in {"successful", "fall", "truncated", "incomplete"}:
        raise ValueError("Invalid demonstration episode outcome")
    return DemonstrationEpisodeSummary(
        source=DemonstrationSource(str(source["path"]), str(source["sha256"])),
        index=index,
        length=length,
        episode_return=episode_return,
        outcome=outcome,
    )
