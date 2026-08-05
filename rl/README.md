# Unity–Gymnasium reinforcement learning

This package connects Gymnasium to `MainScene` through ML-Agents Release 17
and provides the repository's SAC training, evaluation, and human
demonstration replay tools.

## Requirements

- Unity Editor **2020.1.8f1**
- `uv`
- The Unity project opened at this repository root
- A local, trusted machine. The Editor communicator listens on localhost port
  5004, allows one connection at a time, and is not authenticated.

## One-time Unity setup

1. Let Unity finish importing the ML-Agents package.
2. Confirm the Console has no red compile errors.
3. Open `Assets/Scenes/MainScene.unity`.
4. Choose **RL > Configure Main Scene**.
5. You may run the command again; it updates the existing components without
   creating duplicates.
6. Optionally choose **RL > Validate Main Scene**.

The setup adds `GettingOverItAgent`, `BehaviorParameters`,
`DecisionRequester`, and `DemonstrationRecorder` to `Player`. It configures
behavior `GettingOverIt` with 20 observations, two continuous actions, and a
decision every four physics ticks. Demonstration recording is disabled by
default.

With no Python process connected, Play mode keeps using mouse control and does
not automatically end or reset episodes.

## Install the Python environment

From the repository root:

```powershell
uv sync --project rl --dev
```

`uv` installs the pinned Python 3.8.20 interpreter and the locked dependencies
inside `rl/.venv`.

## Communication smoke test

The startup order matters:

1. Make sure Unity is not already in Play mode.
2. Start Python:

   ```powershell
   uv run --project rl python -m getting_over_it_env.smoke --steps 250
   ```

3. When the script says it is waiting, return to Unity and press **Play**
   before the 300-second timeout.
4. Python prints the 20-value reset observation, sends random actions, and
   closes the socket when it finishes.
5. Stop Play mode if Unity remains in it.

Example API:

```python
import gymnasium as gym
import getting_over_it_env  # registers GettingOverItUnity-v0

env = gym.make("GettingOverItUnity-v0")
observation, info = env.reset(seed=7)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

For Editor Play mode, leave `file_name=None` (the default). A later standalone
Unity build can be supplied as `file_name="path/to/executable"` without
changing the rest of the API.

`reset(seed=...)` seeds Gymnasium/Python. The current Unity environment has no
gameplay randomness; Unity physics is deterministic for a fixed action
sequence, while the character's blinking is cosmetic.

## Spaces and episode semantics

- Action: `Box(-1, 1, shape=(2,), dtype=float32)`, normalized hammer direction
- Observation: `Box(-1, 1, shape=(20,), dtype=float32)`
- Reaching waypoint 1 gives `+1` and activates waypoint 2 without ending the
  episode
- Reaching waypoint 2 gives an exact `+10` and is the only successful
  terminal condition
- Falling to `y <= -4.0` gives an exact `-1` terminal reward
- Each unit of new maximum height gives `+0.25`; crossing `y=6.5` does not
  provide a separate bonus or end the episode
- Moving toward the active waypoint gives `0.5 ×` the distance reduction,
  while moving away produces the corresponding negative shaping reward
- Every Unity physics tick gives `-0.0001`
- ML-Agents interrupted terminal steps and the Python 1,250-step limit map to
  `truncated=True`
- `render()` returns `None`; the Unity Game view is the human renderer

Checkpoints made before the waypoint-2-only reward change remain loadable,
but their replay buffers contain obsolete height-goal transitions. Transfer
the network weights with `--initialize-from` instead of fully resuming those
checkpoints:

```powershell
uv run --project rl goi-train `
  --initialize-from rl/checkpoints/latest `
  --checkpoint-path rl/checkpoints/waypoint2-latest `
  --total-steps 100000 `
  --device cuda
```

Training normally uses 5,000 random-action warm-up steps. An explicit
`--warmup-steps` value overrides that setting for new, initialized, or resumed
training and is persisted in the output checkpoint. Omitting the option while
resuming preserves the checkpoint's saved value. A checkpoint that already
has enough online replay can start policy actions and updates immediately:

```powershell
uv run --project rl goi-train `
  --resume-from rl/checkpoints/demo-latest `
  --checkpoint-path rl/checkpoints/demo-latest `
  --warmup-steps 0 `
  --max-episode-steps 2500 `
  --total-steps 100000 `
  --device cuda
```

## Human demonstration replay

Demonstrations are Release 17 `.demo` files recorded from mouse-controlled
gameplay. They are stored beneath the ignored `rl/demonstrations/` directory
and are not committed to Git.

### Record gameplay

1. Make sure Python is not connected and Unity is not in Play mode.
2. Choose **RL > Demonstrations > Enable Recording**.
3. Press **Play** and control the hammer with the mouse. Rewards, waypoint
   progression, terminal conditions, and episode resets remain active while
   recording.
4. Stop Play mode after recording the desired episodes.
5. Choose **RL > Demonstrations > Disable Recording** to restore behavior type
   `Default`.

Unity writes `rl/demonstrations/GOIHuman.demo`. If that file already exists,
ML-Agents adds a numeric suffix instead of overwriting it. Recording and a
Python communicator cannot be active simultaneously.

### Inspect and select episodes

Inspect one or more files before training:

```powershell
uv run --project rl goi-demo-inspect rl/demonstrations/GOIHuman.demo
```

The inspector reports every episode's zero-based index, length, return,
terminal type, and transition count. Outcomes are `successful`, `fall`,
`truncated`, or `incomplete`.

`--demonstration FILE` imports eligible episodes from the entire file. The
default filter is `successful`; use `--demo-filter non-fall` or
`--demo-filter all` to change it. To select a particular episode regardless
of outcome, use `--demo-episode FILE INDEX`. Both options can be repeated.
Duplicate file/episode selections are removed automatically, and training
fails before opening Unity if the files are incompatible or the selection has
fewer than 64 transitions.

### Resume training with demonstrations

Initialize from existing network weights while writing assisted training to a
new checkpoint:

```powershell
uv run --project rl goi-train `
  --initialize-from rl/checkpoints/latest `
  --demonstration rl/demonstrations/GOIHuman.demo `
  --demo-filter successful `
  --checkpoint-path rl/checkpoints/demo-latest `
  --total-steps 100000 `
  --device cuda
```

For the default batch size of 256, every assisted SAC update contains 64
immutable demonstration transitions and 192 online replay transitions. The
combined batch is shuffled before the normal SAC update; demonstration data
does not add behavior-cloning or other losses and cannot be erased by online
replay rollover.

Explicit demonstration options replace demonstrations already embedded in a
resumed checkpoint. If those options are omitted, the embedded demonstration
buffer and its sampling state are preserved, so the original `.demo` files
are not required for later resumes. Existing schema-1 checkpoints such as
`rl/checkpoints/latest` are migrated automatically with an initially empty
demonstration buffer.

Demonstrations that ended at the old `y=6.5` height goal contain obsolete
terminal rewards and should be re-recorded. Incomplete recordings that never
reached that terminal can still be selected explicitly with
`--demo-episode FILE INDEX`.

Run the Python tests without starting Unity:

```powershell
uv run --project rl pytest
```
