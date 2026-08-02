# SAC Implementation Roadmap

## Goal

Implement Soft Actor-Critic from scratch for `GettingOverItUnity-v0` while
using PyTorch only for tensor operations, neural-network layers, automatic
differentiation, and optimization. Do not use Stable-Baselines3 or another
library's SAC implementation.

Complete and test each stage before moving to the next one. Keep the Unity
communication environment independent from the SAC implementation.

## 1. PyTorch foundation

Add a Python 3.8-compatible PyTorch dependency and SAC-specific configuration.
Use CPU by default, with an optional explicit device setting for future GPU
training.

Configuration will eventually include network dimensions, replay capacity,
batch size, learning rates, discount factor, target-update coefficient,
warm-up steps, update frequency, entropy settings, checkpoint frequency, and
random seed.

Completion criteria:

- PyTorch imports in the pinned `uv` environment.
- Configuration validation rejects invalid sizes, rates, and step counts.
- Existing Unity communication tests continue to pass.

## 2. Replay buffer

Implement a fixed-capacity circular replay buffer that stores:

```text
(observation, action, reward, next_observation, terminated)
```

Treat Gymnasium truncation separately from true termination when constructing
Bellman targets. A time-limit truncation should normally allow bootstrapping;
a Unity goal or fall terminal should not.

Completion criteria:

- Stored arrays use `float32` except terminal flags.
- Sampling returns correctly shaped random batches.
- Capacity rollover replaces the oldest transitions.
- Sampling before enough transitions exist fails clearly.

## 3. Neural networks

Implement:

- One stochastic actor producing action means and log standard deviations.
- Two independent Q-value critics.
- One target copy of each critic.
- Multilayer perceptrons suitable for 20 observations and two actions.

Initialize target critics from the online critics and keep their gradients
disabled.

Completion criteria:

- Actor and critic output shapes are correct for single inputs and batches.
- The critics do not share parameters.
- Gradients reach the intended online networks.
- Target-network parameters initially equal their online counterparts.

## 4. Squashed action sampling

Use the reparameterization trick to sample from the actor's Gaussian policy,
then apply `tanh` so both action coordinates remain in `[-1, 1]`. Include the
change-of-variables correction in the action log probability.

Deterministic evaluation should use the `tanh` of the actor mean. Clamp log
standard deviations to a stable configured range.

The Unity controller additionally projects commands outside the unit disk.
Keep that behavior for the first baseline, measure how often projection occurs,
and revisit the action parameterization if it impairs learning.

Completion criteria:

- Sampled and deterministic actions remain within the action space.
- Log probabilities are finite for extreme inputs.
- Reparameterized samples propagate gradients into the actor.

## 5. SAC update equations

Implement one isolated update operation containing:

1. Target-action sampling from the actor.
2. Minimum target-Q calculation from both target critics.
3. Entropy-adjusted Bellman target.
4. Independent losses and optimizer steps for both critics.
5. Actor loss using the minimum online Q estimate.
6. Optional automatic entropy-temperature loss.
7. Soft updates of both target critics.

Do not allow target values to propagate gradients into the actor or target
critics during critic updates.

Completion criteria:

- Every loss is finite on a synthetic batch.
- Only the intended parameters change during each optimizer step.
- Terminal transitions omit bootstrapped value.
- Soft updates follow `target = (1 - tau) * target + tau * online`.

## 6. SAC algorithm class

Create a concrete implementation of the existing `RLAlgorithm` interface:

- `learn(environment, total_steps)`
- `predict(observation, deterministic=True)`
- `save(path)`
- `load(path)`

Update `create_algorithm()` to construct a new SAC agent for training or load
one for evaluation. Save networks, target networks, optimizers, entropy state,
configuration, counters, and random-generator state needed to resume training.

Completion criteria:

- A save/load round trip preserves deterministic predictions.
- Training can resume without resetting optimizer state or counters.
- Loading incompatible observation or action dimensions fails clearly.

## 7. Training loop

Implement the environment interaction loop:

1. Reset the environment with the configured seed.
2. Use random actions during replay-buffer warm-up.
3. Use stochastic actor actions afterward.
4. Store each transition in the replay buffer.
5. Run configured gradient updates once the buffer has enough samples.
6. Reset after termination or truncation.
7. Save periodic and final checkpoints.
8. Close Unity in a `finally` block.

Track episode return, episode length, maximum height, success/fall outcomes,
actor loss, critic losses, entropy temperature, and replay-buffer size.

Completion criteria:

- A fake environment can run several complete episodes.
- Warm-up and learned-action phases occur at the correct steps.
- Termination and truncation are stored and reset correctly.
- Exceptions still close the environment and preserve the latest checkpoint.

## 8. Evaluation

Load a checkpoint and run deterministic episodes without gradient updates or
replay-buffer changes. Report per-episode and aggregate return, maximum height,
episode length, success rate, and fall rate.

Completion criteria:

- Evaluation never changes model parameters.
- Identical deterministic observations produce identical actions.
- The evaluator handles both Unity termination and Python time limits.

## 9. Unity training improvements

After the first end-to-end baseline works:

- Add an ML-Agents engine-configuration channel to accelerate simulation.
- Train without unnecessary rendering when a standalone build is available.
- Introduce bounded reset randomization to reduce overfitting.
- Measure action projection and redesign the action representation if needed.
- Compare results across multiple Python/model seeds.

These improvements should follow a working baseline so environment changes do
not obscure SAC implementation bugs.

## Initial reference defaults

Use these as starting values, not as final tuned hyperparameters:

```text
Actor hidden layers:       256, 256
Critic hidden layers:      256, 256
Replay capacity:           200,000
Warm-up transitions:       5,000
Batch size:                256
Discount factor (gamma):   0.99
Target update (tau):       0.005
Actor learning rate:       0.0003
Critic learning rate:      0.0003
Entropy learning rate:     0.0003
Gradient updates per step: 1
Entropy temperature:       automatically tuned
```

Change one major variable at a time and retain the configuration alongside
every checkpoint so experiments remain reproducible.
