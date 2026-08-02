from gymnasium.envs.registration import register, registry

from .environment import GettingOverItEnv

ENVIRONMENT_ID = "GettingOverItUnity-v0"

if ENVIRONMENT_ID not in registry:
    register(
        id=ENVIRONMENT_ID,
        entry_point="getting_over_it_env.environment:GettingOverItEnv",
    )

__all__ = ["ENVIRONMENT_ID", "GettingOverItEnv"]
