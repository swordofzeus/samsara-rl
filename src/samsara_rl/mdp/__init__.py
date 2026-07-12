from gymnasium.envs.registration import register

register(
    id="gymnasium_env/4x4GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)