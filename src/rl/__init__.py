from gymnasium.envs.registration import register

register(
     id="envs/AlphaFairEnvironment-v0",
     entry_point="src.rl.envs.alpha_fair_env:AlphaFairEnvironment",
     max_episode_steps=300,
     nondeterministic=True
)