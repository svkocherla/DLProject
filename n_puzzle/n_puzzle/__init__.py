from gymnasium.envs.registration import register

register(
     id="n_puzzle/n_puzzle-v0",
     entry_point="n_puzzle.envs:NPuzzleEnv",
     max_episode_steps=300,
)