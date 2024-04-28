import gymnasium as gym
import ray
from n_puzzle.envs import NPuzzleEnv
# from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from architectures import RayFFN
from ray.rllib.algorithms.ppo import PPOConfig
from tqdm import tqdm

ModelCatalog.register_custom_model("rayffn", RayFFN)

config = PPOConfig()
config.environment(env=NPuzzleEnv, env_config={"n": 5, "k": 20})

config.update_from_dict({
    "framework": "torch",
    "model": {
        "custom_model": "rayffn",
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {},
    }
})
config.validate()


ray.init()
algo = config.build()

for _ in tqdm(range(100)):
    algo.train()
# print(algo.evaluate())
print("done training")
ppo_policy = algo.get_policy()
ppo_policy.export_model("checkpoints/ray_test")