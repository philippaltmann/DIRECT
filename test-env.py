import gym
from safety_env import factory
from stable_baselines3.common.env_util import make_vec_env

seed = 0#48#random.randint(0, 100) #42
env_name = "DistributionalShift"
env_spec = 0 # Index of env config to be used (cf. safety_env/__init__.py)
env = factory(env_name, make_vec_env, "train", env_spec)
env.seed(seed)

print(env.reset())
print(env.observation_space.sample())
print(env.action_space)
