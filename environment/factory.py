import gymnasium as gym
from hyphi_gym import named, Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def _make(record_video=False, **spec):
  def _init() -> gym.Env: return Monitor(gym.make(**spec), record_video=record_video)
  return _init

def make_vec(name, seed=None, n_envs=1, **kwargs):
  spec = lambda rank: {**named(name), 'seed': seed+rank, **kwargs}
  return DummyVecEnv([_make(**spec(i)) for i in range(n_envs)])

def factory(env_spec, n_train=4, **kwags):
  assert len(env_spec) > 0, 'Please specify at least one environment for training'
  test_names = ['validation', *[f'evaluation-{i}' for i in range(len(env_spec)-1)]]
  return { 'train': make_vec(env_spec[0], n_envs=n_train, **kwags), 
    'test': {name: make_vec(spec, render_mode='2D', record_video=True, **kwags) for name, spec in zip(test_names, env_spec)}
  }   

if __name__ == "__main__":
  # name = "Maze9Target"
  # name = "Maze13Target"
  name = "Maze15Target"
  # name = "PointMaze11Target"
  # name = "PointMaze13Target"
  # name = "PointMaze15Target"
  
  rt = []
  for seed in [1,2,3,4,5,6,7,8]:
    env = make_vec(name, seed=seed, n_envs=4)
    rt.extend([e.unwrapped.reward_threshold for e in env.envs])
  print(env.envs[0].unwrapped.reward_range)
  print(sum(rt)/len(rt))