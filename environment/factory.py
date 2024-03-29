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
    'test': {name: make_vec(spec, render_mode='3D' if 'Fetch' in spec else '2D', record_video=True, **kwags) 
             for name, spec in zip(test_names, env_spec)}
  } 
