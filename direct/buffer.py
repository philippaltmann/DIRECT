from multiprocessing.sharedctypes import Value
import torch as th; import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import BaseBuffer 
from stable_baselines3.common.preprocessing import preprocess_obs as process
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Tuple, NamedTuple

class Score(): 
  value: float; bias: float; get = lambda self, biased: self.value + self.bias if biased else self.value
  def __init__(self, value:float=0.0, bias:float=0.0): self.value = value; self.bias=bias
    
class DirectBufferSamples(NamedTuple): observations: th.Tensor; actions: th.Tensor; rewards: th.Tensor

class DirectBuffer(BaseBuffer):
  def __init__(self, buffer_size: int, parent: BaseAlgorithm): 
    """ Direct Buffer for Self-Imitation, memory prioritized by biased return
    :param size: (int)  Max number of transitions to store. On overflow, old memory is dropped. """
    self.parent, self.env = parent, parent.env
    n_envs, observation_space, action_space = self.env.num_envs, self.env.observation_space, self.env.action_space 
    super(DirectBuffer, self).__init__(buffer_size, observation_space, action_space, parent.device, n_envs=n_envs)
    self.obs_size, self.scr_size = (self.buffer_size, ) + self.obs_shape, (self.buffer_size)
    self.act_size, self.rew_size = (self.buffer_size, self.action_dim), (self.buffer_size, 1)
    self.observations = np.zeros(self.obs_size, dtype=observation_space.dtype)
    self.actions = np.zeros(self.act_size, dtype=action_space.dtype)
    self.rewards = np.zeros((self.rew_size), dtype=np.float32)
    self.scores = np.full((self.scr_size), Score())
    self.overwrites = 0
    
  def fill(self, obs:np.ndarray, act:np.ndarray, rew:np.ndarray, scr=None):
    """Fill buffer with samples"""
    scr = scr or np.full(rew.shape, Score())
    obs, act, rew, scr = [self.swap_and_flatten(d) for d in (obs, act, rew, scr)]; self.assert_shape(obs, act, rew, scr)
    self.observations, self.actions, self.rewards, self.scores = obs.copy(), act.copy(), rew.copy(), scr.copy()
    self.pos = rew.shape[0]; self.full = self.buffer_size == self.pos; return self

  def _insert(self, obs, act, rew, score, index: float)->int:
    if index >= self.buffer_size or index < 0: return -1
    self.observations = np.insert(self.observations, index, np.array(obs).copy(), axis=0)
    self.actions = np.insert(self.actions, index, np.array(act).copy(), axis=0)
    self.rewards = np.insert(self.rewards, index, np.array(rew).copy(), axis=0)
    self.scores = np.insert(self.scores, index, np.array(score).copy(), axis=0)
    self.pos += 1; self._clean(); return index

  def _clean(self):
    """ Removes buffer items exceeding the intended size """
    if self.pos > self.buffer_size: self.overwrites, self.pos = self.overwrites + 1, self.pos - 1
    self.observations.resize(self.obs_size); self.scores.resize(self.scr_size)
    self.actions.resize(self.act_size); self.rewards.resize(self.rew_size)
    assert self.pos <= self.buffer_size; self.full = self.pos == self.buffer_size

  def assert_shape(self, obs, act, rew, scr) -> Tuple:
    """ Reduce dimensionality if asserting single samples, not lists """
    adapt = lambda shape: shape[(len(self.obs_size)-len(obs.shape)):]
    error = "Missmatching Sample Shape. Got {}, expected {}"
    assert obs.shape == adapt(self.obs_size), error.format(obs.shape, adapt(self.obs_size))
    assert act.shape == adapt(self.act_size), error.format(act.shape, adapt(self.act_size))
    assert rew.shape == adapt(self.rew_size), error.format(rew.shape, adapt(self.rew_size))
    if len(self.obs_size) - len(obs.shape): assert isinstance(scr, Score)
    return (obs, act, rew, scr)
  
  def add(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, scr: np.ndarray) -> int:
    """ add a new observation, action, reward transition to the buffer if within k-highest scores
    prioritizing new experience over old (<=) :returns the index where the sample has been added"""
    obs, act, rew, scr = self.assert_shape(obs, act, rew, scr); index, score = self.size(), scr.get(False)
    while index > 0 and self.scores[(index - 1)].get(False) <= score: index -= 1
    return self._insert(obs, act, rew, scr, index)

  def extend(self, episode) -> list: 
    """ Extends the buffer by sar samples from one episode sorted by their cumulated return
    Returns: list of indices where samples were inserted """
    obs, act, rew = self.shape(*episode.get('history').values(), episode['l'])
    return [self.add(*data) for data in zip(*(obs,act,rew, np.full(episode['l'], Score(episode['r']))))] 

  def _get_samples(self, idx: np.ndarray, env: Optional[VecNormalize] = None) -> DirectBufferSamples:
    return self.prepare(self, self.observations[idx], self.actions[idx], self.rewards[idx], idx.shape[0])

  def shape(self, obs, act, rew, len):
    _s = lambda shape: (len,) + (shape[1:] if isinstance(shape, tuple) else ())
    return np.reshape(obs, _s(self.obs_size)), np.reshape(act, _s(self.act_size)), np.reshape(rew, _s(self.rew_size))

  @classmethod
  def prepare(cls, buffer, observations:np.ndarray, actions:np.ndarray, rewards:np.ndarray, len=1) -> DirectBufferSamples:
    """ Prepares a batch of observatiosn, actions and rewards of size len for being processed by the discriminator by
      reshaping, applying normalization when needed, converting to tensors and packing in DirectBufferSample"""
    observations, actions, rewards = buffer.shape(observations, actions, rewards, len)
    return DirectBufferSamples(
      process(th.as_tensor(buffer._maybe_norm(obs=observations), device=buffer.device), buffer.env.observation_space), 
      process(th.as_tensor(actions, device=buffer.device), buffer.env.action_space), 
      th.as_tensor(buffer._maybe_norm(rew=rewards), device=buffer.device)
    )

  def _maybe_norm(self, obs=None, rew=None):
    if getattr(self.parent, 'normalize', False):
      if obs is not None: return self._normalize_obs(obs, env=self.env) 
      if rew is not None: return self._normalize_reward(rew, env=self.env) 
    else: return obs if obs is not None else rew if rew is not None else None

  def metrics(self): return {"rewards": np.array(self.rewards), "momentum": self.overwrites, "samples": self.size()}
