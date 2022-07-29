import torch as th; import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer 
from stable_baselines3.common.preprocessing import preprocess_obs as process
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Tuple, NamedTuple, Iterable

class DirectBufferSamples(NamedTuple): 
  observations: th.Tensor; actions: th.Tensor; returns: th.Tensor; biases: th.Tensor

class DirectBuffer(BaseBuffer):
  def __init__(self, buffer_size: int, parent: BaseAlgorithm):
    """ Direct Buffer for Self-Imitation, memory prioritized by biased return
    :param size: (int)  Max number of transitions to store. On overflow, old memory is dropped. """
    self.parent, self.env = parent, parent.env
    n_envs, observation_space, action_space = self.env.num_envs, self.env.observation_space, self.env.action_space 
    super(DirectBuffer, self).__init__(buffer_size, observation_space, action_space, parent.device, n_envs=n_envs)
    self.obs_size, self.bias_size = (self.buffer_size, ) + self.obs_shape, (self.buffer_size, 1)
    self.act_size, self.ret_size = (self.buffer_size, self.action_dim), (self.buffer_size, 1)
    self.observations = np.zeros(self.obs_size, dtype=observation_space.dtype)
    self.actions = np.zeros(self.act_size, dtype=action_space.dtype)
    self.returns = np.zeros((self.ret_size), dtype=np.float32)
    self.biases = np.zeros((self.bias_size), dtype=np.float32)
    self.overwrites = 0

  @classmethod
  def fromRollout(self, rb: RolloutBuffer, parent: BaseAlgorithm):
    """ Collect rollout from Rollout Buffer and insert unsorted into DirectBuffer """
    sf = lambda x: self.swap_and_flatten(x)
    buffer = self(buffer_size=rb.buffer_size*rb.n_envs, parent=parent)
    (obs, act, ret, bias) = sf(rb.observations), sf(rb.actions), sf(rb.returns), np.zeros_like(sf(rb.returns))
    buffer.assert_shape(obs, act, ret, bias)
    buffer.observations, buffer.biases = obs.copy(), bias.copy()
    buffer.actions, buffer.returns = act.copy(), ret.copy()
    buffer.pos, buffer.full = True, buffer.buffer_size
    return buffer

  def _score(self, index: Optional[int]=None, data: Optional[DirectBufferSamples]=None) -> float:
    r, b = 0.0, 0.0 
    if index is not None: r,b = self.returns[index], self.biases[index]
    if data: r,b = data.returns, data.biases
    return r - b

  def _insert(self, data: DirectBufferSamples, index: float)->int:
    obs, act, ret, bias = data
    if index >= self.buffer_size or index < 0: return -1
    self.observations = np.insert(self.observations, index, np.array(obs).copy(), axis=0)
    self.actions = np.insert(self.actions, index, np.array(act).copy(), axis=0)
    self.returns = np.insert(self.returns, index, np.array(ret).copy(), axis=0)
    self.biases = np.insert(self.biases, index, np.array(bias).copy(), axis=0)
    self.pos += 1; self._clean(); return index

  def _clean(self):
    if self.pos > self.buffer_size: self.overwrites, self.pos = self.overwrites + 1, self.pos - 1
    self.observations.resize(self.obs_size); self.biases.resize(self.bias_size)
    self.actions.resize(self.act_size); self.returns.resize(self.ret_size)
    assert self.pos <= self.buffer_size; self.full = self.pos == self.buffer_size

  def assert_shape(self, obs, act, ret, bias) -> Tuple:
    # Reduce dimensionality if asseritng single samples, not lists 
    adapt = lambda shape: shape[(len(self.obs_size)-len(obs.shape)):]
    error = "Missmatching Sample Shape. Got {}, expected {}"
    assert obs.shape == adapt(self.obs_size), error.format(obs.shape, adapt(self.obs_size))
    assert act.shape == adapt(self.act_size), error.format(act.shape, adapt(self.act_size))
    assert ret.shape == adapt(self.ret_size), error.format(act.shape, adapt(self.ret_size))
    assert bias.shape == adapt(self.bias_size), error.format(act.shape, adapt(self.bias_size))
    return tuple((obs, act, ret, bias))
  
  def add(self, obs: np.ndarray, act: np.ndarray, ret: np.ndarray, bias: np.ndarray) -> int:
    """ add a new transition of biased observation, action & return to the buffer if within k-highest
    :returns the index where the sample has been added"""
    data = DirectBufferSamples(*self.assert_shape(obs, act, ret, bias))
    index, score = self.size(), self._score(data=data)
    while index > 0 and self._score(index=(index - 1)) <= score: index -= 1
    return self._insert(data, index)

  def extend(self, rollout: RolloutBuffer) -> Tuple[Iterable[int], Iterable[DirectBufferSamples]]:
    sf, rb = lambda x: self.swap_and_flatten(x), rollout
    rollout = sf(rb.observations), sf(rb.actions), sf(rb.returns), np.zeros_like(sf(rb.returns))
    return [self.add(*data) for data in zip(*rollout)] # previous, list of insert idxs

  def _get_samples(self, idx: np.ndarray, env: Optional[VecNormalize] = None) -> DirectBufferSamples:
    e, d, obs_s, act_s = self.env, self.device, self.env.observation_space, self.env.action_space
    obs = lambda o: process(th.as_tensor(self._maybe_norm(env=e, obs=o), device=d), obs_s)
    act = lambda a: process(th.as_tensor(a, device=d), act_s)
    ret = lambda r: th.as_tensor(self._maybe_norm(env=e, ret=r), device=d)
    bias = lambda b: th.as_tensor(b, device=d)        
    return DirectBufferSamples(*tuple(
      (obs(self.observations[idx]), act(self.actions[idx]), ret(self.returns[idx]), bias(self.biases[idx]))
    ))

  def _maybe_norm(self, env, obs=None, ret=None):
    if getattr(self.parent, 'normalize', False):
      if obs is not None: return self._normalize_obs(obs, env=env) 
      if ret is not None: return self._normalize_reward(ret, env=env) 
    else: return obs if obs is not None else ret if ret is not None else None

  def metrics(self): #"buffer/returns": np.mean(self.returns)
    return {"returns": np.array(self.returns), "momentum": self.overwrites, "samples": self.size()}
