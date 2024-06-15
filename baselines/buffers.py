import numpy as np; import random; import operator; import torch as th
import warnings; from typing import Optional, Union
from gymnasium import spaces

try: import psutil
except ImportError: psutil = None

from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import BaseBuffer


class EntReplayBuffer(BaseBuffer):
  """ Replay buffer for unsupervised pretraining from https://github.com/rll-research/BPref
  :param buffer_size: Max number of element in the buffer
  :param observation_space: Observation space
  :param action_space: Action space
  :param device:
  :param n_envs: Number of parallel environments
  :param optimize_memory_usage: Enable a memory efficient variant
      of the replay buffer which reduces by almost a factor two the memory used,
      at a cost of more complexity.
      See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
      and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274"""

  def __init__(
      self,
      buffer_size: int,
      observation_space: spaces.Space,
      action_space: spaces.Space,
      device: Union[th.device, str] = "auto",
      n_envs: int = 1,
      optimize_memory_usage: bool = False,

      # Running Mean Std
      epsilon=1e-4, shape=[1]
  ):
    super(EntReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

    # Check that the replay buffer can fit into the memory
    if psutil is not None:  mem_available = psutil.virtual_memory().available

    self.optimize_memory_usage = optimize_memory_usage
    self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)
    self.n_envs = n_envs

    # Init Running MeanStd
    self.mean = th.zeros(shape, device=device)
    self.var = th.ones(shape, device=device)
    self.count = epsilon
    
    if psutil is not None:
      total_memory_usage = self.observations.nbytes 
      if total_memory_usage > mem_available: # Convert to GB
        total_memory_usage /= 1e9; mem_available /= 1e9;  warnings.warn(
          "This system does not have apparently enough memory to store the complete "
          f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB")


  def compute_state_entropy(self, obs, k=5, norm=False):
    batch_size = 500; idx = lambda i: i * batch_size; k = min(k, self.pos - 1)
    if self.full: full_obs = th.as_tensor(self.observations, device=self.device).float()
    else: full_obs = th.as_tensor(self.observations[:self.pos], device=self.device).float()
    obs = th.as_tensor(obs, device=self.device).float()
    with th.no_grad(): dists = th.cat([ # Calculate knn distance between obs and buffer
        th.norm(obs[:,None,:] - full_obs[None, idx(i):idx(i+1), :], dim=-1, p=2)
      for i in range(len(full_obs) // batch_size + 1)], dim=1)
    state_entropy = th.kthvalue(dists, k=k + 1, dim=1).values#.unsqueeze(1)    
    if norm: state_entropy = self.update_ent_stats(state_entropy)
    return state_entropy


  def update_ent_stats(self, entropy):
    """Normalize entropy by running std"""
    with th.no_grad():
      _mean, _var, _count = th.mean(entropy, axis=0), th.var(entropy, axis=0), entropy.shape[0]
      delta = _mean - self.mean; tot_count = self.count + _count
      self.mean = self.mean + delta + _count / tot_count
      self.var = self.var * self.count + _var * _count + delta**2 * self.count * _count / tot_count
      self.count = tot_count
      return entropy / th.sqrt(self.var)


  def add_obs(self, obs: np.ndarray) -> None:
    # Copy to avoid modification by reference
    next_index = self.pos + self.n_envs
    if next_index >= self.buffer_size:
      self.full = True; maximum_index = self.buffer_size - self.pos
      self.observations[self.pos:] = np.array(obs[:maximum_index]).copy()
      self.pos = self.n_envs - (maximum_index)
      if self.pos > 0: self.observations[0:self.pos] = np.array(obs[maximum_index:]).copy()
    else:
      self.observations[self.pos:next_index] = np.array(obs).copy(); self.pos = next_index

          
  def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
    if self.optimize_memory_usage: next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
    else: next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

    data = (
      self._normalize_obs(self.observations[batch_inds, 0, :], env),
      self.actions[batch_inds, 0, :],
      next_obs,
      self.dones[batch_inds],
      self._normalize_reward(self.rewards[batch_inds], env),
    )
    return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
  



class PrioritizedReplayBuffer(object):
  def __init__(self, size, alpha, beta):
    """ Create Prioritized Replay buffer. Taken from https://github.com/junhyukoh/self-imitation-learning
    size: [int] Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped.
    alpha: [float] how much prioritization is used (0 - no prioritization, 1 - full prioritization) 
    beta: [float] To what degree to use importance weights (0 - no corrections, 1 - full correction)"""       
    self._storage = []; self._maxsize = int(size); self._next_idx = 0
    self.total_steps = []; self.total_rewards = []
    self._alpha = alpha; self._beta = beta; assert alpha > 0
    it_capacity = 1; self._max_priority = 1.0
    while it_capacity < size: it_capacity *= 2
    self._it_sum = SumSegmentTree(it_capacity)
    self._it_min = MinSegmentTree(it_capacity)
    self.default = self._max_priority ** self._alpha
    
  def __len__(self): return len(self._storage)

  def add(self, obs_t, action, R):
    idx = self._next_idx; data = (obs_t, action, R)
    if self._next_idx >= len(self._storage): 
      self._storage.append(data)
    else: 
      self._storage[self._next_idx] = data
    self._next_idx = int((self._next_idx + 1) % self._maxsize)
    self._it_sum[idx] = self.default; self._it_min[idx] = self.default

  def _sample_proportional(self, batch_size):
    res = []
    for _ in range(batch_size):
      mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
      idx = self._it_sum.find_prefixsum_idx(mass)
      res.append(idx)
    return res

  def _encode_sample(self, indexes):
    obses_t, actions, returns= zip(*[self._storage[i] for i in indexes])
    return np.array(obses_t), np.array(actions), np.array(returns)

  def sample(self, batch_size):
    """Sample a batch of experiences. compared to ReplayBuffer.sample
    it also returns importance weights and idxes of sampled experiences.
    batch_size: [int] How many transitions to sample.
    Returns:  obs_batch [np.array]: batch of observations, 
              act_batch [np.array]: batch of actions executed given obs_batch
              R_batch [np.array]: returns received as results of executing act_batch
              weights [np.array]: Array of shape (batch_size,) and dtype np.float32 denoting importance weight of each sampled transition
              idxes: [np.array]: Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences """
    indexes = self._sample_proportional(batch_size)
    samples = self._encode_sample(indexes)
    w = lambda it: ((it / self._it_sum.sum()) * len(self._storage)) ** (-self._beta)

    if self._beta == 0: weights = np.ones_like(indexes, dtype=np.float32)
    else: weights = np.array([w(self._it_sum[i]) / w(self._it_min.min()) for i in indexes])
    return tuple(list(samples) + [weights, indexes])


  def sample_batch(self, batch_size):
    if not len(self) > 100: return None, None, None, None, None
    batch_size = min(batch_size, len(self))
    return self.sample(batch_size)

  def update_priorities(self, idxes, priorities):
      """ Update priorities of sampled transitions. sets priority of transition at index idxes[i] in buffer to priorities[i].
      idxes: [int] List of idxes of sampled transitions
      priorities: [float] List of updated priorities corresponding to  transitions at the sampled idxes denoted by  variable `idxes`. """
      assert len(idxes) == len(priorities)
      for idx, priority in zip(idxes, priorities):
        priority = max(max(priority), 1e-6); 
        assert priority > 0 and 0 <= idx < len(self._storage)
        value = priority ** self._alpha
        self._it_sum[idx] = value; self._it_min[idx] = value
        self._max_priority = max(self._max_priority, priority)

  def update(self, trajectory, gamma):
    obs,act,rew = zip(*trajectory)
    # if not any(r > 0 for r in rew): return 
    # if self.updated: return  #TODO: fix 20 vs 300 steps/s
    ret = discount_with_dones(np.sign(rew), [False]*(len(rew)-1)+[True], gamma)
    [self.add(*t) for t in zip(obs,act,ret)]
        
    self.total_steps.append(len(trajectory)); self.total_rewards.append(np.sum(rew))
    while np.sum(self.total_steps) > self._maxsize and len(self.total_steps) > 1:
      self.total_steps.pop(0); self.total_rewards.pop(0)
  
def discount_with_dones(rewards, dones, gamma):
  discounted = []; r = 0
  for reward, done in zip(rewards[::-1], dones[::-1]):
    r = reward + gamma * r * (1. - done); discounted.append(r)
  return discounted[::-1]




class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer(object):
    """Buffer to store environment transitions for PEBBLE."""
    def __init__(self, obs_shape, action_shape, capacity, device, window=1):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):
        
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.next_obses[self.idx:self.capacity], next_obs[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)
            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.next_obses[0:remain], next_obs[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)
            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.next_obses[self.idx:next_index], next_obs)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)
            self.idx = next_index
        
    def relabel_with_predictor(self, predictor):
        batch_size = 200
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
                
            obses = self.obses[index*batch_size:last_index]
            actions = self.actions[index*batch_size:last_index]
            inputs = np.concatenate([obses, actions], axis=-1)
            
            pred_reward = predictor.r_hat_batch(inputs)
            self.rewards[index*batch_size:last_index] = pred_reward
            
    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = th.as_tensor(self.obses[idxs], device=self.device).float()
        actions = th.as_tensor(self.actions[idxs], device=self.device)
        rewards = th.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = th.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = th.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = th.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = th.as_tensor(self.obses[idxs], device=self.device).float()
        actions = th.as_tensor(self.actions[idxs], device=self.device)
        rewards = th.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = th.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = th.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = th.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = th.as_tensor(full_obs, device=self.device)
        
        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    