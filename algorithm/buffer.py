"""Taken from stable baselines https://github.com/hill-a/stable-baseliness"""
import numpy as np; import random; import operator

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