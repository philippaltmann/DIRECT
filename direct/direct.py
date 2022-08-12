import torch as th
from collections import deque
from stable_baselines3.ppo import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import obs_as_tensor
from typing import Any, Dict, List, Tuple

from util import TrainableAlgorithm
from . import DirectBuffer, Discriminator, DirectCallback

class DIRECT(TrainableAlgorithm, PPO):
  """ Discriminative Reward Co-Training
  Extending the PPO Implementation by stable baselines 3
  :param env: The environment to learn from (if registered in Gym, can be str)
  :param kappa: (int) k-Best Trajectories to be stored in the reward buffer
  :param omega: (float) The frequency to perform updates to the discriminator. 
      [1/1 results in concurrent training of policy and discriminator]
  :param chi: (float): The mixture parameter determining the mixture of real and discriminative reward
      [1 => purely discriminateive reward, 0 => no discriminateive reward, just real reward (pure PPO)]
  :param disc_kwargs: (dict) parameters to be passed the discriminator on creation
      see Discriminator class for full description
  :param **kwargs: further parameters will be passed to TrainableAlgorithm, then PPO"""

  def __init__(self, chi:float=1., kappa:int=512, omega:float=1/1, disc_kwargs:Dict[str,Any]={}, **kwargs):
    self.buffer = None
    self.chi = chi; assert chi <= 1.0
    self.kappa = kappa; assert kappa > 0
    self.omega = omega; assert 0 < omega < 10
    self.discriminator, self.disc_kwargs = None, disc_kwargs  
    self._returns, self._history = None, None #deque(maxlen=100)
    super().__init__(**kwargs)

  def _setup_model(self) -> None: 
    super(DIRECT, self)._setup_model() # Helpers for rolling back discriminative ep return 
    self._returns = [[] for _ in range(self.n_envs)]; self._naming.update({'d': 'direct-100'}) 

    # Policy & Buffer Samples -> twice for same updates, for single batch use self.n_steps*self.n_envs
    self.disc_kwargs.setdefault('batch_size', self.batch_size * 2) 
    self.buffer = DirectBuffer(buffer_size=self.kappa, parent=self)
    self.discriminator = Discriminator(chi=self.chi, parent=self, **self.disc_kwargs).to(self.device)

  def _excluded_save_params(self) -> List[str]:
    return super(DIRECT, self)._excluded_save_params() + ['buffer', '_returns'] 

  def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
    torch, vars = super(DIRECT, self)._get_torch_save_params()
    torch.extend(["discriminator", "discriminator.optimizer"])
    return torch, vars

  def learn(self, reset_num_timesteps: bool = True, **kwargs) -> "TrainableAlgorithm":
    if reset_num_timesteps: self._history = deque(maxlen=100)
    kwargs['callback'] = CallbackList([DirectCallback(), kwargs.pop('callback')]) if 'callback' in kwargs else DirectCallback()
    return super(DIRECT, self).learn(reset_num_timesteps=reset_num_timesteps, **kwargs)

  def train(self) -> None:
    discriminator: Discriminator = self.discriminator
    rollout: RolloutBuffer = self.rollout_buffer  # Grab rollout buffer containing environmental rewards
    # self.logger.record("rewards/environment", rollout.rewards.copy()) 
    real, starts = rollout.rewards.copy(), rollout.episode_starts.copy()
  
    # Update Buffer (previous: self.action_probs as Bias)
    self.buffer.extend(rollout=rollout)
    self.logger.record("buffer", self.buffer.metrics())

    # init buffer to hold rollout trajectories
    policy_buffer = discriminator.process(rollout=rollout)

    # Update Returns from Discriminative Rewards for updating the policy
    with th.no_grad(): _, values, _ = self.policy.forward(obs_as_tensor(self._last_obs, self.device))
    rollout.compute_returns_and_advantage(last_values=values, dones=self._last_episode_starts)
    # self.logger.record("rewards/discriminator", rollout.rewards.copy()) 
    disc = rollout.rewards.copy()
    
    # Rollup to calulate discriminative episode returns
    step = lambda s: (self.num_timesteps / self.n_envs - self.n_steps + s) * self.n_envs
    for r, d, s, i in zip(real, disc, starts, range(self.n_steps)):
      for env, ret in enumerate(self._returns):
        if s[env] and len(ret): self._history.append({**ret.pop(), 't': step(i)})
        if s[env]: ret.append({'d': d[env], 'r': r[env]})
        else: ret[-1]['d'] += d[env]; ret[-1]['r'] += r[env]
    for done, ret in zip(self._last_episode_starts,self._returns): #Check last done flags
      if done: self._history.append({**ret.pop(), 't': step(self.n_steps)}) 
    _intersect = lambda a,b: {k: a[k] for k in a.keys() & b.keys()}
    safe_intersect = lambda a,b: _intersect(a,b) == _intersect(b,a)
    for e,h in zip(self.ep_info_buffer, self._history): # Snyc ep_infos
      if self.normalize: h['r'] = e['r'] # Write unnormalized reward, for comparabiltiy
      assert safe_intersect(e,h), f"History mismatch: {e}, {h}"; e.update(h)

    # Update Discriminator & write stats
    disc_stats = discriminator.train(direct_buffer=self.buffer, policy_buffer=policy_buffer)
    self.logger.record("discriminator", disc_stats)
    
    # Train PPO
    super(DIRECT, self).train()
    