import numpy as np; import torch as th
from baselines import PPO 
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from .bnn import BNN
from .buffers import ReplayPool
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from collections import deque

class VIME(PPO):
  """A preference-based extension to PPO according to: 
  https://arxiv.org/pdf/1706.03741 / https://arxiv.org/pdf/2111.03026"""

  def __init__(self, eta=1e-3, buffer_size=100000, min_pool_size=1000, n_updates_per_sample=500, kl_q_len=10, **kwargs):
    """:param eta: Weight of the intrinsic reward, with r = r_e + eta * r_i"""
    # n_epochs=5, # Default 10 batch_size=32, # Default 64,  ent_coef=0.01, # Default 0.0, learning_rate=7e-4, # Default 3e-4,
    super().__init__(n_epochs=5, batch_size=32, ent_coef=0.01, learning_rate=7e-4, **kwargs); 
    obs_shape, act_dim = get_obs_shape(self.observation_space), get_action_dim(self.action_space)
    self.dynamics, self.eta = BNN(obs_shape[0] + act_dim, obs_shape[0], device=self.device), eta
    self.replay_pool = ReplayPool(buffer_size, self.observation_space, self.action_space, self.device, self.n_envs)
    self.min_pool_size, self.n_updates_per_sample = min_pool_size, n_updates_per_sample
    self._kl_mean, self._kl_std = deque(maxlen=kl_q_len), deque(maxlen=kl_q_len)
    self.kl_q_len, self.kl_previous = kl_q_len, deque(maxlen=kl_q_len)

  
  def learn(self, callback=[], **kwargs) -> "VIME":    
    callback = CallbackList([VIMECallback(), *callback])
    return super(VIME, self).learn(callback=callback, **kwargs)


  def train(self, ) -> None:
    super().train()
    if self.replay_pool.pos < self.min_pool_size: return 
    acc = np.mean([self.train_dynamics() for _ in range(self.n_updates_per_sample)]) 
    self.logger.record("rewards/dynamics_accuracy", acc)


  def train_dynamics(self, batch_size=10):
    inputs, targets = self.replay_pool.sample(batch_size)
    self.dynamics.opt.zero_grad()
    loss = self.dynamics.loss(inputs, targets)
    loss.backward()
    self.dynamics.opt.step()
    with th.no_grad(): return th.mean((self.dynamics(inputs) - targets)**2)



class VIMECallback(BaseCallback):
  """A Callback for Variational Information Maximizing Exploration
  Adapted from https://github.com/mazpie/vime-pytorch"""

  def _on_step(self) -> bool: 
    l = lambda name: self.locals.get(name)
    # Get Observations actions and environment rewards and update replay pool
    obs, next_obs = self.model._last_obs, l('new_obs')
    action, reward, done, infos =  l('actions'), l('rewards'), l('dones'), l('infos')
    self.model.replay_pool.add(obs, next_obs, action, reward, done, infos)


  def _on_rollout_end(self) -> None:
    if self.num_timesteps <= self.model.n_steps * self.model.n_envs:  return True  # Skip initial rollout
    l = lambda name: self.locals.get(name); rollout_buffer, values, dones = l('rollout_buffer'), l('values'), l('dones')
    obs, act, rew = rollout_buffer.observations, rollout_buffer.actions, rollout_buffer.rewards
    obs, act, _ = self.model.replay_pool.normalize(obs, act, obs, 1e-8)
    next_obs = np.concatenate((obs[1:, :, :], self.model._last_obs[np.newaxis,...]), axis=0)
    obs, act, next_obs = (a.reshape(-1, a.shape[-1]) for a in (obs, act, next_obs))
    inputs = th.tensor(np.hstack([obs, act])).to(device=self.model.device)
    targets = th.tensor(next_obs).to(self.model.device)    

    # Add KL as intrinsic reward to external reward
    kl = self.compute_intrinsic_reward(inputs, targets, rew.shape)
    self.model.logger.record("rewards/intrinsic_reward", self.model.eta * kl)
    rollout_buffer.rewards = rollout_buffer.rewards + self.model.eta * kl
    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)


  def compute_intrinsic_reward(self, inputs, targets, shape, kl_batch_size=1, second_order_update=True, n_itr_update=1, 
                                use_replay_pool=True, use_kl_ratio = True, use_kl_ratio_q = True):
    """Iterate over all paths and compute intrinsic reward by updating the model on each observation, 
    calculating the KL divergence of the new params to the old ones, and undoing this operation."""
    def kl_batch(start, end):
      self.model.dynamics.save_old_params() # Save old params for every update.
      for _ in range(n_itr_update): kl_divergence = np.clip(
        self.model.dynamics.train_update_fn(
          inputs[start:end], targets[start:end], second_order_update, step_size=0.01 if second_order_update else None
        ).float().detach(), 0, 1000)
      if use_replay_pool: self.model.dynamics.reset_to_old_params()  # If using replay pool, undo updates.
      return [kl_divergence for _ in range(start, end)]
    
    kl = np.array([kl_batch(k * kl_batch_size, (k + 1) * kl_batch_size) for k in range(inputs.shape[0]//kl_batch_size)]).reshape(shape)
    kl[-1] = kl[-2]  # Last element in KL vector needs to be replaced by second last because actual last has no next.

    if use_kl_ratio and use_kl_ratio_q: # Normalize
      self.model.kl_previous.append(np.median(np.hstack(kl)))
      previous_mean_kl = np.mean(np.asarray(self.model.kl_previous))
      kl = kl / previous_mean_kl

    return kl
    