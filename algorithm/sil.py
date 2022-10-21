import numpy as np; import torch as th
from stable_baselines3.a2c import A2C as StableA2C
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from .sil_buffer import PrioritizedReplayBuffer
from common import TrainableAlgorithm

class SIL(TrainableAlgorithm, StableA2C):
  """A Trainable extension to A2C using Self-Imitation Leanrning (https://arxiv.org/pdf/1806.05635.pdf)
  https://github.com/junhyukoh/self-imitation-learning/blob/master/baselines/common/self_imitation.py
  Implementation inspired by code from https://github.com/TianhongDai/self-imitation-learning-pytorch"""

  def __init__(self, sil_alpha=0.6, sil_beta=0.1, sil_update=4, max_nlogp=5,
    buffer_size=10e+4, sil_batch_size=512, min_batch_size=64, clip=1, w_value = 0.01, 
    _init_setup_model=True, gae_lambda=0.95, ent_coef=0.1, **kwargs):
    self.buffer_size = buffer_size; self.buffer = None; self.updates = 0
    self.sil_alpha = sil_alpha; self.sil_beta = sil_beta; self.sil_update = sil_update; 
    self.max_nlogp = max_nlogp; self.clip = clip; self.w_value = w_value
    self.sil_batch_size = sil_batch_size; self.min_batch_size = min_batch_size
    super(SIL, self).__init__(gae_lambda=gae_lambda, ent_coef=ent_coef, _init_setup_model=False, **kwargs)    
    if _init_setup_model: self._setup_model()

  def _setup_model(self) -> None:
    super(SIL, self)._setup_model(); self.running_episodes = [[] for _ in range(self.n_envs)]
    self.buffer = PrioritizedReplayBuffer(alpha=self.sil_alpha, beta=self.sil_beta, size=self.buffer_size)
    
  def learn(self, reset_num_timesteps: bool = True, **kwargs) -> "TrainableAlgorithm":
    kwargs['callback'] = CallbackList([SILCallback(), kwargs.pop('callback')]) if 'callback' in kwargs else SILCallback()
    return super(SIL, self).learn(reset_num_timesteps=reset_num_timesteps, **kwargs)

  def train(self) -> None:
    self.logger.record("rewards/environment", self.rollout_buffer.rewards.copy()) 
    super(SIL, self).train()
    mean_adv,num_valid_samples = [],[]
    for _ in range(self.sil_update):
      obs, act, ret, wgt, idx = self.buffer.sample_batch(self.sil_batch_size)
      if obs is  None: break
      obs = th.tensor(obs, dtype=th.float32, device=self.device)
      actions = th.tensor(act, dtype=th.float32, device=self.device).unsqueeze(1)
      returns = th.tensor(ret, dtype=th.float32, device=self.device).unsqueeze(1)
      weights = th.tensor(wgt, dtype=th.float32, device=self.device).unsqueeze(1)
      max_nlogp = th.full((len(idx), 1),self.max_nlogp, dtype=th.float32, device=self.device)

      values, action_log_probs, entropy = self.policy.evaluate_actions(obs, actions)
      clipped_nlogp = th.min((-action_log_probs), max_nlogp)
      advantages = (returns - values.flatten()).detach()
      masks = (advantages.cpu().numpy() > 0).astype(np.float32)
      num_valid_samples.append(np.sum(masks))
      num_samples = np.max([np.sum(masks), self.min_batch_size])
      masks = th.tensor(masks, dtype=th.float32, device=self.device)
      clipped_advantages = th.clamp(advantages, 0, self.clip)
      mean_adv.append((th.sum(clipped_advantages) / num_samples ).item())
      action_loss = th.sum(clipped_advantages * weights * clipped_nlogp) / num_samples
      entropy_reg = th.sum(weights * entropy * masks) / num_samples
      policy_loss = action_loss - entropy_reg * self.ent_coef
      delta = (th.clamp(values - returns, -self.clip, 0) * masks).detach()
      value_loss = th.sum(weights * values * delta) / num_samples
      total_loss = policy_loss + 0.5 * self.w_value * value_loss
      self.policy.optimizer.zero_grad(); total_loss.backward()
      th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
      self.policy.optimizer.step(); self.updates +=1
      self.buffer.update_priorities(idx, clipped_advantages.squeeze(1).cpu().numpy())
    if self.writer == None or not self.should_eval(): return 
    self.writer.add_scalar('SIL/Mean Advantage', np.mean(mean_adv), self.num_timesteps)
    self.writer.add_scalar('SIL/Valid Samples', np.mean(num_valid_samples), self.num_timesteps)
    self.writer.add_scalar('SIL/Episodes', len(self.buffer.total_rewards), self.num_timesteps)
    self.writer.add_scalar('SIL/Best Reward', np.max(self.buffer.total_rewards) if len(self.buffer.total_rewards) > 0 else 0, self.num_timesteps)
    self.writer.add_scalar('SIL/Num Steps',  len(self.buffer), self.num_timesteps)
    self.writer.add_scalar('SIL/Updatess', self.updates, self.num_timesteps)


class SILCallback(BaseCallback):      
  def _on_step(self) -> bool: 
    l = lambda name: self.locals.get(name)
    obs, actions, rewards, dones = self.model._last_obs, l('actions'), l('rewards'), l('dones')
    for i, ep in enumerate(self.model.running_episodes):
      ep.append([obs[i], actions[i], rewards[i]])
      if dones[i]: self.model.buffer.update(ep, self.model.gamma); self.model.running_episodes[i] = []
    return True
